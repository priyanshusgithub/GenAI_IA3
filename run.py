"""English-to-Hindi MT experiment: four architectures, common test set.

Run `python run.py --help`. Requires a CUDA GPU for the default training run.
Each model is trained/evaluated in a separate process to release GPU memory.
"""
import argparse
import csv
import hashlib
import json
import random
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"
OUT = ROOT / "results"
MODELS = ("transformer", "mt5", "llama", "gemma")
MODEL_IDS = {"mt5": "google/mt5-small", "llama": "meta-llama/Llama-3.2-1B-Instruct", "gemma": "google/gemma-3-270m-it"}
SEED = 4403


def seed_all(seed=SEED):
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(s):
    return " ".join(unicodedata.normalize("NFC", s).split())


def key(s):
    return hashlib.sha256(s.casefold().encode("utf-8")).hexdigest()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(args):
    from datasets import load_dataset
    DATA.mkdir(exist_ok=True)
    # Official splits; only training split is shuffled/subsampled.
    ds = load_dataset("cfilt/iitb-english-hindi")
    seen_src, seen_pair = set(), set()

    def clean(split, limit, shuffle=False):
        rows = ds[split].shuffle(seed=SEED) if shuffle else ds[split]
        cleaned = []
        for item in rows:
            en, hi = (normalize(item["translation"].get(x, "")) for x in ("en", "hi"))
            if not en or not hi or len(en) > 600 or len(hi) > 600:
                continue
            sig = (key(en), key(hi))
            # Remove repeated sources as well as exact duplicate pairs across all splits.
            if sig[0] in seen_src or sig in seen_pair:
                continue
            seen_src.add(sig[0]); seen_pair.add(sig)
            cleaned.append({"en": en, "hi": hi})
            if len(cleaned) == limit:
                break
        return cleaned

    # Reserve validation and test before sampling training to avoid leakage.
    test = clean("test", args.test_size)
    val = clean("validation", args.val_size)
    train = clean("train", args.train_size, shuffle=True)
    if min(len(train), len(val), len(test)) == 0:
        raise RuntimeError("A split is empty after filtering")
    for name, rows in (("train", train), ("validation", val), ("test", test)):
        save_json(DATA / f"{name}.json", rows)
    save_json(DATA / "manifest.json", {"dataset": "cfilt/iitb-english-hindi", "seed": SEED,
        "sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "filter": "NFC, whitespace collapse, nonempty, <=600 chars, unique English source across splits"})
    print(read_json(DATA / "manifest.json"))


def load_split(name):
    return read_json(DATA / f"{name}.json")


def device_and_dtype():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the full experiment. Use a GPU runtime, then rerun.")
    return torch.device("cuda"), (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


def iter_batches(rows, size, shuffle=False):
    ids = list(range(len(rows)))
    if shuffle:
        random.shuffle(ids)
    for i in range(0, len(ids), size):
        yield [rows[k] for k in ids[i:i + size]]


def train_loop(model, train, val, collate, args, device, mode):
    import torch
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    # mT5 is kept in FP32 on GPUs without BF16 to avoid FP16 instability.
    use_amp = not (mode == "mt5" and args.fp16)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and use_amp)
    model.train(); optimizer.zero_grad(set_to_none=True)
    steps = 0; micro = 0; start = time.perf_counter(); losses = []
    for epoch in range(args.epochs):
        for rows in iter_batches(train, args.batch_size, shuffle=True):
            batch = {k: v.to(device) for k, v in collate(rows).items()}
            with torch.autocast("cuda", dtype=torch.float16 if args.fp16 else torch.bfloat16, enabled=use_amp):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss for {mode}")
            scaler.scale(loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                steps += 1; losses.append(float(loss.detach()))
                if steps % 25 == 0:
                    print(f"{mode}: step {steps}, loss {sum(losses[-25:])/min(25,len(losses)):.3f}", flush=True)
                if steps >= args.max_steps:
                    break
        if steps >= args.max_steps:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    # Validation loss with the same adaptation; no test data used for selection.
    model.eval(); vloss = []
    with torch.inference_mode():
        for rows in iter_batches(val, args.batch_size):
            batch = {k: v.to(device) for k, v in collate(rows).items()}
            with torch.autocast("cuda", dtype=torch.float16 if args.fp16 else torch.bfloat16, enabled=use_amp):
                vloss.append(float(model(**batch).loss))
    return {"train_seconds": elapsed, "steps": steps, "train_loss_last_25": sum(losses[-25:]) / max(1, len(losses[-25:])), "validation_loss": sum(vloss)/len(vloss)}


def make_prompt(tokenizer, en):
    message = [{"role": "user", "content": "Translate the following English text into Hindi. Output only the Hindi translation.\nEnglish: " + en}]
    return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)


def run_hf(args, train, val, test, device, dtype):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model
    name = args.model; mid = MODEL_IDS[name]; seq = name == "mt5"
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model_cls = AutoModelForSeq2SeqLM if seq else AutoModelForCausalLM
    model = model_cls.from_pretrained(mid, torch_dtype=torch.float32 if seq and args.fp16 else dtype).to(device)
    model.config.use_cache = False
    total = sum(p.numel() for p in model.parameters())
    if not seq:
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"], bias="none", task_type=TaskType.CAUSAL_LM))
    turn_end = tok.convert_tokens_to_ids("<|eot_id|>" if name == "llama" else "<end_of_turn>") if not seq else None
    if turn_end is None or turn_end == tok.unk_token_id:
        turn_end = tok.eos_token_id
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def collate(rows):
        if seq:
            x = tok(["translate English to Hindi: " + r["en"] for r in rows],
                padding=True, truncation=True, max_length=128, return_tensors="pt")
            y = tok(text_target=[r["hi"] for r in rows], padding=True,
                truncation=True, max_length=128, return_tensors="pt")["input_ids"]
            y[y == tok.pad_token_id] = -100
            x["labels"] = y
            return x
        enc = []; labs = []
        for r in rows:
            prefix = tok(make_prompt(tok, r["en"]), add_special_tokens=False)["input_ids"]
            target = tok(r["hi"], add_special_tokens=False)["input_ids"][:100] + [turn_end]
            prefix = prefix[-(256 - len(target)):]
            enc.append(prefix + target)
            labs.append([-100] * len(prefix) + target)
        width = max(map(len, enc))
        # Right padding for training; generation uses single unpadded prompts.
        return {"input_ids": torch.tensor([a + [tok.pad_token_id]*(width-len(a)) for a in enc]),
            "attention_mask": torch.tensor([[1]*len(a)+[0]*(width-len(a)) for a in enc]),
            "labels": torch.tensor([a+[-100]*(width-len(a)) for a in labs])}

    metrics = train_loop(model, train, val, collate, args, device, name)
    path = CKPT / name; path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path); tok.save_pretrained(path)
    model.config.use_cache = True
    model.eval()
    def translate(en, temperature=None):
        prompt = "translate English to Hindi: " + en if seq else make_prompt(tok, en)
        x = tok(prompt, return_tensors="pt", truncation=True, max_length=160, add_special_tokens=seq).to(device)
        options = {"max_new_tokens": 96, "do_sample": temperature is not None,
            "pad_token_id": tok.pad_token_id}
        if not seq:
            options["eos_token_id"] = [tok.eos_token_id, turn_end]
        if temperature is not None:
            options["temperature"] = temperature
        with torch.inference_mode():
            out = model.generate(**x, **options)[0]
        return tok.decode(out if seq else out[x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return metrics, total, trainable, translate


def run_transformer(args, train, val, test, device, dtype):
    import torch
    import torch.nn as nn
    import sentencepiece as spm
    from transformer_model import ScratchTransformer, batchify
    CKPT.mkdir(exist_ok=True)
    corpus = DATA / "tokenizer_corpus.txt"
    with corpus.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(r["en"] + "\n" + r["hi"] + "\n")
    prefix = str(CKPT / "joint")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=prefix, vocab_size=4000,
        model_type="unigram", character_coverage=0.9995, hard_vocab_limit=False,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3, num_threads=2)
    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    model = ScratchTransformer(sp.vocab_size()).to(device)
    total = sum(p.numel() for p in model.parameters())
    collate = lambda rows: batchify(rows, sp)
    metrics = train_loop(model, train, val, collate, args, device, "transformer")
    path = CKPT / "transformer.pt"
    torch.save({"state_dict": model.state_dict(), "vocab_size": sp.vocab_size()}, path)
    model.eval()
    def translate(en, temperature=None):
        src = torch.tensor([[sp.bos_id()] + sp.encode(en)[:126] + [sp.eos_id()]], device=device)
        generated = [sp.bos_id()]
        with torch.inference_mode():
            for _ in range(96):
                tgt = torch.tensor([generated], device=device)
                logits = model(src, tgt)[:, -1, :]
                if temperature is None:
                    nxt = int(logits.argmax(-1).item())
                else:
                    nxt = int(torch.multinomial(torch.softmax(logits / temperature, -1), 1).item())
                if nxt == sp.eos_id():
                    break
                generated.append(nxt)
        return sp.decode(generated[1:]).strip()
    return metrics, total, total, translate


def train(args):
    import torch
    seed_all(); device, dtype = device_and_dtype()
    train_rows, val, test = (load_split(x) for x in ("train", "validation", "test"))
    args.fp16 = dtype == torch.float16
    metrics, total, trainable, translate = (run_transformer if args.model == "transformer" else run_hf)(
        args, train_rows, val, test, device, dtype)
    preds = []; timings = []
    for i, row in enumerate(test):
        torch.cuda.synchronize(); t = time.perf_counter()
        prediction = translate(row["en"])
        torch.cuda.synchronize(); timings.append(time.perf_counter()-t)
        preds.append({"id": i, "source": row["en"], "reference": row["hi"], "prediction": prediction})
        if (i+1) % 25 == 0:
            print(f"{args.model}: translated {i+1}/{len(test)}", flush=True)
    import sacrebleu
    refs = [r["reference"] for r in preds]; hypotheses = [r["prediction"] for r in preds]
    metrics.update({"model": args.model, "model_id": MODEL_IDS.get(args.model, "scratch encoder-decoder"),
        "architecture": "encoder-decoder" if args.model in ("transformer", "mt5") else "decoder-only",
        "adaptation": "from scratch" if args.model == "transformer" else ("full fine-tune" if args.model == "mt5" else "LoRA"),
        "parameters_total": total, "parameters_trainable": trainable, "test_size": len(test),
        "bleu": sacrebleu.corpus_bleu(hypotheses, [refs]).score,
        "chrf": sacrebleu.corpus_chrf(hypotheses, [refs]).score,
        "inference_seconds_total": sum(timings), "inference_seconds_per_sentence": sum(timings)/len(timings),
        "gpu": torch.cuda.get_device_name(0), "seed": SEED, "max_new_tokens": 96,
        "decoding": "greedy", "epochs": args.epochs, "max_steps": args.max_steps,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum, "lr": args.lr,
        "training_examples": len(train_rows), "validation_examples": len(val)})
    OUT.mkdir(exist_ok=True)
    save_json(OUT / f"{args.model}_metrics.json", metrics)
    with (OUT / f"{args.model}_translations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "source", "reference", "prediction"])
        writer.writeheader(); writer.writerows(preds)
    # A separate reproducible temperature sweep on the first fixed test examples.
    samples = []
    for temp in (0.3, 0.7, 1.0):
        seed_all()
        for i, row in enumerate(test[:min(10, len(test))]):
            samples.append({"id": i, "source": row["en"], "reference": row["hi"],
                "temperature": temp, "prediction": translate(row["en"], temperature=temp)})
    save_json(OUT / f"{args.model}_temperature.json", samples)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def report(args):
    OUT.mkdir(exist_ok=True)
    missing = [m for m in MODELS if not (OUT / f"{m}_metrics.json").exists()]
    if missing:
        raise SystemExit("Run training for all four models first. Missing: " + ", ".join(missing))
    metrics = [read_json(OUT / f"{m}_metrics.json") for m in MODELS]
    manifest = read_json(DATA / "manifest.json")
    sizes = {m["test_size"] for m in metrics}
    if sizes != {manifest["sizes"]["test"]}:
        raise RuntimeError("Test set sizes differ; comparison aborted")
    headers = ["Model", "Architecture", "BLEU", "chrF", "Train min", "Infer s/sentence", "Params M", "Trainable M"]
    rows = [[m["model"], m["architecture"], f'{m["bleu"]:.2f}', f'{m["chrf"]:.2f}',
        f'{m["train_seconds"]/60:.2f}', f'{m["inference_seconds_per_sentence"]:.3f}',
        f'{m["parameters_total"]/1e6:.1f}', f'{m["parameters_trainable"]/1e6:.1f}'] for m in metrics]
    md = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"]*len(headers)) + "|\n"
    md += "\n".join("| " + " | ".join(row) + " |" for row in rows) + "\n"
    (OUT / "comparison.md").write_text(md, encoding="utf-8")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([m["model"] for m in metrics], [m["chrf"] for m in metrics])
    ax.set_ylabel("chrF (higher is better)"); ax.set_title("English to Hindi: common test set")
    fig.tight_layout(); fig.savefig(OUT / "chrf.png", dpi=180); plt.close(fig)
    print(md)
    print("Inspect translations and temperature JSON before writing qualitative conclusions.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--train-size", type=int, default=5000)
    prep.add_argument("--val-size", type=int, default=200)
    prep.add_argument("--test-size", type=int, default=200)
    tr = sub.add_parser("train")
    tr.add_argument("--model", choices=MODELS, required=True)
    tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--max-steps", type=int, default=300)
    tr.add_argument("--batch-size", type=int, default=4)
    tr.add_argument("--grad-accum", type=int, default=4)
    tr.add_argument("--lr", type=float, default=None)
    sub.add_parser("report")
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        args.lr = args.lr or (3e-4 if args.model in ("transformer", "llama", "gemma") else 5e-5)
        train(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
