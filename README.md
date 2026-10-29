# English–Hindi machine translation: four architectures

**ICT4403 Assignment 3.** Implement and compare a Transformer trained from scratch, fully fine-tuned mT5-small, and LoRA-adapted Llama 3.2 1B Instruct and Gemma 3 270M IT. All four models were trained and evaluated on the same held-out English–Hindi test examples. The measured outputs are in [`results/`](results/); a detailed discussion is in [`ANALYSIS.md`](ANALYSIS.md).

## Dataset and provenance

- **IIT Bombay English–Hindi Parallel Corpus**, by Kunchukuttan, Mehta, and Bhattacharyya (LREC 2018): https://github.com/cfiltnlp/IITB-English-Hindi-PC
- Dataset loader and existing train/validation/test splits: https://huggingface.co/datasets/cfilt/iitb-english-hindi
- Corpus license: CC BY-NC 4.0, as indexed by OPUS: https://opus.nlpl.eu/datasets/IITB . Follow the source's attribution and noncommercial restrictions; do not reupload the corpus to GitHub.

The loader exposes `translation.en` and `translation.hi`. `prepare` reserves the first 200 eligible official test and 200 eligible official validation examples, then samples 5,000 shuffled training examples with seed 4403. It normalizes Unicode to NFC, collapses whitespace, drops empty or >600-character examples, and removes repeated English sources across all three splits. Actual counts are in [`results/data_manifest.json`](results/data_manifest.json). The first 200 test examples are clustered by source news stories, which limits generalization.

## Models and method

| Model | Architecture | Adaptation | Tokenizer |
| --- | --- | --- | --- |
| Small Transformer | encoder–decoder, 3 layers each, 256 hidden, 4 heads | random initialization, supervised training | joint SentencePiece unigram, trained on training data only |
| `google/mt5-small` | encoder–decoder | full supervised fine-tuning | pretrained SentencePiece |
| `meta-llama/Llama-3.2-1B-Instruct` | decoder-only | rank-8 LoRA on query/value projections | pretrained tokenizer and chat template |
| `google/gemma-3-270m-it` | decoder-only | rank-8 LoRA on query/value projections | pretrained tokenizer and chat template |

Official model pages: https://huggingface.co/google/mt5-small ; https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct ; https://huggingface.co/google/gemma-3-270m-it . **Accept the licenses and get access to both gated models through your Hugging Face account before running.** Llama lists Hindi among its supported languages. Gemma's small size is useful for a constrained run, but model size and prior training differ substantially, so architecture alone cannot explain any score gap.

## Run on a CUDA GPU

Use a GPU runtime with enough disk for the three pretrained model downloads and the dataset (plan several GB). In Colab, choose **Runtime → Change runtime type → GPU**, upload/extract this project, and run these commands in notebook cells with a leading `!` (or in a terminal without `!`):

```bash
cd en_hi_mt_assignment
python -m pip install -r requirements.txt
hf auth login
python run.py prepare
python run.py train --model transformer
python run.py train --model mt5
python run.py train --model llama
python run.py train --model gemma
python run.py report
```

If `hf` is unavailable after installation, use `huggingface-cli login`. The recorded runs used 300 optimizer steps for Transformer/Llama/Gemma and **900 steps at learning rate 0.0003 for mT5** (3 epochs, batch size 4, accumulation 4). To reproduce the recorded mT5 run, replace its command with `python run.py train --model mt5 --epochs 3 --max-steps 900 --lr 0.0003`. All others used defaults. Run one model at a time in a fresh process. On Colab with PEFT and preinstalled `torchao 0.10.0`, `pip uninstall -y torchao` fixes the reported incompatible-version error. The tokenizer and model downloads require an unrestricted internet connection.

For a CPU-only *syntax check*: `python -m py_compile run.py transformer_model.py`. Training is intentionally stopped early on non-CUDA devices because the full comparison would be impractically slow.

## Outputs and evaluation

- `data/*.json`: filtered splits and manifest. Keep these local; the corpus is retrieved from the official source.
- `checkpoints/`: trained weights/adapters and joint tokenizer. These files may be large; do not upload them unless permitted and necessary.
- `results/<model>_metrics.json`: BLEU, chrF, total/trainable parameter counts, training time, per-sentence inference time, hardware, and hyperparameters.
- `results/<model>_translations.csv`: **aligned** source, reference, and translation for each common test example.
- `results/<model>_temperature.json`: sampled translations for the first ten test examples at temperatures 0.3, 0.7, and 1.0.
- `results/comparison.md` and `results/scores.png`: measured comparison and plot. Running `python run.py report` regenerates `comparison.md` and a `chrf.png` plot.

All main scores use greedy generation, 96 new-token maximum, one example at a time, and identical test rows. BLEU and chrF come from SacreBLEU on corpus-level predictions. Inference time includes tokenization and generation, excludes loading, and synchronizes CUDA around each sentence. Training time excludes model download and includes optimization only; validation time is separate. Temperature outputs are a qualitative sensitivity analysis and **must not be compared to deterministic test scores as if they used the same decoding**. Read the outputs before drawing conclusions about hallucination or bias.

The same sentence limit is used, but model tokenizers and parameter counts differ. The from-scratch model has no pretrained language knowledge, mT5 is fully updated, and the decoder-only models update LoRA weights. Report those confounders in any architecture comparison. Record the GPU and any out-of-memory adjustments. Checkpoints must correspond to the reported run.

## Measured results (Tesla T4)

| Model | BLEU | chrF | Train min | Inference s/sentence | Total params M | Trainable params M | Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer | 0.08 | 5.09 | 0.62 | 0.46 | 7.58 | 7.58 | 300 |
| mT5-small | 0.50 | 9.46 | 11.85 | 0.84 | 300.18 | 300.18 | 900 |
| Llama 3.2 1B Instruct | **8.87** | **33.89** | 30.36 | 1.54 | 1235.81 | 0.85 | 300 |
| Gemma 3 270M IT | 6.25 | 25.92 | 5.78 | 1.13 | 268.10 | 0.37 | 300 |

**Best quality in this experiment: Llama**, based on both metrics and inspected examples. **Lower compute among the two decoder-only systems: Gemma**, with lower parameter count and shorter measured training/inference times. The encoder–decoder systems performed poorly under the chosen limited training budgets; this does not establish that encoder–decoder architecture is generally worse. mT5 received three times as many optimization steps as the other models. See [`ANALYSIS.md`](ANALYSIS.md) for examples, temperature observations, and limitations.

## Qualitative review and responsible AI

The completed review is in [`ANALYSIS.md`](ANALYSIS.md), and the full source/reference/prediction samples are in each `results/<model>_translations.csv`. The inspection includes names, numbers, a cultural term, and sensitive news. The sampled temperature variants are in the JSON files.

Risks include stereotyped gender or social roles, omission or invention of entities, culturally inappropriate honorifics, and unsafe shifts in medical or legal meaning. Useful controls include representative bilingual review sets, checks for numbers/names/negation, human review for consequential translations, domain-specific testing, uncertainty warnings, and a way to report errors. Do not automatically publish personal or sensitive source text. The corpus is a mixture of domains and may not represent all dialects and communities.

## GitHub and LMS checklist

1. Create a repository such as `ict4403-english-hindi-mt` and upload this project folder. The source, actual results, plot, and analysis are included. `.gitignore` keeps model weights and source corpus out.
2. Verify that the repository works and all references are accessible. If private, give the evaluator access.
3. Create `YourName_RegNum_FGAI_Assignment3.docx` containing your name, registration number, assignment title, the system title above, the dataset name/link, and your **real repository URL**; export to PDF and upload that PDF to LMS.

The submission sheet calls the system “autoregressive text-generation”; this implementation follows the main question's machine-translation task. Its sampling-temperature request is handled by the separate sweep.
