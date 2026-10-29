"""Small trainable encoder-decoder Transformer with joint SentencePiece vocabulary."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, dim=256, max_len=512):
        super().__init__()
        p = torch.arange(max_len).unsqueeze(1)
        d = torch.arange(0, dim, 2)
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(p / (10000 ** (d / dim)))
        pe[:, 1::2] = torch.cos(p / (10000 ** (d / dim)))
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


class ScratchTransformer(nn.Module):
    def __init__(self, vocab_size, dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos = PositionalEncoding(dim)
        self.core = nn.Transformer(d_model=dim, nhead=4, num_encoder_layers=3,
            num_decoder_layers=3, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, src, tgt, labels=None):
        # During training tgt includes BOS + gold tokens. Shift decoder input by one.
        dec = tgt[:, :-1] if labels is not None else tgt
        mask = nn.Transformer.generate_square_subsequent_mask(dec.shape[1], device=dec.device)
        hidden = self.core(self.pos(self.embed(src) * math.sqrt(self.embed.embedding_dim)),
            self.pos(self.embed(dec) * math.sqrt(self.embed.embedding_dim)),
            tgt_mask=mask, src_key_padding_mask=(src == 0),
            tgt_key_padding_mask=(dec == 0), memory_key_padding_mask=(src == 0))
        logits = self.out(hidden)
        if labels is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
        return type("Output", (), {"loss": loss})()


def batchify(rows, sp):
    src = [[sp.bos_id()] + sp.encode(r["en"])[:126] + [sp.eos_id()] for r in rows]
    tgt = [[sp.bos_id()] + sp.encode(r["hi"])[:126] + [sp.eos_id()] for r in rows]
    def pad(seqs, value):
        width = max(map(len, seqs))
        return torch.tensor([s + [value] * (width - len(s)) for s in seqs])
    target = pad(tgt, sp.pad_id())
    labels = target.clone(); labels[labels == sp.pad_id()] = -100
    return {"src": pad(src, sp.pad_id()), "tgt": target, "labels": labels}
