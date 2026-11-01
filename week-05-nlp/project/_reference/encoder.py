"""Ngày 4 · Transformer encoder nhỏ (pre-LN) để phân loại intent, dùng MultiHeadAttention của Ngày 2."""
from __future__ import annotations

import torch
from torch import nn

from intent.attention import MultiHeadAttention, sinusoidal_pe


class TransformerBlock(nn.Module):
    """Pre-LN: x = x + Drop(MHA(LN(x))); x = x + Drop(FFN(LN(x))), FFN = Linear → GELU → Linear."""

    def __init__(self, d: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, h, dropout)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        a, _ = self.attn(self.ln1(x), mask=mask)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.ln2(x)))


class EncoderClassifier(nn.Module):
    """id token (B, T), đệm bằng pad_id → embedding + sinusoidal PE → n_layers block → LN → mean-pool có mask → logit."""

    def __init__(self, vocab_size: int, n_classes: int, d: int = 64, h: int = 4, n_layers: int = 2, d_ff: int = 128,
                 max_len: int = 64, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        self.register_buffer("pe", sinusoidal_pe(max_len, d), persistent=False)
        self.blocks = nn.ModuleList(TransformerBlock(d, h, d_ff, dropout) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)
        self.drop = nn.Dropout(dropout)

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        keep = ids != self.pad_id                                     # (B, T)
        x = self.drop(self.emb(ids) + self.pe[: ids.shape[1]])
        mask = keep[:, None, None, :]                                 # (B, 1, 1, T): mọi query nhìn mọi token thật
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln(x)
        w = keep.unsqueeze(-1).to(x.dtype)
        return (x * w).sum(1) / w.sum(1).clamp_min(1)                # mean-pool chỉ trên token thật

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(ids))
