"""Ngày 3 · Attention trước Transformer: dot-product (Luong), additive (Bahdanau), attention pooling."""
from __future__ import annotations

import math

import torch
from torch import nn


def dot_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None,
                  scale: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """q (..., Lq, d), k (..., Lk, d), v (..., Lk, dv) → (out (..., Lq, dv), weights (..., Lq, Lk)).

    score = q·k / sqrt(d) (nếu scale). `mask` kiểu bool, broadcast được tới (..., Lq, Lk); True = ĐƯỢC nhìn.
    """
    scores = q @ k.transpose(-2, -1)
    if scale:
        scores = scores / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))      # exp(-inf) = 0 → trọng số đúng bằng 0
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class AdditiveAttention(nn.Module):
    """Bahdanau (2015): score_j = vᵀ tanh(W_q q + W_k k_j); context = Σ_j α_j k_j."""

    def __init__(self, d_q: int, d_k: int, d_h: int):
        super().__init__()
        self.W_q = nn.Linear(d_q, d_h, bias=False)
        self.W_k = nn.Linear(d_k, d_h, bias=False)
        self.v = nn.Linear(d_h, 1, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor,
                mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """query (B, d_q), keys (B, L, d_k), mask (B, L) bool → (context (B, d_k), weights (B, L))."""
        scores = self.v(torch.tanh(self.W_q(query).unsqueeze(1) + self.W_k(keys))).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * keys).sum(1), weights


class AttnPool(nn.Module):
    """Gộp T token thành 1 vector bằng MỘT query học được: out = softmax(q·(W_k t)/sqrt(d)) · (W_v t)."""

    def __init__(self, d: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens (B, T, d) → (pooled (B, d), weights (B, T))."""
        q = self.query.expand(tokens.shape[0], 1, -1)              # (B, 1, d)
        out, w = dot_attention(q, self.W_k(tokens), self.W_v(tokens))
        return out.squeeze(1), w.squeeze(1)
