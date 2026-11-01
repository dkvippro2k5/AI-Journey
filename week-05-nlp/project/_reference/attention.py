"""Ngày 2 · Multi-head attention, mask, positional encoding (sinusoidal, RoPE)."""
from __future__ import annotations

import math

import torch
from torch import nn


def causal_mask(T: int) -> torch.Tensor:
    """(T, T) bool, True = ĐƯỢC nhìn: vị trí t chỉ thấy 0..t."""
    return torch.ones(T, T, dtype=torch.bool).tril()


def padding_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """(B, T) bool, True = token thật."""
    return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]


def sinusoidal_pe(T: int, d: int) -> torch.Tensor:
    """PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(...). Shape (T, d), d chẵn."""
    pos = torch.arange(T, dtype=torch.float32)[:, None]
    freq = torch.exp(-math.log(10000.0) * torch.arange(0, d, 2, dtype=torch.float32) / d)
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


def apply_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """RoPE kiểu xen kẽ (Su et al. 2021): xoay cặp (x[2i], x[2i+1]) ở vị trí t một góc t·θ_i, θ_i = base^(-2i/d).
    x có shape (..., T, d), d chẵn."""
    T, d = x.shape[-2], x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d)
    ang = torch.arange(T, dtype=torch.float32, device=x.device)[:, None] * theta     # (T, d/2)
    cos, sin = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class MultiHeadAttention(nn.Module):
    """softmax(Q Kᵀ / √d_head) V cho h head song song, rồi nối lại và chiếu bằng out_proj.

    mask: bool, True = được nhìn, broadcast được tới (B, h, T_q, T_k). Ví dụ (T_q, T_k) cho causal,
    hoặc (B, 1, 1, T_k) cho padding.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        self.h, self.d_head = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:              # (B, T, d) → (B, h, T, d_head)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_head).transpose(1, 2)

    def forward(self, x: torch.Tensor, kv: torch.Tensor | None = None, mask: torch.Tensor | None = None):
        kv = x if kv is None else kv
        q, k, v = self._split(self.q_proj(x)), self._split(self.k_proj(kv)), self._split(self.v_proj(kv))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)     # (B, h, T_q, T_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=-1)
        out = self.drop(w) @ v                                        # (B, h, T_q, d_head)
        B, _, T, _ = out.shape
        out = out.transpose(1, 2).reshape(B, T, self.h * self.d_head)  # nối các head
        return self.out_proj(out), w
