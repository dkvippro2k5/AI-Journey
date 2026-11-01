"""Ngày 3 · Chọn token tiếp theo: greedy, temperature, top-k, top-p (nucleus)."""
from __future__ import annotations

import torch


def softmax_t(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """softmax(logits / T). T < 1: nhọn hơn; T > 1: phẳng hơn."""
    return torch.softmax(logits / temperature, dim=-1)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Giữ các logit ≥ logit lớn thứ k (hòa ở ranh giới thì giữ cả), còn lại đặt -inf. Chiều cuối là vocab."""
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (Holtzman et al. 2020): giữ TẬP NHỎ NHẤT các token xác suất cao nhất có tổng xác suất ≥ p.
    Token xác suất cao nhất luôn được giữ."""
    sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
    cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cum - torch.softmax(sorted_logits, dim=-1) >= p        # tổng TRƯỚC token này đã đủ p → bỏ
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """logits (..., V) → chỉ số token (...). temperature = 0 nghĩa là greedy (argmax).
    Thứ tự: chia temperature → top-k → top-p → lấy mẫu."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    return torch.multinomial(flat, 1, generator=generator).reshape(probs.shape[:-1])


@torch.no_grad()
def generate(step_fn, ids: list[int], max_new_tokens: int, eos_id: int | None = None, **sampling) -> list[int]:
    """step_fn(list id) → logits (V,) cho token kế tiếp. Sinh tới khi gặp eos_id hoặc đủ max_new_tokens."""
    ids = list(ids)
    for _ in range(max_new_tokens):
        nxt = int(sample_next(step_fn(ids), **sampling))
        ids.append(nxt)
        if nxt == eos_id:
            break
    return ids
