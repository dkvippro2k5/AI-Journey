"""Ngày 2 · Optimizer: SGD, Momentum, RMSProp, Adam, gradient clipping, lịch learning rate."""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np


class Optimizer(Protocol):
    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray: ...


class SGD:
    def __init__(self, lr: float):
        self.lr = lr

    def step(self, theta, grad):
        return theta - self.lr * grad


class Momentum:
    """v ← βv + g;  θ ← θ − lr·v  (cách viết của PyTorch)."""

    def __init__(self, lr: float, beta: float = 0.9):
        self.lr, self.beta, self.v = lr, beta, None

    def step(self, theta, grad):
        self.v = grad.copy() if self.v is None else self.beta * self.v + grad
        return theta - self.lr * self.v


class RMSProp:
    """s ← ρs + (1 − ρ)g²;  θ ← θ − lr·g / (√s + eps)."""

    def __init__(self, lr: float, rho: float = 0.9, eps: float = 1e-8):
        self.lr, self.rho, self.eps, self.s = lr, rho, eps, None

    def step(self, theta, grad):
        if self.s is None:
            self.s = np.zeros_like(grad)
        self.s = self.rho * self.s + (1 - self.rho) * grad ** 2
        return theta - self.lr * grad / (np.sqrt(self.s) + self.eps)


class Adam:
    """Kingma & Ba (2015), Algorithm 1, có hiệu chỉnh độ lệch (bias correction)."""

    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, theta, grad):
        if self.m is None:
            self.m, self.v = np.zeros_like(grad), np.zeros_like(grad)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def make_optimizer(name: str, lr: float) -> Optimizer:
    table = {"sgd": SGD, "momentum": Momentum, "rmsprop": RMSProp, "adam": Adam}
    if name not in table:
        raise ValueError(f"optimizer không hỗ trợ: {name!r}")
    return table[name](lr)


def clip_by_norm(grad: np.ndarray, max_norm: float) -> np.ndarray:
    """Co g về độ dài max_norm nếu ‖g‖ > max_norm, giữ nguyên hướng. Không sửa mảng đầu vào."""
    norm = float(np.linalg.norm(grad))
    return grad * (max_norm / norm) if norm > max_norm else grad.copy()


def cosine_lr(step: int, total: int, base_lr: float, warmup: int = 0, min_lr: float = 0.0) -> float:
    """Warmup tuyến tính trong `warmup` bước đầu, sau đó giảm theo cosine từ base_lr về min_lr ở bước `total`."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
