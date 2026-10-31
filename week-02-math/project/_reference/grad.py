"""Ngày 2 · Gradient: kiểm tra bằng sai phân, loss MSE và logistic kèm gradient."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def numerical_grad(f: Callable[[np.ndarray], float], theta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Gradient bằng sai phân trung tâm (sai số O(eps²)). Không làm thay đổi theta."""
    theta = np.array(theta, dtype=float)          # bản sao riêng
    g = np.zeros_like(theta)
    for i in range(theta.size):
        old = theta.flat[i]
        theta.flat[i] = old + eps
        fp = f(theta)
        theta.flat[i] = old - eps
        fm = f(theta)
        theta.flat[i] = old
        g.flat[i] = (fp - fm) / (2 * eps)
    return g


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    """‖a − b‖ / (‖a‖ + ‖b‖), có mẫu số tối thiểu để tránh chia cho 0."""
    return float(np.linalg.norm(a - b) / max(1e-12, np.linalg.norm(a) + np.linalg.norm(b)))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """σ(z) không bị tràn số với |z| lớn."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def mse_loss(theta: np.ndarray, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> tuple[float, np.ndarray]:
    """L = (1/2n)‖Xθ − y‖² + (l2/2)‖θ‖²,  ∇L = Xᵀ(Xθ − y)/n + l2·θ."""
    n = len(y)
    r = X @ theta - y
    loss = 0.5 * float(r @ r) / n + 0.5 * l2 * float(theta @ theta)
    return loss, X.T @ r / n + l2 * theta


def logistic_loss(theta: np.ndarray, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> tuple[float, np.ndarray]:
    """BCE trung bình tính từ logit z = Xθ:  mean(log(1 + e^z) − y·z) + (l2/2)‖θ‖².

    ∇L = Xᵀ(σ(z) − y)/n + l2·θ. Dùng logaddexp để không tràn số khi |z| lớn.
    """
    z = X @ theta
    loss = float(np.mean(np.logaddexp(0.0, z) - y * z)) + 0.5 * l2 * float(theta @ theta)
    return loss, X.T @ (sigmoid(z) - y) / len(y) + l2 * theta
