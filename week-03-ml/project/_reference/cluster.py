"""Ngày 3 · K-Means (thuật toán Lloyd + khởi tạo k-means++), viết bằng NumPy."""
from __future__ import annotations

import numpy as np


def _sqdist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Bình phương khoảng cách (n, k), dùng ‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b (Tuần 1, final B1)."""
    d = (X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * X @ C.T
    return np.maximum(d, 0.0)


def kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++: tâm đầu chọn đều; tâm tiếp theo chọn với xác suất ∝ D(x)², D = khoảng cách tới tâm gần nhất."""
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = _sqdist(X, np.array(centers)).min(1)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(len(X), 1 / len(X))
        centers.append(X[rng.choice(len(X), p=probs)])
    return np.array(centers, dtype=float)


def _lloyd(X, centers, max_iter, tol):
    for _ in range(max_iter):
        labels = _sqdist(X, centers).argmin(1)
        new = centers.copy()
        for j in range(len(centers)):
            members = X[labels == j]
            if len(members):
                new[j] = members.mean(0)
            else:                                         # cụm rỗng: đặt lại vào điểm xa tâm của nó nhất
                new[j] = X[_sqdist(X, centers).min(1).argmax()]
        shift = np.sqrt(((new - centers) ** 2).sum(1)).max()
        centers = new
        if shift <= tol:
            break
    labels = _sqdist(X, centers).argmin(1)
    inertia = float(_sqdist(X, centers)[np.arange(len(X)), labels].sum())
    return centers, labels, inertia


def kmeans(X, k: int, *, n_init: int = 10, max_iter: int = 300, tol: float = 1e-6, seed: int = 0, init=None):
    """Trả về (centers (k, d), labels (n,), inertia) của lần chạy có inertia nhỏ nhất.

    `init` (mảng (k, d)) thì chỉ chạy một lần từ đúng các tâm đó.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if not 1 <= k <= len(X):
        raise ValueError("cần 1 <= k <= n")
    if init is not None:
        return _lloyd(X, np.asarray(init, dtype=float).reshape(k, -1), max_iter, tol)
    rng = np.random.default_rng(seed)
    runs = [_lloyd(X, kmeans_pp_init(X, k, rng), max_iter, tol) for _ in range(n_init)]
    return min(runs, key=lambda r: r[2])
