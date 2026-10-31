"""Dữ liệu tổng hợp cho các thí nghiệm (cho sẵn, không cần sửa)."""
from __future__ import annotations

import numpy as np


def make_regression(n: int = 500, d: int = 2, *, scales=None, offsets=None, noise: float = 0.1,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """y = 1 + X @ w + nhiễu Gauss.

    Cột j của X có độ lệch chuẩn `scales[j]` và trung bình `offsets[j]`. Đặt scales khác nhau
    để tạo bài toán có điều kiện kém (ill-conditioned).
    Trả về (X, y, theta_true), trong đó theta_true = [bias, w...] ứng với add_bias(X).
    """
    rng = np.random.default_rng(seed)
    scales = np.ones(d) if scales is None else np.asarray(scales, dtype=float)
    offsets = np.zeros(d) if offsets is None else np.asarray(offsets, dtype=float)
    X = rng.standard_normal((n, d)) * scales + offsets
    w = np.linspace(1.0, -1.0, d) / scales           # mọi feature đóng góp cùng cỡ vào y
    y = 1.0 + X @ w + noise * rng.standard_normal(n)
    return X, y, np.concatenate([[1.0], w])


def make_classification(n: int = 400, *, sep: float = 1.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Hai cụm Gauss 2 chiều, tâm (±sep/2, ±sep/2), nhãn 0/1 (float)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(float)
    centers = np.where(y[:, None] == 1, sep / 2, -sep / 2)
    X = centers + rng.standard_normal((n, 2))
    return X, y


def rosenbrock(theta: np.ndarray, X=None, y=None, a: float = 1.0, b: float = 100.0) -> tuple[float, np.ndarray]:
    """f(x, y) = (a - x)^2 + b (y - x^2)^2, cực tiểu tại (a, a^2). Nhận X, y để khớp chữ ký loss_fn."""
    x0, x1 = float(theta[0]), float(theta[1])
    f = (a - x0) ** 2 + b * (x1 - x0 ** 2) ** 2
    g = np.array([-2 * (a - x0) - 4 * b * x0 * (x1 - x0 ** 2), 2 * b * (x1 - x0 ** 2)])
    return f, g
