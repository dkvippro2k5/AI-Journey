"""Ngày 1 · Đại số tuyến tính: chuẩn hóa, bình phương tối thiểu, điều kiện, PCA."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def add_bias(X: np.ndarray) -> np.ndarray:
    """Thêm cột 1 vào đầu: (n, d) -> (n, d+1)."""
    return np.column_stack([np.ones(len(X)), X])


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z = (X - mean) / std theo từng cột. Cột hằng (std = 0) được chia cho 1 thay vì 0."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nghiệm bình phương tối thiểu qua SVD (ổn định số học, chạy được cả khi X thiếu hạng)."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


def normal_equation(X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> np.ndarray:
    """Giải (XᵀX/n + l2·I) θ = Xᵀy/n bằng solve — KHÔNG tính nghịch đảo."""
    n, d = X.shape
    return np.linalg.solve(X.T @ X / n + l2 * np.eye(d), X.T @ y / n)


def condition_number(X: np.ndarray) -> float:
    """κ(X) = s_max / s_min (giá trị kỳ dị).

    Trả về inf nếu X thiếu hạng theo cùng ngưỡng với np.linalg.matrix_rank:
    s_min <= s_max · max(n, d) · eps của float64.
    """
    s = np.linalg.svd(X, compute_uv=False)
    if s[-1] <= s[0] * max(X.shape) * np.finfo(float).eps:
        return float(np.inf)
    return float(s[0] / s[-1])


def hessian_eigs_mse(X: np.ndarray) -> np.ndarray:
    """Trị riêng (tăng dần) của Hessian XᵀX/n của loss (1/2n)‖Xθ − y‖²."""
    return np.linalg.eigvalsh(X.T @ X / len(X))


@dataclass(frozen=True)
class PCAResult:
    mean: np.ndarray                      # (d,)
    components: np.ndarray                # (k, d), mỗi hàng là một trục chính, độ dài 1
    explained_variance: np.ndarray        # (k,), phương sai mẫu (chia n − 1)
    explained_variance_ratio: np.ndarray  # (k,)


def pca_fit(X: np.ndarray, k: int) -> PCAResult:
    """PCA qua SVD của X đã trừ trung bình. Dấu: phần tử có |giá trị| lớn nhất của mỗi trục là dương."""
    n, d = X.shape
    if not 1 <= k <= min(n, d):
        raise ValueError(f"k phải trong [1, {min(n, d)}]")
    mean = X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    var = s ** 2 / (n - 1)
    comps = Vt[:k].copy()
    idx = np.argmax(np.abs(comps), axis=1)
    comps *= np.sign(comps[np.arange(k), idx])[:, None]
    return PCAResult(mean, comps, var[:k], var[:k] / var.sum())


def pca_transform(res: PCAResult, X: np.ndarray) -> np.ndarray:
    """Chiếu X lên k trục chính: (n, d) -> (n, k)."""
    return (X - res.mean) @ res.components.T
