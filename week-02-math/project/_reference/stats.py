"""Ngày 3 · Xác suất & thông tin: softmax ổn định, entropy, KL, khoảng tin cậy."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as st


def logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """log Σ exp(a) theo `axis`, không tràn số: m + log Σ exp(a − m)."""
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)          # cả hàng là -inf thì không trừ
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return z - np.expand_dims(logsumexp(z, axis=axis), axis)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.exp(log_softmax(z, axis=axis))


def _check(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if np.any(p < 0) or not math.isclose(p.sum(), 1.0, abs_tol=1e-9):
        raise ValueError("không phải phân phối xác suất")
    return p


def entropy(p, base: float = math.e) -> float:
    """H(p) = −Σ p log p, quy ước 0·log 0 = 0."""
    p = _check(p)
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum() / math.log(base))


def cross_entropy(p, q, base: float = math.e) -> float:
    """H(p, q) = −Σ p log q. Bằng +inf nếu q(x) = 0 tại x có p(x) > 0."""
    p, q = _check(p), _check(q)
    nz = p > 0
    if np.any(q[nz] == 0):
        return math.inf
    return float(-(p[nz] * np.log(q[nz])).sum() / math.log(base))


def kl_divergence(p, q, base: float = math.e) -> float:
    """KL(p‖q) = Σ p log(p/q) = H(p, q) − H(p) ≥ 0."""
    p, q = _check(p), _check(q)
    nz = p > 0
    if np.any(q[nz] == 0):
        return math.inf
    return float((p[nz] * np.log(p[nz] / q[nz])).sum() / math.log(base))


def mean_ci(samples, level: float = 0.95) -> tuple[float, float, float]:
    """(trung bình, cận dưới, cận trên) theo phân phối t với n − 1 bậc tự do."""
    x = np.asarray(samples, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("cần ít nhất 2 mẫu")
    m = float(x.mean())
    half = float(st.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / math.sqrt(n))
    return m, m - half, m + half


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy Wilson cho tỉ lệ k/n. Vẫn dùng được khi k = 0 hoặc k = n."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("cần 0 <= k <= n và n > 0")
    ph = k / n
    den = 1 + z * z / n
    center = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)
