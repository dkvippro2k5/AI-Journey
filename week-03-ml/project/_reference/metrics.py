"""Ngày 1 · Metric phân loại và chia fold, viết từ đầu (đối chiếu với scikit-learn trong test)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def confusion(y_true, y_pred) -> np.ndarray:
    """Ma trận nhầm lẫn nhị phân [[TN, FP], [FN, TP]] (cùng bố cục với sklearn)."""
    y, p = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred) -> tuple[float, float, float]:
    """Mẫu số bằng 0 thì metric tương ứng bằng 0 (giống zero_division=0 của sklearn)."""
    (_, fp), (fn, tp) = confusion(y_true, y_pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def roc_auc(y_true, score) -> float:
    """AUC = P(score của mẫu dương > score của mẫu âm), hòa tính 1/2. Tính qua tổng hạng (Mann–Whitney)."""
    y, s = np.asarray(y_true), np.asarray(score, dtype=float)
    n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("cần có cả mẫu dương và mẫu âm")
    ranks = rankdata(s)                                   # hạng trung bình khi hòa
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y_true, score) -> float:
    """AP = Σ (R_i − R_{i−1}) · P_i trên các ngưỡng là các giá trị score phân biệt (định nghĩa của sklearn)."""
    y, s = np.asarray(y_true), np.asarray(score, dtype=float)
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]   # vị trí cuối của mỗi nhóm score bằng nhau
    tp = np.cumsum(y == 1)[last]
    fp = np.cumsum(y == 0)[last]
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def best_threshold(y_true, score, cost_fp: float = 1.0, cost_fn: float = 1.0) -> tuple[float, float]:
    """Ngưỡng t (dự đoán 1 khi score >= t) có tổng chi phí nhỏ nhất.

    Ứng viên: mọi giá trị score phân biệt và +inf (không dự đoán dương nào).
    Hòa thì chọn ngưỡng LỚN hơn. Trả về (t, tổng chi phí).
    """
    y, s = np.asarray(y_true), np.asarray(score, dtype=float)
    best = (np.inf, cost_fn * float(np.sum(y == 1)))
    for t in np.unique(s)[::-1]:                          # từ lớn tới nhỏ → hòa giữ ngưỡng lớn hơn
        pred = s >= t
        cost = cost_fp * np.sum(pred & (y == 0)) + cost_fn * np.sum(~pred & (y == 1))
        if cost < best[1]:
            best = (float(t), float(cost))
    return best


def stratified_kfold(y, k: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Chia k fold giữ tỉ lệ lớp: xáo từng lớp, chia mỗi lớp thành k phần bằng np.array_split."""
    y = np.asarray(y)
    if k < 2:
        raise ValueError("k phải >= 2")
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for i, chunk in enumerate(np.array_split(idx, k)):
            parts[i].extend(chunk.tolist())
    all_idx = np.arange(len(y))
    folds = []
    for i in range(k):
        test = np.sort(np.array(parts[i], dtype=int))
        folds.append((np.setdiff1d(all_idx, test), test))
    return folds
