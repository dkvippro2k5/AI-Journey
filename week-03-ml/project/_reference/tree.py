"""Ngày 2 · Tiêu chí chia của cây quyết định và gradient boosting cho hồi quy."""
from __future__ import annotations

import math

import numpy as np
from sklearn.tree import DecisionTreeRegressor


def gini(y) -> float:
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / y.size
    return float(1.0 - np.sum(p ** 2))


def entropy(y) -> float:
    """Entropy theo bit."""
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / y.size
    return float(-np.sum(p * np.log2(p)))


def best_split(x, y, criterion: str = "gini") -> tuple[float, float]:
    """Ngưỡng tốt nhất trên MỘT đặc trưng số: trái là x <= t.

    Ứng viên là trung điểm của hai giá trị phân biệt liên tiếp (sau khi sắp xếp).
    gain = impurity(cha) − Σ (n_con/n) · impurity(con). Hòa thì chọn ngưỡng nhỏ nhất.
    Không có ứng viên (x hằng) thì trả về (nan, 0.0).
    """
    imp = {"gini": gini, "entropy": entropy}[criterion]
    x, y = np.asarray(x, dtype=float), np.asarray(y)
    values = np.unique(x)
    if values.size < 2:
        return math.nan, 0.0
    parent, n = imp(y), y.size
    best_t, best_gain = math.nan, -np.inf
    for t in (values[:-1] + values[1:]) / 2:
        left, right = y[x <= t], y[x > t]
        gain = parent - (left.size * imp(left) + right.size * imp(right)) / n
        if gain > best_gain + 1e-12:
            best_t, best_gain = float(t), float(gain)
    return best_t, best_gain


class L2Boost:
    """Gradient boosting với loss bình phương: mỗi cây khớp phần dư y − F."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3, random_state: int = 0):
        self.n_estimators, self.learning_rate = n_estimators, learning_rate
        self.max_depth, self.random_state = max_depth, random_state

    def fit(self, X, y) -> "L2Boost":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.init_ = float(y.mean())
        self.trees_, self.train_mse_ = [], []
        F = np.full(y.shape, self.init_)
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X, y - F)                                   # phần dư = −gradient của ½(y − F)²
            F = F + self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
            self.train_mse_.append(float(np.mean((y - F) ** 2)))
        return self

    def staged_predict(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.init_)
        for tree in self.trees_:
            F = F + self.learning_rate * tree.predict(X)
            yield F.copy()

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.init_ + self.learning_rate * sum(t.predict(X) for t in self.trees_)
