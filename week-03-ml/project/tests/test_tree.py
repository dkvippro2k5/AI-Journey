import math

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from quality.tree import L2Boost, best_split, entropy, gini

LAT = np.array([1.2, 2.0, 2.5, 3.1, 4.0, 4.8, 5.5, 6.2, 7.0, 8.5])    # ví dụ tính tay của bài 06
BAD = np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 1])


def test_impurities():
    assert gini(BAD) == pytest.approx(0.48)
    assert entropy(BAD) == pytest.approx(0.970950594)
    assert gini([1, 1, 1]) == 0 and entropy([0, 0]) == 0
    assert gini([0, 1, 2]) == pytest.approx(2 / 3) and entropy([0, 1, 2, 3]) == pytest.approx(2.0)


def test_best_split_hand_example():
    t, g = best_split(LAT, BAD, "gini")
    assert t == pytest.approx(5.15) and g == pytest.approx(0.48 - 0.31666667)
    t, g = best_split(LAT, BAD, "entropy")
    assert t == pytest.approx(2.8) and g == pytest.approx(0.28129, abs=1e-5)   # hai tiêu chí chọn hai ngưỡng khác nhau


def test_best_split_edge_cases():
    t, g = best_split([3.0, 3.0, 3.0], [0, 1, 0])
    assert math.isnan(t) and g == 0.0
    t, g = best_split([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1])
    assert t == 2.5 and g == pytest.approx(0.5)


def test_l2boost_hand_example():
    x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([1.0, 1.2, 1.1, 3.0, 3.2, 3.1])
    m = L2Boost(n_estimators=3, learning_rate=0.5, max_depth=1).fit(x, y)
    assert m.init_ == pytest.approx(2.1)
    assert m.train_mse_ == pytest.approx([0.256667, 0.069167, 0.022292], abs=1e-5)
    assert np.allclose(m.predict(x), [1.225] * 3 + [2.975] * 3)


def test_l2boost_matches_sklearn_and_staged_predict():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (300, 3))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.3, 300)
    ours = L2Boost(n_estimators=50, learning_rate=0.1, max_depth=3).fit(X, y)
    ref = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=0).fit(X, y)
    assert np.allclose(ours.predict(X), ref.predict(X), atol=1e-8)
    staged = list(ours.staged_predict(X))
    assert len(staged) == 50 and np.allclose(staged[-1], ours.predict(X))
    assert all(a >= b - 1e-12 for a, b in zip(ours.train_mse_, ours.train_mse_[1:]))   # MSE train không tăng
