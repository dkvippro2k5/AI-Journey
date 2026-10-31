"""Test phần C — final Tuần 2. Chạy từ thư mục tuần: python -m pytest -q code/test_final.py"""
import math

import numpy as np
import pytest

from final_lib import (bootstrap_ci, fit_softmax, paired_bootstrap_diff, pca_power, predict_softmax,
                       softmax_ce_loss)


def numerical_grad(f, W, eps=1e-6):
    W = W.astype(float).copy()
    g = np.zeros_like(W)
    for i in range(W.size):
        old = W.flat[i]
        W.flat[i] = old + eps
        fp = f(W)
        W.flat[i] = old - eps
        fm = f(W)
        W.flat[i] = old
        g.flat[i] = (fp - fm) / (2 * eps)
    return g


# ---------- C1 ----------

def data_with_spectrum(seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    return rng.standard_normal((2000, 5)) * np.sqrt([9.0, 4.0, 1.0, 0.25, 0.04]) @ Q.T + 7.0


def test_pca_power_matches_eigh():
    X = data_with_spectrum()
    comps, vals = pca_power(X, 3)
    w, V = np.linalg.eigh(np.cov(X, rowvar=False))
    assert comps.shape == (3, 5) and vals.shape == (3,)
    assert np.allclose(vals, w[::-1][:3], rtol=1e-6)
    for i in range(3):
        assert abs(abs(comps[i] @ V[:, -1 - i]) - 1) < 1e-6
    assert np.allclose(comps @ comps.T, np.eye(3), atol=1e-6)


def test_pca_power_sign_convention_and_hand_example():
    P = np.array([[-3.0, -1.0], [-1.0, -3.0], [1.0, 3.0], [3.0, 1.0]])
    comps, vals = pca_power(P, 2)
    assert np.allclose(vals, [32 / 3, 8 / 3])
    assert np.allclose(comps[0], [2 ** -0.5, 2 ** -0.5])
    assert all(c[np.argmax(np.abs(c))] > 0 for c in comps)


def test_pca_power_does_not_modify_input_and_validates_k():
    X = data_with_spectrum(1)
    X0 = X.copy()
    pca_power(X, 2)
    assert np.array_equal(X, X0)
    with pytest.raises(ValueError):
        pca_power(X, 6)


# ---------- C2 ----------

def blobs(seed=0, n=300):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    centers = np.array([[0.0, 3.0], [-3.0, -2.0], [3.0, -2.0]])
    X = centers[y] + rng.standard_normal((n, 2))
    return np.column_stack([np.ones(n), X]), y


def test_softmax_loss_at_zero_and_gradient_check():
    X, y = blobs()
    loss, g = softmax_ce_loss(np.zeros((3, 3)), X, y)
    assert loss == pytest.approx(math.log(3))
    rng = np.random.default_rng(1)
    W = rng.standard_normal((3, 3))
    for l2 in (0.0, 0.1):
        _, g = softmax_ce_loss(W, X[:20], y[:20], l2)
        num = numerical_grad(lambda w: softmax_ce_loss(w, X[:20], y[:20], l2)[0], W)
        assert np.linalg.norm(g - num) / (np.linalg.norm(g) + np.linalg.norm(num)) < 1e-7


def test_softmax_loss_is_stable():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    W = np.array([[1000.0, 1001.0], [0.0, 0.0]])
    with np.errstate(over="raise", invalid="raise"):
        loss, g = softmax_ce_loss(W, X, np.array([0, 1]))
    assert loss == pytest.approx(1.0032044340)
    assert np.allclose(g, [[-0.36552929, 0.36552929], [0.25, -0.25]], atol=1e-6)   # X = I nên ∇W = (P − Y)/n


def test_fit_softmax_learns_blobs():
    X, y = blobs(2)
    W = fit_softmax(X, y, 3)
    assert W.shape == (3, 3)
    Xt, yt = blobs(3)
    assert np.mean(predict_softmax(W, Xt) == yt) > 0.93
    assert softmax_ce_loss(W, X, y)[0] < 0.3


# ---------- C3 ----------

def test_bootstrap_ci_close_to_normal_theory():
    x = np.array([1.0] * 320 + [0.0] * 80)
    lo, hi = bootstrap_ci(x, n_boot=4000, seed=0)
    half = 1.96 * math.sqrt(0.8 * 0.2 / 400)
    assert lo < 0.8 < hi
    assert lo == pytest.approx(0.8 - half, abs=0.01) and hi == pytest.approx(0.8 + half, abs=0.01)


def test_bootstrap_is_reproducible_and_validates():
    x = np.random.default_rng(5).exponential(size=50)
    assert bootstrap_ci(x, seed=3) == bootstrap_ci(x, seed=3)
    lo90, hi90 = bootstrap_ci(x, level=0.90, seed=3)
    lo99, hi99 = bootstrap_ci(x, level=0.99, seed=3)
    assert lo99 < lo90 < hi90 < hi99
    lo, hi = bootstrap_ci(x, stat=np.median, seed=3)
    assert lo <= np.median(x) <= hi
    with pytest.raises(ValueError):
        bootstrap_ci([1.0])


def test_paired_bootstrap_detects_difference():
    # 280 cùng đúng, 20 chỉ A đúng, 40 chỉ B đúng, 60 cùng sai (giống quiz Ngày 3, T3)
    a = np.array([1] * 280 + [1] * 20 + [0] * 40 + [0] * 60)
    b = np.array([1] * 280 + [0] * 20 + [1] * 40 + [0] * 60)
    diff, lo, hi = paired_bootstrap_diff(a, b, n_boot=4000, seed=0)
    assert diff == pytest.approx(0.05)
    assert 0 < lo < diff < hi
    with pytest.raises(ValueError):
        paired_bootstrap_diff(a, b[:-1])
