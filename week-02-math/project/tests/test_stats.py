import math

import numpy as np
import pytest

from optlab.stats import (cross_entropy, entropy, kl_divergence, log_softmax, logsumexp, mean_ci, softmax,
                          wilson_interval)

P = [0.7, 0.2, 0.1]
Q = [0.5, 0.3, 0.2]


def test_logsumexp_is_stable_and_respects_axis():
    with np.errstate(over="raise", invalid="raise"):
        assert logsumexp(np.array([1000.0, 1001.0])) == pytest.approx(1001.3132616875)
    a = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    assert np.allclose(logsumexp(a, axis=1), [3.40760596, math.log(3)])
    assert np.allclose(logsumexp(a, axis=0), np.log(np.exp(a).sum(0)))


def test_softmax_and_log_softmax():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(p, [0.09003057, 0.24472847, 0.66524096])
    with np.errstate(over="raise", invalid="raise"):
        big = softmax(np.array([[1000.0, 1001.0]]), axis=1)
    assert np.allclose(big, [[0.26894142, 0.73105858]])
    assert np.allclose(log_softmax(np.array([1.0, 2.0, 3.0])), np.log(p))


def test_entropy():
    assert entropy([0.5, 0.5], base=2) == pytest.approx(1.0)
    assert entropy([1.0, 0.0]) == 0.0                          # 0·log 0 = 0
    assert entropy(P, base=2) == pytest.approx(1.15677965)
    assert entropy(P) == pytest.approx(0.80181855)             # nats


def test_cross_entropy_and_kl():
    assert cross_entropy(P, Q, base=2) == pytest.approx(1.27958593)
    assert kl_divergence(P, Q, base=2) == pytest.approx(0.12280628)
    assert kl_divergence(Q, P, base=2) == pytest.approx(0.13277534)   # KL không đối xứng
    assert cross_entropy(P, Q) == pytest.approx(entropy(P) + kl_divergence(P, Q))
    assert kl_divergence(P, P) == 0
    assert kl_divergence([0.5, 0.5], [1.0, 0.0]) == math.inf
    assert kl_divergence([1.0, 0.0], [0.5, 0.5]) == pytest.approx(math.log(2))


def test_invalid_distribution():
    with pytest.raises(ValueError):
        entropy([0.5, 0.6])
    with pytest.raises(ValueError):
        kl_divergence([1.2, -0.2], [0.5, 0.5])


def test_mean_ci():
    m, lo, hi = mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    half = 2.7764451 * math.sqrt(2.5) / math.sqrt(5)           # t_{0.975, 4} · s / √n
    assert m == 3 and lo == pytest.approx(3 - half) and hi == pytest.approx(3 + half)
    with pytest.raises(ValueError):
        mean_ci([1.0])


def test_wilson_interval():
    lo, hi = wilson_interval(0, 50)
    assert lo == 0 and hi == pytest.approx(0.07135, abs=1e-5)
    lo, hi = wilson_interval(160, 200)
    assert lo == pytest.approx(0.73914, abs=1e-5) and hi == pytest.approx(0.84955, abs=1e-5)
    with pytest.raises(ValueError):
        wilson_interval(5, 3)
