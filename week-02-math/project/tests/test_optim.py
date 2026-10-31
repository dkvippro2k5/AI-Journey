import numpy as np
import pytest

from optlab.optim import SGD, Adam, Momentum, RMSProp, clip_by_norm, cosine_lr, make_optimizer


def run(opt, grads, theta0=None):
    theta = np.zeros_like(grads[0]) if theta0 is None else theta0
    out = []
    for g in grads:
        theta = opt.step(theta, g)
        out.append(theta)
    return out


def test_sgd():
    assert np.allclose(SGD(0.1).step(np.array([1.0, 2.0]), np.array([10.0, -10.0])), [0.0, 3.0])


def test_momentum_constant_gradient():
    out = run(Momentum(0.1, beta=0.9), [np.array([1.0])] * 3)
    assert np.allclose([t[0] for t in out], [-0.1, -0.29, -0.561])   # v = 1, 1.9, 2.71


def test_rmsprop_first_step_without_bias_correction():
    (theta,) = run(RMSProp(0.01, rho=0.9), [np.array([0.1, 10.0])])
    assert np.allclose(theta, -0.01 * np.array([1, 1]) / np.sqrt(0.1), rtol=1e-6)   # ≈ 3.16·lr


def test_adam_first_two_steps():
    opt = Adam(0.01)
    t1, t2 = run(opt, [np.array([0.1, 10.0]), np.array([0.1, -10.0])])
    assert np.allclose(t1, [-0.01, -0.01], rtol=1e-6)        # bước đầu ≈ lr·sign(g), bất kể độ lớn g
    assert np.allclose(t2 - t1, [-0.01, 0.01 * 0.0526316], rtol=1e-4)
    assert opt.t == 2


@pytest.mark.parametrize("cls", [SGD, Momentum, RMSProp, Adam])
def test_step_does_not_mutate_inputs(cls):
    opt = cls(0.1)
    theta, g = np.array([1.0, 2.0]), np.array([0.5, -0.5])
    opt.step(theta, g)
    new = opt.step(theta, g)
    assert np.array_equal(theta, [1.0, 2.0]) and np.array_equal(g, [0.5, -0.5])
    assert new is not theta


def test_clip_by_norm():
    g = np.array([3.0, 4.0])
    assert np.allclose(clip_by_norm(g, 1.0), [0.6, 0.8])
    small = clip_by_norm(g, 10.0)
    assert np.array_equal(small, g) and small is not g
    assert np.array_equal(g, [3.0, 4.0])


def test_cosine_lr():
    assert cosine_lr(0, 110, 1.0, warmup=10) == pytest.approx(0.1)
    assert cosine_lr(9, 110, 1.0, warmup=10) == pytest.approx(1.0)
    assert cosine_lr(10, 110, 1.0, warmup=10) == pytest.approx(1.0)
    assert cosine_lr(60, 110, 1.0, warmup=10) == pytest.approx(0.5)
    assert cosine_lr(110, 110, 1.0, warmup=10, min_lr=0.1) == pytest.approx(0.1)
    assert cosine_lr(500, 110, 1.0, warmup=10, min_lr=0.1) == pytest.approx(0.1)


def test_make_optimizer():
    for name, cls in [("sgd", SGD), ("momentum", Momentum), ("rmsprop", RMSProp), ("adam", Adam)]:
        assert isinstance(make_optimizer(name, 0.1), cls)
    with pytest.raises(ValueError):
        make_optimizer("adagrad", 0.1)
