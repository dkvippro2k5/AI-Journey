import math

import numpy as np

from optlab.data import make_regression, rosenbrock
from optlab.grad import mse_loss
from optlab.linalg import add_bias, hessian_eigs_mse, lstsq
from optlab.optim import SGD, Momentum
from optlab.train import History, train


def regression():
    X, y, _ = make_regression(200, 2, seed=0)
    Xb = add_bias(X)
    return Xb, y, lstsq(Xb, y), hessian_eigs_mse(Xb)[-1]


def test_full_batch_gd_converges_to_lstsq():
    Xb, y, theta_star, L = regression()
    theta, h = train(mse_loss, np.zeros(3), Xb, y, SGD(1 / L), epochs=500, theta_star=theta_star, tol=1e-8)
    assert isinstance(h, History) and h.reached_tol and not h.diverged
    assert np.allclose(theta, theta_star, atol=1e-7)
    assert h.losses[0] > h.losses[-1] and len(h.dists) == len(h.losses) == len(h.thetas)


def test_lengths_and_step_count_with_minibatch():
    Xb, y, _, L = regression()
    _, h = train(mse_loss, np.zeros(3), Xb, y, SGD(0.1 / L), epochs=4, batch_size=64)
    assert len(h.losses) == 5                                  # trước epoch 1 + sau mỗi epoch
    assert h.steps == 4 * math.ceil(200 / 64)
    assert h.dists == []                                       # không có theta_star thì không ghi khoảng cách


def test_divergence_stops_early():
    Xb, y, _, L = regression()
    _, h = train(mse_loss, np.zeros(3), Xb, y, SGD(2.5 / L), epochs=1000)
    assert h.diverged and len(h.losses) < 1001
    assert not h.reached_tol


def test_seed_controls_shuffling():
    Xb, y, _, L = regression()
    run = lambda s: train(mse_loss, np.zeros(3), Xb, y, SGD(0.5 / L), epochs=2, batch_size=16, seed=s)[0]
    assert np.array_equal(run(1), run(1))
    assert not np.array_equal(run(1), run(2))


def test_clip_limits_each_step():
    Xb, y, _, _ = regression()
    theta, h = train(mse_loss, np.full(3, 50.0), Xb, y, SGD(1.0), epochs=3, clip=0.01)
    steps = np.diff(np.array(h.thetas), axis=0)
    assert np.all(np.linalg.norm(steps, axis=1) <= 0.01 + 1e-12)


def test_works_without_data_on_rosenbrock():
    theta, h = train(rosenbrock, np.array([-1.5, 2.0]), None, None, Momentum(1e-3), epochs=3000,
                     theta_star=np.array([1.0, 1.0]))
    assert h.steps == 3000 and h.dists[-1] < 1e-3
