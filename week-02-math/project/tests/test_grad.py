import math

import numpy as np
import pytest

from optlab.grad import logistic_loss, mse_loss, numerical_grad, rel_error, sigmoid


def test_numerical_grad_hand_example_and_no_mutation():
    f = lambda t: t[0] ** 2 + 3 * t[0] * t[1] + t[1] ** 2
    theta = np.array([1.0, 2.0])
    g = numerical_grad(f, theta)
    assert np.allclose(g, [8, 7], atol=1e-6)
    assert np.array_equal(theta, [1.0, 2.0])


def test_rel_error():
    a = np.array([1.0, 2.0])
    assert rel_error(a, a) == 0
    assert rel_error(np.zeros(2), np.zeros(2)) == 0
    assert rel_error(a, -a) == pytest.approx(1.0)


def test_sigmoid_is_stable():
    with np.errstate(over="raise", invalid="raise"):          # overflow / nan sẽ thành exception
        s = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.allclose(s, [0.0, 0.5, 1.0])
    assert sigmoid(np.array([0.1]))[0] == pytest.approx(0.52497918747894)


@pytest.mark.parametrize("l2", [0.0, 0.3])
def test_mse_loss_value_and_gradient(l2):
    rng = np.random.default_rng(0)
    X, y, theta = rng.standard_normal((30, 4)), rng.standard_normal(30), rng.standard_normal(4)
    loss, g = mse_loss(theta, X, y, l2=l2)
    r = X @ theta - y
    assert loss == pytest.approx(r @ r / 60 + l2 / 2 * theta @ theta)
    assert rel_error(g, numerical_grad(lambda t: mse_loss(t, X, y, l2)[0], theta)) < 1e-7


@pytest.mark.parametrize("l2", [0.0, 0.3])
def test_logistic_loss_value_and_gradient(l2):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 3))
    y = (rng.random(30) < 0.5).astype(float)
    assert logistic_loss(np.zeros(3), X, y)[0] == pytest.approx(math.log(2))
    theta = rng.standard_normal(3)
    _, g = logistic_loss(theta, X, y, l2=l2)
    assert rel_error(g, numerical_grad(lambda t: logistic_loss(t, X, y, l2)[0], theta)) < 1e-7


def test_logistic_loss_huge_logits_are_finite():
    X = np.array([[1.0], [1.0]])
    y = np.array([0.0, 1.0])
    with np.errstate(over="raise"):
        loss, g = logistic_loss(np.array([800.0]), X, y)
    assert loss == pytest.approx(400.0)                        # (800 − 0)/2: mẫu y=0 phạt 800, mẫu y=1 phạt 0
    assert np.allclose(g, [0.5])
