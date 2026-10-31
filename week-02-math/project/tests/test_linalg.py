import numpy as np
import pytest

from optlab.linalg import (add_bias, condition_number, hessian_eigs_mse, lstsq, normal_equation,
                           pca_fit, pca_transform, standardize)

X3 = np.array([[0.0], [1.0], [2.0]])
Y3 = np.array([1.0, 3.0, 4.0])
P4 = np.array([[-3.0, -1.0], [-1.0, -3.0], [1.0, 3.0], [3.0, 1.0]])


def test_add_bias():
    Xb = add_bias(X3)
    assert Xb.shape == (3, 2)
    assert np.all(Xb[:, 0] == 1) and np.all(Xb[:, 1] == X3[:, 0])


def test_standardize_and_constant_column():
    X = np.array([[1.0, 5.0, 7.0], [3.0, 5.0, 9.0], [5.0, 5.0, 11.0]])
    Z, mean, std = standardize(X)
    assert np.allclose(mean, [3, 5, 9])
    assert np.allclose(Z.mean(0), 0)
    assert np.allclose(Z[:, [0, 2]].std(0), 1)
    assert np.all(Z[:, 1] == 0) and not np.isnan(Z).any()     # cột hằng → 0, không phải nan


def test_lstsq_hand_example_and_orthogonal_residual():
    Xb = add_bias(X3)
    theta = lstsq(Xb, Y3)
    assert np.allclose(theta, [7 / 6, 1.5])
    r = Y3 - Xb @ theta
    assert np.allclose(Xb.T @ r, 0)                            # phần dư vuông góc với mọi cột


def test_normal_equation_matches_lstsq_and_ridge_formula():
    rng = np.random.default_rng(0)
    X = add_bias(rng.standard_normal((50, 3)))
    y = rng.standard_normal(50)
    assert np.allclose(normal_equation(X, y), lstsq(X, y))
    l2 = 0.5
    expected = np.linalg.solve(X.T @ X / 50 + l2 * np.eye(4), X.T @ y / 50)
    assert np.allclose(normal_equation(X, y, l2=l2), expected)


def test_condition_number():
    assert condition_number(np.diag([1.0, 100.0])) == pytest.approx(100)
    assert condition_number(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])) == np.inf


def test_hessian_eigs_mse():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 3)) * [1, 2, 3]
    ev = hessian_eigs_mse(X)
    assert np.all(np.diff(ev) >= 0)                            # tăng dần
    assert np.allclose(ev, np.linalg.eigvalsh(X.T @ X / 40))


def test_pca_hand_example():
    res = pca_fit(P4, 2)
    assert np.allclose(res.components[0], [2 ** -0.5, 2 ** -0.5])
    assert np.allclose(res.explained_variance, [32 / 3, 8 / 3])
    assert np.allclose(res.explained_variance_ratio, [0.8, 0.2])
    Z = pca_transform(res, P4)
    assert np.allclose(Z[:, 0], [-8 ** 0.5, -8 ** 0.5, 8 ** 0.5, 8 ** 0.5])


def test_pca_random_data_properties():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 5)) @ rng.standard_normal((5, 5)) + 10
    res = pca_fit(X, 3)
    C = res.components
    assert np.allclose(C @ C.T, np.eye(3))                     # trực chuẩn
    w = np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1]
    assert np.allclose(res.explained_variance, w[:3])
    assert np.all(np.abs(C).argmax(1) == np.argmax(C, axis=1))  # phần tử lớn nhất theo |.| là dương
    with pytest.raises(ValueError):
        pca_fit(X, 6)
