import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from quality.cluster import kmeans, kmeans_pp_init

X1 = np.array([1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 25.0])                  # ví dụ tính tay của bài 08


def test_hand_example_with_given_init():
    centers, labels, inertia = kmeans(X1, 2, init=[[2.0], [3.0]])
    assert np.allclose(np.sort(centers[:, 0]), [2.0, 13.0])
    assert inertia == pytest.approx(196.0)
    assert labels.tolist() == [0, 0, 0, 1, 1, 1, 1]


def test_outlier_gets_own_cluster_with_k3():
    centers, _, inertia = kmeans(X1, 3, seed=0)
    assert np.allclose(np.sort(centers[:, 0]), [2.0, 9.0, 25.0]) and inertia == pytest.approx(4.0)


def test_matches_sklearn_on_blobs():
    X, _ = make_blobs(600, centers=4, cluster_std=1.0, random_state=0)
    _, labels, inertia = kmeans(X, 4, seed=0)
    ref = KMeans(4, n_init=10, random_state=0).fit(X)
    assert inertia <= ref.inertia_ * 1.001
    assert labels.shape == (600,) and set(labels.tolist()) == {0, 1, 2, 3}


def test_kmeans_pp_init_picks_distinct_points_from_data():
    X, _ = make_blobs(200, centers=3, random_state=1)
    C = kmeans_pp_init(X, 3, np.random.default_rng(0))
    assert C.shape == (3, 2)
    assert all(any(np.allclose(c, x) for x in X) for c in C)
    assert len({tuple(c) for c in C}) == 3


def test_validation_and_reproducibility():
    X, _ = make_blobs(100, centers=3, random_state=2)
    a, b = kmeans(X, 3, seed=5), kmeans(X, 3, seed=5)
    assert np.array_equal(a[1], b[1]) and a[2] == b[2]
    with pytest.raises(ValueError):
        kmeans(X, 101)
