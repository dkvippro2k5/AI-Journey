import numpy as np
import pandas as pd
import pytest
import torch

from qnet.baseline import split_xy
from qnet.data import FEATURES_NUM, make_feedback
from qnet.layers import BatchNorm1d, Dropout, LayerNorm
from qnet.tabular import TabAttn, TabMLP, TabularPrep, build_net

TOY = pd.DataFrame({"a": [1.0, np.nan, 3.0, 5.0], "b": [10.0, 10.0, 10.0, 10.0], "c": ["x", "y", "x", None]})


def test_prep_hand_example():
    prep = TabularPrep(num_cols=["a", "b"], cat_cols=["c"]).fit(TOY)
    # a: median 3 → [1, 3, 3, 5], mean 3, std sqrt(2) ; b hằng số → std thay bằng 1 ; chỉ "a" có NaN
    assert prep.missing_cols_ == ["a"] and prep.n_num == 3
    assert prep.vocab_ == {"c": {"x": 1, "y": 2}} and prep.cardinalities == [3]
    assert prep.token_names == ["a", "b", "a_missing", "c"]
    x_num, x_cat = prep.transform(TOY)
    assert x_num.dtype == torch.float32 and x_cat.dtype == torch.int64
    s = 2 ** 0.5
    assert torch.allclose(x_num, torch.tensor([[-2 / s, 0, 0], [0, 0, 1], [0, 0, 0], [2 / s, 0, 0]]))
    assert x_cat.flatten().tolist() == [1, 2, 1, 0]                           # None → 0
    new = pd.DataFrame({"a": [np.nan], "b": [12.0], "c": ["z"]})              # giá trị lạ → 0
    x_num, x_cat = prep.transform(new)
    assert x_num.tolist() == [[0.0, 2.0, 1.0]] and x_cat.tolist() == [[0]]


def test_prep_uses_train_statistics_only():
    X, _ = split_xy(make_feedback(1000, seed=0))
    prep = TabularPrep().fit(X.iloc[:800])
    x_tr, _ = prep.transform(X.iloc[:800])
    k = len(FEATURES_NUM)
    assert np.allclose(x_tr[:, :k].mean(0), 0, atol=1e-5) and np.allclose(x_tr[:, :k].std(0, unbiased=False), 1, atol=1e-4)
    shifted = X.iloc[800:].copy()
    shifted["latency_s"] += 100
    before = prep.median_.copy()
    prep.transform(shifted)
    assert np.array_equal(prep.median_, before)                              # transform không được học lại
    assert prep.missing_cols_ == ["retrieval_score", "latency_s"]


def _batch(n=16, seed=0):
    X, _ = split_xy(make_feedback(200, seed=seed))
    prep = TabularPrep().fit(X)
    return prep, prep.transform(X.iloc[:n])


def test_tabmlp_shapes_params_and_own_layers():
    prep, (xn, xc) = _batch()
    m = TabMLP(prep.n_num, prep.cardinalities)
    assert m(xn, xc).shape == (16,)
    assert sum(p.numel() for p in m.parameters()) == 5417                   # tính tay ở project.md
    assert any(isinstance(x, Dropout) for x in m.modules())
    assert any(isinstance(x, BatchNorm1d) for x in TabMLP(prep.n_num, prep.cardinalities, norm="batch").modules())
    assert any(isinstance(x, LayerNorm) for x in TabMLP(prep.n_num, prep.cardinalities, norm="layer").modules())
    with pytest.raises(ValueError):
        TabMLP(prep.n_num, prep.cardinalities, norm="group")


def test_tabattn_shapes_and_attention():
    torch.manual_seed(0)
    prep, (xn, xc) = _batch()
    m = TabAttn(prep.n_num, prep.cardinalities)
    assert m(xn, xc).shape == (16,)
    assert sum(p.numel() for p in m.parameters()) == 2433
    w = m.attention(xn, xc)
    assert w.shape == (16, len(prep.token_names)) == (16, 11)
    assert torch.allclose(w.sum(1), torch.ones(16))
    assert isinstance(build_net("attn", prep), TabAttn) and isinstance(build_net("mlp", prep, hidden=(8,)), TabMLP)
    with pytest.raises(ValueError):
        build_net("cnn", prep)
