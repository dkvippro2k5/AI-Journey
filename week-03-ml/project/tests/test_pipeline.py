import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from quality.data import make_feedback
from quality.pipeline import build_model, build_preprocessor, split_xy

DF = make_feedback(800, seed=3)


def test_split_xy_drops_id_and_leakage():
    X, y = split_xy(DF)
    assert "session_id" not in X and "escalated" not in X and "bad" not in X
    assert len(X.columns) == 9 and y.name == "bad" and len(X) == len(DF)


def test_preprocessors_output_no_nan_and_have_indicators():
    X, _ = split_xy(DF)
    dense = lambda Z: Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    for kind in ("linear", "tree"):
        Z = dense(build_preprocessor(kind).fit_transform(X))
        assert not np.isnan(Z).any()
        assert Z.shape[1] == 7 + 2 + 5 + 3                              # 7 số + 2 cột chỉ báo thiếu + one-hot
    Zl = dense(build_preprocessor("linear").fit_transform(X))
    assert np.allclose(Zl[:, :7].mean(0), 0, atol=1e-9)                 # cột số đã chuẩn hóa
    with pytest.raises(ValueError):
        build_preprocessor("svm")


def test_build_model_structure_and_override():
    m = build_model("hgb", learning_rate=0.2)
    assert isinstance(m, Pipeline) and list(m.named_steps) == ["pre", "clf"]
    assert isinstance(m["clf"], HistGradientBoostingClassifier) and m["clf"].learning_rate == 0.2
    assert m["clf"].max_leaf_nodes == 7                                 # các mặc định khác giữ nguyên
    with pytest.raises(ValueError):
        build_model("svm")


@pytest.mark.parametrize("name", ["logreg", "rf", "hgb"])
def test_models_fit_and_handle_unknown_category(name):
    X, y = split_xy(DF)
    m = build_model(name).fit(X, y)
    new = X.head(3).copy()
    new.loc[new.index[0], "intent"] = "loại_mới_chưa_gặp"
    new.loc[new.index[1], "retrieval_score"] = np.nan
    proba = m.predict_proba(new)
    assert proba.shape == (3, 2) and np.allclose(proba.sum(1), 1)
