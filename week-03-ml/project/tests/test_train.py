import json

import joblib
import numpy as np
import pytest

from quality.data import make_feedback
from quality.pipeline import build_model, split_xy
from quality.train import choose_threshold, compare_models, evaluate, main, run, tune

DF = make_feedback(2000, seed=1)
X, Y = split_xy(DF)


def test_compare_models_table_and_ranking():
    t = compare_models(X, Y, k=3)
    assert list(t.columns) == ["model", "auc_mean", "auc_std", "ap_mean", "time_s"]
    assert t["model"].tolist() == ["logreg", "rf", "hgb"]
    auc = t.set_index("model")["auc_mean"]
    assert auc["hgb"] > auc["logreg"] and auc["rf"] > auc["logreg"]     # có tương tác/ngưỡng → model cây thắng
    assert (t["ap_mean"] > Y.mean()).all()                              # AP luôn phải hơn mức ngẫu nhiên (tỉ lệ dương)


def test_tune_returns_plain_params():
    params, score = tune(X, Y, "hgb", n_iter=3, k=3)
    assert set(params) == {"learning_rate", "max_leaf_nodes", "min_samples_leaf", "max_iter"}
    assert 0.5 < score < 1
    build_model("hgb", **params)                                         # dùng lại được ngay


def test_choose_threshold_prefers_low_threshold_when_fn_expensive():
    m = build_model("logreg")
    t_cheap_fn, _ = choose_threshold(m, X, Y, cost_fp=1, cost_fn=1, k=3)
    t_costly_fn, cost = choose_threshold(m, X, Y, cost_fp=1, cost_fn=5, k=3)
    assert t_costly_fn < t_cheap_fn
    assert 0 < cost < 1000 * 5 * Y.mean()                               # rẻ hơn "không bắt gì cả"


def test_evaluate_reports_ci():
    m = build_model("logreg").fit(X.iloc[:1500], Y.iloc[:1500])
    r = evaluate(m, X.iloc[1500:], Y.iloc[1500:], threshold=0.2, n_boot=200)
    assert r["auc_lo"] < r["auc"] < r["auc_hi"]
    assert set(r) == {"auc", "auc_lo", "auc_hi", "ap", "precision", "recall", "f1", "cost_per_1000"}


def test_run_saves_artifacts(tmp_path):
    report = run(make_feedback(1500, seed=2), n_iter=2, out=tmp_path)
    assert (tmp_path / "model.joblib").exists() and (tmp_path / "report.json").exists()
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["threshold"] == pytest.approx(report["threshold"])
    model = joblib.load(tmp_path / "model.joblib")
    Xn, _ = split_xy(make_feedback(50, seed=9))
    assert model.predict_proba(Xn).shape == (50, 2)
    assert set(report["importance"]) == set(X.columns)


def test_main_cli(tmp_path, capsys):
    main(["--n", "1200", "--n-iter", "2", "--out", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert {"best_params", "cv_auc", "threshold", "test", "importance"} <= set(out)
    assert np.isfinite(out["test"]["auc"])
