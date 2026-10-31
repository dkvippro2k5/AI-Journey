import json

import numpy as np
from sklearn.metrics import roc_auc_score

from qnet.baseline import split_xy
from qnet.data import make_feedback
from qnet.train import compare_seeds, load_bundle, main, predict_proba, run, train_net

DF = make_feedback(2000, seed=1)
X, Y = split_xy(DF)


def test_train_net_learns_and_restores_best():
    model, prep, hist = train_net("mlp", X.iloc[:1200], Y.iloc[:1200], X.iloc[1200:1600], Y.iloc[1200:1600],
                                  max_epochs=60, patience=5)
    p = predict_proba(model, prep, X.iloc[1600:])
    assert p.shape == (400,) and ((p > 0) & (p < 1)).all()
    assert roc_auc_score(Y.iloc[1600:], p) > 0.7
    assert hist["best_val"] == min(hist["val_loss"]) and hist["stopped_epoch"] <= 59
    assert {"train_loss", "val_loss", "best_epoch", "best_val", "stopped_epoch", "time_s"} <= set(hist)


def test_run_report_and_bundle(tmp_path):
    report = run(DF, out=tmp_path)
    assert set(report["test"]) == {"logreg", "hgb", "mlp", "attn"}
    assert all(0.6 < r["auc"] < 1 and r["ap"] > Y.mean() for r in report["test"].values())
    assert report["nets"]["mlp"]["n_params"] == 5417 and report["n_test"] == 400
    att = report["attention"]
    assert len(att) == 11 and abs(sum(att.values()) - 1) < 1e-4
    assert list(att.values()) == sorted(att.values(), reverse=True)
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["test"]["mlp"]["auc"] == report["test"]["mlp"]["auc"]
    model, prep = load_bundle(tmp_path / "mlp")
    Xn, _ = split_xy(make_feedback(30, seed=9))
    Xn.loc[0, "intent"] = "intent_mới"                                       # giá trị lạ không làm hỏng dự đoán
    p = predict_proba(model, prep, Xn)
    assert p.shape == (30,) and np.isfinite(p).all()


def test_compare_seeds_is_paired():
    t = compare_seeds(make_feedback(1500, seed=2), seeds=(0, 1), archs=("mlp",))
    assert list(t.columns) == ["model", "auc_mean", "auc_std", "diff_vs_hgb", "wins_vs_hgb"]
    assert t["model"].tolist() == ["logreg", "hgb", "mlp"]
    row = t.set_index("model")
    assert row.loc["hgb", "diff_vs_hgb"] == 0 and row.loc["hgb", "wins_vs_hgb"] == 0
    assert abs(row.loc["mlp", "diff_vs_hgb"] - (row.loc["mlp", "auc_mean"] - row.loc["hgb", "auc_mean"])) < 1e-9


def test_main_cli(tmp_path, capsys):
    main(["--n", "1200", "--out", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert {"test", "nets", "attention"} <= set(out)
    assert (tmp_path / "attn.pt").exists() and (tmp_path / "mlp.prep.joblib").exists()
