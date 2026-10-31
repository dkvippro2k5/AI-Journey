"""Ngày 4 · So sánh model, tune, chọn ngưỡng theo chi phí, đánh giá trên test, lưu model.

Chạy: python -m quality.train [--n 6000] [--seed 0] [--n-iter 20] [--out artifacts]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate,
                                     train_test_split)

from quality.data import make_feedback
from quality.metrics import best_threshold, precision_recall_f1
from quality.pipeline import build_model, split_xy

log = logging.getLogger("quality")

SEARCH_SPACES = {
    "logreg": {"clf__C": loguniform(1e-3, 1e2)},
    "rf": {"clf__max_features": ["sqrt", 0.5, 1.0], "clf__min_samples_leaf": randint(1, 50)},
    "hgb": {"clf__learning_rate": loguniform(0.01, 0.3), "clf__max_leaf_nodes": randint(4, 64),
            "clf__min_samples_leaf": randint(5, 200), "clf__max_iter": randint(50, 500)},
}


def _cv(k: int, seed: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)


def compare_models(X, y, names=("logreg", "rf", "hgb"), k: int = 5, seed: int = 0) -> pd.DataFrame:
    rows = []
    for name in names:
        t = time.perf_counter()
        r = cross_validate(build_model(name), X, y, cv=_cv(k, seed), scoring=["roc_auc", "average_precision"])
        rows.append({"model": name, "auc_mean": r["test_roc_auc"].mean(), "auc_std": r["test_roc_auc"].std(),
                     "ap_mean": r["test_average_precision"].mean(), "time_s": time.perf_counter() - t})
    return pd.DataFrame(rows)


def tune(X, y, name: str = "hgb", n_iter: int = 20, k: int = 5, seed: int = 0) -> tuple[dict, float]:
    """RandomizedSearchCV; trả về (tham số tốt nhất KHÔNG có tiền tố 'clf__', AUC CV tốt nhất)."""
    search = RandomizedSearchCV(build_model(name), SEARCH_SPACES[name], n_iter=n_iter, cv=_cv(k, seed),
                                scoring="roc_auc", random_state=seed, n_jobs=-1)
    search.fit(X, y)
    params = {key.removeprefix("clf__"): (v.item() if hasattr(v, "item") else v) for key, v in search.best_params_.items()}
    return params, float(search.best_score_)


def choose_threshold(model, X, y, cost_fp: float = 1.0, cost_fn: float = 5.0, k: int = 5, seed: int = 0):
    """Chọn ngưỡng trên dự đoán out-of-fold của tập TRAIN (không bao giờ trên test).

    Trả về (ngưỡng, chi phí trung bình cho 1000 lượt trả lời).
    """
    proba = cross_val_predict(model, X, y, cv=_cv(k, seed), method="predict_proba")[:, 1]
    t, cost = best_threshold(y, proba, cost_fp, cost_fn)
    return t, 1000 * cost / len(y)


def evaluate(model, X, y, threshold: float, cost_fp: float = 1.0, cost_fn: float = 5.0,
             n_boot: int = 1000, seed: int = 0) -> dict:
    """Metric trên tập test, kèm khoảng tin cậy bootstrap 95% cho AUC."""
    y = np.asarray(y)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    p, r, f1 = precision_recall_f1(y, pred)
    cost = cost_fp * np.sum((pred == 1) & (y == 0)) + cost_fn * np.sum((pred == 0) & (y == 1))
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():                  # mẫu bootstrap phải có cả 2 lớp
            aucs.append(roc_auc_score(y[idx], proba[idx]))
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    return {"auc": float(roc_auc_score(y, proba)), "auc_lo": float(lo), "auc_hi": float(hi),
            "ap": float(average_precision_score(y, proba)), "precision": p, "recall": r, "f1": f1,
            "cost_per_1000": float(1000 * cost / len(y))}


def run(df: pd.DataFrame, *, seed: int = 0, n_iter: int = 20, out: Path | None = None) -> dict:
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    table = compare_models(X_tr, y_tr, seed=seed)
    log.info("so sánh model:\n%s", table.to_string(index=False))
    params, cv_auc = tune(X_tr, y_tr, "hgb", n_iter=n_iter, seed=seed)
    model = build_model("hgb", **params)
    threshold, cv_cost = choose_threshold(model, X_tr, y_tr, seed=seed)
    model.fit(X_tr, y_tr)
    metrics = evaluate(model, X_te, y_te, threshold, seed=seed)
    imp = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=seed)
    importance = dict(sorted(zip(X.columns, imp.importances_mean.round(4).tolist()), key=lambda kv: -kv[1]))
    report = {"compare": table.round(4).to_dict(orient="records"), "best_params": params, "cv_auc": cv_auc,
              "threshold": threshold, "cv_cost_per_1000": cv_cost, "test": metrics, "importance": importance}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out / "model.joblib")
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="quality.train")
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = run(make_feedback(args.n, args.seed), seed=args.seed, n_iter=args.n_iter, out=args.out)
    print(json.dumps({k: report[k] for k in ("best_params", "cv_auc", "threshold", "test", "importance")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
