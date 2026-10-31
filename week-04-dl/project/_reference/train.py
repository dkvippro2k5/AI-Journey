"""Ngày 4 · Huấn luyện TabMLP / TabAttn, so sánh với logistic regression và HGB của Tuần 3 trên CÙNG split.

Chạy: python -m qnet.train [--n 6000] [--seed 0] [--repeats 0] [--out artifacts]
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
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

from qnet.baseline import build_baseline, split_xy
from qnet.data import make_feedback
from qnet.loop import count_params, fit, make_loader, predict, set_seed
from qnet.tabular import TabularPrep, build_net

log = logging.getLogger("qnet")


def train_net(arch: str, X_fit: pd.DataFrame, y_fit, X_val: pd.DataFrame, y_val, *, seed: int = 0,
              lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 128, max_epochs: int = 200,
              patience: int = 10, **net_kw) -> tuple[nn.Module, TabularPrep, dict]:
    """Fit TabularPrep trên X_fit, huấn luyện bằng AdamW + BCEWithLogitsLoss, early stopping trên (X_val, y_val)."""
    set_seed(seed)
    prep = TabularPrep().fit(X_fit)
    model = build_net(arch, prep, **net_kw)
    y_fit_t = torch.tensor(np.asarray(y_fit), dtype=torch.float32)
    y_val_t = torch.tensor(np.asarray(y_val), dtype=torch.float32)
    train_loader = make_loader(*prep.transform(X_fit), y_fit_t, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(*prep.transform(X_val), y_val_t, batch_size=1024)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    t0 = time.perf_counter()
    hist = fit(model, train_loader, val_loader, nn.BCEWithLogitsLoss(), opt, max_epochs=max_epochs, patience=patience)
    hist["time_s"] = time.perf_counter() - t0
    return model, prep, hist


def predict_proba(model: nn.Module, prep: TabularPrep, X: pd.DataFrame) -> np.ndarray:
    return torch.sigmoid(predict(model, *prep.transform(X))).numpy()


def scores(y, p) -> dict:
    return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p))}


def attention_summary(model: nn.Module, prep: TabularPrep, X: pd.DataFrame) -> dict:
    """Trọng số attention trung bình của từng token trên X, sắp giảm dần."""
    w = model.attention(*prep.transform(X)).mean(0).tolist()
    return dict(sorted(zip(prep.token_names, w), key=lambda kv: -kv[1]))


def save_bundle(path: Path, arch: str, net_kw: dict, model: nn.Module, prep: TabularPrep) -> None:
    torch.save({"arch": arch, "net_kw": net_kw, "state_dict": model.state_dict()}, path.with_suffix(".pt"))
    joblib.dump(prep, path.with_suffix(".prep.joblib"))


def load_bundle(path: Path) -> tuple[nn.Module, TabularPrep]:
    state = torch.load(path.with_suffix(".pt"), weights_only=True)
    prep = joblib.load(path.with_suffix(".prep.joblib"))
    model = build_net(state["arch"], prep, **state["net_kw"])
    model.load_state_dict(state["state_dict"])
    return model.eval(), prep


def _split(df: pd.DataFrame, seed: int):
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=seed)
    return X_tr, X_te, y_tr, y_te, X_fit, X_val, y_fit, y_val


def run(df: pd.DataFrame, *, seed: int = 0, archs=("mlp", "attn"), out: Path | None = None) -> dict:
    """Một split 80/20: baseline fit trên toàn bộ train; mạng fit trên 80% train, early stopping trên 20% còn lại."""
    X_tr, X_te, y_tr, y_te, X_fit, X_val, y_fit, y_val = _split(df, seed)
    report = {"n_train": len(X_tr), "n_test": len(X_te), "test": {}, "nets": {}}
    for name in ("logreg", "hgb"):
        t0 = time.perf_counter()
        base = build_baseline(name).fit(X_tr, y_tr)
        report["test"][name] = scores(y_te, base.predict_proba(X_te)[:, 1])
        report["test"][name]["time_s"] = time.perf_counter() - t0
    for arch in archs:
        model, prep, hist = train_net(arch, X_fit, y_fit, X_val, y_val, seed=seed)
        report["test"][arch] = scores(y_te, predict_proba(model, prep, X_te))
        report["test"][arch]["time_s"] = hist["time_s"]
        report["nets"][arch] = {"n_params": count_params(model), "best_epoch": hist["best_epoch"],
                                "stopped_epoch": hist["stopped_epoch"], "best_val": hist["best_val"]}
        if arch == "attn":
            report["attention"] = attention_summary(model, prep, X_te)
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            save_bundle(out / arch, arch, {}, model, prep)
        log.info("%s: %s", arch, report["test"][arch])
    if out is not None:
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def compare_seeds(df: pd.DataFrame, seeds=(0, 1, 2, 3, 4), archs=("mlp", "attn")) -> pd.DataFrame:
    """Lặp lại `run` với nhiều split; so sánh GHÉP CẶP với HGB trên từng split."""
    rows = []
    for s in seeds:
        rep = run(df, seed=s, archs=archs)
        rows += [{"seed": s, "model": m, "auc": r["auc"]} for m, r in rep["test"].items()]
    t = pd.DataFrame(rows).pivot(index="seed", columns="model", values="auc")
    diff = t.sub(t["hgb"], axis=0)
    return pd.DataFrame({"auc_mean": t.mean(), "auc_std": t.std(ddof=1), "diff_vs_hgb": diff.mean(),
                         "wins_vs_hgb": (diff > 0).sum()}).loc[["logreg", "hgb", *archs]].reset_index()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="qnet.train")
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=0, help="số split để so sánh ghép cặp (0 = bỏ qua)")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = make_feedback(args.n, args.seed)
    report = run(df, seed=args.seed, out=args.out)
    if args.repeats > 0:
        table = compare_seeds(df, seeds=range(args.repeats))
        log.info("so sánh ghép cặp qua %d split:\n%s", args.repeats, table.round(4).to_string(index=False))
        report["repeated"] = table.round(4).to_dict(orient="records")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
