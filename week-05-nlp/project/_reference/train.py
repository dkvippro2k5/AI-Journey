"""Ngày 4 · Router intent: TF-IDF + logistic regression so với Transformer encoder tự viết, trên CÙNG split.

Chạy: python -m intent.train [--n 6000] [--seeds 5] [--split template|random]
"""
from __future__ import annotations

import argparse
import copy
import functools
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch import nn

from intent.bpe import BPE, train_bpe
from intent.data import INTENTS, make_messages
from intent.encoder import EncoderClassifier
from intent.metrics import prf
from intent.textprep import normalize

FALLBACK = "fallback"                       # chuyển cho tổng đài viên


# ---------- chia dữ liệu ----------
def random_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0):
    return train_test_split(df, test_size=test_frac, stratify=df["intent"], random_state=seed)


def template_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0):
    """Giữ lại ~test_frac số TEMPLATE của mỗi intent gốc (ít nhất 1) làm test: câu diễn đạt chưa từng thấy."""
    rng = np.random.default_rng(seed)
    origin = df["template_id"].str.split("/").str[0]
    test_t = []
    for name in INTENTS:
        temps = sorted(df.loc[origin == name, "template_id"].unique())
        k = max(1, round(test_frac * len(temps)))
        test_t += list(rng.choice(temps, size=k, replace=False))
    is_test = df["template_id"].isin(test_t)
    return df[~is_test], df[is_test]


def make_split(df, kind: str, test_frac: float = 0.2, seed: int = 0):
    return (template_split if kind == "template" else random_split)(df, test_frac, seed)


# ---------- baseline TF-IDF ----------
def build_tfidf(kind: str = "char", strip: bool = False, C: float = 10.0) -> Pipeline:
    """kind: "word" (âm tiết 1–2 gram) hoặc "char" (char_wb 2–5 gram). Tiền xử lý bằng normalize() của Ngày 1."""
    prep = functools.partial(normalize, strip=strip)
    vec = (TfidfVectorizer(preprocessor=prep, token_pattern=r"<\w+>|\w+", ngram_range=(1, 2)) if kind == "word"
           else TfidfVectorizer(preprocessor=prep, analyzer="char_wb", ngram_range=(2, 5)))
    return Pipeline([("tfidf", vec), ("clf", LogisticRegression(C=C, max_iter=3000))])


# ---------- Transformer encoder ----------
def encode_texts(bpe: BPE, texts, max_len: int = 64) -> torch.Tensor:
    """<cls> + id BPE của normalize(text), cắt ở max_len, đệm <pad> (id 0) → (N, T_max)."""
    cls = bpe.vocab["<cls>"]
    seqs = [[cls] + bpe.encode(normalize(t))[: max_len - 1] for t in texts]
    T = max(map(len, seqs))
    return torch.tensor([s + [0] * (T - len(s)) for s in seqs])


def train_encoder(train_texts, train_y, val_texts, val_y, *, seed: int = 0, num_merges: int = 300, epochs: int = 40,
                  patience: int = 5, lr: float = 1e-3, batch_size: int = 64, **net_kw):
    """Học BPE trên train → huấn luyện EncoderClassifier bằng AdamW, early stopping theo val loss.
    Trả về (model, bpe, history)."""
    torch.manual_seed(seed)
    bpe = train_bpe([normalize(t) for t in train_texts], num_merges)
    label_id = {l: i for i, l in enumerate(INTENTS)}
    X, Xv = encode_texts(bpe, train_texts), encode_texts(bpe, val_texts)
    y = torch.tensor([label_id[l] for l in train_y])
    yv = torch.tensor([label_id[l] for l in val_y])
    model = EncoderClassifier(len(bpe.vocab), len(INTENTS), **net_kw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    hist = {"train_loss": [], "val_loss": [], "best_epoch": -1}
    best, best_state, wait = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        perm, total = torch.randperm(len(X), generator=g), 0.0
        for i in range(0, len(X), batch_size):
            b = perm[i:i + batch_size]
            loss = loss_fn(model(X[b]), y[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xv), yv).item()
        hist["train_loss"].append(total / len(X))
        hist["val_loss"].append(val)
        if val < best - 1e-4:
            best, best_state, wait, hist["best_epoch"] = val, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), bpe, hist


@torch.no_grad()
def encoder_proba(model, bpe, texts) -> np.ndarray:
    return torch.softmax(model(encode_texts(bpe, texts)), -1).numpy()


# ---------- định tuyến có ngưỡng từ chối ----------
def choose_threshold(proba: np.ndarray, y_true, target_precision: float = 0.95) -> float:
    """Ngưỡng NHỎ NHẤT trên max-probability sao cho các tin nhắn được nhận (max ≥ ngưỡng) đạt precision ≥ mục tiêu.
    Không ngưỡng nào đạt thì trả 1.0 (chuyển hết cho người)."""
    conf = proba.max(1)
    correct = np.array(INTENTS)[proba.argmax(1)] == np.asarray(y_true)
    for t in np.unique(conf):                                        # tăng dần
        keep = conf >= t
        if correct[keep].mean() >= target_precision:
            return float(t)
    return 1.0


def route(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Nhãn có xác suất lớn nhất, hoặc FALLBACK nếu max-probability < threshold."""
    pred = np.array(INTENTS, dtype=object)[proba.argmax(1)]
    pred[proba.max(1) < threshold] = FALLBACK
    return pred


def scores(y_true, proba: np.ndarray, threshold: float | None = None) -> dict:
    pred = np.array(INTENTS)[proba.argmax(1)]
    out = {"macro_f1": prf(list(y_true), list(pred), labels=INTENTS)[2],
           "acc": float(np.mean(pred == np.asarray(y_true)))}
    if threshold is not None:
        r = route(proba, threshold)
        kept = r != FALLBACK
        out["coverage"] = float(kept.mean())                          # tỉ lệ tự động xử lý
        out["routed_precision"] = float(np.mean(r[kept] == np.asarray(y_true)[kept])) if kept.any() else 1.0
    return out


# ---------- một split và nhiều split ----------
def run(df: pd.DataFrame, *, seed: int = 0, split: str = "template", models=("word", "char", "encoder")) -> dict:
    """Train/val/test: test theo `split`; val tách khỏi train CÙNG kiểu split (để ngưỡng và early stopping trung thực)."""
    tr, te = make_split(df, split, 0.2, seed)
    fit, val = make_split(tr, split, 0.2, seed + 100)
    report = {"n_train": len(tr), "n_test": len(te), "test": {}}
    for name in models:
        t0 = time.perf_counter()
        if name == "encoder":
            model, bpe, hist = train_encoder(fit.text, fit.intent, val.text, val.intent, seed=seed)
            p_val, p_te = encoder_proba(model, bpe, val.text), encoder_proba(model, bpe, te.text)
            extra = {"best_epoch": hist["best_epoch"], "n_params": sum(p.numel() for p in model.parameters())}
        else:
            model = build_tfidf(name).fit(fit.text, fit.intent)
            order = [list(model.classes_).index(l) for l in INTENTS]
            p_val, p_te = model.predict_proba(val.text)[:, order], model.predict_proba(te.text)[:, order]
            extra = {}
        thr = choose_threshold(p_val, val.intent)
        report["test"][name] = {**scores(te.intent, p_te, thr), "threshold": thr,
                                "time_s": time.perf_counter() - t0, **extra}
    return report


def compare(df: pd.DataFrame, seeds=range(5), split: str = "template", models=("word", "char", "encoder")):
    """So sánh GHÉP CẶP qua nhiều split: chênh macro-F1 so với "char" trên từng split."""
    rows = []
    for s in seeds:
        rep = run(df, seed=s, split=split, models=models)
        rows += [{"seed": s, "model": m, **r} for m, r in rep["test"].items()]
    t = pd.DataFrame(rows)
    f1 = t.pivot(index="seed", columns="model", values="macro_f1")
    diff = f1.sub(f1["char"], axis=0)
    out = t.groupby("model")[["macro_f1", "coverage", "routed_precision"]].mean()
    out["f1_std"] = f1.std(ddof=1)
    out["diff_vs_char"] = diff.mean()
    out["wins_vs_char"] = (diff > 0).sum()
    return out.loc[list(models)].reset_index()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="intent.train")
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--split", choices=["template", "random"], default="template")
    args = parser.parse_args(argv)
    df = make_messages(args.n, 0)
    table = compare(df, seeds=range(args.seeds), split=args.split)
    print(json.dumps(table.round(4).to_dict(orient="records"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
