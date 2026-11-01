"""Ngày 3 · Metric NLP viết bằng Python thuần: P/R/F1, BLEU, ROUGE, EM, token-F1, perplexity, Cohen's kappa."""
from __future__ import annotations

import math
import re
import string
from collections import Counter


def prf(y_true, y_pred, labels=None, average: str | None = "macro"):
    """(precision, recall, f1) như precision_recall_fscore_support(zero_division=0) của scikit-learn.
    average: None (trả list theo từng nhãn), "macro", "micro", "weighted"."""
    labels = sorted(set(y_true) | set(y_pred)) if labels is None else list(labels)
    tp, fp, fn = Counter(), Counter(), Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    def div(a, b):
        return a / b if b else 0.0

    def f1(p, r):
        return div(2 * p * r, p + r)

    if average == "micro":
        TP, FP, FN = (sum(c[l] for l in labels) for c in (tp, fp, fn))
        p, r = div(TP, TP + FP), div(TP, TP + FN)
        return p, r, f1(p, r)
    ps = [div(tp[l], tp[l] + fp[l]) for l in labels]
    rs = [div(tp[l], tp[l] + fn[l]) for l in labels]
    fs = [f1(p, r) for p, r in zip(ps, rs)]
    if average is None:
        return ps, rs, fs
    if average == "macro":
        w = [1.0] * len(labels)
    elif average == "weighted":
        w = [tp[l] + fn[l] for l in labels]                         # số mẫu thật của mỗi nhãn
    else:
        raise ValueError(average)
    tot = sum(w)
    return tuple(sum(wi * x for wi, x in zip(w, xs)) / tot for xs in (ps, rs, fs))


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _bleu_stats(cand, refs, max_n):
    clipped, totals = [], []
    for n in range(1, max_n + 1):
        c = _ngrams(cand, n)
        max_ref = Counter()
        for ref in refs:
            for g, k in _ngrams(ref, n).items():
                max_ref[g] = max(max_ref[g], k)
        clipped.append(sum(min(k, max_ref[g]) for g, k in c.items()))
        totals.append(max(len(cand) - n + 1, 0))
    r = min((abs(len(ref) - len(cand)), len(ref)) for ref in refs)[1]    # độ dài ref gần nhất, hòa → ref ngắn
    return clipped, totals, len(cand), r


def _combine(clipped, totals, c, r, smooth):
    logs = []
    for n, (m, t) in enumerate(zip(clipped, totals), start=1):
        if smooth and n > 1:
            m, t = m + 1, t + 1
        if m == 0 or t == 0:
            return 0.0
        logs.append(math.log(m / t))
    bp = 1.0 if c > r else math.exp(1 - r / c) if c > 0 else 0.0
    return bp * math.exp(sum(logs) / len(logs))


def bleu(candidate: list[str], references: list[list[str]], max_n: int = 4, smooth: bool = False) -> float:
    """Sentence BLEU (Papineni et al. 2002): trung bình nhân của modified precision 1..max_n gram
    (đếm bị chặn bởi số lần xuất hiện nhiều nhất trong một ref), nhân brevity penalty.
    smooth=True: cộng 1 vào tử và mẫu với n ≥ 2 (Lin & Och 2004)."""
    return _combine(*_bleu_stats(candidate, references, max_n), smooth)


def corpus_bleu(candidates, references_list, max_n: int = 4) -> float:
    """Corpus BLEU: CỘNG số đếm của mọi câu rồi mới chia (khác trung bình của sentence BLEU)."""
    C, T, c_len, r_len = [0] * max_n, [0] * max_n, 0, 0
    for cand, refs in zip(candidates, references_list):
        clipped, totals, c, r = _bleu_stats(cand, refs, max_n)
        C = [a + b for a, b in zip(C, clipped)]
        T = [a + b for a, b in zip(T, totals)]
        c_len, r_len = c_len + c, r_len + r
    return _combine(C, T, c_len, r_len, False)


def _prf_counts(overlap, n_cand, n_ref):
    p = overlap / n_cand if n_cand else 0.0
    r = overlap / n_ref if n_ref else 0.0
    return {"p": p, "r": r, "f": 2 * p * r / (p + r) if p + r else 0.0}


def rouge_n(candidate: list[str], reference: list[str], n: int = 1) -> dict:
    """ROUGE-N (Lin 2004): số n-gram trùng (đã chặn) chia cho số n-gram của ref (recall) / của candidate (precision)."""
    c, r = _ngrams(candidate, n), _ngrams(reference, n)
    return _prf_counts(sum((c & r).values()), sum(c.values()), sum(r.values()))


def lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: list[str], reference: list[str]) -> dict:
    """ROUGE-L: dựa trên độ dài dãy con chung dài nhất (LCS), không cần liên tiếp. F với β = 1."""
    return _prf_counts(lcs_len(candidate, reference), len(candidate), len(reference))


def normalize_answer(s: str) -> str:
    """Chuẩn hóa kiểu SQuAD: chữ thường, bỏ dấu câu, gộp khoảng trắng."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def perplexity(log_probs) -> float:
    """exp(-trung bình log p) với log tự nhiên: 'model phân vân giữa bao nhiêu lựa chọn đều nhau'."""
    log_probs = list(log_probs)
    return math.exp(-sum(log_probs) / len(log_probs))


def cohen_kappa(a, b) -> float:
    """Độ đồng thuận giữa hai người gán nhãn, đã trừ phần trùng do ngẫu nhiên: (p_o - p_e) / (1 - p_e)."""
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum(ca[k] * cb[k] for k in ca) / n ** 2
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0
