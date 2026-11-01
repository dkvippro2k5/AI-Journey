import math

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

from intent.metrics import (bleu, cohen_kappa, corpus_bleu, exact_match, lcs_len, perplexity, prf, rouge_l, rouge_n,
                            token_f1)

REF = "tôi muốn hủy gói cước này".split()


@pytest.mark.parametrize("average", [None, "macro", "micro", "weighted"])
def test_prf_matches_sklearn(average):
    rng = np.random.default_rng(0)
    labels = ["a", "b", "c", "d"]
    y = rng.choice(labels, 300, p=[0.5, 0.3, 0.15, 0.05]).tolist()
    p = [t if rng.random() < 0.7 else rng.choice(labels[:3]) for t in y]           # "d" không bao giờ được đoán
    mine = prf(y, p, labels=labels, average=average)
    sk = precision_recall_fscore_support(y, p, labels=labels, average=average, zero_division=0)[:3]
    assert np.allclose(np.array(mine, dtype=float), np.array(sk, dtype=float))


def test_bleu_hand_example():
    cand = "tôi muốn hủy gói này".split()
    # p1 = 5/5, p2 = 3/4, p3 = 2/3, p4 = 1/2; BP = exp(1 - 6/5)
    expected = math.exp(1 - 6 / 5) * (1 * 0.75 * (2 / 3) * 0.5) ** 0.25
    assert bleu(cand, [REF]) == pytest.approx(expected)
    assert bleu(REF, [REF]) == pytest.approx(1.0)
    assert bleu("hủy hủy hủy hủy".split(), [REF], max_n=1) == pytest.approx(0.25 * math.exp(1 - 6 / 4))   # bị chặn
    assert bleu("gói cước".split(), [REF]) == 0.0                                    # không có 3-gram trùng
    assert bleu("gói cước".split(), [REF], smooth=True) > 0


def test_bleu_picks_closest_reference_length():
    cand = "a b c".split()
    assert bleu(cand, [["a", "b", "c", "d", "e"], ["a", "b", "c"]], max_n=1) == pytest.approx(1.0)   # ref dài 3


def test_corpus_bleu_is_not_mean_of_sentence_bleu():
    cands = [["a", "b", "c", "d"], ["x", "y"]]
    refs = [[["a", "b", "c", "d"]], [["x", "z"]]]
    assert np.mean([bleu(c, r, max_n=2) for c, r in zip(cands, refs)]) == pytest.approx(0.5)
    assert corpus_bleu(cands, refs, max_n=2) == pytest.approx(math.sqrt(5 / 6 * 3 / 4))


def test_rouge_hand_example():
    cand = "hủy gói cước giúp tôi".split()
    assert rouge_n(cand, REF, 1) == pytest.approx({"p": 4 / 5, "r": 4 / 6, "f": 8 / 11})
    assert rouge_n(cand, REF, 2)["r"] == pytest.approx(2 / 5)
    assert lcs_len(cand, REF) == 3                                                   # hủy gói cước
    assert rouge_l(cand, REF) == pytest.approx({"p": 3 / 5, "r": 3 / 6, "f": 6 / 11})


def test_exact_match_and_token_f1():
    assert exact_match("Gói GX30.", "gói gx30") == 1.0
    assert exact_match("gói GX30 tháng", "gói gx30") == 0.0
    assert token_f1("gói GX30 tháng", "gói gx30") == pytest.approx(0.8)
    assert token_f1("abc", "xyz") == 0.0


def test_perplexity_and_kappa():
    assert perplexity([math.log(0.5), math.log(0.25), math.log(0.125)]) == pytest.approx(4.0)
    assert perplexity([math.log(0.1)] * 7) == pytest.approx(10.0)
    a = ["x", "x", "y", "y", "x", "y", "x", "x"]
    b = ["x", "y", "y", "y", "x", "x", "x", "x"]
    assert cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b)) == pytest.approx(0.21875 / 0.46875)
