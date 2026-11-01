"""Test phần C của final Tuần 5. Viết lời giải vào code/final_lib.py rồi chạy:
python -m pytest -q code/test_final.py   (ở thư mục week-05-nlp)
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from final_lib import BigramLM, causal_mha, decode_step, init_cache, mlm_mask, wordpiece_tokenize  # noqa: E402

VOCAB = {"goi", "go", "dang", "##ky", "##n", "##i", "hu", "##y", "huy", "c", "##u", "##o", "##c"}


# ---------- C1 · WordPiece ----------
def test_wordpiece_longest_match_first():
    assert wordpiece_tokenize("dangky", VOCAB) == ["dang", "##ky"]
    assert wordpiece_tokenize("gon", VOCAB) == ["go", "##n"]
    assert wordpiece_tokenize("goi", VOCAB) == ["goi"]                           # không tách khi cả từ có trong vocab
    assert wordpiece_tokenize("huyy", VOCAB) == ["huy", "##y"]
    assert wordpiece_tokenize("cuoc", VOCAB) == ["c", "##u", "##o", "##c"]


def test_wordpiece_unknown_and_too_long():
    assert wordpiece_tokenize("xin", VOCAB) == ["[UNK]"]                         # không có tiền tố nào khớp
    assert wordpiece_tokenize("goix", VOCAB) == ["[UNK]"]                        # phần sau không tách được → cả từ UNK
    assert wordpiece_tokenize("go" * 60, VOCAB | {"##go"}, max_chars=100) == ["[UNK]"]
    assert wordpiece_tokenize("gogo", VOCAB | {"##go"}) == ["go", "##go"]


# ---------- C2 · causal MHA + KV-cache ----------
def _weights(d, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 1 / math.sqrt(d), (d, d)) for _ in range(4)]


def test_causal_mha_hand_example_single_head_identity():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    I = np.eye(2)
    out = causal_mha(X, I, I, I, I, n_heads=1)
    assert np.allclose(out[0], [1, 0])
    assert np.allclose(out[1], [0.3302, 0.6698], atol=1e-4)
    assert np.allclose(out[2], [0.7517, 0.7517], atol=1e-4)


def test_causal_mha_ignores_future_and_shape():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 8))
    W = _weights(8)
    a = causal_mha(X, *W, n_heads=2)
    Y = X.copy()
    Y[4:] = rng.normal(size=(2, 8))
    b = causal_mha(Y, *W, n_heads=2)
    assert a.shape == (6, 8)
    assert np.allclose(a[:4], b[:4]) and not np.allclose(a[4:], b[4:])


def test_decode_step_matches_full_and_cache_grows():
    rng = np.random.default_rng(2)
    T, d, h = 7, 12, 3
    X = rng.normal(size=(T, d))
    W = _weights(d, seed=3)
    full = causal_mha(X, *W, n_heads=h)
    cache = init_cache(h, d // h)
    outs = []
    for t in range(T):
        o, cache = decode_step(X[t], cache, *W, n_heads=h)
        outs.append(o)
    assert np.allclose(np.stack(outs), full, atol=1e-10)
    assert cache["k"].shape == (h, T, d // h) and cache["v"].shape == (h, T, d // h)


def test_multi_head_differs_from_single_head():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(5, 8))
    W = _weights(8, seed=5)
    assert not np.allclose(causal_mha(X, *W, n_heads=1), causal_mha(X, *W, n_heads=4))


# ---------- C3 · Bigram LM ----------
TRAIN = [["hủy", "gói"], ["đăng", "ký", "gói"], ["hủy", "gói", "cước"]]


def test_bigram_probabilities_hand_example():
    lm = BigramLM(k=1.0).fit(TRAIN)
    assert lm.vocab_ == sorted(["hủy", "gói", "đăng", "ký", "cước", "</s>"])        # V = 6, không có <s>
    assert lm.prob("<s>", "hủy") == pytest.approx((2 + 1) / (3 + 6))
    assert lm.prob("gói", "</s>") == pytest.approx((2 + 1) / (3 + 6))
    assert lm.prob("gói", "cước") == pytest.approx((1 + 1) / (3 + 6))
    assert lm.prob("cước", "hủy") == pytest.approx((0 + 1) / (1 + 6))


def test_bigram_distribution_sums_to_one():
    lm = BigramLM(k=0.5).fit(TRAIN)
    for prev in ["<s>", "hủy", "gói", "cước", "ký"]:
        assert sum(lm.prob(prev, w) for w in lm.vocab_) == pytest.approx(1.0)


def test_bigram_perplexity():
    lm = BigramLM(k=1.0).fit(TRAIN)
    expected = math.exp(-(math.log(3 / 9) + math.log(3 / 8) + math.log(3 / 9)) / 3)    # <s> hủy gói </s>
    assert lm.perplexity([["hủy", "gói"]]) == pytest.approx(expected)
    big_k = BigramLM(k=1e9).fit(TRAIN)
    assert big_k.perplexity([["hủy", "gói"]]) == pytest.approx(6.0, rel=1e-6)          # k lớn → gần phân phối đều
    assert lm.perplexity(TRAIN) < lm.perplexity([["cước", "đăng", "hủy"]])


# ---------- C4 · MLM masking ----------
def test_mlm_mask_labels_and_specials():
    ids = np.array([0, 5, 6, 7, 8, 2, 1, 1])                                            # 0 = <s>, 2 = </s>, 1 = <pad>
    inputs, labels = mlm_mask(ids, mask_id=4, vocab_size=50, special_ids=(0, 1, 2), p=1.0,
                              rng=np.random.default_rng(0))
    assert labels.tolist() == [-100, 5, 6, 7, 8, -100, -100, -100]                     # chọn mọi token thường
    assert inputs[[0, 5, 6, 7]].tolist() == [0, 2, 1, 1]                                # special giữ nguyên
    assert ids.tolist() == [0, 5, 6, 7, 8, 2, 1, 1]                                     # không sửa đầu vào


def test_mlm_mask_rates():
    rng = np.random.default_rng(1)
    ids = rng.integers(10, 1000, size=200_000)
    inputs, labels = mlm_mask(ids, mask_id=3, vocab_size=1000, special_ids=(0, 1, 2), p=0.15, rng=rng)
    chosen = labels != -100
    assert chosen.mean() == pytest.approx(0.15, abs=0.005)
    assert np.all(labels[chosen] == ids[chosen])
    masked = inputs[chosen] == 3
    same = inputs[chosen] == ids[chosen]
    assert masked.mean() == pytest.approx(0.8, abs=0.01)
    assert same.mean() == pytest.approx(0.1, abs=0.01)                                  # (+ ngẫu nhiên trùng, rất hiếm)
    assert np.all(inputs[~chosen] == ids[~chosen])
