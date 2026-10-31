"""Test phần C — final Tuần 4. Chạy từ thư mục tuần: python -m pytest -q code/test_final.py"""
import itertools
import math
import zlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from final_lib import beam_search, conv2d, conv_out_size, lstm_forward, maxpool2d


# ---------- C1 ----------

def test_conv_out_size():
    assert conv_out_size(224, 7, stride=2, padding=3) == 112
    assert conv_out_size(112, 3, stride=2, padding=1) == 56
    assert conv_out_size(32, 5) == 28
    assert conv_out_size(10, 3, dilation=2) == 6
    assert conv_out_size(7, 3, stride=2) == 3


def test_conv2d_hand_example():
    x = np.array([[1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2]], dtype=float).reshape(1, 1, 4, 4)
    k = np.array([[1, 0, -1]] * 3, dtype=float).reshape(1, 1, 3, 3)            # ví dụ bài 07
    assert conv2d(x, k).reshape(2, 2).tolist() == [[0, 2], [-1, -1]]
    assert conv2d(x, k, stride=2, padding=1).reshape(2, 2).tolist() == [[-3, 1], [-2, -1]]


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1), (3, 2)])
def test_conv2d_matches_torch(stride, padding):
    rng = np.random.default_rng(stride * 10 + padding)
    x, w, b = rng.normal(size=(2, 3, 9, 7)), rng.normal(size=(4, 3, 3, 2)), rng.normal(size=4)
    ref = F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b), stride=stride, padding=padding).numpy()
    out = conv2d(x, w, b, stride=stride, padding=padding)
    assert out.shape == ref.shape and np.allclose(out, ref)
    assert np.allclose(conv2d(x, w, stride=stride, padding=padding) + b[None, :, None, None], out)


def test_maxpool_and_channel_check():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2, 3, 8, 6))
    for k, s in [(2, None), (3, 2), (2, 1)]:
        ref = F.max_pool2d(torch.tensor(x), k, stride=s).numpy()
        assert np.allclose(maxpool2d(x, k, s), ref)
    with pytest.raises(ValueError):
        conv2d(x, rng.normal(size=(4, 2, 3, 3)))                                 # 3 kênh vào, kernel 2 kênh


# ---------- C2 ----------

def test_lstm_hand_example():
    W_ih = np.ones((4, 1))                                                        # thứ tự i, f, g, o
    out, (h, c) = lstm_forward(np.ones((1, 1, 1)), np.zeros((1, 1)), np.full((1, 1), 0.5),
                               W_ih, np.zeros((4, 1)), np.array([0.0, 1.0, 0.0, 0.0]), np.zeros(4))
    assert h.item() == pytest.approx(0.5559, abs=1e-4) and c.item() == pytest.approx(0.9972, abs=1e-4)
    assert out.shape == (1, 1, 1)


def test_lstm_matches_torch():
    torch.manual_seed(0)
    lstm = torch.nn.LSTM(4, 6).double()
    x, h0, c0 = torch.randn(5, 3, 4).double(), torch.randn(1, 3, 6).double(), torch.randn(1, 3, 6).double()
    ref, (hr, cr) = lstm(x, (h0, c0))
    p = {n: t.detach().numpy() for n, t in lstm.named_parameters()}
    out, (h, c) = lstm_forward(x.numpy(), h0[0].numpy(), c0[0].numpy(),
                               p["weight_ih_l0"], p["weight_hh_l0"], p["bias_ih_l0"], p["bias_hh_l0"])
    assert out.shape == (5, 3, 6)
    assert np.allclose(out, ref.detach().numpy()) and np.allclose(h, hr[0].detach().numpy())
    assert np.allclose(c, cr[0].detach().numpy())


def test_lstm_forget_bias_keeps_memory():
    H = 2
    b = np.zeros(4 * H)
    b[H:2 * H] = 10.0                                                             # f ≈ 1
    b[:H] = -10.0                                                                 # i ≈ 0
    _, (_, c) = lstm_forward(np.random.default_rng(0).normal(size=(50, 1, 3)), np.zeros((1, H)),
                             np.array([[1.0, -2.0]]), np.zeros((4 * H, 3)), np.zeros((4 * H, H)), b, np.zeros(4 * H))
    assert np.allclose(c, [[1.0, -2.0]], atol=0.01)                               # cell giữ nguyên qua 50 bước


# ---------- C3 ----------

LN = math.log
TABLE = {("<s>",): {"A": LN(0.5), "B": LN(0.4), "</s>": LN(0.1)},
         ("<s>", "A"): {"x": LN(0.3), "y": LN(0.3), "z": LN(0.4)},
         ("<s>", "B"): {"x": LN(0.9), "y": LN(0.05), "z": LN(0.05)}}


def toy(prefix):
    return TABLE.get(prefix, {"</s>": 0.0})                                       # sau 2 token: chắc chắn kết thúc


def test_beam_hand_example_from_lesson():
    seq, score = beam_search(toy, "<s>", "</s>", beam_width=1, max_len=5)
    assert seq == ["A", "z", "</s>"] and score == pytest.approx(LN(0.2))         # greedy
    seq, score = beam_search(toy, "<s>", "</s>", beam_width=2, max_len=5)
    assert seq == ["B", "x", "</s>"] and score == pytest.approx(LN(0.36))


def _random_model(V=3, seed=0):
    def f(prefix):
        rng = np.random.default_rng(zlib.crc32(repr((seed, prefix)).encode()))
        lp = np.log(rng.dirichlet(np.ones(V)))
        return {t: float(lp[t]) for t in range(V)}                                # không có eos
    return f


def test_beam_width_one_is_greedy_and_wide_beam_is_exhaustive():
    f = _random_model()
    prefix, total = ("<s>",), 0.0
    for _ in range(4):                                                            # greedy tự viết
        d = f(prefix)
        t = max(d, key=d.get)
        prefix, total = prefix + (t,), total + d[t]
    seq, score = beam_search(f, "<s>", "</s>", beam_width=1, max_len=4)
    assert seq == list(prefix[1:]) and score == pytest.approx(total)
    def lp(s):
        return sum(f(("<s>",) + tuple(s[:i]))[s[i]] for i in range(len(s)))
    best = max(itertools.product(range(3), repeat=4), key=lp)
    seq, score = beam_search(f, "<s>", "</s>", beam_width=81, max_len=4)
    assert tuple(seq) == best and score == pytest.approx(lp(best))
    assert beam_search(f, "<s>", "</s>", beam_width=2, max_len=4)[1] <= score + 1e-12


def test_length_normalization_changes_choice():
    table = {("<s>",): {"a": LN(0.6), "</s>": LN(0.4)}, ("<s>", "a"): {"</s>": LN(0.6), "b": LN(0.4)}}
    def f(prefix):
        return table[prefix]
    assert beam_search(f, "<s>", "</s>", beam_width=2, max_len=2, alpha=0.0)[0] == ["</s>"]
    seq, score = beam_search(f, "<s>", "</s>", beam_width=2, max_len=2, alpha=1.0)
    assert seq == ["a", "</s>"] and score == pytest.approx(LN(0.36) / 2)
