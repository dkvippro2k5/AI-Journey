import math

import pytest
import torch

from intent.decoding import generate, sample_next, softmax_t, top_k_filter, top_p_filter

NEG = float("-inf")


def test_temperature():
    z = torch.tensor([2.0, 1.0, 0.0])
    assert torch.allclose(softmax_t(z, 1.0), torch.softmax(z, -1))
    sharp, flat = softmax_t(z, 0.5), softmax_t(z, 2.0)
    assert sharp[0] > softmax_t(z, 1.0)[0] > flat[0]
    assert sharp[0].item() == pytest.approx(math.exp(4) / (math.exp(4) + math.exp(2) + 1))


def test_top_k_keeps_ties_at_boundary():
    z = torch.tensor([3.0, 1.0, 3.0, 2.0])
    assert top_k_filter(z, 1).tolist() == [3.0, NEG, 3.0, NEG]
    assert top_k_filter(z, 3).tolist() == [3.0, NEG, 3.0, 2.0]
    assert top_k_filter(z[None].repeat(2, 1), 2).shape == (2, 4)                     # chạy theo batch


def test_top_p_smallest_set():
    z = torch.log(torch.tensor([0.05, 0.5, 0.15, 0.3]))                              # thứ tự cố tình bị xáo
    kept = lambda p: torch.isfinite(top_p_filter(z, p)).tolist()
    assert kept(0.45) == [False, True, False, False]                                 # 0.5 đã đủ
    assert kept(0.75) == [False, True, False, True]                                  # 0.5 + 0.3
    assert kept(0.85) == [False, True, True, True]                                   # + 0.15
    assert kept(0.01) == [False, True, False, False]                                 # luôn giữ token tốt nhất
    assert torch.allclose(top_p_filter(z, 0.75)[[1, 3]], z[[1, 3]])                  # giá trị giữ nguyên


def test_greedy_and_filters_restrict_samples():
    z = torch.tensor([0.0, 2.0, 1.0, -1.0])
    assert sample_next(z, temperature=0).item() == 1
    g = torch.Generator().manual_seed(0)
    draws = [sample_next(z, top_k=2, generator=g).item() for _ in range(300)]
    assert set(draws) == {1, 2}
    draws = [sample_next(z, top_p=0.6, generator=g).item() for _ in range(100)]
    assert set(draws) == {1}                                                          # p(1) = 0.64 ≥ 0.6


def test_sampling_frequencies_follow_temperature():
    z = torch.tensor([1.0, 0.0])
    g = torch.Generator().manual_seed(1)
    freq = torch.stack([sample_next(z.expand(20000, 2), temperature=t, generator=g) for t in (0.5, 2.0)]).float()
    p0 = (freq == 0).float().mean(1)
    assert p0[0].item() == pytest.approx(1 / (1 + math.exp(-2)), abs=0.015)          # T = 0.5 → logit nhân 2
    assert p0[1].item() == pytest.approx(1 / (1 + math.exp(-0.5)), abs=0.015)


def test_generate_stops_at_eos():
    table = {0: 2, 2: 1, 1: 3}                                                        # 0 → 2 → 1 → 3 (eos)

    def step(ids):
        out = torch.full((5,), -10.0)
        out[table.get(ids[-1], 4)] = 10.0
        return out

    assert generate(step, [0], 10, eos_id=3, temperature=0) == [0, 2, 1, 3]
    assert generate(step, [0], 2, temperature=0) == [0, 2, 1]
