import math

import pytest
import tiktoken

from gateway.tokens import (Price, chinchilla_optimal, cost_usd, count_chat_tokens, count_tokens, decode_tps_upper_bound,
                            kv_cache_bytes, latency_s, param_count, render_chatml, train_flops)

ENC = tiktoken.get_encoding("o200k_base")


def test_count_tokens_matches_tiktoken_and_special_tokens_are_text():
    for s in ["", "tôi muốn hủy gói cước", "I want to cancel my plan", "kiem tra so du"]:
        assert count_tokens(s) == len(ENC.encode(s))
    s = "xin chào <|endoftext|> bạn"                                              # không được ném lỗi
    assert count_tokens(s) == len(ENC.encode(s, disallowed_special=()))
    assert count_tokens("<|endoftext|>") > 1                                      # bị tách như văn bản, không phải 1 token đặc biệt


def test_count_chat_tokens_formula():
    msgs = [{"role": "system", "content": "Bạn là trợ lý."},
            {"role": "user", "content": "hủy gói GX30", "name": "an"}]
    n = lambda s: len(ENC.encode(s))  # noqa: E731
    expected = 3 + (3 + n("system") + n("Bạn là trợ lý.")) + (3 + n("user") + n("hủy gói GX30") + n("an") + 1)
    assert count_chat_tokens(msgs) == expected
    assert count_chat_tokens([], reply_priming=0) == 0


def test_render_chatml():
    msgs = [{"role": "system", "content": "Bạn là trợ lý."}, {"role": "user", "content": "hủy gói"}]
    assert render_chatml(msgs) == ("<|im_start|>system\nBạn là trợ lý.<|im_end|>\n"
                                   "<|im_start|>user\nhủy gói<|im_end|>\n<|im_start|>assistant\n")
    assert render_chatml(msgs, add_generation_prompt=False).endswith("hủy gói<|im_end|>\n")


def test_cost_usd():
    p = Price(3.0, 15.0, cached_input=0.3)
    assert cost_usd(1_000_000, 0, p) == pytest.approx(3.0)
    assert cost_usd(2000, 500, p) == pytest.approx((2000 * 3 + 500 * 15) / 1e6)
    assert cost_usd(2000, 500, p, cached_tokens=1500) == pytest.approx((500 * 3 + 1500 * 0.3 + 500 * 15) / 1e6)
    assert cost_usd(100, 0, Price(1.0, 2.0), cached_tokens=100) == pytest.approx(100 / 1e6)  # không có giá cache
    with pytest.raises(ValueError):
        cost_usd(100, 0, p, cached_tokens=101)


def test_param_count_matches_published_models():
    # cấu hình công bố của hai model mở (config.json trên Hugging Face)
    assert param_count(151_936, 896, 24, 14, 2, 4864, tied=True, qkv_bias=True) == 494_032_768      # Qwen2.5-0.5B
    assert param_count(128_256, 4096, 32, 32, 8, 14_336, tied=False) == 8_030_261_248              # Llama-3-8B
    # GQA làm giảm tham số so với MHA cùng kích thước
    assert param_count(1000, 512, 4, 8, 2, 1024) < param_count(1000, 512, 4, 8, 8, 1024)


def test_kv_cache_bytes():
    assert kv_cache_bytes(24, 2, 64, 1024) == 12_582_912                                          # 12 MiB, fp16
    assert kv_cache_bytes(32, 8, 128, 131_072) == 16 * 2 ** 30                                    # 16 GiB
    assert kv_cache_bytes(32, 8, 128, 1000, batch=4, bytes_per_value=1) == 2 * 32 * 8 * 128 * 1000 * 4


def test_scaling_and_latency_estimates():
    assert train_flops(7e9, 2e12) == pytest.approx(8.4e22)
    n, d = chinchilla_optimal(5.76e23)                                           # ngân sách của Chinchilla
    assert n == pytest.approx(6.93e10, rel=0.01) and d == pytest.approx(20 * n)
    assert 6 * n * d == pytest.approx(5.76e23)
    ttft, total = latency_s(2000, 300, prefill_tps=4000, decode_tps=50, overhead_s=0.2)
    assert ttft == pytest.approx(0.7) and total == pytest.approx(6.7)
    assert decode_tps_upper_bound(7e9, 2, 100) == pytest.approx(100e9 / 14e9)
    assert math.isclose(decode_tps_upper_bound(7e9, 0.5, 100), 4 * decode_tps_upper_bound(7e9, 2, 100))
