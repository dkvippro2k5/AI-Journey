"""Ngày 1 · Đếm token, chat template, chi phí, và các phép ước lượng nhanh (tham số, FLOP, bộ nhớ KV, độ trễ)."""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import tiktoken

ENCODING = "o200k_base"


@functools.lru_cache(maxsize=8)
def get_encoding(name: str = ENCODING) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding: str = ENCODING) -> int:
    """Số token của `text`. Chuỗi trông giống special token (ví dụ '<|endoftext|>') được đếm như văn bản thường."""
    return len(get_encoding(encoding).encode(text, disallowed_special=()))


def count_chat_tokens(messages: list[dict], encoding: str = ENCODING, *, per_message: int = 3, per_name: int = 1,
                      reply_priming: int = 3) -> int:
    """Ước lượng token của một danh sách message kiểu Chat Completions (công thức của OpenAI cookbook):
    mỗi message tốn `per_message` token khung + token của mọi giá trị (role, content, name); có `name` thì thêm
    `per_name`; cộng `reply_priming` token mở đầu lượt trả lời của assistant."""
    total = reply_priming
    for m in messages:
        total += per_message
        for key, value in m.items():
            total += count_tokens(value, encoding)
            if key == "name":
                total += per_name
    return total


def render_chatml(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Chat template kiểu ChatML (họ Qwen dùng định dạng này):
    '<|im_start|>{role}\\n{content}<|im_end|>\\n' cho từng message, rồi '<|im_start|>assistant\\n' nếu cần sinh tiếp."""
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


@dataclass(frozen=True)
class Price:
    """Giá theo USD cho 1 triệu token (GIÁ GIẢ ĐỊNH trong tài liệu). `cached_input` = giá token đầu vào trúng cache."""
    input: float
    output: float
    cached_input: float | None = None


def cost_usd(input_tokens: int, output_tokens: int, price: Price, cached_tokens: int = 0) -> float:
    """Chi phí một lần gọi. `cached_tokens` (≤ input_tokens) là phần đầu vào được tính theo giá cache."""
    if not 0 <= cached_tokens <= input_tokens:
        raise ValueError("cached_tokens phải nằm trong [0, input_tokens]")
    cached_price = price.input if price.cached_input is None else price.cached_input
    return ((input_tokens - cached_tokens) * price.input + cached_tokens * cached_price
            + output_tokens * price.output) / 1e6


def param_count(vocab: int, d: int, n_layers: int, n_heads: int, n_kv_heads: int, d_ff: int, *,
                tied: bool = True, qkv_bias: bool = False) -> int:
    """Số tham số của decoder kiểu LLaMA/Qwen: attention có GQA, FFN SwiGLU (3 ma trận), RMSNorm (chỉ có trọng số),
    không bias trừ khi `qkv_bias`. `tied=True`: lm_head dùng chung ma trận với embedding."""
    hd = d // n_heads
    kv = n_kv_heads * hd
    attn = d * d + 2 * d * kv + d * d + (d + 2 * kv if qkv_bias else 0)
    layer = attn + 3 * d * d_ff + 2 * d
    return vocab * d * (1 if tied else 2) + n_layers * layer + d


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, seq_len: int, batch: int = 1,
                   bytes_per_value: int = 2) -> int:
    """Bộ nhớ KV-cache: 2 (K và V) × lớp × head KV × d_head × độ dài × batch × số byte mỗi giá trị."""
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_value


def train_flops(n_params: float, n_tokens: float) -> float:
    """FLOP để train: C ≈ 6·N·D (2ND cho forward, 4ND cho backward)."""
    return 6 * n_params * n_tokens


def chinchilla_optimal(compute: float, tokens_per_param: float = 20.0) -> tuple[float, float]:
    """Cặp (N, D) tối ưu theo Chinchilla với ngân sách `compute`: D = k·N và C = 6·N·D ⇒ N = sqrt(C / 6k)."""
    n = math.sqrt(compute / (6 * tokens_per_param))
    return n, tokens_per_param * n


def latency_s(input_tokens: int, output_tokens: int, prefill_tps: float, decode_tps: float,
              overhead_s: float = 0.0) -> tuple[float, float]:
    """(TTFT, tổng thời gian): TTFT = overhead + đầu vào / tốc độ prefill; tổng = TTFT + đầu ra / tốc độ decode."""
    ttft = overhead_s + input_tokens / prefill_tps
    return ttft, ttft + output_tokens / decode_tps


def decode_tps_upper_bound(n_params: float, bytes_per_param: float, mem_bandwidth_gbs: float) -> float:
    """Trần tốc độ decode với batch 1: mỗi token phải đọc toàn bộ trọng số từ bộ nhớ một lần (memory-bound)."""
    return mem_bandwidth_gbs * 1e9 / (n_params * bytes_per_param)
