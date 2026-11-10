"""Ngày 3 · Quản lý context window: bỏ bớt lượt cũ cho vừa ngân sách token, cắt giữa một văn bản dài,
tóm tắt phần hội thoại cũ."""
from __future__ import annotations

from typing import Callable

from gateway.tokens import count_chat_tokens, count_tokens


def fit_history(messages: list[dict], budget: int,
                count: Callable[[list[dict]], int] = count_chat_tokens) -> tuple[list[dict], int]:
    """Bỏ các message cũ nhất (ngay sau phần system đầu danh sách) cho tới khi `count(messages) <= budget`.
    Luôn giữ: mọi message system ở đầu, và message cuối cùng. Sau khi bỏ, phần lịch sử còn lại không được bắt đầu bằng
    một message assistant (bỏ luôn nó để giữ đúng thứ tự user → assistant).
    Trả về (danh sách mới, số message đã bỏ). Không đủ chỗ ngay cả khi chỉ còn system + message cuối thì trả về đúng hai
    phần đó (người gọi phải cắt nội dung)."""
    n_sys = 0
    while n_sys < len(messages) - 1 and messages[n_sys]["role"] == "system":
        n_sys += 1
    head, middle, last = messages[:n_sys], list(messages[n_sys:-1]), messages[-1:]
    dropped = 0
    while middle and count(head + middle + last) > budget:
        middle.pop(0)
        dropped += 1
        while middle and middle[0]["role"] == "assistant":
            middle.pop(0)
            dropped += 1
    return head + middle + last, dropped


def truncate_middle(text: str, max_tokens: int, marker: str = " […] ") -> str:
    """Giữ phần đầu và phần cuối của `text`, bỏ phần giữa, sao cho `count_tokens(kết quả) <= max_tokens`.
    Cắt theo KÝ TỰ (không theo token, để không làm vỡ ký tự tiếng Việt), giữ k ký tự đầu và k ký tự cuối với k lớn nhất
    tìm bằng chia đôi. Văn bản đã vừa thì trả nguyên."""
    if count_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text) // 2
    while lo < hi:
        k = (lo + hi + 1) // 2
        if count_tokens(text[:k] + marker + text[-k:]) <= max_tokens:
            lo = k
        else:
            hi = k - 1
    return text[:lo] + marker + (text[-lo:] if lo else "")


def summarize_old(messages: list[dict], keep_last: int, summarizer: Callable[[list[dict]], str]) -> list[dict]:
    """Thay các message cũ (trừ system ở đầu và `keep_last` message cuối) bằng MỘT message system
    "Tóm tắt hội thoại trước: …" do `summarizer` tạo (ngoài đời là một lần gọi LLM rẻ). Không có gì để tóm tắt thì trả nguyên."""
    n_sys = 0
    while n_sys < len(messages) and messages[n_sys]["role"] == "system":
        n_sys += 1
    body = messages[n_sys:]
    if len(body) <= keep_last:
        return list(messages)
    old, recent = body[:len(body) - keep_last], body[len(body) - keep_last:]
    summary = {"role": "system", "content": f"Tóm tắt hội thoại trước: {summarizer(old)}"}
    return messages[:n_sys] + [summary] + recent
