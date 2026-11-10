"""Ngày 2 · Bảo mật đầu vào/đầu ra: che PII trước khi ghi log, phát hiện prompt injection (heuristic),
kiểm tra rò rỉ chuỗi canary, che API key khi in ra."""
from __future__ import annotations

import re
import unicodedata

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b")
_ID = re.compile(r"(?<!\d)(\d{12}|\d{9})(?!\d)")                         # CCCD 12 số, CMND 9 số
_PHONE = re.compile(r"(?<!\d)(\+84|0)(\d[\s.]?){8,9}\d(?!\d)")


def redact_pii(text: str) -> str:
    """Thay email → <email>, CCCD/CMND → <id>, số điện thoại → <phone>. Thứ tự: email, rồi số giấy tờ, rồi điện thoại
    (để số CCCD 12 chữ số không bị nhận nhầm thành số điện thoại)."""
    text = _EMAIL.sub("<email>", text)
    text = _ID.sub("<id>", text)
    return _PHONE.sub("<phone>", text)


def _fold(text: str) -> str:
    """Chữ thường, bỏ dấu tiếng Việt, gộp khoảng trắng: 'Bỏ  QUA' → 'bo qua'."""
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text)


INJECTION_PATTERNS = [
    r"(bo qua|quen|ignore|disregard|forget)\b.{0,40}\b(huong dan|chi dan|quy tac|instruction|rule|prompt)",
    r"system prompt|prompt he thong|huong dan he thong",
    r"\bban (bay gio|gio) la\b|\byou are now\b",
    r"(tiet lo|in ra|cho .{0,15}xem|reveal|print).{0,30}(ma noi bo|bi mat|secret|api key|password|mat khau)",
    r"<\|?\s*(im_start|im_end|endoftext|system)\s*\|?>|<\s*/?\s*customer_message\s*>",
]
_INJ = [re.compile(p) for p in INJECTION_PATTERNS]


def looks_like_injection(text: str) -> bool:
    """Heuristic: có khớp một trong INJECTION_PATTERNS sau khi `_fold`. Chỉ là tầng phát hiện thô (có bỏ sót, có báo
    nhầm); không thay được việc tách quyền và kiểm tra đầu ra."""
    folded = _fold(text)
    return any(p.search(folded) for p in _INJ)


def leaks_canary(output: str, canary: str) -> bool:
    """Đầu ra có chứa chuỗi canary (bí mật cài trong system prompt) không, kể cả khi bị chèn khoảng trắng/gạch ngang."""
    squash = lambda s: re.sub(r"[\s\-_.]", "", s.lower())  # noqa: E731
    return squash(canary) in squash(output)


def mask_secret(secret: str, keep: int = 4) -> str:
    """'sk-abcdef123456' → 'sk-…3456'. Chuỗi ngắn (≤ 2·keep) → '****'. Dùng khi ghi log cấu hình."""
    if len(secret) <= 2 * keep:
        return "****"
    return f"{secret[:3]}…{secret[-keep:]}"
