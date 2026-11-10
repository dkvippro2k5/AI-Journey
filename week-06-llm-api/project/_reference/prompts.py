"""Ngày 2 · Prompt template có version, registry, chia A/B ổn định, bọc dữ liệu không tin cậy, bỏ phiếu đa số."""
from __future__ import annotations

import hashlib
import json
import re
import string
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """Một phiên bản prompt. `system` và `user` dùng cú pháp `str.format` ({ten_bien}); `few_shot` là các cặp
    (user, assistant) chèn giữa system và câu hỏi thật."""
    name: str
    version: str
    system: str
    user: str
    few_shot: tuple[tuple[str, str], ...] = ()
    changelog: str = ""

    def fields(self) -> set[str]:
        """Tên các biến xuất hiện trong `system` và `user`."""
        names = set()
        for part in (self.system, self.user):
            names |= {f for _, f, _, _ in string.Formatter().parse(part) if f}
        return names

    def render(self, **variables) -> list[dict]:
        """Danh sách message: system, các cặp few-shot (user/assistant), rồi user.
        Thiếu biến → KeyError; thừa biến (thường do gõ sai tên) → ValueError."""
        need = self.fields()
        missing, extra = need - variables.keys(), variables.keys() - need
        if missing:
            raise KeyError(f"thiếu biến: {sorted(missing)}")
        if extra:
            raise ValueError(f"biến không dùng: {sorted(extra)}")
        msgs = [{"role": "system", "content": self.system.format(**variables)}]
        for q, a in self.few_shot:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        return msgs + [{"role": "user", "content": self.user.format(**variables)}]

    @property
    def fingerprint(self) -> str:
        """12 ký tự hex đầu của SHA-256 trên nội dung (system, user, few_shot). Đổi một ký tự là đổi fingerprint."""
        payload = json.dumps([self.system, self.user, [list(p) for p in self.few_shot]], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def parse_version(version: str) -> tuple[int, ...]:
    """'1.10.0' → (1, 10, 0), để '1.10.0' > '1.9.0' (so chuỗi thì sai)."""
    return tuple(int(x) for x in version.split("."))


class PromptRegistry:
    def __init__(self):
        self._items: dict[tuple[str, str], PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        """Một (name, version) chỉ ứng với MỘT nội dung: đăng ký lại cùng nội dung thì bỏ qua, khác nội dung → ValueError."""
        key = (template.name, template.version)
        old = self._items.get(key)
        if old is not None and old.fingerprint != template.fingerprint:
            raise ValueError(f"{key} đã tồn tại với nội dung khác; hãy tăng version")
        self._items[key] = template

    def versions(self, name: str) -> list[str]:
        return sorted((v for n, v in self._items if n == name), key=parse_version)

    def get(self, name: str, version: str | None = None) -> PromptTemplate:
        """`version=None` → bản mới nhất theo `parse_version`. Không có → KeyError."""
        if version is None:
            vs = self.versions(name)
            if not vs:
                raise KeyError(name)
            version = vs[-1]
        return self._items[(name, version)]

    @staticmethod
    def assign(name: str, user_id: str, weights: dict[str, float]) -> str:
        """Chia A/B ổn định: cùng (name, user_id) luôn rơi vào cùng version.
        u = 8 hex đầu của sha256(f"{name}:{user_id}") / 2**32; duyệt version theo `parse_version`, cộng dồn trọng số
        (đã chuẩn hóa về tổng 1), trả version đầu tiên có tổng cộng dồn > u."""
        u = int(hashlib.sha256(f"{name}:{user_id}".encode()).hexdigest()[:8], 16) / 2 ** 32
        total = sum(weights.values())
        acc = 0.0
        ordered = sorted(weights, key=parse_version)
        for v in ordered:
            acc += weights[v] / total
            if u < acc:
                return v
        return ordered[-1]


def wrap_untrusted(text: str, tag: str = "customer_message") -> str:
    """Bọc dữ liệu không tin cậy trong thẻ <tag>…</tag> ("spotlighting"). Xóa mọi thẻ mở/đóng cùng tên có sẵn trong
    `text` (không phân biệt hoa thường, cho phép khoảng trắng) để người dùng không thể "đóng thẻ" sớm."""
    clean = re.sub(rf"<\s*/?\s*{re.escape(tag)}\s*>", "", text, flags=re.IGNORECASE)
    return f"<{tag}>\n{clean}\n</{tag}>"


def majority_vote(answers: list) -> tuple[object, float]:
    """(đáp án nhiều phiếu nhất, tỉ lệ phiếu) trên các đáp án khác None. Hòa → đáp án xuất hiện trước.
    Không có đáp án hợp lệ → (None, 0.0)."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 0.0
    counts = Counter(valid)
    best = max(counts.values())
    winner = next(a for a in valid if counts[a] == best)
    return winner, best / len(valid)
