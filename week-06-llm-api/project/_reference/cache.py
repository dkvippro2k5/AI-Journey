"""Ngày 3 · Cache câu trả lời: khóa SHA-256 trên JSON chuẩn hóa, bộ nhớ đệm LRU có thời hạn (TTL)."""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict

# Chỉ các tham số làm thay đổi đầu ra mới được đưa vào khóa. "stream", "user", "timeout"… không làm đổi nội dung.
CACHE_PARAMS = ("temperature", "top_p", "max_tokens", "response_format", "seed", "stop", "tools")


def canonical_json(obj) -> str:
    """JSON chuẩn hóa: sắp xếp khóa, không khoảng trắng thừa, giữ nguyên tiếng Việt. Hai dict bằng nhau → cùng chuỗi."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def cache_key(model: str, messages: list[dict], params: dict | None = None) -> str:
    """sha256 (hex) của {model, messages (chỉ role và content), params (chỉ các khóa trong CACHE_PARAMS)}."""
    params = params or {}
    payload = {"model": model,
               "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
               "params": {k: params[k] for k in CACHE_PARAMS if k in params}}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class TTLCache:
    """Tối đa `max_items` phần tử; phần tử hết hạn sau `ttl` giây kể từ lúc `set`. Đầy thì bỏ phần tử dùng lâu nhất (LRU).
    Đếm `hits` / `misses` (phần tử hết hạn tính là miss)."""

    def __init__(self, max_items: int = 10_000, ttl: float = 3600.0, clock=time.monotonic):
        self.max_items, self.ttl, self.clock = max_items, ttl, clock
        self._data: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key: str):
        item = self._data.get(key)
        if item is None or self.clock() >= item[0]:
            if item is not None:
                del self._data[key]
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return item[1]

    def set(self, key: str, value) -> None:
        self._data[key] = (self.clock() + self.ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
