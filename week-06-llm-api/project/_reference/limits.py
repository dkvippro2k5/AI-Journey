"""Ngày 3 · Giới hạn tốc độ: token bucket (cho từng khách), đọc header Retry-After, chọn thời gian chờ trước khi thử lại."""
from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


class TokenBucket:
    """Xô chứa tối đa `capacity` token, được thêm `rate` token mỗi giây. Một request tốn `n` token.
    Cho phép burst tới `capacity`, tốc độ trung bình dài hạn không vượt `rate`. `clock` trả về số giây (đơn điệu)."""

    def __init__(self, rate: float, capacity: float, clock=time.monotonic):
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate và capacity phải dương")
        self.rate, self.capacity, self.clock = rate, capacity, clock
        self.tokens = float(capacity)
        self.last = clock()

    def _refill(self) -> None:
        now = self.clock()
        elapsed = max(0.0, now - self.last)               # đồng hồ lùi (hiếm, nhưng có) thì coi như 0
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last = max(self.last, now)

    def try_acquire(self, n: float = 1) -> bool:
        """Đủ token thì trừ và trả True; không đủ thì không trừ gì, trả False. `n > capacity` → ValueError."""
        if n > self.capacity:
            raise ValueError("n lớn hơn capacity: request này không bao giờ được phép")
        self._refill()
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

    def wait_time(self, n: float = 1) -> float:
        """Số giây phải chờ để có đủ `n` token (0 nếu đã đủ)."""
        self._refill()
        return max(0.0, (n - self.tokens) / self.rate)


class KeyedLimiter:
    """Mỗi khóa (user_id, API key, …) có một TokenBucket riêng, tạo khi gặp lần đầu."""

    def __init__(self, rate: float, capacity: float, clock=time.monotonic):
        self.rate, self.capacity, self.clock = rate, capacity, clock
        self.buckets: dict[str, TokenBucket] = {}

    def _bucket(self, key: str) -> TokenBucket:
        if key not in self.buckets:
            self.buckets[key] = TokenBucket(self.rate, self.capacity, self.clock)
        return self.buckets[key]

    def allow(self, key: str, n: float = 1) -> bool:
        return self._bucket(key).try_acquire(n)

    def retry_after(self, key: str, n: float = 1) -> float:
        return self._bucket(key).wait_time(n)


def parse_retry_after(headers, now: datetime | None = None) -> float | None:
    """Số giây server yêu cầu chờ. Đọc `retry-after-ms` (mili giây) nếu có, nếu không thì `retry-after`: hoặc là số giây,
    hoặc là một HTTP-date (RFC 9110, mục 10.2.3). Không có / không đọc được → None; thời điểm đã qua → 0."""
    h = {k.lower(): v for k, v in dict(headers).items()}
    ms = h.get("retry-after-ms")
    if ms is not None:
        try:
            return max(0.0, float(ms) / 1000)
        except ValueError:
            pass
    v = h.get("retry-after")
    if v is None:
        return None
    v = v.strip()
    try:
        return max(0.0, float(v))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(v)
    except (TypeError, ValueError):
        return None
    now = now or datetime.now(timezone.utc)
    return max(0.0, (when - now).total_seconds())


def next_delay(attempt: int, retry_after: float | None = None, base: float = 0.5, cap: float = 8.0,
               rng: random.Random | None = None) -> float:
    """Thời gian chờ trước lần thử thứ `attempt + 1` (attempt bắt đầu từ 0).
    Server có nói Retry-After → làm đúng theo đó (không chờ ít hơn). Không → full jitter (Tuần 1, bài 09):
    uniform(0, min(cap, base·2^attempt))."""
    if retry_after is not None:
        return retry_after
    return (rng or random).uniform(0, min(cap, base * 2 ** attempt))
