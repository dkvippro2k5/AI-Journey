import random
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from gateway.limits import KeyedLimiter, TokenBucket, next_delay, parse_retry_after


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_token_bucket_burst_then_rate():
    c = Clock()
    b = TokenBucket(rate=2, capacity=5, clock=c)
    assert [b.try_acquire() for _ in range(6)] == [True] * 5 + [False]           # burst 5, cái thứ 6 bị chặn
    assert b.wait_time() == pytest.approx(0.5)
    c.t = 0.5
    assert b.try_acquire() and not b.try_acquire()
    c.t = 100.0
    assert b.tokens <= 5 and b.wait_time(5) == 0                                  # không tích quá capacity
    assert b.try_acquire(5) and not b.try_acquire(0.5)
    with pytest.raises(ValueError):
        b.try_acquire(6)


def test_token_bucket_long_run_rate():
    c = Clock()
    b = TokenBucket(rate=3, capacity=10, clock=c)
    ok = 0
    for i in range(10_000):                                                       # 100 request/giây trong 100 giây
        c.t = i * 0.01
        ok += b.try_acquire()
    assert ok == pytest.approx(10 + 3 * 100, abs=2)


def test_keyed_limiter_isolates_users():
    c = Clock()
    lim = KeyedLimiter(rate=1, capacity=2, clock=c)
    assert [lim.allow("a") for _ in range(3)] == [True, True, False]
    assert lim.allow("b") and lim.allow("b")                                      # b không bị ảnh hưởng bởi a
    assert lim.retry_after("a") == pytest.approx(1.0)


def test_parse_retry_after():
    assert parse_retry_after({"Retry-After": "7"}) == 7.0
    assert parse_retry_after({"retry-after-ms": "1500", "retry-after": "9"}) == 1.5
    now = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after({"retry-after": format_datetime(now + timedelta(seconds=30), usegmt=True)}, now) == 30
    assert parse_retry_after({"retry-after": format_datetime(now - timedelta(seconds=30), usegmt=True)}, now) == 0
    assert parse_retry_after({}) is None and parse_retry_after({"retry-after": "mai nhé"}) is None


def test_next_delay():
    assert next_delay(0, retry_after=12.0) == 12.0                               # server bảo chờ bao lâu thì chờ bấy lâu
    rng = random.Random(0)
    ds = [next_delay(3, base=0.5, cap=8, rng=rng) for _ in range(2000)]
    assert 0 <= min(ds) and max(ds) <= 4.0 and 1.8 < sum(ds) / len(ds) < 2.2    # uniform(0, 4)
    assert max(next_delay(10, base=0.5, cap=8, rng=rng) for _ in range(500)) <= 8
