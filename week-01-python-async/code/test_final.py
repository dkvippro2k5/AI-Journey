import asyncio
import itertools

import pytest

from final_lib import iter_sse_data, map_bounded, retry


class FakeSleep:
    def __init__(self):
        self.calls = []

    async def __call__(self, seconds):
        self.calls.append(seconds)


def make_flaky(n_failures, exc=ConnectionError):
    state = {"calls": 0}

    async def fn(x):
        state["calls"] += 1
        if state["calls"] <= n_failures:
            raise exc("boom")
        return x * 2

    return fn, state


def test_succeeds_after_transient_failures():
    sleep = FakeSleep()
    fn, state = make_flaky(2)
    wrapped = retry(max_attempts=5, base=0.5, cap=20, sleep=sleep, rng=lambda: 1.0)(fn)
    assert asyncio.run(wrapped(21)) == 42
    assert state["calls"] == 3
    assert sleep.calls == [0.5, 1.0]


def test_non_retryable_raises_immediately():
    sleep = FakeSleep()
    fn, state = make_flaky(1, exc=ValueError)
    wrapped = retry(max_attempts=5, sleep=sleep)(fn)
    with pytest.raises(ValueError):
        asyncio.run(wrapped(1))
    assert state["calls"] == 1
    assert sleep.calls == []


def test_exhausts_and_reraises_last_error():
    sleep = FakeSleep()
    fn, state = make_flaky(100)
    wrapped = retry(max_attempts=4, base=1, cap=3, sleep=sleep, rng=lambda: 1.0)(fn)
    with pytest.raises(ConnectionError):
        asyncio.run(wrapped(1))
    assert state["calls"] == 4
    assert sleep.calls == [1, 2, 3]


def test_full_jitter_bounds():
    sleep = FakeSleep()
    fn, _ = make_flaky(5)
    wrapped = retry(max_attempts=6, base=0.5, cap=4, sleep=sleep, rng=lambda: 0.5)(fn)
    asyncio.run(wrapped(1))
    assert sleep.calls == [0.25, 0.5, 1.0, 2.0, 2.0]


def test_preserves_metadata():
    async def call_llm(prompt: str) -> str:
        """doc"""
        return prompt

    wrapped = retry()(call_llm)
    assert wrapped.__name__ == "call_llm"
    assert wrapped.__doc__ == "doc"



def test_map_bounded_preserves_order():
    async def fn(x):
        await asyncio.sleep(0.001 * (10 - x))
        return x * 10

    assert asyncio.run(map_bounded(fn, list(range(10)), 3)) == [x * 10 for x in range(10)]


def test_map_bounded_respects_limit():
    state = {"now": 0, "peak": 0}

    async def fn(x):
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.005)
        state["now"] -= 1
        return x

    asyncio.run(map_bounded(fn, list(range(20)), 5))
    assert state["peak"] == 5


def test_map_bounded_captures_exceptions_in_place():
    async def fn(x):
        if x == 3:
            raise ValueError("bad item")
        return x

    out = asyncio.run(map_bounded(fn, [1, 2, 3, 4], 2))
    assert out[:2] == [1, 2] and out[3] == 4
    assert isinstance(out[2], ValueError)


def test_map_bounded_edge_cases():
    async def fn(x):
        return x

    assert asyncio.run(map_bounded(fn, [], 4)) == []
    with pytest.raises(ValueError):
        asyncio.run(map_bounded(fn, [1], 0))


def test_sse_basic_and_multiline():
    lines = ["data: a", "", "data: line1", "data: line2", "", ]
    assert list(iter_sse_data(lines)) == ["a", "line1\nline2"]


def test_sse_ignores_comments_and_other_fields():
    lines = [": keep-alive", "event: delta", "id: 7", "data: x", "", "retry: 1000", ""]
    assert list(iter_sse_data(lines)) == ["x"]


def test_sse_strips_only_one_leading_space_and_crlf():
    lines = ["data:no-space\r\n", "\r\n", "data:  two\r\n", "\r\n"]
    assert list(iter_sse_data(lines)) == ["no-space", " two"]


def test_sse_stops_at_done_lazily():
    def explode():
        raise AssertionError("read past [DONE]")
        yield

    lines = itertools.chain(["data: 1", "", "data: [DONE]", ""], explode())
    assert list(iter_sse_data(lines)) == ["1"]


def test_sse_discards_incomplete_last_event():
    assert list(iter_sse_data(["data: 1", "", "data: partial"])) == ["1"]

