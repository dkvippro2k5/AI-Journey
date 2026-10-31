import asyncio
import inspect
import itertools
import logging
import re

import pytest

from multi_caller.utils import batched, read_jsonl, timed

TOOK = re.compile(r"^(\w+) took (\d+\.\d{3})s$")


def _took(caplog):
    msgs = [r.getMessage() for r in caplog.records if r.name == "multi_caller"]
    assert msgs, "timed must log via logger 'multi_caller'"
    m = TOOK.match(msgs[-1])
    assert m, f"bad log format: {msgs[-1]!r}"
    return m.group(1), float(m.group(2))


def test_timed_sync(caplog):
    caplog.set_level(logging.INFO, logger="multi_caller")

    @timed
    def add(a, b):
        """doc"""
        return a + b

    assert add(1, 2) == 3
    assert add.__name__ == "add" and add.__doc__ == "doc"
    name, _ = _took(caplog)
    assert name == "add"


def test_timed_async_measures_real_duration(caplog):
    caplog.set_level(logging.INFO, logger="multi_caller")

    @timed
    async def call_api():
        await asyncio.sleep(0.05)
        return "ok"

    assert inspect.iscoroutinefunction(call_api)
    assert asyncio.run(call_api()) == "ok"
    name, secs = _took(caplog)
    assert name == "call_api" and secs >= 0.04


def test_timed_logs_on_error(caplog):
    caplog.set_level(logging.INFO, logger="multi_caller")

    @timed
    def boom():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        boom()
    assert _took(caplog)[0] == "boom"


def test_read_jsonl_is_lazy_and_skips_bad_lines(tmp_path, caplog):
    p = tmp_path / "x.jsonl"
    p.write_text('{"id": 1}\n\nnot json\n{"id": 2}\n', encoding="utf-8")
    gen = read_jsonl(p)
    assert inspect.isgenerator(gen)
    with caplog.at_level(logging.WARNING, logger="multi_caller"):
        assert list(gen) == [{"id": 1}, {"id": 2}]
    assert sum(r.levelno == logging.WARNING for r in caplog.records) == 1


def test_batched_basic():
    assert list(batched(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(batched([], 3)) == []


def test_batched_is_lazy_on_infinite_input():
    first = next(batched(itertools.count(), 2))
    assert first == [0, 1]


def test_batched_rejects_bad_n():
    with pytest.raises(ValueError):
        list(batched([1, 2], 0))
