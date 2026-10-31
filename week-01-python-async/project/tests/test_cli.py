import asyncio
import json
from pathlib import Path

import pytest

from multi_caller.cli import load_providers, main, run_all, summarize_results, write_results
from multi_caller.fake import FakeLLM
from multi_caller.models import CallResult

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [{"id": i, "prompt": f"câu hỏi {i}"} for i in range(4)]


async def no_sleep(_):
    return None


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("ALPHA_API_KEY", "a")
    monkeypatch.setenv("BETA_API_KEY", "b")


@pytest.fixture
def providers():
    return load_providers(ROOT / "providers.example.toml")


def _run(fake, providers, concurrency, **kw):
    async def go():
        async with fake.client() as client:
            return await run_all(client, providers, PROMPTS, concurrency=concurrency, sleep=no_sleep, **kw)
    return asyncio.run(go())


def test_load_providers(providers):
    assert [p.name for p in providers] == ["alpha", "beta"]
    assert providers[0].price_out == 15.0


def test_run_all_order_and_concurrency(providers):
    fake = FakeLLM(fail_rate=0.0, min_latency=0.02, max_latency=0.05)
    results = _run(fake, providers, concurrency=3)
    assert [(r.prompt_id, r.provider) for r in results] == [
        (i, p) for i in range(4) for p in ("alpha", "beta")
    ]
    assert all(r.ok for r in results)
    assert fake.peak_in_flight == 3


def test_run_all_failures_are_results_not_exceptions(providers):
    fake = FakeLLM(fail_rate=1.0, min_latency=0.0, max_latency=0.0)
    results = _run(fake, providers, concurrency=4, max_attempts=2)
    assert len(results) == 8 and not any(r.ok for r in results)
    assert all(r.attempts == 2 for r in results)


def test_summarize_results():
    rs = [
        CallResult("a", 0, latency_s=1.0, input_tokens=10, output_tokens=5, cost_usd=0.1, attempts=1),
        CallResult("a", 1, latency_s=3.0, input_tokens=10, output_tokens=5, cost_usd=0.1, attempts=1),
        CallResult("b", 0, latency_s=2.0, attempts=3, error="HTTP 503"),
    ]
    s = summarize_results(rs)
    assert list(s.columns) == ["n", "success_rate", "p50_latency_s", "p95_latency_s", "total_tokens", "total_cost_usd"]
    assert s.loc["a", "n"] == 2 and s.loc["a", "success_rate"] == 1.0
    assert s.loc["a", "p50_latency_s"] == pytest.approx(2.0)
    assert s.loc["a", "total_tokens"] == 30 and s.loc["a", "total_cost_usd"] == pytest.approx(0.2)
    assert s.loc["b", "success_rate"] == 0.0


def test_write_results_roundtrip(tmp_path):
    path = tmp_path / "r.jsonl"
    write_results([CallResult("a", 1, text="xin chào")], path)
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["provider"] == "a" and row["text"] == "xin chào" and row["error"] is None


def test_main_end_to_end_fake(tmp_path, capsys):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in PROMPTS), encoding="utf-8")
    out = tmp_path / "out.jsonl"
    main([str(prompts), "--providers", str(ROOT / "providers.example.toml"), "--out", str(out), "--fake", "--concurrency", "4"])
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8
    printed = capsys.readouterr().out
    assert "alpha" in printed and "beta" in printed and "success_rate" in printed
