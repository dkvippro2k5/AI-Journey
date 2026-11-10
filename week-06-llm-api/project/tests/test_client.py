import asyncio

import pytest

from gateway.client import LLMClient, ProviderError
from gateway.clock import SimClock
from gateway.fake import FakeLLM

MSGS = [{"role": "user", "content": "xin chào tổng đài"}]


def run(coro):
    return asyncio.run(coro)


def make(fake_kw=None, **client_kw):
    clock = SimClock()
    fake = FakeLLM(clock=clock, **(fake_kw or {}))
    return fake, LLMClient(fake.client(), "test-key", clock=clock, **client_kw), clock


def test_chat_non_stream_usage_and_latency():
    fake, client, clock = make()
    c = run(client.chat("model-small", MSGS, temperature=0))
    assert c.text.startswith("[model-small]") and c.finish_reason == "stop" and c.attempts == 1
    assert c.input_tokens > 0 and c.output_tokens > 0
    spec = fake.models["model-small"]
    expected = spec.overhead_s + c.input_tokens / spec.prefill_tps + c.output_tokens / spec.decode_tps
    assert c.latency_s == pytest.approx(expected) and c.ttft_s == pytest.approx(c.latency_s)
    c2 = run(client.chat("model-small", MSGS, max_tokens=3))
    assert c2.finish_reason == "length" and c2.output_tokens == 3


def test_stream_deltas_ttft_and_usage():
    fake, client, clock = make()
    deltas = []
    c = run(client.chat("model-large", MSGS, stream=True, on_delta=deltas.append))
    assert "".join(deltas) == c.text and len(deltas) > 3                          # parser chịu được byte bị cắt vụn
    assert c.output_tokens > 0 and c.input_tokens > 0                             # usage từ chunk cuối
    spec = fake.models["model-large"]
    prefill = spec.overhead_s + c.input_tokens / spec.prefill_tps
    assert prefill < c.ttft_s < prefill + 0.25                                    # TTFT = prefill + sinh mảnh đầu tiên
    assert c.latency_s > c.ttft_s


def test_retry_after_is_respected_and_non_retryable_fails_fast():
    fake, client, clock = make({"rate_limit_rate": 1.0, "retry_after": 7}, max_attempts=3)
    with pytest.raises(ProviderError) as e:
        run(client.chat("model-small", MSGS))
    assert e.value.status == 429 and e.value.attempts == 3 and fake.calls == 3
    assert clock.now() == pytest.approx(14.0)                                     # chờ 7 s × 2 lần, đúng Retry-After
    fake, client, clock = make()
    with pytest.raises(ProviderError) as e:
        run(client.chat("khong-co-model", MSGS))
    assert e.value.status == 404 and not e.value.retryable and fake.calls == 1
    bad = LLMClient(fake.client(), "sai-key", clock=clock)
    with pytest.raises(ProviderError) as e:
        run(bad.chat("model-small", MSGS))
    assert e.value.status == 401 and fake.calls == 2


def test_transient_errors_and_timeouts_are_retried():
    fake, client, clock = make({"error_rate": 0.3, "seed": 3}, max_attempts=6)
    results = [run(client.chat("model-small", MSGS)) for _ in range(20)]
    assert all(r.text for r in results) and max(r.attempts for r in results) > 1
    fake, client, clock = make({"timeout_rate": 1.0, "timeout_s": 10}, max_attempts=2, base=0.5)
    with pytest.raises(ProviderError) as e:
        run(client.chat("model-small", MSGS))
    assert e.value.retryable and e.value.attempts == 2 and clock.now() >= 20
