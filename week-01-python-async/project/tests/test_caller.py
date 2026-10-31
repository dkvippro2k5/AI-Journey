import asyncio
import json

import httpx
import pytest

from multi_caller.caller import ask
from multi_caller.models import ProviderConfig

PROVIDER = ProviderConfig(
    name="p1", base_url="https://p1.test/v1", model="MODEL_ID",
    api_key_env="P1_API_KEY", price_in=3.0, price_out=15.0,
)

OK_BODY = {
    "choices": [{"message": {"role": "assistant", "content": "xin chào"}}],
    "usage": {"prompt_tokens": 2000, "completion_tokens": 500},
}


class NoSleep:
    def __init__(self):
        self.calls = []

    async def __call__(self, s):
        self.calls.append(s)


def run(responses, **kwargs):
    seen = []

    def handler(request):
        seen.append(request)
        item = responses[min(len(seen), len(responses)) - 1]
        if isinstance(item, Exception):
            raise item
        return httpx.Response(item, json=OK_BODY if item == 200 else {"error": "x"})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await ask(client, PROVIDER, 7, "Xin chào", sleep=NoSleep(), **kwargs)

    return asyncio.run(go()), seen


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("P1_API_KEY", "sk-test")


def test_success_parses_text_tokens_cost():
    res, seen = run([200])
    assert res.ok and res.text == "xin chào" and res.attempts == 1
    assert (res.input_tokens, res.output_tokens) == (2000, 500)
    assert res.cost_usd == pytest.approx(0.0135)
    assert res.provider == "p1" and res.prompt_id == 7 and res.latency_s >= 0


def test_request_shape():
    _, seen = run([200])
    req = seen[0]
    assert req.method == "POST" and str(req.url) == "https://p1.test/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    body = json.loads(req.content)
    assert body["model"] == "MODEL_ID"
    assert body["messages"][-1] == {"role": "user", "content": "Xin chào"}


def test_retries_transient_then_succeeds():
    res, seen = run([503, 429, 200])
    assert res.ok and res.attempts == 3 and len(seen) == 3


def test_transport_error_is_retried():
    res, seen = run([httpx.ConnectError("refused"), 200])
    assert res.ok and res.attempts == 2


def test_permanent_error_not_retried():
    res, seen = run([401, 200])
    assert not res.ok and res.attempts == 1 and len(seen) == 1
    assert "401" in res.error


def test_gives_up_after_max_attempts():
    res, seen = run([503], max_attempts=4)
    assert not res.ok and res.attempts == 4 and len(seen) == 4
    assert "503" in res.error


def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("P1_API_KEY")
    res, seen = run([200])
    assert not res.ok and len(seen) == 0 and "P1_API_KEY" in res.error
