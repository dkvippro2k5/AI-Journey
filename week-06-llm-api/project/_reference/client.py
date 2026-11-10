"""Ngày 4 · Client async cho API kiểu Chat Completions: retry có Retry-After và jitter, streaming SSE, đo TTFT.

Phát triển từ `ask` của Tuần 1 (bài 09): thêm streaming, Retry-After, phân biệt lỗi nên thử lại và lỗi không nên.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import httpx

from gateway.clock import RealClock
from gateway.limits import next_delay, parse_retry_after
from gateway.stream import SSEParser, StreamAccumulator

RETRYABLE = {408, 409, 429, 500, 502, 503, 504, 529}


class ProviderError(Exception):
    def __init__(self, message: str, status: int | None = None, retryable: bool = False,
                 retry_after: float | None = None):
        super().__init__(message)
        self.status, self.retryable, self.retry_after = status, retryable, retry_after
        self.attempts = 0


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None
    latency_s: float
    ttft_s: float
    attempts: int


class LLMClient:
    """`http` là một httpx.AsyncClient đã đặt `base_url` (ví dụ https://api.groq.com/openai/v1 hoặc FakeLLM.client()).
    API key chỉ nằm ở đây, phía server; không bao giờ gửi về trình duyệt (bài 09)."""

    def __init__(self, http: httpx.AsyncClient, api_key: str, *, clock=None, max_attempts: int = 3,
                 base: float = 0.5, cap: float = 8.0, rng: random.Random | None = None):
        self.http, self.api_key = http, api_key
        self.clock = clock or RealClock()
        self.max_attempts, self.base, self.cap = max_attempts, base, cap
        self.rng = rng or random.Random(0)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _raise_for(resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            msg = resp.json()["error"]["message"]
        except Exception:                                   # noqa: BLE001 — thân lỗi không phải JSON
            msg = resp.text[:200]
        raise ProviderError(f"HTTP {resp.status_code}: {msg}", resp.status_code, resp.status_code in RETRYABLE,
                            parse_retry_after(resp.headers))

    async def chat(self, model: str, messages: list[dict], *, stream: bool = False, on_delta=None,
                   **params) -> Completion:
        """Gọi một lần (có thử lại). Thành công → Completion; thất bại sau `max_attempts` lần, hoặc gặp lỗi không nên
        thử lại (400, 401, 404…) → ProviderError (thuộc tính `attempts` = số lần đã gọi).
        `stream=True`: mỗi đoạn văn bản mới được đưa vào `on_delta(text)`; TTFT = lúc có đoạn văn bản đầu tiên.
        Đã gửi delta cho người dùng rồi mà luồng hỏng thì KHÔNG thử lại (tránh lặp chữ), báo lỗi luôn."""
        body = {"model": model, "messages": messages, **params}
        start = self.clock.now()
        for attempt in range(self.max_attempts):
            emitted = [False]
            try:
                if stream:
                    text, usage, finish, ttft = await self._stream_once(body, start, on_delta, emitted)
                else:
                    text, usage, finish = await self._post_once(body)
                    ttft = self.clock.now() - start
                return Completion(text, model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                                  finish, self.clock.now() - start, ttft, attempt + 1)
            except ProviderError as exc:
                exc.attempts = attempt + 1
                if not exc.retryable or emitted[0] or attempt == self.max_attempts - 1:
                    raise
                await self.clock.sleep(next_delay(attempt, exc.retry_after, self.base, self.cap, self.rng))
        raise AssertionError("unreachable")

    async def _post_once(self, body: dict) -> tuple[str, dict, str | None]:
        try:
            resp = await self.http.post("/chat/completions", json=body, headers=self._headers())
        except httpx.TransportError as exc:                 # timeout, mất kết nối…
            raise ProviderError(f"{type(exc).__name__}: {exc}", None, True) from exc
        self._raise_for(resp)
        data = resp.json()
        choice = data["choices"][0]
        return choice["message"].get("content") or "", data.get("usage") or {}, choice.get("finish_reason")

    async def _stream_once(self, body: dict, start: float, on_delta, emitted: list) -> tuple[str, dict, str | None, float]:
        body = {**body, "stream": True, "stream_options": {"include_usage": True}}
        parser, acc, ttft = SSEParser(), StreamAccumulator(), None
        try:
            async with self.http.stream("POST", "/chat/completions", json=body, headers=self._headers()) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    self._raise_for(resp)
                async for chunk in resp.aiter_bytes():
                    for ev in parser.feed(chunk):
                        delta = acc.add(ev)
                        if delta:
                            if ttft is None:
                                ttft = self.clock.now() - start
                            emitted[0] = True
                            if on_delta is not None:
                                on_delta(delta)
                parser.close()
        except httpx.TransportError as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}", None, True) from exc
        if not acc.done:
            raise ProviderError("luồng SSE kết thúc mà không có [DONE]", None, True)
        return acc.text, acc.usage or {}, acc.finish_reason, (ttft if ttft is not None else self.clock.now() - start)
