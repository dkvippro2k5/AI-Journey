"""Cho sẵn — không cần sửa. Server giả lập API kiểu OpenAI Chat Completions, chạy offline."""
import asyncio
import json
import random

import httpx


class FakeLLM:
    def __init__(self, fail_rate=0.1, min_latency=0.05, max_latency=0.3, seed=0):
        self.fail_rate = fail_rate
        self.min_latency = min_latency
        self.max_latency = max_latency
        self.rng = random.Random(seed)
        self.in_flight = 0
        self.peak_in_flight = 0
        self.calls = 0

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.rng.uniform(self.min_latency, self.max_latency))
            if self.rng.random() < self.fail_rate:
                return httpx.Response(503, json={"error": {"message": "overloaded"}})
            body = json.loads(request.content)
            prompt = body["messages"][-1]["content"]
            answer = f"[{request.url.host}] {prompt[:40]}"
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": max(1, len(prompt) // 4), "completion_tokens": max(1, len(answer) // 4)},
            })
        finally:
            self.in_flight -= 1

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
