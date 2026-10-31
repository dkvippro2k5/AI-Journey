import asyncio
import os
import random
import time

import httpx

from multi_caller.models import CallResult, ProviderConfig

RETRYABLE = {408, 429, 500, 502, 503, 504, 529}


class TemporaryError(Exception):
    pass


async def _send(client: httpx.AsyncClient, provider: ProviderConfig, api_key: str, prompt: str) -> dict:
    try:
        resp = await client.post(
            f"{provider.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": provider.model, "messages": [{"role": "user", "content": prompt}]},
        )
    except httpx.TransportError as exc:
        raise TemporaryError(f"{type(exc).__name__}: {exc}") from exc
    if resp.status_code in RETRYABLE:
        raise TemporaryError(f"HTTP {resp.status_code}")
    resp.raise_for_status()
    return resp.json()


async def ask(client, provider: ProviderConfig, prompt_id: int, prompt: str, *,
              max_attempts: int = 3, base: float = 0.5, cap: float = 4.0,
              sleep=asyncio.sleep) -> CallResult:
    result = CallResult(provider=provider.name, prompt_id=prompt_id)
    api_key = os.environ.get(provider.api_key_env)
    if not api_key:
        result.error = f"missing env var {provider.api_key_env}"
        return result
    start = time.perf_counter()
    try:
        for attempt in range(max_attempts):
            result.attempts = attempt + 1
            try:
                data = await _send(client, provider, api_key, prompt)
                break
            except TemporaryError as exc:
                if attempt == max_attempts - 1:
                    result.error = str(exc)
                    return result
                await sleep(random.uniform(0, min(cap, base * 2 ** attempt)))
            except httpx.HTTPStatusError as exc:
                result.error = f"HTTP {exc.response.status_code}"
                return result
        usage = data.get("usage", {})
        result.text = data["choices"][0]["message"]["content"]
        result.input_tokens = usage.get("prompt_tokens", 0)
        result.output_tokens = usage.get("completion_tokens", 0)
        result.cost_usd = (result.input_tokens * provider.price_in + result.output_tokens * provider.price_out) / 1e6
        return result
    finally:
        result.latency_s = time.perf_counter() - start
