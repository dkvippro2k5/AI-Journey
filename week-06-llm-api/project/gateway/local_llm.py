"""LLM nhỏ chạy local làm "provider thật" (cho sẵn, TÙY CHỌN; cần torch + transformers và model đã tải).

Nói cùng giao thức Chat Completions như FakeLLM, chạy trong tiến trình qua `httpx.MockTransport`, nên `LLMClient`
và `Gateway` dùng được y nguyên:

    local = LocalLLM()                                  # nạp model một lần (~1 GB RAM với fp32)
    client = LLMClient(local.client(), api_key="local")
    c = await client.chat("local", [{"role": "user", "content": "xin chào"}], max_tokens=50)

- Mọi tên model đều được phục vụ bằng cùng một model (để gateway có thể gọi "model-small" lẫn "model-large").
- `temperature=0` → greedy; >0 → lấy mẫu (có `top_p`). `max_tokens` mặc định 256.
- `response_format` bị BỎ QUA: model nhỏ không có constrained decoding ở đây, nên đầu ra có thể sai JSON thật.
- Stream: token được đẩy ra ngay khi sinh (TextIteratorStreamer chạy ở thread riêng).
"""
from __future__ import annotations

import asyncio
import json
import threading
import time

import httpx

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"                # model instruct ~0.5B tham số (trong tài liệu gọi là MODEL_ID)


class LocalLLM:
    def __init__(self, model_id: str = MODEL_ID, threads: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
        self.lock = threading.Lock()                     # một lần sinh tại một thời điểm (CPU)
        self.calls = 0

    def client(self, base_url: str = "https://local.llm/v1") -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler), base_url=base_url,
                                 timeout=httpx.Timeout(600.0))

    def _inputs(self, messages):
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

    def _gen_kwargs(self, body, inputs):
        t = float(body.get("temperature", 1.0) or 0.0)
        kw = dict(**inputs, max_new_tokens=int(body.get("max_tokens") or 256), pad_token_id=self.tok.eos_token_id)
        if t > 0:
            kw.update(do_sample=True, temperature=t, top_p=float(body.get("top_p", 1.0)), top_k=0)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        return kw

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        body = json.loads(request.content)
        inputs = self._inputs(body["messages"])
        n_in = int(inputs["input_ids"].shape[1])
        kw = self._gen_kwargs(body, inputs)
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=self._sse(body, kw, n_in))

        def run():
            with self.lock, self.torch.no_grad():
                return self.model.generate(**kw)
        out = await asyncio.to_thread(run)
        new = out[0, n_in:]
        text = self.tok.decode(new, skip_special_tokens=True)
        finish = "length" if len(new) >= kw["max_new_tokens"] and new[-1] != self.tok.eos_token_id else "stop"
        n_out = int((new != self.tok.eos_token_id).sum())
        return httpx.Response(200, json={
            "id": f"local-{self.calls}", "object": "chat.completion", "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}})

    async def _sse(self, body, kw, n_in):
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        counter = {"n": 0}

        def run():
            with self.lock, self.torch.no_grad():
                out = self.model.generate(**kw, streamer=streamer)
                counter["n"] = int(out.shape[1] - n_in)
        th = threading.Thread(target=run, daemon=True)
        th.start()

        def chunk(delta, finish=None):
            return {"id": f"local-{self.calls}", "object": "chat.completion.chunk", "model": body.get("model"),
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        def ev(obj):
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()

        yield ev(chunk({"role": "assistant", "content": ""}))
        it = iter(streamer)
        while True:
            piece = await asyncio.to_thread(next, it, None)
            if piece is None:
                break
            if piece:
                yield ev(chunk({"content": piece}))
        await asyncio.to_thread(th.join)
        yield ev(chunk({}, "length" if counter["n"] >= kw["max_new_tokens"] else "stop"))
        if (body.get("stream_options") or {}).get("include_usage"):
            yield ev({"choices": [], "usage": {"prompt_tokens": n_in, "completion_tokens": counter["n"],
                                               "total_tokens": n_in + counter["n"]}})
        yield b"data: [DONE]\n\n"


def tokens_per_second(local: LocalLLM, prompt_tokens: int = 256, new_tokens: int = 64, repeat: int = 3) -> dict:
    """Đo nhanh tốc độ prefill (token/s) và decode (token/s) trên máy bạn với một prompt ngẫu nhiên."""
    torch = local.torch
    ids = torch.randint(1000, 20000, (1, prompt_tokens))
    best_pre, best_dec = float("inf"), float("inf")
    with torch.no_grad():
        for _ in range(repeat):
            t = time.perf_counter()
            local.model.generate(ids, max_new_tokens=1, min_new_tokens=1, do_sample=False,
                                 pad_token_id=local.tok.eos_token_id)
            pre = time.perf_counter() - t
            t = time.perf_counter()
            local.model.generate(ids, max_new_tokens=new_tokens + 1, min_new_tokens=new_tokens + 1, do_sample=False,
                                 pad_token_id=local.tok.eos_token_id)
            best_pre, best_dec = min(best_pre, pre), min(best_dec, (time.perf_counter() - t - pre) / new_tokens)
    return {"prefill_tps": prompt_tokens / best_pre, "decode_tps": 1 / best_dec}
