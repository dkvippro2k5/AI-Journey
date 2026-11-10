"""Provider LLM giả (cho sẵn, KHÔNG cần sửa). Nói đúng giao thức Chat Completions kiểu OpenAI (giao thức mà
Groq, OpenRouter, Gemini (endpoint tương thích), vLLM, Ollama… đều hỗ trợ), chạy offline qua `httpx.MockTransport`.

    fake = FakeLLM(oracle=build_oracle(workload), clock=SimClock())
    http = fake.client()                      # httpx.AsyncClient, base_url = https://fake.llm/v1
    r = await http.post("/chat/completions", headers={"Authorization": "Bearer test-key"}, json={...})

Có những gì:
- `POST /chat/completions`, `stream` true/false (SSE, chunk `chat.completion.chunk`, `[DONE]`,
  `stream_options.include_usage`), `max_tokens` (cắt → finish_reason "length"), `response_format`
  (`json_object`: luôn đúng cú pháp JSON nhưng vẫn có thể sai schema; `json_schema` + `strict`: luôn đúng schema),
  `usage`.
- Lỗi hạ tầng: 401 sai key, 404 sai model, 429 kèm `retry-after`, 500/503, timeout, khoảng "sập" theo thời gian.
- "Bộ não" giả: model được hỏi trả JSON {reply, intent, slots}. Nó biết đáp án đúng qua `oracle` (văn bản → nhãn),
  trả đúng với xác suất `accuracy` của model, hỏng định dạng với xác suất `format_error`, làm theo câu prompt
  injection với xác suất `obey_injection` (thấp hơn khi câu đó nằm trong thẻ <customer_message>).
- Byte của luồng SSE được cắt thành mảnh ngẫu nhiên 1–48 byte (có khi cắt giữa ký tự UTF-8) để kiểm tra parser.

Mọi xác suất và tốc độ là GIẢ ĐỊNH để mô phỏng; số đo thật trên một LLM nhỏ nằm ở các bài 02–06.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass

import httpx
import pandas as pd
import tiktoken

from gateway.clock import RealClock
from gateway.data import INJECTIONS, INTENTS

_ENC = tiktoken.get_encoding("o200k_base")


def _ntok(s: str) -> int:
    return len(_ENC.encode(s, disallowed_special=()))


@dataclass(frozen=True)
class FakeModel:
    name: str
    accuracy: float            # P(intent và slot đúng)
    format_error: float        # P(đầu ra hỏng định dạng) khi không có response_format
    obey_injection: float      # P(làm theo injection) khi câu injection KHÔNG được bọc thẻ
    obey_wrapped: float        # P(làm theo injection) khi câu injection nằm trong <customer_message>
    overhead_s: float          # độ trễ mạng + hàng đợi
    prefill_tps: float         # token đầu vào xử lý mỗi giây
    decode_tps: float          # token đầu ra sinh mỗi giây


FAKE_MODELS = {
    "model-small": FakeModel("model-small", 0.90, 0.12, 0.40, 0.15, 0.15, 5000, 150),
    "model-large": FakeModel("model-large", 0.97, 0.04, 0.15, 0.03, 0.40, 2500, 50),
}

REPLIES = {
    "tra_cuoc": "Dạ, anh/chị soạn KT gửi 1234 (giả định) để xem số dư và data còn lại ạ.",
    "nap_tien": "Dạ, anh/chị nạp tiền qua ứng dụng hoặc soạn NAP <mã thẻ> gửi 1234 (giả định) ạ.",
    "dang_ky_goi": "Dạ, để đăng ký anh/chị soạn DK {goi} gửi 1234 (giả định) ạ.",
    "huy_goi": "Dạ, để hủy gói anh/chị soạn HUY {goi} gửi 1234 (giả định) ạ.",
    "loi_mang": "Dạ, em rất tiếc. Anh/chị thử khởi động lại máy; nếu vẫn lỗi em sẽ tạo phiếu kỹ thuật ạ.",
    "doi_sim": "Dạ, anh/chị mang CCCD tới cửa hàng gần nhất để đổi SIM, hoặc đổi eSIM trên ứng dụng ạ.",
    "khieu_nai": "Dạ, em xin lỗi về sự cố. Em đã ghi nhận khiếu nại và sẽ phản hồi trong 24 giờ ạ.",
    "gap_tvv": "Dạ, em đang chuyển anh/chị tới tổng đài viên, xin chờ trong giây lát ạ.",
    "khac": "Dạ, em chỉ hỗ trợ các dịch vụ viễn thông. Anh/chị cần hỗ trợ gì về thuê bao ạ?",
}
SYNTAX_ERRORS = ["fence", "preface", "trailing_comma", "single_quotes", "truncated"]
SCHEMA_ERRORS = ["missing_field", "bad_enum", "extra_field"]
_TAG = re.compile(r"<\s*/?\s*customer_message\s*>", re.IGNORECASE)
_WRAPPED = re.compile(r"<customer_message>\s*(.*?)\s*</customer_message>", re.DOTALL)


def _key(text: str) -> str:
    return _TAG.sub("", text).strip()


def build_oracle(df) -> dict:
    """Từ một DataFrame có cột text, intent, goi, sdt, tien → {văn bản: đáp án} cho FakeLLM."""
    out = {}
    for r in df.itertuples():
        tien = None if pd.isna(r.tien) else int(r.tien)
        goi = r.goi if isinstance(r.goi, str) else None
        sdt = r.sdt if isinstance(r.sdt, str) else None
        out[_key(r.text)] = {"intent": r.intent, "goi": goi, "sdt": sdt, "tien": tien}
    return out


def _is_injection(text: str) -> bool:
    low = text.lower()
    return any(p.lower() in low for p in INJECTIONS) or "system prompt" in low


class FakeLLM:
    def __init__(self, oracle: dict | None = None, *, models: dict | None = None, clock=None, seed: int = 0,
                 api_key: str = "test-key", rate_limit_rate: float = 0.0, retry_after: float = 1.0,
                 error_rate: float = 0.0, timeout_rate: float = 0.0, timeout_s: float = 10.0,
                 outages: dict | None = None, chunk_bytes: tuple[int, int] = (1, 48)):
        self.oracle = oracle or {}
        self.models = models or FAKE_MODELS
        self.clock = clock or RealClock()
        self.rng = random.Random(seed)
        self.api_key = api_key
        self.rate_limit_rate, self.retry_after = rate_limit_rate, retry_after
        self.error_rate, self.timeout_rate, self.timeout_s = error_rate, timeout_rate, timeout_s
        self.outages = outages or {}                       # {model: [(t0, t1), ...]} → 503 trong [t0, t1)
        self.chunk_bytes = chunk_bytes
        self.calls = 0
        self.status = Counter()
        self.tokens = Counter()                            # (model, "in"/"out") → tổng token

    # ---------- HTTP ----------
    def client(self, base_url: str = "https://fake.llm/v1") -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler), base_url=base_url)

    def _error(self, status: int, message: str, kind: str, headers: dict | None = None) -> httpx.Response:
        self.status[status] += 1
        return httpx.Response(status, json={"error": {"message": message, "type": kind}}, headers=headers)

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if request.headers.get("authorization") != f"Bearer {self.api_key}":
            return self._error(401, "invalid api key", "authentication_error")
        if not request.url.path.endswith("/chat/completions"):
            return self._error(404, "not found", "not_found")
        body = json.loads(request.content)
        spec = self.models.get(body.get("model"))
        if spec is None:
            return self._error(404, f"model {body.get('model')!r} not found", "model_not_found")
        now = self.clock.now()
        if any(t0 <= now < t1 for t0, t1 in self.outages.get(spec.name, [])):
            await self.clock.sleep(spec.overhead_s)
            return self._error(503, "service unavailable", "overloaded")
        if self.rng.random() < self.rate_limit_rate:
            return self._error(429, "rate limit exceeded", "rate_limit", {"retry-after": f"{self.retry_after:g}"})
        if self.rng.random() < self.error_rate:
            return self._error(self.rng.choice([500, 503]), "internal error", "server_error")
        if self.rng.random() < self.timeout_rate:
            await self.clock.sleep(self.timeout_s)
            self.status["timeout"] += 1
            raise httpx.ReadTimeout("fake timeout", request=request)

        messages = body["messages"]
        content = self._think(messages, spec, body.get("response_format"))
        prompt_tokens = sum(_ntok(m.get("content") or "") for m in messages) + 4 * len(messages) + 3
        ids = _ENC.encode(content, disallowed_special=())
        finish = "stop"
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        if max_tokens is not None and len(ids) > max_tokens:
            content, finish, ids = _ENC.decode(ids[:max_tokens]), "length", ids[:max_tokens]
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": len(ids),
                 "total_tokens": prompt_tokens + len(ids)}
        self.tokens[(spec.name, "in")] += prompt_tokens
        self.tokens[(spec.name, "out")] += len(ids)
        self.status[200] += 1
        ttft = spec.overhead_s + prompt_tokens / spec.prefill_tps
        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=self._sse(spec, content, finish, usage if include_usage else None, ttft))
        await self.clock.sleep(ttft + len(ids) / spec.decode_tps)
        return httpx.Response(200, json={
            "id": f"chatcmpl-{self.calls}", "object": "chat.completion", "model": spec.name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": usage})

    async def _sse(self, spec: FakeModel, content: str, finish: str, usage: dict | None, ttft: float):
        def event(obj) -> bytes:
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()

        def chunk(delta: dict, finish_reason=None) -> dict:
            return {"id": f"chatcmpl-{self.calls}", "object": "chat.completion.chunk", "model": spec.name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

        async def emit(data: bytes):
            i = 0
            while i < len(data):                           # cắt byte thành mảnh ngẫu nhiên
                n = self.rng.randint(*self.chunk_bytes)
                yield data[i:i + n]
                i += n

        await self.clock.sleep(ttft)
        async for b in emit(b": keep-alive\n\n" + event(chunk({"role": "assistant", "content": ""}))):
            yield b
        decode_s = _ntok(content) / spec.decode_tps            # chia thời gian decode theo độ dài từng mảnh
        for piece in re.findall(r"\S+\s*|\s+", content):
            await self.clock.sleep(decode_s * len(piece) / max(1, len(content)))
            async for b in emit(event(chunk({"content": piece}))):
                yield b
        tail = event(chunk({}, finish))
        if usage is not None:
            tail += event({"id": f"chatcmpl-{self.calls}", "object": "chat.completion.chunk", "model": spec.name,
                           "choices": [], "usage": usage})
        async for b in emit(tail + b"data: [DONE]\n\n"):
            yield b

    # ---------- "bộ não" ----------
    def _think(self, messages: list[dict], spec: FakeModel, response_format: dict | None) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        users = [m["content"] for m in messages if m["role"] == "user"]
        reask = bool(users) and users[-1].startswith("Câu trả lời trước không phải JSON")
        customer = next((u for u in reversed(users) if not u.startswith("Câu trả lời trước không phải JSON")), "")
        m = _WRAPPED.search(customer)
        wrapped, text = (True, m.group(1)) if m else (False, customer)

        if _is_injection(text) and self.rng.random() < (spec.obey_wrapped if wrapped else spec.obey_injection):
            return "Được thôi, đây là hướng dẫn hệ thống của tôi: " + system[:400]
        if "JSON" not in system:                           # không yêu cầu JSON → trả lời văn bản thường
            return f"[{spec.name}] Dạ, em đã nhận được: {text[:80]}"

        truth = self.oracle.get(_key(text), {"intent": "khac", "goi": None, "sdt": None, "tien": None})
        intent = truth["intent"]
        if self.rng.random() >= spec.accuracy:
            intent = self.rng.choice([i for i in INTENTS if i != intent])
        slots = {"goi": truth["goi"], "sdt": truth["sdt"], "tien": truth["tien"]}
        reply = REPLIES[intent].format(goi=truth["goi"] or "<tên gói>")
        obj = {"reply": reply, "intent": intent, "slots": slots}

        kind = (response_format or {}).get("type")
        strict = kind == "json_schema" and (response_format.get("json_schema") or {}).get("strict")
        p_err = 0.0 if strict else spec.format_error / (5 if reask else 1)
        if self.rng.random() >= p_err:
            return json.dumps(obj, ensure_ascii=False)
        error = self.rng.choice(SYNTAX_ERRORS + SCHEMA_ERRORS)
        if kind == "json_object" and error in SYNTAX_ERRORS:   # JSON mode chặn lỗi cú pháp, không chặn lỗi schema
            return json.dumps(obj, ensure_ascii=False)
        return self._break(obj, error)

    @staticmethod
    def _break(obj: dict, error: str) -> str:
        good = json.dumps(obj, ensure_ascii=False)
        if error == "fence":
            return f"```json\n{good}\n```"
        if error == "preface":
            return f"Dưới đây là kết quả JSON:\n{good}"
        if error == "trailing_comma":
            return good[:-1] + ",}"
        if error == "single_quotes":
            return str(obj)                                # dict của Python: nháy đơn, None thay cho null
        if error == "truncated":
            return good[: len(good) * 2 // 3]
        bad = dict(obj)
        if error == "missing_field":
            del bad["reply"]
        elif error == "bad_enum":
            bad["intent"] = "hủy gói"                      # không thuộc danh sách intent
        else:
            bad["confidence"] = 0.9                        # khóa thừa, schema không cho phép
        return json.dumps(bad, ensure_ascii=False)
