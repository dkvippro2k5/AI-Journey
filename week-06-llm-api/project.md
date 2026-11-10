# Project Tuần 6 · `gateway`: gateway LLM cho tổng đài

> **Ngày 4:** ghép các mảnh thành gateway (~5h).
> **Ngày 5:** chạy thí nghiệm, viết `REPORT.md` (2h), sau đó làm final (3h).
> **Câu hỏi của tuần:** đặt một LLM sau router intent của Tuần 5 thì chi phí, độ trễ, độ tin cậy và an toàn thay đổi thế nào, và mỗi lớp của gateway (định tuyến, cache, retry, fallback, schema, chặn đầu ra) đóng góp bao nhiêu?
> **Vị trí trong hệ thống chung:** đây là lớp gọi LLM của nền tảng RAG + Agent:
> - Tuần 7 thêm circuit breaker, nhiều nhà cung cấp, LangGraph.
> - Tuần 8 thêm tài liệu truy xuất vào prompt (và injection gián tiếp).
> - Tuần 10 biến `action` thành tool call của agent.
> - Tuần 13 đóng gói thành service có Redis, giám sát.

## 1. Gateway làm gì

```
khách ──► server.py (FastAPI: xác thực token khách, kiểm tra đầu vào, 429 + Retry-After)
            │
            ▼
         app.py · Gateway.handle(user_id, text, history)
            1. rate limit theo khách (KeyedLimiter)                    → "rate_limited"
            2. router intent Tuần 5 (TF-IDF, < 1 ms) + ngưỡng tin cậy
            3. chọn đường: gap_tvv → "human" · khac → "refuse"
                           intent khác → model-small · khieu_nai hoặc router không chắc → model-large
            4. dựng prompt: template có version + bọc <customer_message> + cắt lịch sử
            5. cache (khóa SHA-256, chuẩn hóa nhẹ)                     → "cache"
            6. LLMClient: retry (Retry-After, full jitter), stream SSE, đo TTFT
               tier chính hỏng → tier còn lại (fallback)               → "error" nếu hỏng hết
            7. parse + Pydantic; sai thì sửa nhẹ, rồi hỏi lại một lần
            8. canary trong đầu ra → "blocked"
            9. ghi log (đã che PII): token, chi phí, độ trễ, route, version…
```

**Dữ liệu** (`gateway/data.py`, cho sẵn): luồng 2 000 request trong khoảng 100 giây.
- **Tin nhắn:** sinh bằng chính generator của Tuần 5, có thêm đáp án slot (gói, số điện thoại, số tiền VND).
- **Câu hay gặp:** 35% là câu hay gặp được gõ lại với nhiễu. Văn bản thô chỉ trùng 3.4%, sau khi chuẩn hóa thì trùng 16%.
- **Khách:** rút theo Zipf; khách nhiều nhất gửi 16% lưu lượng.
- **Prompt injection:** 3% request có câu tấn công (7 kiểu, 2 kiểu lách được heuristic).

**Provider** (`gateway/fake.py`, cho sẵn): nói giao thức Chat Completions; có SSE cắt byte ngẫu nhiên, 429/503/timeout, khoảng sập theo thời gian.
- "Bộ não" giả biết đáp án đúng, và trả đúng với xác suất cho trước: `model-small` 0.90, `model-large` 0.97.
- Hỏng định dạng, làm theo injection, tốc độ prefill/decode đều theo tham số **giả định**.
- Vì vậy các con số chất lượng trong E1–E7 là **mô phỏng**, dùng để so sánh các lớp của gateway, không phải để đánh giá một LLM thật. Số đo trên LLM thật nằm ở bài 02–06 và ở E8.

**Đồng hồ:** test và benchmark dùng `SimClock` (thời gian mô phỏng), nên 2 000 request với hàng giờ "độ trễ" chạy trong vài giây và cho kết quả lặp lại được.

## 2. Thư mục project

```
project/
├── pyproject.toml
├── gateway/
│   ├── data.py         # cho sẵn: tin nhắn + slot (Tuần 5), luồng request, câu injection
│   ├── router.py       # cho sẵn: router intent Tuần 5 + ngưỡng
│   ├── clock.py        # cho sẵn: RealClock, SimClock
│   ├── fake.py         # cho sẵn: provider giả kiểu Chat Completions
│   ├── local_llm.py    # cho sẵn, tùy chọn: LLM nhỏ chạy local cùng giao thức
│   ├── bench.py        # cho sẵn: thí nghiệm E1–E8
│   ├── tokens.py       # Ngày 1: đếm token, ChatML, chi phí, tham số, KV, FLOP, độ trễ
│   ├── prompts.py      # Ngày 2: PromptTemplate, PromptRegistry (A/B), wrap_untrusted, majority_vote
│   ├── schema.py       # Ngày 2: Reply (Pydantic), extract_json, light_repair, parse_reply, repair_prompt
│   ├── security.py     # Ngày 2: redact_pii, looks_like_injection, leaks_canary, mask_secret
│   ├── stream.py       # Ngày 3: SSEParser, StreamAccumulator, sse_encode, FieldStreamer
│   ├── context.py      # Ngày 3: fit_history, truncate_middle, summarize_old
│   ├── limits.py       # Ngày 3: TokenBucket, KeyedLimiter, parse_retry_after, next_delay
│   ├── cache.py        # Ngày 3: canonical_json, cache_key, TTLCache
│   ├── client.py       # Ngày 4: LLMClient (retry, Retry-After, stream, TTFT)
│   ├── app.py          # Ngày 4: Gateway, GatewayConfig, Result, định tuyến, prompt mặc định
│   └── server.py       # Ngày 4: FastAPI backend proxy, /v1/chat, /v1/chat/stream
├── tests/              # 49 test (cho sẵn)
└── _reference/         # lời giải tham khảo — chỉ mở khi kẹt quá 1h
```

## 3. Ngày 4 — `client.py`, `app.py`, `server.py`

Kiểm tra bằng `tests/test_client.py`, `tests/test_app.py`, `tests/test_server.py` (4 + 5 + 2 = 11 test).

### 3.1 `gateway/client.py`: phát triển từ `ask` của Tuần 1

1. `RETRYABLE = {408, 409, 429, 500, 502, 503, 504, 529}`.
2. `ProviderError(message, status=None, retryable=False, retry_after=None)` có thêm thuộc tính `attempts`.
3. `Completion`: `text, model, input_tokens, output_tokens, finish_reason, latency_s, ttft_s, attempts`.
4. `LLMClient(http, api_key, *, clock=None, max_attempts=3, base=0.5, cap=8.0, rng=None)`:
   - `http` là `httpx.AsyncClient` đã đặt `base_url`; gửi `POST /chat/completions` với header `Authorization: Bearer <key>`.
   - `async chat(model, messages, *, stream=False, on_delta=None, **params) -> Completion`:
     - Mã ≥ 400 → `ProviderError` (đọc `error.message` nếu có, `retryable` theo `RETRYABLE`, `retry_after` bằng `parse_retry_after`).
     - `httpx.TransportError` (gồm timeout) → `ProviderError(retryable=True)`.
     - Thử lại tối đa `max_attempts` lần, chờ `next_delay(attempt, e.retry_after, base, cap, rng)` bằng `clock.sleep`. Hết lượt, hoặc lỗi không nên thử lại → ném `ProviderError` với `attempts` đúng.
     - `stream=True`: gửi thêm `"stream": True, "stream_options": {"include_usage": True}`; đọc `resp.aiter_bytes()` → `SSEParser` → `StreamAccumulator`; gọi `on_delta(delta)` với mỗi đoạn văn bản mới; `ttft_s` = lúc có delta đầu tiên (tính từ đầu lời gọi, gồm cả các lần thử trước).
     - Luồng kết thúc mà không có `[DONE]` → lỗi có thể thử lại, **trừ khi** đã gửi delta cho người dùng.
     - `latency_s` tính từ đầu lời gọi tới khi xong; không stream thì `ttft_s = latency_s`.

### 3.2 `gateway/app.py`: Gateway

Phần cho sẵn trong `_reference/app.py`, bạn chép nguyên: `CANARY`, `TIERS` (giá giả định), `ROUTES`, các câu trả lời mẫu, `SYSTEM_V1`, `FEW_SHOT`, `TEMPLATES` (`answer@1.0.0` zero-shot, `answer@1.1.0` few-shot), `default_registry()`. Phần bạn viết:

1. `cache_text(text)`: chuẩn hóa **nhẹ** (NFC, chữ thường, `TONE_FIX`, chỉ giữ `\w+`, `TEENCODE`). Không thay số bằng placeholder.
2. `GatewayConfig` (dataclass): `routing` (`"intent"`, `"all-large"`, `"all-small"`), `cache`, `cache_key_fn` (`"raw"`, `"light"`, `"aggressive"`), `fallback`, `reask`, `structured` (`"none"`, `"json_object"`, `"json_schema"`), `wrap_input`, `output_guard`, `stream`, `max_tokens = 200`, `history_budget = 1500`, `template_version`.
3. `Result` (dataclass): các trường như trong `_reference` (`route`, `reply`, `intent`, `slots`, `router_intent`, `router_conf`, `model`, `template`, token, `cost_usd`, `latency_s`, `ttft_s`, `attempts`, `calls`, `fallback_used`, `repaired`, `reasked`, `valid`, `injection_flag`, `leaked`, `errors`).
4. `Gateway(client, router, registry=None, *, clock, config, limiter, cache, canary, log_path)`. `handle(user_id, text, history=None, *, on_reply_delta=None) -> Result` theo đúng thứ tự ở mục 1:
   - **Định tuyến:** router tự tin thì theo `ROUTES`; không tự tin thì `large`. Có `fallback` thì thử thêm tier còn lại.
   - **Prompt:** `template.render(canary=…, intent_hint=…, message=wrap_untrusted(text) hoặc text)`; có `history` thì chèn trước tin nhắn cuối rồi `fit_history(…, history_budget)`.
   - **Tham số gọi:** `temperature=0`, `max_tokens`, `response_format` theo `structured`.
   - **Cache** chỉ khi: có `cache`, `config.cache`, không có lịch sử, không bị gắn cờ injection.
     - Khóa = `cache_key(model của tier chính, [{"role": "system", "content": fingerprint}, {"role": "user", "content": khóa văn bản}], tham số)`.
     - Chỉ lưu câu trả lời hợp lệ.
   - **Mỗi tier:** gọi (`stream` theo cấu hình; với stream thì đưa delta qua `FieldStreamer("reply")` rồi `on_reply_delta`) → `parse_reply`. Sai thì (nếu `reask`, và đầu ra không chứa canary) gửi thêm assistant + `repair_prompt(error)` và gọi lại **một** lần, không stream.
   - **Canary:** đầu ra chứa canary thì `leaked=True`; có `output_guard` thì route `"blocked"` với câu trả lời mẫu, không thì trả nguyên văn cho khách (để đo E7).
   - **Ghi sổ:** cộng `calls`, `attempts`, token, `cost_usd` (theo giá của tier), `ttft_s` của lần gọi đầu; `ProviderError` thì cộng `attempts`, ghi vào `errors`, sang tier sau.
   - **Ghi log:** luôn ghi (kể cả `rate_limited`, `human`) vào `self.records`, và vào file JSONL nếu có `log_path`; `text` và `reply` đã qua `redact_pii`.

### 3.3 `gateway/server.py`: backend proxy

`create_app(gateway, user_tokens) -> FastAPI`:
- **Xác thực:** header `Authorization: Bearer <token của khách>`, tra `user_tokens` ra `user_id`; sai thì 401. Đây **không** phải API key của nhà cung cấp.
- **Kiểm tra đầu vào bằng Pydantic:** `text` dài 1–2 000 ký tự; `history` tối đa 20 message, `role` chỉ là `user`/`assistant` (khách không được tự chèn `system`).
- `POST /v1/chat` trả `{"reply", "intent", "route"}`; bị rate limit thì 429 kèm `Retry-After`.
- `POST /v1/chat/stream`:
  - chạy `gateway.handle(…, on_reply_delta=queue.put_nowait)` trong một task;
  - stream các sự kiện `event: delta` (`{"delta": …}`), rồi một sự kiện `event: done` với kết quả công khai và `replace` (phần đã stream khác câu trả lời cuối).
- `GET /healthz`.

## 4. Chạy thử

```
cd project
python -m pytest -q                        # 49 test, khoảng 5 giây
python -m gateway.bench                    # E1–E7 trên FakeLLM, khoảng 2 phút
GATEWAY_DEMO=1 uvicorn gateway.server:app  # thử bằng tay (PowerShell: $env:GATEWAY_DEMO="1"; uvicorn …)
```

Thử server bằng `curl` (token khách demo là `demo-token`):
```
curl -N -X POST localhost:8000/v1/chat/stream -H "Authorization: Bearer demo-token" -H "Content-Type: application/json" -d "{\"text\": \"tôi muốn hủy gói GX30\"}"
```

### 4.1 Kết quả mong đợi (FakeLLM, 2 000 request, đo trên máy bạn bằng lời giải tham khảo)

Số chất lượng (intent đúng, tỉ lệ hợp lệ, tỉ lệ làm lộ) là **mô phỏng** theo tham số giả định của `fake.py`. Số chi phí là giá giả định. Độ trễ là thời gian mô phỏng. Cái đáng tin là **so sánh giữa các cấu hình**.

**E1 · Định tuyến** (`intent_acc` tính trên các request được trả lời):

| Cấu hình | USD / 1 000 request | Lần gọi LLM / request | LLM | Cache | Người | Từ chối | Intent đúng | p50 / p95 (s) |
|---|---|---|---|---|---|---|---|---|
| Mọi tin → model lớn | 2.374 | 1.018 | 100% | — | — | — | 0.971 | 1.85 / 1.95 |
| Mọi tin → model nhỏ | 0.112 | 1.042 | 99.7% | — | — | — | 0.911 | 0.66 / 0.71 |
| Định tuyến theo intent | 0.475 | 0.890 | 85.5% | — | 7.4% | 7.0% | 0.936 | 0.68 / 1.88 |
| Định tuyến + cache | **0.442** | 0.744 | 71.2% | 14.3% | 7.4% | 7.0% | 0.934 | 0.68 / 1.88 |

Trên các request đi qua LLM (cấu hình "định tuyến theo intent"): intent của **router** đúng 0.973, intent do **LLM** trả về chỉ đúng 0.924. Request đi model lớn chiếm 19.6% số lần gọi LLM, nhưng chiếm phần lớn chi phí (2.4 so với 0.11 milli-USD mỗi lần gọi). Mỗi lần gọi trung bình 483 token vào, 67 token ra.

**E2 · Khóa cache:**

| Khóa | Tỉ lệ trúng | Trúng nhưng trả sai slot | USD / 1 000 request |
|---|---|---|---|
| Văn bản thô | 3.3% | 0 | 0.466 |
| Chuẩn hóa nhẹ (`cache_text`) | 14.3% | 0 | 0.442 |
| Chuẩn hóa mạnh (`normalize` của router) | 14.6% | **6** | 0.436 |

**E3 · Model nhỏ sập trong khoảng 30–60 s** (cộng 5% lỗi 500/503 ngẫu nhiên):

| Cấu hình | Được trả lời | Được trả lời trong lúc sập | Dùng fallback | p95 (s) | USD / 1 000 request |
|---|---|---|---|---|---|
| 1 lần, không fallback | 78.9% | 39.9% | — | 1.89 | 0.405 |
| 3 lần, không fallback | 82.3% | 41.4% | — | 1.94 | 0.433 |
| 3 lần + fallback | **99.8%** | **100%** | 17% | 3.42 | 0.829 |

**E4 · Streaming** (chỉ các request đi qua LLM, không cache):

| | Chữ đầu tiên p50 / p95 (s) | Tổng p50 / p95 (s) |
|---|---|---|
| Không stream | 0.675 / 1.884 | 0.675 / 1.884 |
| Stream (qua `FieldStreamer`) | **0.283 / 0.705** | 0.675 / 1.884 |

**E5 · Structured output** (không fallback, không cache; tính trên request có gọi LLM):

| `structured`, hỏi lại | Hợp lệ | Sửa cục bộ | Phải hỏi lại | Lần gọi / request | USD / 1 000 request |
|---|---|---|---|---|---|
| none, không | 0.938 | 4.0% | — | 1.000 | 0.465 |
| none, có | 0.998 | 4.3% | 5.4% | 1.054 | 0.474 |
| json_object, không | 0.954 | — | — | 1.000 | 0.465 |
| json_object, có | 0.999 | — | 4.1% | 1.041 | 0.475 |
| json_schema (strict), không | 0.997 | — | — | 1.000 | 0.465 |
| json_schema (strict), có | 0.997 | — | — | 1.000 | 0.465 |

(0.3% còn lại ở `json_schema` là các ca model bị injection làm lộ system prompt và bị chặn.)

**E6 · Rate limit theo khách** (0.5 request/s, burst 10):

| | Request | Bị chặn |
|---|---|---|
| Tất cả | 2 000 | 21.5% |
| Khách gửi nhiều nhất (`u000`) | 322 | **81.4%** |
| Các khách còn lại | 1 678 | 10.1% |

**E7 · Prompt injection** (luồng riêng: 20% request có câu tấn công, 395 request bị tấn công):

| Bọc thẻ | Chặn đầu ra | Heuristic gắn cờ | Model làm lộ canary | Lộ tới tay khách | Gắn cờ nhầm tin sạch |
|---|---|---|---|---|---|
| không | không | 76.7% | 32.4% | 32.4% | 0% |
| không | có | 76.7% | 32.4% | **0%** | 0% |
| có | không | 76.7% | 11.6% | 11.6% | 0% |
| có | có | 76.7% | 11.6% | **0%** | 0% |

`python -m pytest -q` chạy 49 test trong khoảng 5–10 giây; `python -m gateway.bench` mất khoảng 2 phút (máy cắm sạc).

### 4.2 E8 · Cùng gateway, một LLM thật chạy local (tùy chọn)

```
python -m gateway.bench --provider local --n 100
```

Dùng model 0.5B của bài 02–06 cho **cả hai** tier, không cache, không giới hạn tốc độ, `structured="none"` (model local không có JSON mode), streaming bật. 100 request đầu của luồng (seed 0):

| Chỉ số | Giá trị |
|---|---|
| Request / request có gọi LLM | 100 / 86 (14 tin đi đường "người" hoặc "từ chối") |
| JSON hợp lệ (sau sửa cục bộ và hỏi lại) | **96.5%** |
| Phải hỏi lại | 3.5% |
| Intent đúng (LLM, trên câu hợp lệ) | **0.253** |
| Intent đúng của router TF-IDF trên cùng các request | **0.965** |
| Slot đúng cả 3 | 0.880 (phần lớn tin không có slot, đoán `null` là đúng) |
| TTFT p50 | 1.96 s |
| Tổng thời gian p50 / p95 | 5.17 / 8.71 s |
| Token ra trung bình mỗi lần gọi | 51 |

- **Định dạng:** model 0.5B viết JSON đúng định dạng khá tốt nhờ few-shot (`answer@1.1.0`) và vòng hỏi lại.
- **Intent:** model 0.5B đúng chưa bằng **một phần tư** số tin, trong khi router đúng 96.5%. Đây là bằng chứng thật cho gợi ý ở REPORT câu 1: khi router tự tin, giữ intent của router.
- **Độ trễ:** TTFT gần 2 s dù prompt chỉ khoảng 450 token, vì prefill trên CPU chậm (bài 02). Ở FakeLLM, TTFT giả định chỉ 0.25–0.6 s.
- Mô phỏng ở E1–E7 **lạc quan hơn nhiều** về chất lượng (giả định model nhỏ đúng 90%).

### 4.3 E8 với API thật (tùy chọn, cần key miễn phí)

1. Lấy key miễn phí của Groq, OpenRouter, hoặc Gemini; tạo file `project/.env` (đã bị `.gitignore` loại):
   ```
   GROQ_API_KEY=...
   ```
2. Nạp biến môi trường (PowerShell: `Get-Content .env | ForEach-Object { $k, $v = $_ -split '=', 2; Set-Item "env:$k" $v }`), rồi chạy:
   ```
   python -m gateway.bench --provider openai --base-url https://api.groq.com/openai/v1 --api-key-env GROQ_API_KEY --small <MODEL_ID nhỏ> --large <MODEL_ID lớn> --n 50
   ```
   - OpenRouter: `--base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY`.
   - Gemini: `--base-url https://generativelanguage.googleapis.com/v1beta/openai --api-key-env GEMINI_API_KEY`.
   - Tên model lấy từ trang docs của nhà cung cấp.
3. Gói miễn phí có rate limit thấp: chạy `--n` nhỏ. Gateway sẽ thử lại theo `Retry-After`.
4. **Không** gửi dữ liệu thật của khách hàng lên API bên ngoài; luồng request ở đây là dữ liệu tổng hợp.

## 5. Ngày 5 — `REPORT.md`

Trả lời bằng **số liệu và phép tính**. Mỗi câu nên có một bảng nhỏ.

1. **Định tuyến (E1).** Lập bảng chi phí / 1 000 request, intent đúng, p50/p95 cho bốn cấu hình.
   - Tách chi phí của định tuyến theo nguồn: bao nhiêu phần trăm tiết kiệm đến từ việc không gọi LLM (human/refuse), bao nhiêu từ model nhỏ?
   - Trên cùng các request đi qua LLM, so độ chính xác intent của router (tham khảo: 0.973) với intent do LLM trả về (tham khảo: 0.924; với model 0.5B thật ở E8: router 0.965, LLM 0.253). Đề xuất cách dùng cả hai (ví dụ: router tự tin thì giữ intent của router, LLM chỉ viết câu trả lời và trích slot).
2. **Khóa cache (E2).** Vì sao khóa thô gần như không trúng? Vì sao khóa chuẩn hóa mạnh trả sai slot? Viết một quy tắc khi nào được cache, khi nào không (lịch sử, injection, `temperature`, dữ liệu cá nhân, thông tin hay thay đổi như giá gói).
3. **Độ tin cậy (E3).** Vì sao retry gần như không giúp trong khoảng sập 30 giây, còn fallback thì giúp? Tính p95 tăng bao nhiêu và chi phí tăng bao nhiêu khi bật fallback. Thiết kế circuit breaker (ngưỡng, thời gian mở mạch) để giảm p95 trong khoảng sập (sẽ làm ở Tuần 7).
4. **Streaming (E4).** So thời điểm khách thấy chữ đầu tiên và tổng thời gian. Vì sao tổng thời gian không đổi? Với model local (E8), TTFT chiếm bao nhiêu phần trăm tổng thời gian?
5. **Structured output (E5).** Lập bảng `valid` theo `structured` × hỏi lại. Vì sao `json_object` không đạt `valid` = 1? Chi phí thêm của vòng hỏi lại là bao nhiêu phần trăm? Nối với bài 05: sinh có ràng buộc trên model thật đạt bao nhiêu.
6. **Rate limit (E6) và injection (E7).**
   - E6: tỉ lệ bị chặn của khách nhiều nhất so với các khách khác.
   - E7: giải thích vì sao bọc thẻ giảm tỉ lệ model làm lộ, nhưng chỉ lớp chặn đầu ra mới đưa tỉ lệ **tới tay khách** về 0. Heuristic phát hiện được bao nhiêu phần trăm câu tấn công, và hai câu nào lọt? Liên hệ với số đo thật ở bài 06.
7. **(Nếu chạy E8)** Model 0.5B thật: tỉ lệ JSON hợp lệ, intent đúng, slot đúng, độ trễ. So với FakeLLM, chỗ nào mô phỏng quá lạc quan? Bạn sẽ dùng model này cho tầng nào của gateway, hay không dùng?

**Checklist hoàn thành:**
- [ ] `python -m pytest -q`: cả 49 test pass.
- [ ] `python -m gateway.bench` chạy hết E1–E7.
- [ ] `REPORT.md` trả lời đủ câu 1–6, có bảng.
- [ ] Gọi thử `/v1/chat/stream` bằng `curl` (hoặc `httpx`), dán 3 ví dụ: một câu bình thường, một câu "gặp nhân viên", một câu injection.

**Mở rộng (không bắt buộc):**
- **Nhiều nhà cung cấp:** thêm adapter Anthropic Messages (chiều đi và chiều về, kể cả sự kiện stream có tên) và chạy cùng bộ test hợp đồng trên một fake kiểu Anthropic.
- **Rate limit theo token:** thêm một `TokenBucket` theo TPM cho từng model, trừ trước bằng ước lượng (`count_chat_tokens` + `max_tokens`) rồi hoàn lại phần dư theo `usage`.
- **Prompt A/B thật:** dùng `PromptRegistry.assign` chia khách giữa `answer@1.0.0` và `answer@1.1.0` trên model local, đo tỉ lệ JSON hợp lệ và slot đúng; tính khoảng tin cậy của chênh lệch (bài 06).
- **(Roadmap Phase 6, Easy project)** Lưu lịch sử hội thoại theo `user_id` trong một kho key-value (in-memory, Tuần 13 đổi sang Redis), và dùng `summarize_old` khi quá dài.
