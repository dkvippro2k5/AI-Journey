# Tuần 6 · LLM và AI API

**Mục tiêu tuần:**
- Hiểu LLM ở mức đủ để **ước lượng**: số tham số, FLOP, bộ nhớ KV, độ trễ prefill/decode, số token và chi phí.
- Biết LLM được train thế nào (pretrain → SFT → RLHF/DPO) và vì sao điều đó quyết định cách bạn viết prompt.
- Viết prompt cho production: có cấu trúc, có version, đầu ra kiểm tra được bằng schema, có lớp chống prompt injection.
- Gọi được API của nhiều nhà cung cấp và xử lý đúng: streaming SSE, giới hạn token, rate limit, retry, fallback, cache, chi phí, bảo mật key.

**Sản phẩm cuối tuần:** `project/gateway/`, một **gateway LLM** cho tổng đài:
- **Router intent của Tuần 5** chọn đường đi cho mỗi tin nhắn: model nhỏ, model lớn, tổng đài viên, hoặc từ chối.
- Client async **phát triển từ `multi_caller` của Tuần 1**, có thêm streaming, Retry-After và fallback.
- Đầu ra JSON được kiểm tra bằng Pydantic; có cache, rate limit theo khách, log đã che PII, và backend proxy FastAPI.
- Mọi test chạy **offline** trên một provider giả. Có thêm hai backend tùy chọn: một LLM nhỏ chạy local, hoặc API thật qua key miễn phí.

**Loại tài liệu:**
- Ngày 1 là **Kết hợp**: công thức chỉ cho cơ chế lõi (đếm tham số, $6ND$, bộ nhớ KV, ALiBi, Bradley–Terry, DPO).
- Ngày 2–3 là **Kỹ thuật**: toán chỉ ở mức ước lượng nhanh. Phần B của final là bài ước lượng.

**Ứng với roadmap:** Phase 6 (6.1–6.5), cộng các mục dời từ Tuần 5: bảng các LLM cụ thể, `tiktoken`, chi phí theo token.

## Lịch 5 ngày (khoảng 5 giờ mỗi ngày)

| Ngày | Đọc (~2h45–3h) | Quiz (45') | Làm (~2h) |
|---|---|---|---|
| 1 | [ngay-1.md](ngay-1.md): **01** LLM ở quy mô lớn, scaling laws, MoE, bảng các LLM · **02** Context window: vị trí, sliding window, prefill/decode, token tiếng Việt · **03** Huấn luyện LLM: pretrain → SFT → RLHF → Constitutional AI → DPO, chat template | [ngay-1.quiz.md](ngay-1.quiz.md) | `gateway/tokens.py` |
| 2 | [ngay-2.md](ngay-2.md): **04** Giải phẫu prompt và các kỹ thuật prompting · **05** Structured output và định dạng tool call · **06** Prompt trong production: version, A/B test, nén prompt, prompt injection | [ngay-2.quiz.md](ngay-2.quiz.md) | `gateway/prompts.py`, `gateway/schema.py`, `gateway/security.py` |
| 3 | [ngay-3.md](ngay-3.md): **07** Bản đồ API các nhà cung cấp · **08** Streaming SSE và quản lý context · **09** Độ tin cậy, chi phí, bảo mật | [ngay-3.quiz.md](ngay-3.quiz.md) | `gateway/stream.py`, `gateway/context.py`, `gateway/limits.py`, `gateway/cache.py` |
| 4 | — | — | **Project:** [project.md](project.md), viết `gateway/client.py`, `gateway/app.py`, `gateway/server.py` |
| 5 | — | — | `REPORT.md` (2h) · **Final 180'**: [final-exam.md](final-exam.md) |

Tuần này **dài về số mục** (roadmap Phase 6 rất rộng), nhưng mỗi mảnh code đều ngắn. Nếu thiếu thời gian, đọc lướt bảng nhà cung cấp ở bài 07 (chỉ cần nắm được các điểm khác nhau) và các mục "Đọc thêm".

## Chuẩn bị một lần (10 phút)

```
cd "week-06-llm-api/project"
python -m venv .venv
.venv\Scripts\activate
python -m pip install numpy pandas scikit-learn httpx "pydantic>=2" tiktoken fastapi uvicorn pytest ruff
python -m pip install torch transformers          # TÙY CHỌN: chỉ cho LLM nhỏ chạy local (bài 02–06, E8)
python -m pytest -q                                # lúc đầu sẽ báo lỗi vì chưa có code của bạn: bình thường
```

- **Không cần API key.** Mọi test và mọi kết quả của project chạy trên `gateway/fake.py`, một provider giả nói đúng giao thức Chat Completions (có SSE, lỗi 429/503, timeout).
- **`tiktoken`** tải file encoding (vài MB) ở lần dùng đầu tiên, sau đó chạy offline.
- **LLM nhỏ chạy local (tùy chọn).** Model instruct khoảng 0.5B tham số, khoảng 1 GB; trong tài liệu gọi là `MODEL_ID`, tên thật nằm trong `gateway/local_llm.py`. Model **đã được tải** về cache Hugging Face trên máy bạn. Các số "đo trên máy bạn" ở bài 02–06 dùng model này. Không test nào cần nó.
- **API thật (tùy chọn).** Lấy một key miễn phí của Gemini, Groq hoặc OpenRouter (cả ba đều có endpoint kiểu OpenAI). Đặt key vào `.env` (đã có trong `.gitignore`), **không commit**. Chạy theo hướng dẫn ở [project.md](project.md), mục 4.3.
- **Cắm sạc khi đo.** Khi chạy pin, laptop hạ xung nhịp (Tuần 4, README).
- Mỗi ngày chỉ chạy test của ngày đó, ví dụ `python -m pytest -q tests/test_tokens.py`.

## Cách học mỗi bài

Vẫn **hai tầng** như các tuần trước. Các cơ chế khó hình dung có khối **"🐢 Đi chậm từng bước"** ngay trước công thức hoặc trước code:
- đếm tham số một lớp decoder (01.2);
- vì sao decode chậm hơn prefill rất nhiều (02.2);
- ALiBi và sliding window (02.3);
- SFT chỉ tính loss trên câu trả lời; DPO tính tay (03.2, 03.3);
- self-consistency: vì sao bỏ phiếu giúp (04.3);
- sinh có ràng buộc theo schema (05.2);
- vì sao prompt injection xảy ra (06.2);
- một request đi qua ba nhà cung cấp (07.2);
- parser SSE nhận byte bị cắt vụn (08.2);
- token bucket (09.2).

Hãy đọc khối đó trước, rồi mới đọc công thức.

**Những "bất ngờ" được đo thật tuần này:**
- Model 0.5B chạy trên CPU của bạn: prefill khoảng **170 token/s**, decode khoảng **12 token/s**. Prompt 4 096 token mất **26 s** trước khi có chữ đầu tiên (bài 02).
- Cùng một câu, tiếng Việt tốn **1.21×** token so với tiếng Anh với tokenizer đời mới, nhưng **2.05×** với tokenizer đời cũ. Bỏ dấu **không** làm rẻ hơn với tokenizer mới (bài 02).
- Gửi prompt không qua chat template thì model instruct trả lời lan man; số liệu ở bài 03.
- Bọc tin nhắn khách trong thẻ giúp giảm tỉ lệ lộ bí mật, nhưng không về 0. Chỉ lớp kiểm tra đầu ra mới chặn được hết (bài 06, project E7).
- Router TF-IDF của Tuần 5 rẻ hơn và đúng hơn LLM 0.5B ở việc phân loại intent. LLM nhỏ đáng giá ở chỗ viết câu trả lời và trích slot (bài 04–05).
- Định tuyến theo intent rẻ hơn khoảng **5 lần** so với gửi mọi thứ cho model lớn (project E1). Khi model nhỏ sập 30 giây, retry không cứu được, còn fallback thì cứu được (project E3).

**Quy ước:**
- Dùng `MODEL_ID` hoặc tên tier (`model-small`, `model-large`) thay cho tên model API thật. Mọi mức giá là **giá giả định**. Thông tin về nhà cung cấp (tham số, giới hạn, tính năng) ghi theo tài liệu công khai **tại thời điểm viết**; API thay đổi nhanh, hãy kiểm tra lại docs trước khi dùng.
- Số liệu "đo trên máy bạn" dùng PyTorch 2.14 (CPU), `transformers` 5.18, `tiktoken` 0.14.

## Đối chiếu với roadmap

| Roadmap | Bài |
|---|---|
| 6.1 Transformer ở quy mô lớn (1B → 100B) | 01 |
| 6.1 Context window, tokenization at scale | 02 |
| 6.1 KV-cache, FlashAttention, GQA | Tuần 5 bài 16 (khái niệm); bài 02 ôn lại và tính bộ nhớ |
| 6.1 Positional encoding: absolute, relative, RoPE, ALiBi | 02 (absolute và RoPE: Tuần 5 bài 15) |
| 6.1 Sliding window attention | 02 |
| 6.1 Pretraining, instruction tuning, RLHF, Constitutional AI, DPO | 03 (thực hành SFT/DPO ở **Tuần 11**) |
| 6.1 Scaling laws | 01 |
| 6.2 Prompt anatomy (system, user, assistant, few-shot) | 04 |
| 6.2 Zero/one/few-shot, CoT, self-consistency, ReAct, ToT, role prompting, prompt chaining | 04 (thực hành ReAct ở **Tuần 10**) |
| 6.2 Structured output (JSON, XML) | 05 |
| 6.2 Production: chỉ dẫn rõ, nói điều không được làm, ví dụ, ràng buộc đầu ra | 04 |
| 6.2 Versioning, changelog, A/B test, nén prompt, chống prompt injection | 06 |
| 6.2 PromptLayer, LangSmith, OpenAI Playground, Anthropic Console | 06 (mô tả; LangSmith dùng thật ở **Tuần 7**) |
| 6.3 OpenAI: Chat Completions, function calling, JSON mode / structured outputs, streaming, embeddings, vision, Assistants API, Batch API, `tiktoken`, rate limit | 05, 07, 08 |
| 6.3 Anthropic, Gemini, Mistral, LLaMA qua HF/Ollama, Cohere, NVIDIA NIM, Groq, Together, Replicate | 07 (fine-tune LLaMA bằng PEFT ở **Tuần 11**) |
| 6.4 Giới hạn token: đếm trước khi gửi, cắt bớt, quản lý context, tóm tắt lịch sử | 02, 08 |
| 6.4 Streaming SSE, phản hồi từng phần, hiển thị phía client, độ trễ cảm nhận | 08 |
| 6.4 Rate limit và retry: backoff có jitter, quota, hàng đợi | 09 (backoff: Tuần 1 bài 09). **Circuit breaker** dời sang **Tuần 7** (cùng phần fallback 7.3) |
| 6.4 Kiểm soát chi phí: log token theo khách/tính năng, định tuyến theo độ khó, nén prompt, cache SHA-256, pipeline bất đồng bộ | 09, project |
| 6.4 Xử lý lỗi và fallback | 09, project |
| 6.5 Bảo mật: không lộ key ra frontend, `.env` và secret manager, backend proxy, rate limit theo khách (Redis), xoay vòng key, log và giám sát | 09, project (Redis thật ở **Tuần 13**; tuần này dùng bản in-memory) |
| 5.7 (dời từ Tuần 5) Bảng các LLM cụ thể | 01 |
| Phase 6 projects: chatbot nhiều provider + SSE · Resume ranker · Middleware AI production | Project `gateway` là bản thu gọn của dự án **Hard** (middleware). Chatbot + SSE: `server.py`. Resume ranker (PDF): phần đọc tài liệu ở **Tuần 8**. Redis, PostgreSQL, Docker: **Tuần 13** |

## Kiểm tra

| Loại | Khi nào | Thời lượng | Đạt khi |
|---|---|---|---|
| Quiz ngày | Ngay sau khi đọc xong file của ngày đó | 45' (15' quick test + 30' tự luận) | ≥ 21/30 |
| Test bài tập | Sau khi làm bài tập | — | Pass hết |
| Final tuần | Ngày 5 | 180' | ≥ 70/100 |
| [Ôn phỏng vấn](on-phong-van.md) | Sau final; ôn lại sau 1 tuần và sau 1 tháng | ~45' (16 câu, nói to, mỗi câu 1') | ≥ 12/16 |

Quiz và final **không có câu chứng minh**. Phần tự luận tập trung vào:
- **ước lượng:** tham số, FLOP, bộ nhớ KV, độ trễ, token, chi phí, rate limit;
- **phân tích** số đo;
- **thiết kế:** prompt, schema, chống injection, gateway.

Phần C của final: validator cho một phần JSON Schema, sliding-window rate limiter, chia ngân sách token cho các phần của prompt, và tích lũy tool call trong luồng stream.

## Nhận từ tuần trước, chuyển sang tuần sau

| Từ | Nội dung | Dùng ở bài |
|---|---|---|
| Tuần 1, bài 07–09 | asyncio, `httpx`, Semaphore, retry, full jitter, `multi_caller` | 08, 09, project (`client.py`) |
| Tuần 1, phụ lục P6 | `.env`, cấu hình | 09 |
| Tuần 3, bài 03–04 | So ghép cặp, ngưỡng theo chi phí | 04, 06 (A/B test) |
| Tuần 4, bài 06 | 16 byte/tham số khi train, số byte theo dtype | 01, 02 |
| Tuần 5, bài 14 | BPE, byte-level BPE, special token | 02, 03, 06 |
| Tuần 5, bài 15–16 | RoPE, causal mask, KV-cache, GQA, đếm tham số | 01, 02 |
| Tuần 5, bài 17 | Temperature, top-k, top-p, perplexity | 04, 05, 07 |
| Tuần 5, project | Router intent, `normalize`, ngưỡng chuyển người | Project (`router.py`) |

**Dời sang tuần sau:**
- Circuit breaker, chuỗi fallback nhiều provider, semantic cache, LangChain/LangGraph, MCP: **Tuần 7**.
- Embeddings dùng thật (semantic search, RAG): **Tuần 8**.
- Vòng lặp tool use, ReAct thực hành: **Tuần 10**.
- SFT, LoRA, DPO thực hành: **Tuần 11**.
- vLLM, PagedAttention, quantization: **Tuần 12**.
- Redis, PostgreSQL, Docker, giám sát: **Tuần 13**.

## Tuần này nối với các tuần sau thế nào

| Kiến thức Tuần 6 | Được dùng lại ở |
|---|---|
| `LLMClient` (retry, Retry-After, stream), FakeLLM | Tuần 7 (fallback nhiều provider, circuit breaker), Tuần 10 (agent gọi LLM) |
| Prompt template có version, A/B | Tuần 7 (LangSmith), Tuần 13 (theo dõi phiên bản prompt trong production) |
| Structured output, định dạng tool call | Tuần 7 (MCP), Tuần 10 (tool use) |
| Chống prompt injection, canary, che PII | Tuần 8 (injection gián tiếp qua tài liệu RAG), Tuần 10 (agent có quyền hành động) |
| Ước lượng token, chi phí, bộ nhớ KV, prefill/decode | Tuần 8 (chi phí ngữ cảnh RAG), Tuần 12 (vLLM, batching), Tuần 15 (system design) |
| Cache, rate limit, log | Tuần 13 (Redis, giám sát) |
