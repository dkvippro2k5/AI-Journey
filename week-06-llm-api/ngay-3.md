# Ngày 3 · API các nhà cung cấp, streaming và quản lý context, độ tin cậy / chi phí / bảo mật

> **Loại tài liệu: Kỹ thuật.** Trọng tâm là thuật ngữ chuẩn, cơ chế, code và quyết định thiết kế. Toán chỉ ở mức ước lượng nhanh.
> **Hôm nay gồm 3 bài:**
> - **07** Bản đồ API các nhà cung cấp: OpenAI, Anthropic, Gemini, Mistral, model mở (Ollama, Hugging Face), các nền tảng suy luận
> - **08** Streaming SSE và quản lý context: parser, phản hồi từng phần, hiển thị phía client, cắt bớt và tóm tắt lịch sử
> - **09** Độ tin cậy, chi phí, bảo mật: rate limit, retry, fallback, cache, theo dõi chi phí, backend proxy, key, log
>
> **Thời gian:** đọc ~2h45 · quiz 45' · bài tập ~2h
> **Lưu ý:** thông tin về API ghi theo tài liệu công khai **tại thời điểm viết**. Tên endpoint và tham số đổi khá thường xuyên, nên trước khi code thật hãy mở docs. Những gì cần nhớ lâu là **khái niệm** và **cách ánh xạ** giữa các nhà cung cấp. Mọi mức giá là giá giả định.

---

## 07 · Bản đồ API các nhà cung cấp

> ⏱ Tầng 1: 10' · Tầng 2: 50'
> **Cần biết trước:** `httpx` async (Tuần 1, bài 08); structured output, định dạng tool call (bài 05); prefill và decode (bài 02); embedding câu (Tuần 5, bài 18).

> **📖 Từ mới**
> - **Chat Completions API:** API kiểu OpenAI nhận một mảng `messages` (system, user, assistant, tool). Rất nhiều nhà cung cấp hỗ trợ cùng định dạng này ("OpenAI-compatible").
> - **Responses API:** API mới hơn của OpenAI, gộp hội thoại có trạng thái và công cụ có sẵn (tìm kiếm web, tìm trong file). Thay cho Assistants API.
> - **Messages API:** API của Anthropic; system prompt là trường riêng, `max_tokens` bắt buộc, nội dung là danh sách "block".
> - **`generateContent`:** API của Gemini; hội thoại là `contents` gồm các `parts`, vai trò là `user` và `model`.
> - **Embeddings API:** trả vector cho văn bản (dùng ở RAG, Tuần 8).
> - **Batch API:** gửi một file nhiều request, nhận kết quả trong vòng tối đa 24 giờ, giá thấp hơn.
> - **Prompt / context caching:** nhà cung cấp giữ lại KV-cache của phần đầu prompt (bài 02); lần sau gửi cùng phần đầu thì tính giá thấp hơn và TTFT nhỏ hơn.
> - **Grounding:** model được nối với công cụ tìm kiếm và trả lời kèm nguồn.
> - **Inference provider:** dịch vụ chạy model mở cho bạn (Groq, Together, NVIDIA NIM, Replicate, OpenRouter…).

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Đặt cùng một món ăn ở ba nhà hàng, mỗi nhà có một mẫu phiếu gọi món khác nhau: nhà này ghi "ghi chú cho bếp" ở đầu phiếu, nhà kia bắt buộc ghi số suất, nhà thứ ba gọi khách là "thực khách". Món ăn vẫn vậy. Bạn chỉ cần một **người phiên dịch** (provider abstraction) chuyển yêu cầu của mình sang đúng mẫu phiếu của từng nhà.

**Ví dụ có số (ước lượng với giá giả định).** Một request tổng đài: system prompt + few-shot 1 200 token (giống nhau mọi request), tin nhắn 300 token, câu trả lời 150 token. Giá model lớn: 3 USD / 1M token vào, 15 USD / 1M token ra; token trúng cache tính 10% giá vào; Batch giảm 50%.

| Cách gọi | Chi phí mỗi request | 1 triệu request |
|---|---|---|
| Gọi thường | $1\,500 \times 3 + 150 \times 15 = 6\,750$ micro-USD | 6 750 USD |
| Prompt caching phần 1 200 token chung | $1\,200 \times 0.3 + 300 \times 3 + 150 \times 15 = 3\,510$ | 3 510 USD (−48%) |
| Caching + Batch (việc không cần trả lời ngay) | $3\,510 \times 0.5$ | 1 755 USD (−74%) |

(Chưa tính phụ phí ghi cache: một số nhà cung cấp tính lần ghi đầu cao hơn giá vào thường.)

**Ba ý cốt lõi:**
1. **Ba "hình dạng" request.** OpenAI Chat Completions, Anthropic Messages, Gemini `generateContent`. Phần lớn các nhà cung cấp khác (Groq, OpenRouter, Together, Mistral, vLLM, Ollama, và cả Gemini qua endpoint tương thích) nói được định dạng OpenAI. Vì vậy một client với `base_url` thay đổi được là đã phủ được phần lớn.
2. **Cùng một bộ tính năng, khác tên gọi:** tool use, structured output, streaming, vision, embeddings, batch, caching. Hãy học cách ánh xạ.
3. **Đòn bẩy chi phí ở phía nhà cung cấp:** chọn model, prompt caching (phần đầu prompt cố định), Batch (việc không gấp), giới hạn `max_tokens`.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 07.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Chat Completions** | `POST /v1/chat/completions`; `messages`, `tools`, `response_format`, `stream`; trả `choices[].message`, `usage` | OpenAI API Reference |
| **Responses API** | `POST /v1/responses`; có trạng thái hội thoại, công cụ dựng sẵn (web search, file search); thay Assistants API | OpenAI API Reference, 2025 |
| **Messages API** | `POST /v1/messages`; header `x-api-key`, `anthropic-version`; `system` là trường riêng; `max_tokens` bắt buộc | Anthropic API Reference |
| **`generateContent`** | `POST …/models/{model}:generateContent`; `contents[].parts[]`, `systemInstruction`, `generationConfig` | Google Gemini API Reference |
| **Batch API** | File JSONL nhiều request; kết quả trong tối đa 24 giờ; khoảng 50% giá | OpenAI Batch API; Anthropic Message Batches |
| **Prompt caching** | OpenAI: tự động với prompt từ 1 024 token; Anthropic: đánh dấu `cache_control` trên block; Gemini: `cachedContents` có TTL, tính phí lưu theo giờ | Docs của từng nhà cung cấp |
| **OpenAI-compatible endpoint** | Cùng đường dẫn và định dạng JSON với Chat Completions, chỉ khác `base_url` và key | Docs của Groq, OpenRouter, Together, vLLM, Ollama, Gemini |
| **GGUF** | Định dạng file model đã lượng tử hóa cho llama.cpp/Ollama, chạy được trên CPU | llama.cpp (Tuần 12) |

#### 07.2 🐢 Đi chậm từng bước: một request đi qua ba nhà cung cấp

Yêu cầu: system "Bạn là trợ lý tổng đài", user "hủy gói GX30", nhiệt độ 0, tối đa 200 token.

*Bước 1 · Request.*

| Thành phần | OpenAI Chat Completions | Anthropic Messages | Gemini `generateContent` |
|---|---|---|---|
| Xác thực | `Authorization: Bearer <key>` | `x-api-key: <key>` + `anthropic-version` | `x-goog-api-key: <key>` |
| System prompt | message `{"role": "system", …}` đầu mảng | trường `"system"` ở cấp ngoài | `"systemInstruction": {"parts": [{"text": …}]}` |
| Lượt hội thoại | `messages: [{role, content}]` | `messages: [{role, content}]`, chỉ `user`/`assistant`, phải xen kẽ | `contents: [{role, parts: [{text}]}]`, vai trò `user`/`model` |
| Giới hạn đầu ra | `max_tokens` (hoặc `max_completion_tokens`), tùy chọn | `max_tokens`, **bắt buộc** | `generationConfig.maxOutputTokens` |
| Nhiệt độ | `temperature` | `temperature` | `generationConfig.temperature` |

*Bước 2 · Response.*

| Thành phần | OpenAI | Anthropic | Gemini |
|---|---|---|---|
| Văn bản | `choices[0].message.content` | `content[0].text` (danh sách block) | `candidates[0].content.parts[0].text` |
| Lý do dừng | `finish_reason`: `stop`, `length`, `tool_calls` | `stop_reason`: `end_turn`, `max_tokens`, `tool_use` | `finishReason`: `STOP`, `MAX_TOKENS` |
| Token | `usage.prompt_tokens`, `completion_tokens` | `usage.input_tokens`, `output_tokens` | `usageMetadata.promptTokenCount`, `candidatesTokenCount` |
| Token trúng cache | `usage.prompt_tokens_details.cached_tokens` | `usage.cache_read_input_tokens` | `usageMetadata.cachedContentTokenCount` |

*Bước 3 · Tool use* (chi tiết ở bài 05, mục 05.5):

| | OpenAI | Anthropic | Gemini |
|---|---|---|---|
| Khai báo | `tools: [{"type": "function", "function": {name, description, parameters}}]` | `tools: [{name, description, input_schema}]` | `tools: [{"functionDeclarations": [...]}]` |
| Model yêu cầu gọi | `message.tool_calls[]`, đối số là **chuỗi JSON** | block `{"type": "tool_use", id, name, input}`, đối số là **object** | part `functionCall {name, args}` |
| Trả kết quả | message `role: "tool"` + `tool_call_id` | block `tool_result` + `tool_use_id`, nằm trong một message **user** | part `functionResponse` |

*Bước 4 · Streaming* (chi tiết ở bài 08):
- OpenAI: SSE, mỗi sự kiện là một `chat.completion.chunk`, kết thúc bằng `data: [DONE]`.
- Anthropic: SSE có tên sự kiện: `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`, cộng `ping` và `error`.
- Gemini: `streamGenerateContent?alt=sse`, mỗi sự kiện là một response rút gọn.

*Bước 5 · Hệ quả cho code.* Viết **một** lớp trừu tượng (`complete(messages, **params) -> Completion`), và mỗi nhà cung cấp một adapter chuyển đổi hai chiều. Gateway của bạn chỉ làm việc với lớp trừu tượng. Vì phần lớn nhà cung cấp hỗ trợ định dạng OpenAI, `LLMClient` của project chỉ cần `base_url` và key.

#### 07.3 Từng nhà cung cấp: những gì roadmap yêu cầu

**OpenAI.**
- Chat Completions (mục 07.2); **function calling** (bài 05).
- **JSON mode** (`response_format={"type": "json_object"}`, prompt phải có chữ "JSON") và **structured outputs** (`json_schema` + `strict`).
- **Streaming** (bài 08); **embeddings** (`POST /v1/embeddings`, `text-embedding-3-*` có tham số `dimensions`).
- **Vision:** phần nội dung `{"type": "image_url", "image_url": {"url": …, "detail": "low|high"}}`.
- **Assistants API** (thread, file search) đã bị **thay thế** bởi Responses API: công cụ `file_search` trên vector store thay cho tìm kiếm file của Assistants.
- **Batch API**, `tiktoken` (bài 02).
- **Rate limit** theo RPM và TPM, theo từng tier; header `x-ratelimit-remaining-requests`, `x-ratelimit-remaining-tokens`, `x-ratelimit-reset-*` (bài 09).

**Anthropic.**
- Messages API (mục 07.2); system prompt là trường riêng.
- **Context dài** (200K token); **vision** bằng block `image` (base64 hoặc URL).
- **Tool use** bằng block `tool_use`/`tool_result`; **streaming** bằng sự kiện có tên.
- Prompt caching bằng `cache_control`; đếm token bằng `POST /v1/messages/count_tokens`.
- Mã lỗi riêng **529** (overloaded), nên được thử lại.

**Google Gemini.**
- Các dòng Pro / Flash / Flash-Lite (đánh đổi chất lượng và giá).
- **Đa phương thức từ đầu:** văn bản, ảnh, video, âm thanh, PDF (`inline_data` hoặc Files API).
- **Grounding với Google Search** (công cụ `google_search`; trả về nguồn).
- **Context caching** (`cachedContents`, có TTL và phí lưu theo token-giờ); context tới 1–2 triệu token.
- Có endpoint tương thích OpenAI.

**Mistral.**
- API của Mistral theo định dạng gần với OpenAI; có function calling, JSON mode.
- Model mở: Mistral 7B, Mixtral 8x7B và 8x22B (MoE, bài 01). Model thương mại: Mistral Large.
- Model mở chạy được qua Ollama.

**LLaMA và model mở (Hugging Face, Ollama).**
- Tải trọng số từ Hugging Face Hub (cần đồng ý giấy phép với Llama), chạy bằng `transformers` (như `gateway/local_llm.py`) hoặc bằng server suy luận (vLLM, Tuần 12).
- **Ollama:** `ollama run <model>` tải bản GGUF đã lượng tử hóa, mở server ở `localhost:11434` với cả API riêng (`/api/chat`) lẫn endpoint kiểu OpenAI (`/v1/chat/completions`). Chạy được trên laptop không GPU với model vài tỉ tham số int4.
- Fine-tune LLaMA bằng PEFT/LoRA: **Tuần 11**.

**Các nền tảng khác.**

| Nền tảng | Điểm mạnh | Ghi chú |
|---|---|---|
| **Groq** | Chip LPU, tốc độ decode rất cao với model mở | Endpoint kiểu OpenAI; có gói miễn phí |
| **NVIDIA NIM** | Container suy luận tối ưu cho GPU NVIDIA (TensorRT-LLM), tự host được | API kiểu OpenAI; hợp với hạ tầng on-prem |
| **Together AI** | Chạy và fine-tune nhiều model mở | Endpoint kiểu OpenAI |
| **Replicate** | Chạy model (cả ảnh, âm thanh) qua API, tính theo giây | API riêng (prediction) |
| **OpenRouter** | Một key, định tuyến tới rất nhiều nhà cung cấp và model | Endpoint kiểu OpenAI; có model miễn phí |
| **Cohere** | Embedding và **rerank** cho RAG (Tuần 8), model Command | API riêng |

#### 07.4 Ước lượng nhanh: ảnh, cache, batch

**Token của ảnh** (theo docs tại thời điểm viết):
- **OpenAI**, `detail="high"`: co ảnh cho vừa 2048 × 2048, rồi co cạnh ngắn về 768; chia ô 512 × 512; mỗi ô 170 token, cộng 85. `detail="low"`: 85 token.
- **Anthropic:** khoảng $\text{rộng} \times \text{cao} / 750$ token (ảnh quá lớn bị co lại trước).

**Ví dụ tính tay.** Ảnh chụp hóa đơn 1024 × 1024:
- OpenAI high: co cạnh ngắn về 768, ảnh thành 768 × 768, chia được 2 × 2 = 4 ô, tức $85 + 4 \times 170 =$ **765 token**. Low: 85 token.
- Anthropic: $1024 \times 1024 / 750 \approx$ **1 398 token**.

Một ảnh tốn token bằng khoảng 50–100 tin nhắn tổng đài. Nếu chỉ cần đọc số tiền trên hóa đơn, hãy cắt vùng ảnh hoặc dùng `detail="low"`.

**Prompt caching:**
- Chỉ phần **đầu** prompt giống hệt nhau mới được cache (KV-cache tính tuần tự, bài 02). Hãy xếp prompt theo thứ tự: system → few-shot → tài liệu tĩnh → lịch sử → câu hỏi mới. Thay đổi một ký tự ở đầu prompt là mất cache toàn bộ.
- **Đo trên máy bạn** (model 0.5B, prompt few-shot 421 token, trong đó 403 token là phần chung): prefill không cache **3 048 ms**; dùng lại KV-cache của phần chung **727 ms** (−76%). Nhà cung cấp làm đúng việc này ở quy mô lớn, nên họ giảm được giá và TTFT cho token trúng cache.

**Batch:** dùng cho việc không cần trả lời ngay: gán nhãn lại log, sinh dữ liệu đánh giá, tóm tắt cuộc gọi cuối ngày. Không dùng cho chat thời gian thực.

#### 07.5 Code chuẩn: một client, nhiều nhà cung cấp

```python
PROVIDERS = {   # cùng giao thức Chat Completions, chỉ khác base_url và biến môi trường chứa key
    "groq":       ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "local":      ("http://localhost:11434/v1", None),                     # Ollama
}
base_url, env = PROVIDERS["groq"]
client = LLMClient(httpx.AsyncClient(base_url=base_url, timeout=60), api_key=os.environ[env])
c = await client.chat(MODEL_ID, messages, temperature=0, max_tokens=200)

# Adapter chiều đi cho Anthropic: tách system, đặt max_tokens, đổi tên header
def to_anthropic(messages, max_tokens=200, **params):
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    return {"system": system, "messages": rest, "max_tokens": max_tokens, **params}
```

#### 07.6 Cạm bẫy

1. **Gửi message `system` trong mảng `messages` cho Anthropic:** lỗi 400. System là trường riêng.
2. **Quên `max_tokens` với Anthropic:** lỗi 400. Với OpenAI không bắt buộc, nhưng nên luôn đặt để chặn chi phí.
3. **Đọc đối số tool call của OpenAI như object:** đó là chuỗi JSON, phải `json.loads`, và có thể sai cú pháp.
4. **Dùng JSON mode mà prompt không nhắc tới "JSON":** API từ chối, hoặc model sinh khoảng trắng tới hết `max_tokens`.
5. **Đặt phần thay đổi (ngày giờ, tên khách) ở đầu system prompt:** mất prompt caching ở mọi request.
6. **So giá theo "USD / 1M token" giữa các nhà cung cấp mà quên tokenizer khác nhau** (bài 02): cùng một câu tiếng Việt có thể chênh 1.7 lần số token.
7. **Hard-code tên model.** Model bị ngừng hỗ trợ theo lịch của nhà cung cấp; đặt tên model trong cấu hình.

**Sắc thái phỏng vấn:**
- *"Làm sao để đổi nhà cung cấp mà không sửa code nghiệp vụ?"* Lớp trừu tượng + adapter; ưu tiên giao thức kiểu OpenAI; cấu hình model theo tier, không hard-code; có bộ test hợp đồng (contract test) chạy trên provider giả.
- *"Prompt caching hoạt động thế nào, viết prompt sao cho trúng cache?"* Nhà cung cấp giữ KV-cache của phần đầu prompt; chỉ phần đầu **giống hệt** mới trúng; để phần tĩnh lên đầu, phần thay đổi xuống cuối.
- *"Khi nào dùng Batch API?"* Việc không cần trả lời ngay, khối lượng lớn: rẻ khoảng một nửa, không tốn rate limit thời gian thực.

#### 07.7 Đọc thêm

- OpenAI API Reference: *Chat Completions*, *Responses*, *Batch*, *Prompt caching*, *Vision*.
- Anthropic docs: *Messages API*, *Tool use*, *Prompt caching*, *Streaming messages*.
- Google AI for Developers: *Gemini API*, *Context caching*, *Grounding with Google Search*, *OpenAI compatibility*.
- Ollama docs: *OpenAI compatibility*.

---

## 08 · Streaming SSE và quản lý context

> ⏱ Tầng 1: 15' · Tầng 2: 45'
> **Cần biết trước:** async generator, `httpx` stream (Tuần 1, bài 08); UTF-8 nhiều byte (Tuần 1, phụ lục P1); TTFT và TPOT (bài 02); `count_chat_tokens` (bài tập Ngày 1).

> **📖 Từ mới**
> - **SSE (Server-Sent Events):** chuẩn để server đẩy một chuỗi sự kiện văn bản qua **một** HTTP response kéo dài, `Content-Type: text/event-stream`.
> - **Sự kiện SSE:** vài dòng `field: value` (`data`, `event`, `id`, `retry`), kết thúc bằng **một dòng trống**.
> - **Delta:** phần văn bản mới trong một chunk của luồng.
> - **Chunk (mảnh byte):** đơn vị dữ liệu mà mạng giao tới; **không** trùng với ranh giới sự kiện hay ranh giới ký tự.
> - **Incremental decoder:** bộ giải mã UTF-8 giữ lại các byte dở dang của một ký tự cho lần sau.
> - **Truncation:** cắt bớt phần context cho vừa ngân sách token.
> - **Summary memory:** thay phần hội thoại cũ bằng một bản tóm tắt.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Xem bóng đá **trực tiếp** so với chờ xem **bản ghi** sau trận. Trận đấu dài bằng nhau, nhưng xem trực tiếp thì bạn thấy hình ảnh ngay từ phút đầu. Streaming không làm model viết nhanh hơn; nó cho người dùng **thấy chữ sớm hơn**.

**Ví dụ có số:**
- **Gateway trên FakeLLM** (project, E4): không stream, khách chờ p50 **0.68 s** mới thấy câu trả lời; có stream, khách thấy chữ đầu tiên sau p50 **0.28 s**. Tổng thời gian như nhau.
- **Model 0.5B chạy local qua `LLMClient` + SSE** (5 câu hỏi, mỗi câu trả lời 150 token): chữ đầu tiên sau **0.34 s**, câu trả lời xong sau **8.4 s** (decode khoảng 19 token/s). Không stream thì khách nhìn màn hình trống 8 giây; có stream thì chờ dưới nửa giây (TTFT chỉ chiếm 4% tổng thời gian).
- **Parser SSE viết sai** (300 luồng của FakeLLM, byte bị cắt ngẫu nhiên 1–48 byte, giống mạng thật):

| Cách đọc luồng | Luồng đúng hoàn toàn |
|---|---|
| Giải mã từng mảnh, tách `\n\n` trên từng mảnh | **0 / 300** (gần như mọi sự kiện bị cắt ngang giữa hai mảnh) |
| Có bộ đệm dòng, nhưng vẫn giải mã UTF-8 từng mảnh | 95 / 300 (sinh ra 771 ký tự `�`, vì ký tự tiếng Việt 2–3 byte bị cắt đôi) |
| Parser đúng chuẩn (bài tập hôm nay) | **300 / 300** |

**Ba ý cốt lõi:**
1. Streaming giảm **độ trễ cảm nhận** (thời gian tới chữ đầu tiên), không giảm tổng thời gian và không giảm chi phí.
2. Luồng byte phải được ghép lại đúng ba tầng: **byte → ký tự** (incremental decoder), **ký tự → dòng**, **dòng → sự kiện** (dòng trống). Mạng có thể cắt ở bất kỳ đâu.
3. Context có hạn. Phải chủ động quyết định giữ gì: system luôn giữ; lượt cũ thì bỏ hoặc tóm tắt; tài liệu dài thì cắt giữa hoặc chọn đoạn liên quan (RAG, Tuần 8).

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 08.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Server-sent events** | Định dạng `text/event-stream`: dòng kết thúc bằng CRLF, LF hoặc CR; dòng trống phát sự kiện; `:` mở đầu chú thích; nhiều dòng `data` nối bằng `\n` | WHATWG HTML Living Standard, mục 9.2 |
| **`EventSource`** | API trình duyệt cho SSE; chỉ hỗ trợ GET, không đặt được header tùy ý, tự kết nối lại theo `retry` và `Last-Event-ID` | WHATWG HTML |
| **Chunked transfer encoding** | HTTP/1.1 gửi thân response theo từng khúc, không cần biết trước độ dài | RFC 9112, mục 7.1 |
| **UTF-8** | Mỗi ký tự 1–4 byte; chữ Việt có dấu thường 2–3 byte ("ủ" = `E1 BB A7`) | RFC 3629 |
| **Backpressure** | Bên nhận chậm thì bên gửi phải chậm lại (hoặc đệm), tránh tràn bộ nhớ | Tài liệu asyncio streams |

#### 08.2 🐢 Đi chậm từng bước: parser SSE nhận byte bị cắt vụn

Server gửi hai sự kiện:
```
data: {"t": "hủy"}\n\ndata: [DONE]\n\n
```
Chữ "ủ" là 3 byte `E1 BB A7`. Mạng giao thành 3 mảnh: `data: {"t": "h\xE1\xBB` · `\xA7y"}\n` · `\ndata: [DONE]\n\n`.

*Bước 1 · Byte → ký tự.* Mảnh 1 kết thúc giữa chữ "ủ".
- Decoder thường (`chunk.decode("utf-8", errors="replace")`) biến 2 byte dở dang thành `�`, và mảnh 2 bắt đầu bằng một byte lẻ, cũng thành `�`.
- **Incremental decoder** (`codecs.getincrementaldecoder("utf-8")()`) giữ 2 byte `E1 BB` lại, chờ mảnh 2 mang byte `A7` tới rồi mới trả ra "ủ".

*Bước 2 · Ký tự → dòng.* Sau mảnh 1, bộ đệm là `data: {"t": "h` (chưa có `\n`), nên chưa có dòng nào hoàn chỉnh. Sau mảnh 2, bộ đệm có `data: {"t": "hủy"}` + `\n`: được một dòng. Phần `\n` thứ hai nằm ở mảnh 3.

*Bước 3 · `\r\n` bị cắt đôi.* Nếu một mảnh kết thúc bằng `\r`, chưa thể biết đó là dòng kết thúc bằng `\r` hay nửa đầu của `\r\n`. Phải **chờ** ký tự tiếp theo, nếu không sẽ sinh ra một dòng trống giả và phát sự kiện sai.

*Bước 4 · Dòng → sự kiện.* Mảnh 3 mở đầu bằng `\n`: một dòng **trống**, nên phát sự kiện với `data = {"t": "hủy"}`. Tiếp theo là `data: [DONE]` và một dòng trống: phát sự kiện thứ hai.

*Bước 5 · Các luật nhỏ của chuẩn.*
- `data:x` và `data: x` cho cùng giá trị `x` (chỉ bỏ **một** dấu cách đầu).
- Hai dòng `data:` liên tiếp được nối bằng `\n`.
- Dòng `: keep-alive` là chú thích, bỏ qua (server gửi để giữ kết nối không bị proxy đóng).
- Sự kiện không có dòng `data` thì không phát.
- Hết luồng mà sự kiện chưa có dòng trống kết thúc thì bỏ sự kiện đó.

**Ba tầng (tóm tắt).** `bytes →(incremental UTF-8)→ str →(tách CR/LF/CRLF, chờ CR cuối)→ dòng →(dòng trống)→ sự kiện → JSON → delta`.

#### 08.3 Định dạng luồng của các nhà cung cấp

| | OpenAI Chat Completions | Anthropic Messages | Gemini |
|---|---|---|---|
| Bật stream | `"stream": true`; muốn có `usage` thì thêm `stream_options: {"include_usage": true}` | `"stream": true` | Endpoint `streamGenerateContent?alt=sse` |
| Một sự kiện | `data: {"object": "chat.completion.chunk", "choices": [{"delta": {"content": "…"}}]}` | `event: content_block_delta` + `data: {"delta": {"type": "text_delta", "text": "…"}}` | `data: {"candidates": [...]}` |
| Kết thúc | `data: [DONE]` | `event: message_stop` | Đóng kết nối |
| Lỗi giữa luồng | Chunk có khóa `error` | `event: error` | Sự kiện có `error` |
| Đối số tool call | `delta.tool_calls[i].function.arguments` đến **từng mẩu chuỗi** | `input_json_delta` từng mẩu JSON | Thường đến trọn vẹn |

Tool call đến từng mẩu, nên phải **tích lũy theo `index`** rồi mới `json.loads` (final, phần C4).

#### 08.4 Phía client và phía server

**Hiển thị ở trình duyệt:**
- `EventSource` chỉ GET, không đặt được header `Authorization`. Chat thường dùng POST có thân JSON, nên dùng `fetch()` + `response.body.getReader()` và một parser SSE như của bạn.
- Hiện từng delta ngay khi tới. Có nút "Dừng" gọi `AbortController.abort()`.

**Server của bạn (backend proxy):**
- Trả `StreamingResponse(media_type="text/event-stream")` (FastAPI).
- Thêm header `Cache-Control: no-cache`. Sau nginx thì tắt buffering (`X-Accel-Buffering: no`), nếu không proxy sẽ gom cả luồng rồi mới gửi.
- Gửi chú thích `: ping` mỗi 15–30 s để giữ kết nối qua proxy và load balancer.
- **Khách ngắt kết nối thì hủy request tới nhà cung cấp:** tiền vẫn bị tính cho token đã sinh, nhưng dừng sớm thì tiết kiệm phần còn lại.

**Retry khi đang stream:** chỉ thử lại nếu **chưa** gửi delta nào cho người dùng. Đã gửi rồi mà thử lại thì người dùng thấy chữ bị lặp. `LLMClient` của project làm đúng như vậy.

#### 08.5 Stream một đầu ra có cấu trúc

Gateway yêu cầu model trả JSON `{"reply": …, "intent": …, "slots": …}`. Khách chỉ cần thấy `reply`.
- **Đặt `reply` ở đầu schema** (bài 05). `FieldStreamer("reply")` đọc dần luồng JSON và chỉ trả ra các ký tự của giá trị `reply`, đã giải escape (`\"`, `\n`, `\uXXXX`, kể cả khi escape bị cắt giữa hai mảnh).
- Nếu model trả về văn bản không phải JSON (ví dụ bị prompt injection và in system prompt, bài 06), `FieldStreamer` không tìm thấy khóa `reply`, nên **không có gì** được hiện cho khách. Lớp chặn đầu ra (canary) xử lý phần còn lại.
- **Cái giá của streaming:** không thể "rút lại" chữ đã hiện. Nếu kiểm duyệt nội dung phải chạy trên toàn bộ câu trả lời, hãy chọn giữa stream có kiểm tra theo câu và không stream.

#### 08.6 Quản lý context: cắt bớt và tóm tắt

**Ngân sách.** context window = system + few-shot + tài liệu + lịch sử + câu hỏi + `max_tokens` dành cho câu trả lời.

**Ba chiến lược (bài tập hôm nay):**

| Chiến lược | Làm gì | Mất gì |
|---|---|---|
| `fit_history`: cửa sổ trượt | Giữ system và câu cuối; bỏ lượt cũ nhất theo **cặp** user/assistant tới khi vừa ngân sách | Mất hẳn thông tin cũ ("số thuê bao tôi nói ở đầu") |
| `truncate_middle` | Giữ đầu và cuối một văn bản dài (đầu có tiêu đề, cuối có kết luận), bỏ giữa | Mất chi tiết ở giữa |
| `summarize_old` | Thay các lượt cũ bằng **một** message tóm tắt do một lần gọi LLM rẻ tạo ra | Tốn thêm một lần gọi; bản tóm tắt có thể sai |

**Ví dụ tính tay.** Model context 8 192 token, `max_tokens` 1 000; system + few-shot 1 300; câu hỏi mới 60; mỗi lượt hỏi đáp cũ khoảng 180 token.
- Còn lại cho lịch sử: $8\,192 - 1\,000 - 1\,300 - 60 = 5\,832$ token, tức $\lfloor 5\,832 / 180 \rfloor = 32$ lượt.
- Hội thoại dài 50 lượt thì phải bỏ 18 lượt cũ nhất, hoặc tóm tắt chúng (khoảng 200 token), và khi đó giữ được hết 32 lượt gần nhất.

**Thông tin quan trọng không được nằm trong phần có thể bị bỏ.** Số thuê bao, tên gói khách đang hỏi: hãy trích ra thành **trạng thái có cấu trúc** (slot) và đưa lại vào system prompt mỗi lượt, thay vì trông vào lịch sử.

#### 08.7 Code chuẩn

```python
class SSEParser:
    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")()     # tầng 1: byte → ký tự
        self._buf, self._data, self._event = "", [], ""

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        self._buf += self._decoder.decode(chunk)
        events = []
        while True:                                                   # tầng 2: ký tự → dòng
            i = min((k for k in (self._buf.find("\r"), self._buf.find("\n")) if k >= 0), default=-1)
            if i < 0 or (self._buf[i] == "\r" and i == len(self._buf) - 1):
                break                                                 # chưa đủ một dòng / chờ xem có "\n" sau "\r"
            line = self._buf[:i]
            self._buf = self._buf[i + (2 if self._buf.startswith("\r\n", i) else 1):]
            if (ev := self._line(line)) is not None:                  # tầng 3: dòng trống → sự kiện
                events.append(ev)
        return events
```

```python
# Server: stream câu trả lời của gateway cho trình duyệt
@app.post("/v1/chat/stream")
async def chat_stream(body: ChatIn, user: str = Depends(current_user)):
    queue = asyncio.Queue()
    task = asyncio.create_task(gateway.handle(user, body.text, on_reply_delta=queue.put_nowait))
    async def events():
        while not (task.done() and queue.empty()):
            ...                                                       # chờ delta hoặc task xong
            yield sse_encode({"delta": d}, event="delta")
        yield sse_encode(public(task.result()), event="done")
    return StreamingResponse(events(), media_type="text/event-stream")
```

#### 08.8 Cạm bẫy

1. **Giải mã UTF-8 từng mảnh** (`chunk.decode()`): ký tự `�` xuất hiện ngẫu nhiên, nhất là với tiếng Việt.
2. **Tách sự kiện theo `\n\n` trên từng mảnh:** sự kiện bị cắt giữa hai mảnh thì mất. Phải dùng bộ đệm.
3. **Chỉ xử lý `\n`:** server dùng `\r\n` thì mọi dòng có `\r` ở cuối; `\r\n` bị cắt đôi thì sinh ra sự kiện rỗng.
4. **`json.loads` đối số tool call ở từng mẩu:** chỉ parse khi luồng đã kết thúc (`finish_reason == "tool_calls"`).
5. **Thử lại giữa chừng luồng:** chữ bị lặp.
6. **Proxy gom luồng** (nginx, một số CDN): streaming "không chạy" dù code đúng.
7. **Cắt lịch sử bằng cách bỏ đúng một message:** để lại một assistant đứng đầu, hoặc một kết quả tool không có lời gọi tool đi trước. Nhiều API báo lỗi 400 vì thứ tự vai trò sai.
8. **Cắt văn bản theo token rồi decode:** có thể cắt giữa một ký tự nhiều byte. `truncate_middle` cắt theo **ký tự**, rồi kiểm tra lại số token.

**Sắc thái phỏng vấn:**
- *"Vì sao chat dùng SSE mà không dùng WebSocket?"* Luồng chỉ một chiều (server → client); SSE là HTTP thường nên đi qua proxy, load balancer, xác thực sẵn có; đơn giản hơn. WebSocket hợp với giao tiếp hai chiều thời gian thực (voice, Tuần 12).
- *"Streaming có giảm chi phí không?"* Không. Nó giảm độ trễ cảm nhận. Chi phí chỉ giảm nếu người dùng bấm dừng và bạn hủy request.
- *"Hội thoại dài vượt context thì xử lý thế nào?"* Cửa sổ trượt theo cặp lượt; tóm tắt phần cũ; trích thông tin quan trọng thành trạng thái có cấu trúc; với tài liệu thì truy xuất đoạn liên quan (RAG).

#### 08.9 Đọc thêm

- WHATWG HTML Living Standard, mục *Server-sent events*.
- OpenAI docs, *Streaming API responses*; Anthropic docs, *Streaming Messages*.
- MDN, *Using server-sent events*; *ReadableStream*.
- Python docs, `codecs.getincrementaldecoder`.

---

## 09 · Độ tin cậy, chi phí, bảo mật

> ⏱ Tầng 1: 15' · Tầng 2: 45'
> **Cần biết trước:** retry, exponential backoff, full jitter, Little's law, phân loại lỗi (Tuần 1, bài 08–09); `.env` (Tuần 1, phụ lục P6); hashing SHA-256 (Tuần 1).

> **📖 Từ mới**
> - **RPM / TPM:** số request / số token tối đa mỗi phút mà nhà cung cấp cho phép với key của bạn.
> - **Token bucket:** xô chứa tối đa $b$ "vé", được thêm $r$ vé mỗi giây; mỗi request lấy một vé (hoặc nhiều vé). Cho phép dồn dập tới $b$, trung bình không quá $r$.
> - **Retry-After:** header server gửi kèm 429/503 cho biết phải chờ bao lâu.
> - **Fallback chain:** danh sách phương án dự phòng: model khác → cache → câu trả lời mặc định → chuyển người.
> - **Graceful degradation:** hệ thống kém đi một phần nhưng vẫn phục vụ được, thay vì sập hẳn.
> - **Cache key:** chuỗi định danh một request; hai request có cùng khóa thì dùng chung câu trả lời.
> - **Backend proxy:** trình duyệt gọi server của bạn, server của bạn mới gọi nhà cung cấp; key chỉ nằm ở server.
> - **Secret manager:** dịch vụ lưu và phát key có kiểm soát quyền, có nhật ký truy cập, hỗ trợ xoay vòng.
> - **PII (personally identifiable information):** số điện thoại, email, số CCCD…

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Trạm thu phí cao tốc có số làn giới hạn (rate limit của nhà cung cấp). Bạn điều hành một đội xe:
- Phát vé cho từng tài xế để không ai chiếm hết làn (rate limit theo khách).
- Làn kẹt thì chờ đúng thời gian biển báo ghi (Retry-After).
- Trạm đóng cửa thì đi đường khác (fallback).
- Ghi sổ mỗi chuyến tốn bao nhiêu tiền và của ai (theo dõi chi phí).
- **Không bao giờ** đưa thẻ thu phí của công ty cho khách giữ (không lộ API key ra frontend).

**Ví dụ có số (project, FakeLLM, 2 000 request, đo bằng thời gian mô phỏng):**

| Thí nghiệm | Kết quả |
|---|---|
| Gửi mọi tin nhắn cho model lớn | 2.37 USD / 1 000 request, intent đúng 0.971 |
| Định tuyến theo intent (model nhỏ, model lớn, người, từ chối) | **0.48 USD** / 1 000 request (rẻ hơn khoảng 5 lần), intent đúng 0.936 |
| Định tuyến + cache với khóa chuẩn hóa nhẹ | 0.44 USD / 1 000 request, 14.3% trúng cache |
| Model nhỏ sập 30 s: không retry, không fallback | 79% request được trả lời; trong lúc sập chỉ 40% |
| Như trên, có retry 3 lần | 82%; trong lúc sập chỉ 41% (retry không cứu được sự cố kéo dài) |
| Retry + fallback sang model lớn | **99.8%**; trong lúc sập **100%**; p95 tăng từ 1.9 s lên 3.4 s, chi phí gấp đôi |

**Ba ý cốt lõi:**
1. **Tin cậy theo lớp:** timeout → retry có jitter (đúng Retry-After) cho lỗi thoáng qua → fallback cho sự cố kéo dài → câu trả lời mặc định hoặc chuyển người. Retry và fallback giải quyết **hai loại lỗi khác nhau**.
2. **Chi phí phải đo được thì mới giảm được:** ghi token và tiền theo từng khách, tính năng, model, phiên bản prompt. Đòn bẩy lớn nhất là **định tuyến** (không gửi mọi thứ cho model lớn), sau đó mới đến cache, nén prompt, batch.
3. **Bảo mật:** key chỉ ở server (backend proxy, secret manager, xoay vòng); giới hạn tốc độ theo khách; log có che PII và không bao giờ chứa key.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 09.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **429 Too Many Requests** | Client gửi quá nhiều; có thể kèm `Retry-After` | RFC 6585, mục 4 |
| **Retry-After** | Số giây, hoặc một HTTP-date | RFC 9110, mục 10.2.3 |
| **Token bucket** | Thuật toán định hình lưu lượng: tốc độ $r$, độ dồn dập $b$ | Tanenbaum, *Computer Networks*; RFC 2697 (đo lưu lượng) |
| **Full jitter** | Chờ `uniform(0, min(cap, base·2^n))` | AWS Architecture Blog, *Exponential Backoff and Jitter*, 2015 |
| **Timeout budget / deadline** | Tổng thời gian cho cả request (gồm mọi lần thử) bị chặn trên | Google SRE Book, ch. 22 *Addressing Cascading Failures* |
| **Retry storm** | Mọi client cùng thử lại một lúc làm sự cố nặng thêm | Google SRE Book, ch. 22 |
| **Graceful degradation** | Giảm chất lượng có kiểm soát thay vì sập | Google SRE Book, ch. 22 |
| **Secret manager** | Vault, AWS Secrets Manager, GCP Secret Manager | Tài liệu từng sản phẩm |
| **Bảo vệ dữ liệu cá nhân (VN)** | Nghị định 13/2023/NĐ-CP: xử lý dữ liệu cá nhân phải có căn cứ, có biện pháp bảo vệ | Chính phủ Việt Nam, 2023 |

#### 09.2 🐢 Đi chậm từng bước: token bucket (đọc trước code)

Xô có `capacity = 5`, `rate = 2` vé mỗi giây. Ban đầu đầy (5 vé).

| Thời điểm | Sự kiện | Vé trước (sau khi nạp thêm) | Kết quả | Vé sau |
|---|---|---|---|---|
| 0.0 s | 6 request liên tiếp | 5 | 5 được, cái thứ 6 bị chặn | 0 |
| 0.0 s | Hỏi phải chờ bao lâu cho 1 vé | 0 | $(1 - 0)/2 =$ 0.5 s | 0 |
| 0.5 s | 2 request | $0 + 0.5 \times 2 = 1$ | 1 được, 1 bị chặn | 0 |
| 100 s | 5 request | $\min(5, 0 + 99.5 \times 2) = 5$ (không tích quá xô) | 5 được | 0 |

**Trong 100 giây gửi 100 request mỗi giây,** số request được đi qua là 5 (đầy xô ban đầu) cộng $2 \times 100$ (nạp thêm) = **205**. Test `test_token_bucket_long_run_rate` kiểm tra đúng con số này với `rate = 3`, `capacity = 10`.

**Vì sao không đếm "mỗi phút tối đa 120"?** Cửa sổ cố định (fixed window) cho phép 120 request ở giây 59 và thêm 120 ở giây 61: 240 request trong 2 giây. Token bucket và sliding window (final, C2) không có lỗ hổng ở ranh giới cửa sổ này.

**Không lưu thời gian của từng request.** Chỉ cần hai số (`tokens`, `last`), nạp thêm "lười" lúc có request tới. Đồng hồ phải đơn điệu (`time.monotonic`), và nếu đồng hồ lùi thì coi như không trôi.

#### 09.3 Rate limit của nhà cung cấp: ước lượng nhanh

**Ví dụ tính tay.** Key có giới hạn 500 RPM và 200 000 TPM. Mỗi request trung bình 483 token vào + 67 token ra (số đo trong project) = 550 token.
- Giới hạn theo TPM: $200\,000 / 550 = 363$ request/phút.
- Giới hạn theo RPM: 500 request/phút.
- **TPM là cổ chai:** khoảng 6 request/giây.
- Với 80 tin nhắn/giây lúc cao điểm, cần khoảng $80/6 \approx 14$ key/tier/nhà cung cấp, hoặc giảm token (prompt ngắn hơn, cache, router không gửi lên LLM), hoặc tự host.

**Mức đồng thời** (Little's law, Tuần 1 bài 08): độ trễ trung bình 1.5 s × 6 request/giây = 9 request đang chạy cùng lúc. Đặt `Semaphore(10)` cùng với một token bucket tính theo **token** (ước lượng trước khi gửi bằng `count_chat_tokens` cộng `max_tokens`, rồi điều chỉnh theo `usage` thật).

**Đọc header của nhà cung cấp:** `x-ratelimit-remaining-tokens`, `x-ratelimit-reset-tokens` (OpenAI), `anthropic-ratelimit-*` (Anthropic). Chủ động chậm lại khi sắp chạm trần, đừng chờ tới 429.

#### 09.4 Retry đúng cách

| Loại lỗi | Ví dụ | Thử lại? |
|---|---|---|
| Thoáng qua | 408, 409, 429, 500, 502, 503, 504, 529, timeout, mất kết nối | Có, với jitter; nếu có Retry-After thì chờ **đúng** thời gian đó |
| Do request | 400 (sai định dạng), 401/403 (key), 404 (sai model), 413/422 | **Không**: gửi lại vẫn lỗi y như cũ |
| Hết quota (429 kèm thông báo hết hạn mức) | Hết tiền, hết quota ngày | Không: báo động, chuyển nhà cung cấp |

**Bốn quy tắc:**
1. **Ngân sách thời gian:** tổng thời gian mọi lần thử ≤ deadline của request. Khách chat không chờ 30 s.
2. **Giới hạn số lần** (3–4), và chỉ thử lại ở **một** tầng. Nếu cả gateway lẫn SDK đều thử lại 3 lần, mỗi request thành 9 lần gọi khi có sự cố (retry storm).
3. **Không thử lại khi đã stream** một phần cho người dùng (bài 08).
4. **Tính idempotency của tác dụng phụ:** gọi lại LLM để sinh văn bản thì an toàn. Nhưng nếu LLM vừa yêu cầu một tool **có tác dụng phụ** (hủy gói, chuyển tiền) thì thử lại có thể làm việc đó hai lần. Dùng idempotency key cho hành động (Tuần 10).

#### 09.5 Fallback và giảm cấp có kiểm soát

```
model chính (tier theo intent) → tier còn lại → câu trả lời từ cache (kể cả hơi cũ) → câu trả lời mặc định / chuyển người
```

- **E3 của project:** retry không giúp được khi model nhỏ sập 30 giây (chỉ từ 79% lên 82%), vì mọi lần thử đều rơi vào khoảng sập. Fallback sang model lớn đưa tỉ lệ trả lời lên 99.8%, với cái giá là chi phí gấp đôi và p95 tăng (phải thử hết các lần với model nhỏ rồi mới chuyển).
- **Cải tiến tự nhiên: circuit breaker** (Tuần 7). Sau N lỗi liên tiếp với một model, "ngắt mạch" một thời gian, gửi thẳng sang dự phòng, không phí thời gian thử lại.
- **Fallback khác chất lượng:** model dự phòng có thể kém hơn hoặc khác định dạng. Phải chạy cùng bộ đánh giá trên **mọi** model trong chuỗi, và ghi lại `fallback_used` để biết chất lượng thực tế.

#### 09.6 Kiểm soát chi phí

**Ghi log từng request:** `user_id`, tính năng, model, phiên bản prompt, token vào/ra, token trúng cache, chi phí, độ trễ, route. Không có log này thì không trả lời được câu "tháng này vì sao hóa đơn tăng 40%?".

**Các đòn bẩy, xếp theo tác dụng thường gặp:**
1. **Định tuyến** (project E1): router TF-IDF dưới 1 ms quyết định đường đi. Tin "gặp nhân viên" và tin ngoài phạm vi (15% lưu lượng) không cần gọi LLM. Phần lớn còn lại đi model nhỏ. Kết quả: 2.37 → 0.48 USD / 1 000 request.
2. **Cache câu trả lời** (E2): chỉ cache khi `temperature = 0`, không có lịch sử, không có dấu hiệu injection. Khóa = SHA-256 của JSON chuẩn hóa gồm model, fingerprint của prompt, văn bản đã chuẩn hóa, và các tham số ảnh hưởng tới đầu ra.

| Khóa cache | Tỉ lệ trúng | Trúng nhưng trả sai slot |
|---|---|---|
| Văn bản thô | 3.3% | 0 |
| Chuẩn hóa nhẹ (`cache_text`: NFC, chữ thường, kiểu dấu, bỏ dấu câu, teencode) | **14.3%** | **0** |
| Chuẩn hóa mạnh (`normalize` của router: thay số điện thoại, số tiền bằng placeholder) | 14.6% | **6** |

   Chuẩn hóa mạnh gộp "nạp 100k cho số 0912…" với "nạp 200k cho số 0987…" vào **cùng một khóa**, nên trả cho khách thứ hai số tiền và số điện thoại của khách thứ nhất. Thêm 0.3% tỉ lệ trúng không đáng với lỗi đó.
3. **Giới hạn `max_tokens`** và yêu cầu câu trả lời ngắn.
4. **Nén prompt** (bài 06): bỏ few-shot thừa, rút gọn chỉ dẫn. Đo lại chất lượng sau mỗi lần nén.
5. **Prompt caching** (bài 07) cho phần đầu cố định; **Batch** cho việc không gấp.
6. **Ngân sách và cảnh báo:** giới hạn chi phí theo khách/ngày, cảnh báo khi vượt dự báo.

**Ví dụ tính tay.** 1 triệu tin nhắn/ngày:
- Gửi hết cho model lớn: $1\,000 \times 2.37 = 2\,370$ USD/ngày, khoảng 71 000 USD/tháng.
- Định tuyến + cache: $1\,000 \times 0.44 = 440$ USD/ngày, khoảng 13 200 USD/tháng.

Với lưu lượng ổn định ở mức này, nên tính tiếp phương án **tự host** một model nhỏ (Tuần 12).

#### 09.7 Bảo mật khi tích hợp API

1. **Không bao giờ để key ở frontend** (JavaScript, app di động): ai cũng trích ra được. Mọi lời gọi đi qua **backend proxy** (`server.py`): khách xác thực bằng phiên đăng nhập của hệ thống bạn, server dùng key của nhà cung cấp.
2. **Lưu key:**
   - Khi phát triển: `.env` (có trong `.gitignore`), nạp bằng `python-dotenv` hoặc biến môi trường.
   - Production: secret manager, cấp quyền theo dịch vụ, có nhật ký truy cập. **Không** đặt key trong image Docker, trong code, trong log.
3. **Xoay vòng key** (key rotation):
   - Luôn có **hai** key hợp lệ: tạo key mới → triển khai cho mọi instance → theo dõi lưu lượng key cũ về 0 → thu hồi key cũ.
   - Xoay định kỳ (ví dụ 90 ngày), và **ngay lập tức** khi nghi bị lộ (lỡ commit lên git: coi như đã lộ, kể cả khi đã xóa commit).
4. **Rate limit theo khách:** chặn một khách (hoặc một bot) đốt hết quota và tiền của cả hệ thống. Project E6: với 0.5 request/s, burst 10, khách gửi nhiều nhất (16% lưu lượng) bị chặn 81% request, còn các khách khác chỉ bị chặn 10%.
   - **Nhiều instance gateway** phải dùng chung trạng thái, ví dụ Redis: `INCR` + `EXPIRE` cho fixed window, hoặc một script Lua cho token bucket (Tuần 13). `KeyedLimiter` in-memory chỉ đúng khi có một instance.
5. **Kiểm tra đầu vào ở biên:** độ dài tối đa, số lượt lịch sử tối đa, khách **không được** tự gửi message `system` (`server.py` kiểm tra bằng Pydantic).
6. **Log và giám sát:**
   - Che PII trước khi ghi log (`redact_pii`); không ghi key (`mask_secret` khi phải in cấu hình); giữ log có thời hạn.
   - Giám sát tỉ lệ lỗi theo nhà cung cấp, p95, chi phí theo giờ, tỉ lệ bị chặn.
   - Với dữ liệu khách hàng ở Việt Nam, xử lý dữ liệu cá nhân phải tuân thủ Nghị định 13/2023/NĐ-CP. Đây cũng là lý do hay gặp để chọn tự host thay vì gửi dữ liệu ra nhà cung cấp nước ngoài.

#### 09.8 Code chuẩn

```python
class TokenBucket:
    def __init__(self, rate, capacity, clock=time.monotonic):
        self.rate, self.capacity, self.clock = rate, capacity, clock
        self.tokens, self.last = float(capacity), clock()

    def try_acquire(self, n=1) -> bool:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + max(0.0, now - self.last) * self.rate)
        self.last = max(self.last, now)
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

def cache_key(model, messages, params):
    payload = {"model": model, "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
               "params": {k: params[k] for k in CACHE_PARAMS if k in params}}         # bỏ "stream", "user"…
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()

async def chat(self, model, messages, **params):            # LLMClient, rút gọn
    for attempt in range(self.max_attempts):
        try:
            return await self._post_once(...)
        except ProviderError as e:
            if not e.retryable or attempt == self.max_attempts - 1:
                raise
            await self.clock.sleep(next_delay(attempt, e.retry_after, self.base, self.cap, self.rng))
```

#### 09.9 Cạm bẫy

1. **Thử lại 400/401/404.** Lãng phí, và với 401 còn có thể làm key bị khóa.
2. **Bỏ qua Retry-After,** tự backoff ngắn hơn: bị 429 tiếp, có khi bị chặn lâu hơn.
3. **Retry ở nhiều tầng** (SDK + gateway + client): số lần gọi nhân lên khi có sự cố.
4. **Fixed window** "mỗi phút N request": burst gấp đôi ở ranh giới cửa sổ.
5. **Rate limit in-memory khi chạy nhiều instance:** mỗi instance có xô riêng, nên giới hạn thực tế nhân với số instance.
6. **Cache khi `temperature > 0`,** hoặc khóa thiếu model / phiên bản prompt: đổi prompt mà vẫn trả câu trả lời cũ.
7. **Khóa cache chuẩn hóa quá mạnh:** trả câu trả lời của khách này cho khách khác (E2).
8. **Key trong frontend, trong git, trong log, trong thông báo lỗi** trả về cho khách.
9. **Không ghi `fallback_used`:** chất lượng thực tế giảm mà không ai biết vì sao.

**Sắc thái phỏng vấn:**
- *"Thiết kế xử lý lỗi khi gọi LLM API."* Phân loại lỗi; timeout với deadline; retry có jitter và Retry-After cho lỗi thoáng qua; fallback (model khác, nhà cung cấp khác, cache, mặc định, người); circuit breaker; log và cảnh báo.
- *"Giảm chi phí LLM 5 lần thế nào?"* Đo trước (log theo khách/tính năng); định tuyến theo độ khó (classifier rẻ trước); cache; prompt ngắn hơn; `max_tokens`; prompt caching; batch; cân nhắc tự host khi lưu lượng lớn và ổn định.
- *"Làm sao không lộ API key trong app mobile?"* Không để key trong app. App gọi backend của bạn bằng token phiên; backend giữ key trong secret manager, giới hạn tốc độ theo khách, ghi log.

#### 09.10 Đọc thêm

- M. Brooker, *Exponential Backoff And Jitter*, AWS Architecture Blog, 2015.
- B. Beyer et al., *Site Reliability Engineering*, ch. 21–22 (*Handling Overload*, *Addressing Cascading Failures*).
- OWASP, *Secrets Management Cheat Sheet*.
- OpenAI docs, *Rate limits*; Anthropic docs, *Rate limits*, *Errors*.
- Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân.

---

## Bài tập Ngày 3 · `stream.py`, `context.py`, `limits.py`, `cache.py` (~2h)

Lệnh test: `python -m pytest -q tests/test_stream.py tests/test_context.py tests/test_limits.py tests/test_cache.py` (6 + 4 + 5 + 3 = 18 test).

**`gateway/stream.py`** (~50'):

| Thành phần | Yêu cầu |
|---|---|
| `SSEEvent(data, event="message", id=None)` | dataclass |
| `SSEParser.feed(chunk: bytes) -> list[SSEEvent]`, `.close()`, `.last_id`, `.retry` | Mục 08.2: incremental UTF-8; dòng kết thúc bằng `\r\n`/`\n`/`\r` (chờ nếu `\r` ở cuối bộ đệm); `:` là chú thích; bỏ **một** dấu cách đầu value; nhiều `data` nối bằng `\n`; không có `data` thì không phát; `id` được nhớ cho các sự kiện sau; `retry` số nguyên. Kết quả phải **giống nhau với mọi cách cắt byte** |
| `StreamAccumulator.add(event) -> str` | `[DONE]` → `done=True`; chunk có `error` → `StreamError`; cộng `choices[].delta.content`; lưu `finish_reason`, `usage`; `.text` |
| `sse_encode(data, event=None) -> bytes` | dict → JSON (`ensure_ascii=False`); chuỗi nhiều dòng → nhiều dòng `data:`; kết thúc bằng dòng trống |
| `FieldStreamer(field="reply").feed(delta) -> str` | Ký tự **mới** của giá trị chuỗi của khóa `field`, đã giải escape (`\"`, `\\`, `\n`, `\uXXXX`…), chịu được escape bị cắt giữa hai mảnh; `.closed` khi gặp dấu `"` đóng |

**`gateway/context.py`** (~25'): `fit_history`, `truncate_middle`, `summarize_old` như docstring trong `_reference` (mục 08.6). Chú ý `fit_history` bỏ theo **cặp**: sau khi bỏ một message, nếu message đầu phần lịch sử là assistant thì bỏ luôn.

**`gateway/limits.py`** (~25'):

| Thành phần | Yêu cầu |
|---|---|
| `TokenBucket(rate, capacity, clock=time.monotonic)` | `try_acquire(n=1)`, `wait_time(n=1)`; `n > capacity` → `ValueError`; đồng hồ lùi coi như không trôi |
| `KeyedLimiter(rate, capacity, clock)` | `allow(key, n=1)`, `retry_after(key, n=1)`; mỗi khóa một xô |
| `parse_retry_after(headers, now=None)` | `retry-after-ms` (ưu tiên) hoặc `retry-after` (giây hoặc HTTP-date); không phân biệt hoa thường; không đọc được → `None`; đã qua → 0 |
| `next_delay(attempt, retry_after=None, base=0.5, cap=8.0, rng=None)` | Có Retry-After thì trả đúng giá trị đó; không thì full jitter |

**`gateway/cache.py`** (~20'): `canonical_json`, `cache_key` (chỉ các tham số trong `CACHE_PARAMS`; message chỉ lấy `role`, `content`), `TTLCache(max_items, ttl, clock)` với `get`/`set`, LRU, hết hạn khi `now >= thời điểm set + ttl`, đếm `hits`/`misses`, `hit_rate`.

**Gợi ý:**
- `test_sse_parser_any_chunking_gives_same_events` cắt luồng thành mảnh 1–7 byte, kể cả cắt giữa `\r\n` và giữa ký tự UTF-8. Nếu test này lỗi, hãy in bộ đệm sau mỗi lần `feed`.
- `FieldStreamer`: giữ một con trỏ `pos` vào bộ đệm; chỉ tiến khi đủ dữ liệu (một escape `\uXXXX` cần đủ 6 ký tự).
- `TokenBucket`: đừng dùng `time.time()` (đồng hồ hệ thống có thể bị chỉnh lùi).
