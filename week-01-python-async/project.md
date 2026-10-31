# Project Tuần 1 · `multi_caller`

> **Ngày 4:** ghép các phần đã có thành một công cụ hoàn chỉnh (~5h).
> **Ngày 5:** hoàn thiện project (2h), sau đó làm final (3h).
> Project này là **viên gạch đầu tiên** của hệ thống chung. Gateway ở Tuần 6 sẽ dùng lại `ask`, retry và cách giới hạn số request đồng thời viết trong tuần này.

## 1. Bài toán

Gửi cùng một bộ prompt tới **nhiều LLM provider cùng lúc**. Gặp lỗi tạm thời thì retry. Sau đó lưu toàn bộ kết quả và so sánh các provider theo **tỉ lệ thành công, độ trễ p50/p95, token và chi phí**.

```
data/prompts_raw.csv ──prepare.py──▶ data/prompts.jsonl ──cli.py──▶ results.jsonl
                                                         │
                                         run_all: Semaphore(concurrency)
                                          ├─ ask(alpha, prompt_0) ─┐
                                          ├─ ask(beta,  prompt_0)  ├─ retry + jitter (bài 09)
                                          └─ ...                  ─┘
                                                         ▼
                                       bảng tóm tắt theo provider (pandas)
```

## 2. Thư mục project

```
project/
├── pyproject.toml            # cấu hình pytest, ruff, mypy (cho sẵn)
├── providers.example.toml    # 2 provider giả định, giá giả định (cho sẵn)
├── .env.example              # mẫu tên biến môi trường cho API key (cho sẵn)
├── data/prompts_raw.csv      # dữ liệu bẩn (cho sẵn)
├── multi_caller/
│   ├── models.py             # ProviderConfig, CallResult (cho sẵn)
│   ├── fake.py               # server giả lập, chạy offline (cho sẵn)
│   ├── utils.py              # Ngày 1: timed, read_jsonl, batched
│   ├── prepare.py            # Ngày 2: làm sạch CSV → JSONL
│   ├── caller.py             # Ngày 3: ask() có retry
│   └── cli.py                # Ngày 4: ghép tất cả lại
├── tests/                    # test cho từng ngày (cho sẵn)
└── _reference/               # lời giải tham khảo — chỉ mở khi kẹt quá 1h
```

## 3. Ngày 4 — viết `multi_caller/cli.py`

Các hàm dưới đây được kiểm tra bởi `tests/test_cli.py`.

1. **`load_providers(path) -> list[ProviderConfig]`**
   - Đọc file TOML bằng `tomllib` (có sẵn trong thư viện chuẩn từ Python 3.11).
2. **`async run_all(client, providers, prompts, *, concurrency, **ask_kwargs) -> list[CallResult]`**
   - Gọi `ask` cho **mọi cặp** (prompt, provider).
   - Số lời gọi chạy cùng lúc **không vượt quá `concurrency`**.
   - Kết quả **sắp theo thứ tự prompt, rồi đến thứ tự provider**.
   - Một lời gọi lỗi **không được làm hỏng cả lô**.
3. **`summarize_results(results) -> DataFrame`**
   - Index là `provider`.
   - Các cột, đúng thứ tự: `n`, `success_rate`, `p50_latency_s`, `p95_latency_s`, `total_tokens`, `total_cost_usd`.
4. **`write_results(results, path)`**
   - Ghi JSONL, mỗi dòng là `dataclasses.asdict(result)`, mã hóa UTF-8.
5. **`main(argv=None)`**
   - Lệnh chạy: `python -m multi_caller.cli <prompts.jsonl> --providers <toml> --out <jsonl> [--concurrency N] [--fake] [--seed S]`.
   - Cấu hình `logging` **tại đây** (đây là điểm vào của chương trình).
   - Hạ mức log của logger `httpx` xuống `WARNING`. Nếu không, mỗi request đều in ra một dòng `INFO`.
   - Gọi `load_dotenv()` nếu có cài `python-dotenv`.
   - Bọc hàm chạy chính bằng `@timed`.
   - Cờ `--fake` thì dùng `FakeLLM().client()` và tự đặt API key giả.
   - Cuối cùng in bảng tóm tắt ra màn hình.

## 4. Chạy thử

```
cd project
python -m pytest -q                                   # 28 test của cả 4 ngày
python -m multi_caller.prepare data/prompts_raw.csv data/prompts.jsonl
python -m multi_caller.cli data/prompts.jsonl --providers providers.example.toml --out results.jsonl --fake
```

**Kết quả mong đợi** (chạy bằng `--fake`, có 8 prompt và 2 provider; con số cụ thể trên máy bạn sẽ hơi khác):

```
          n  success_rate  p50_latency_s  p95_latency_s  total_tokens  total_cost_usd
provider
alpha     8           1.0         0.2205         0.2868           189          0.0018
beta      8           1.0         0.1656         0.2437           183          0.0002
```

**Gọi API thật (không bắt buộc):**
1. Copy `providers.example.toml` thành `providers.toml`.
2. Điền `base_url` của một endpoint **tương thích OpenAI Chat Completions** và điền `model`. Tra tài liệu chính thức của nhà cung cấp để lấy đúng hai giá trị này.
3. Copy `.env.example` thành `.env` và điền key. File `.env` **không bao giờ được commit**.

## 5. Ngày 5 — hoàn thiện (2h) và trả lời bằng số liệu

Dùng `--fake` để chạy các thí nghiệm sau, rồi ghi kết quả vào `project/REPORT.md`.

1. **Mức đồng thời.** Chạy với `--concurrency` bằng 1, 4 và 16, ghi lại thời gian của `_run`. Giải thích kết quả bằng Little's law.
   - Tham khảo, đo trên máy tác giả: **3.46 s → 0.88 s → 0.29 s**.
   - Vì sao từ 4 lên 16 vẫn còn nhanh hơn? Vì chỉ có 16 job, nên ở mức 16 mọi job chạy cùng lúc.
2. **Chi phí.** Dựa vào `total_cost_usd`, ước lượng chi phí khi chạy **10 000 prompt** qua mỗi provider. Ghi rõ đây là giá giả định.
3. **Retry.** Sửa `FakeLLM(fail_rate=0.3)` rồi đọc file `results.jsonl`:
   - Có bao nhiêu lời gọi phải retry?
   - Có lời gọi nào thất bại hẳn không?
   - p95 latency thay đổi ra sao? Vì sao retry làm **đuôi phân phối latency** (p95) tăng mạnh hơn p50?

**Checklist hoàn thành:**
- [ ] `python -m pytest -q`: cả 28 test pass.
- [ ] `python -m ruff check .` không còn lỗi; `python -m mypy multi_caller` không còn lỗi (cài theo hướng dẫn ở `ngay-2.md`, bài 10).
- [ ] Không còn `print` để debug trong code; mọi thông báo đi qua `logging`, trừ bảng tóm tắt in ra cuối cùng.
- [ ] Đã viết `REPORT.md` trả lời 3 câu hỏi trên.
- [ ] `.env` và `results.jsonl` có trong `.gitignore`.

**Mở rộng (không bắt buộc):**
- Thêm cờ `--stream`, dùng `stream_text` của bài 08 và ghi lại **TTFT**.
- Đặt semaphore **riêng cho từng provider**, vì mỗi provider có rate limit riêng.
