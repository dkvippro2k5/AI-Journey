# Tuần 1 · Python nền tảng & Async cho AI API

**Mục tiêu tuần:** viết Python sạch; hiểu NumPy và Pandas; gọi được nhiều AI API cùng lúc một cách an toàn.
**Sản phẩm cuối tuần:** `project/multi_caller/`. Công cụ này gửi cùng một bộ prompt tới nhiều provider cùng lúc, có retry, rồi so sánh tỉ lệ thành công, độ trễ và chi phí.
**Loại tài liệu: Kỹ thuật.** Tuần này nặng về code và thiết kế. Phần toán đầy đủ bắt đầu từ Tuần 2.

## Lịch 5 ngày (khoảng 5 giờ mỗi ngày)

| Ngày | Đọc (~2–2.5h) | Quiz (45') | Làm (~2h) |
|---|---|---|---|
| 1 | [ngay-1.md](ngay-1.md): **01** Data model · **02** Function & decorator · **03** OOP & dataclass · **04** Generator | [ngay-1.quiz.md](ngay-1.quiz.md) | `multi_caller/utils.py` |
| 2 | [ngay-2.md](ngay-2.md): **05** NumPy · **06** Pandas · **10** Code quality | [ngay-2.quiz.md](ngay-2.quiz.md) | `multi_caller/prepare.py` |
| 3 | [ngay-3.md](ngay-3.md): **07** Thread/Process/GIL · **08** asyncio & httpx · **09** Retry | [ngay-3.quiz.md](ngay-3.quiz.md) | `multi_caller/caller.py` |
| 4 | — | — | **Project:** [project.md](project.md), ghép các phần thành `multi_caller/cli.py` |
| 5 | — | — | Hoàn thiện project (2h) · **Final 180'**: [final-exam.md](final-exam.md) |

Bài tập mỗi ngày là **một mảnh của project**. Đến ngày 4 bạn chỉ phải ghép các mảnh lại.

## Chuẩn bị một lần (15 phút)

```
cd "week-01-python-async/project"
python -m venv .venv
.venv\Scripts\activate
python -m pip install httpx pandas numpy pytest ruff mypy python-dotenv
python -m pytest -q        # lúc đầu sẽ báo lỗi vì chưa có code của bạn — đó là điều bình thường
```

## Cách học mỗi bài

Mỗi bài có **hai tầng**:
- **Tầng 1 · Trực giác** (10–15'): ví dụ đời thường, ba ý cốt lõi, một ví dụ nhỏ có số.
- **Tầng 2 · Kiến thức chuẩn** (25–35', **bắt buộc**): bảng thuật ngữ chuẩn kèm nguồn (PEP, RFC, tài liệu chính thức), cơ chế hoạt động, các con số cần biết, code chuẩn, lý do thiết kế, cạm bẫy và câu hỏi hay gặp khi phỏng vấn. **Đề thi chấm theo tầng 2.**

Gặp từ lạ thì tra ô **📖 Từ mới** ở đầu bài. Nếu có từ nào chưa được giải thích, hãy ghi lại để bổ sung.

**Quy ước:**
- Tên model ghi là `MODEL_ID`.
- Mọi mức giá là **giá giả định**.
- Mọi đoạn code và mọi con số đo được (thời gian, bộ nhớ) đều đã chạy thật trên Python 3.14, NumPy 2.5 và pandas 3.0. Máy bạn có thể cho ra con số khác, nhưng tỉ lệ giữa các con số sẽ tương tự.

## Phụ lục

[phu-luc.md](phu-luc.md) bổ sung những mục của roadmap mà 5 ngày ở trên chưa phủ: 6 bài P1–P6: cú pháp nền và chuỗi tiếng Việt (NFC), `collections`/`functools`/`itertools` và độ phức tạp, đọc traceback và `pdb`, profiling (`timeit`, `cProfile`, `tracemalloc`), `np.stack`/`np.where`/`concat`/`melt`/`to_datetime` và múi giờ, `tqdm`/config YAML/`aiohttp`/Jupyter–Colab. Thời lượng ~2h15 đọc + quiz 30', có đáp án trong `<details>`. Đọc vào một buổi trống trong tuần, hoặc trước khi sang tuần sau; không bắt buộc cho final tuần này.

## Kiểm tra

| Loại | Khi nào | Thời lượng | Đạt khi |
|---|---|---|---|
| Quiz ngày | Ngay sau khi đọc xong file của ngày đó | 45' (15' quick test + 30' tự luận) | ≥ 21/30 |
| Test bài tập | Sau khi làm bài tập | — | Pass hết |
| Final tuần | Ngày 5 | 180' | ≥ 70/100 |

Không đạt quiz thì đọc lại **đúng phần làm sai** trước khi sang ngày tiếp theo.

## Phase 0 · Định hướng: AI Engineer làm gì? (đọc 10')

| | AI Engineer | ML Engineer / Research |
|---|---|---|
| Công việc chính | Tích hợp model vào sản phẩm: gọi API, RAG, agent, gateway, đánh giá chất lượng, vận hành | Huấn luyện và tối ưu model, xây pipeline dữ liệu, thiết kế thí nghiệm |
| Kỹ năng lõi | Kỹ thuật phần mềm, hệ thống phân tán, đánh giá chất lượng, tối ưu chi phí và độ trễ | Toán, thống kê, deep learning, xử lý dữ liệu lớn |
| Điểm chung | **Cả hai đều cần hiểu model hoạt động thế nào.** Vì vậy lộ trình vẫn học Toán, ML, DL và Transformer ở Tuần 2–5 trước khi quay lại phần LLM Engineering |

Tuần 1 xây phần móng cho cả hai hướng: một kỹ sư AI dành phần lớn thời gian viết Python để gọi API **đồng thời** và **chịu lỗi**, đúng như những gì `multi_caller` làm.
