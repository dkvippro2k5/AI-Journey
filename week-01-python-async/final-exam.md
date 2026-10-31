# Final · Tuần 1 · Python Foundation & Async cho AI API

> ⏱ **180 phút** · **100 điểm** · trung bình – khó · Đạt khi ≥ 70 · Đáp án: [final-solutions.md](final-solutions.md)
> **Được dùng:** giấy nháp và máy tính cầm tay (phần A, B); Python + pytest (phần C).
> **Không được dùng:** AI, Internet, tài liệu.
> **Phần A và B làm trên giấy, không chạy code.**

| Phần | Nội dung | Điểm | Thời gian gợi ý |
|---|---|---|---|
| A | Quick test: đoán output, trắc nghiệm | 24 | 30' |
| B | Tính tay | 20 | 35' |
| C | Code (có test sẵn) | 32 | 75' |
| D | Thiết kế | 24 | 40' |

**Bài nào được kiểm tra ở câu nào:**
- 01 Data model: A1, A2
- 02 Function/closure/decorator: A3, C1
- 03 OOP & dataclass: A10
- 04 Generator: A4, C3
- 05 NumPy: A5, A6, B1
- 06 Pandas: A7, D2
- 07 GIL/thread/process: A8, A9
- 08 asyncio/httpx/SSE: B2, C2, C3, D1
- 09 Retry: A11, B3, C1, D3
- 10 Code quality: A12, D2

---

## A. Quick test (12 câu × 2đ)

Với các câu đoán output, ghi **chính xác** những gì được in ra. Nếu code báo lỗi, ghi tên exception.

**A1.**
```python
def add(x, items=[]):
    items.append(x)
    return items
print(add(1), add(2))
```

**A2.**
```python
a = [[0] * 3] * 2
a[0][0] = 1
print(a)
```

**A3.**
```python
fs = [lambda: i for i in range(3)]
print([f() for f in fs])
```

**A4.**
```python
g = (x * x for x in range(4))
print(sum(g), sum(g))
```

**A5.** Biết `a.shape == (4, 1, 3)` và `b.shape == (2, 1)`. Hỏi `(a + b).shape` là gì?
A. lỗi broadcast · B. `(4, 2, 3)` · C. `(4, 1, 3)` · D. `(2, 4, 3)`

**A6.**
```python
x = np.arange(6)
y = x[1:4];    y[0] = 100
z = x[[1, 2]]; z[0] = -1
print(x[1])
```

**A7.**
```python
df = pd.DataFrame({"g": ["a", "b", "a", None], "v": [1, 2, 3, 4]})
print(df.groupby("g")["v"].sum().to_dict())
```

**A8.** Đoạn code sau chạy hết khoảng bao nhiêu giây? A. 1 · B. 2 · C. 3 · D. 4
```python
async def a(): await asyncio.sleep(1)
async def b(): time.sleep(1)
async def main(): await asyncio.gather(a(), a(), b(), b())
```

**A9.** Trên CPython bản mặc định (có GIL), máy 4 core, một hàm CPU-bound viết bằng Python thuần. Chạy hàm này 4 lần tuần tự mất 8 s. Hỏi chạy bằng `ThreadPoolExecutor(4)` mất khoảng bao lâu? Chạy bằng `ProcessPoolExecutor(4)` mất khoảng bao lâu?

**A10.**
```python
@dataclass
class Cfg:
    tags: list = []
```
Điều gì xảy ra? A. chạy bình thường, mọi instance dùng chung một list · B. `ValueError` ngay khi định nghĩa class · C. `TypeError` khi tạo instance · D. chạy bình thường, mỗi instance có list riêng

**A11.** Bạn bọc client OpenAI SDK (để mặc định `max_retries`) bằng decorator retry tự viết với `max_attempts=4`. Khi API sập, mỗi lời gọi tạo ra tối đa bao nhiêu HTTP request?

**A12.**
```python
def f(x: int) -> int:
    return x * 2
print(f("ab"))
```

---

## B. Tính tay (20đ)

**B1 — NumPy và bộ nhớ (6đ).** Ma trận embedding `X` có kiểu float32, shape `(10000, 768)`.
- (a) `X` chiếm bao nhiêu MB?
- (b) Ma trận cosine similarity giữa mọi cặp hàng của `X` có shape gì và chiếm bao nhiêu MB?
- (c) Một bạn tính khoảng cách L2 bằng `X[:, None, :] - X[None, :, :]`. Mảng trung gian chiếm bao nhiêu GB? Viết lại công thức sao cho không cần mảng trung gian này, và giải thích vì sao cách viết lại tiết kiệm bộ nhớ.

**B2 — Đồng thời và rate limit (7đ).** Có 600 prompt cần xử lý. Mỗi lần gọi API mất 2 s. Rate limit là 300 RPM.
- (a) Chạy tuần tự mất bao lâu?
- (b) Dùng `Semaphore(10)` và giả sử **không có** rate limit: mất bao lâu?
- (c) **Có** rate limit: cần tối thiểu bao nhiêu request chạy đồng thời để tận dụng hết rate limit? Dùng **định luật Little**. Khi đó mất bao lâu?
- (d) Tăng lên `Semaphore(50)` thì có nhanh hơn không? Chuyện gì sẽ xảy ra?

**B3 — Ước lượng retry (7đ).** Lưu lượng là 20 req/s. Gateway retry 2 lần thử, service retry 3 lần thử, còn SDK để mặc định (`max_retries=2`).
- (a) Khi backend sập, mỗi lời gọi tạo ra tối đa bao nhiêu request? Tải lên backend là bao nhiêu req/s?
- (b) Service dùng base = 0.5 s, cap = 4 s, 5 lần thử, mỗi lần thử timeout 2 s. Tính tổng thời gian tệ nhất của một lời gọi.
- (c) Sửa cấu hình để tổng thời gian ≤ 8 s **và** tải ≤ 3 lần lưu lượng bình thường. Ghi rõ phép tính.

---

## C. Code (32đ)

Viết file `code/final_lib.py`, tức đặt cùng thư mục với file test. Đứng ở thư mục của tuần, chạy `python -m pytest -q code/test_final.py` phải qua hết. **Không được sửa test.**

**C1 — `retry` (12đ).** Viết decorator cho hàm async:

```python
retry(*, max_attempts=5, base=0.5, cap=20.0, retry_on=(ConnectionError, TimeoutError),
      sleep=asyncio.sleep, rng=random.random)
```

- Thời gian chờ: `rng() * min(cap, base * 2**attempt)`.
- Exception không thuộc `retry_on` thì raise ngay.
- Hết lượt thì raise lại exception của lần thử cuối; không ngủ sau lần thử cuối.
- Giữ nguyên metadata của hàm gốc.
- `max_attempts < 1` thì raise `ValueError`.

**C2 — `map_bounded` (12đ).** Viết `async def map_bounded(fn, items, limit) -> list`:
- Chạy `fn(item)` cho mọi item, với **tối đa `limit`** lời gọi đang chạy cùng lúc.
- Kết quả **giữ đúng thứ tự đầu vào**.
- Item nào lỗi thì vị trí tương ứng trong kết quả chứa chính exception đó; các item khác vẫn chạy bình thường.
- `limit < 1` thì raise `ValueError`.

Viết thêm 2 câu: với 1 triệu item, cách cài đặt của bạn gặp vấn đề gì, và sửa thế nào?

**C3 — `iter_sse_data` (8đ).** Viết **generator** `iter_sse_data(lines)` đọc luồng Server-Sent Events và yield phần `data` của từng event:
- Các event phân tách nhau bằng dòng trống. Event có nhiều dòng `data:` thì nối các dòng bằng `"\n"`.
- Bỏ qua dòng comment (bắt đầu bằng `:`) và các field khác (`event`, `id`, `retry`).
- Sau dấu `:` của field, chỉ bỏ **một** dấu cách đầu tiên. Xử lý được cả dòng kết thúc bằng `\r\n`.
- Gặp `data: [DONE]` thì dừng ngay, **không đọc thêm** dòng nào.
- Event cuối cùng không có dòng trống kết thúc thì **bỏ qua** (đúng theo spec SSE).

---

## D. Thiết kế (24đ) — Pipeline embedding 1 triệu chunk

**Bối cảnh:**
- 1 000 000 chunk văn bản, trung bình 300 token/chunk, đọc từ file CSV 4 GB.
- Embedding API cho phép: **3 000 RPM**, **1 000 000 TPM**, tối đa **100 input mỗi request**, độ trễ khoảng **1.5 s/request**.
- Khoảng 1% request gặp lỗi tạm thời.
- Thỉnh thoảng có chunk vượt giới hạn token, làm **cả batch** bị lỗi `400`.
- Máy chạy pipeline có thể bị tắt giữa chừng.

**D1 (8đ).** Tính toán:
- Giới hạn nào là nút cổ chai, RPM hay TPM? Thời gian chạy tối thiểu là bao nhiêu?
- Với batch 100, cần bao nhiêu request chạy đồng thời? Tính lại với batch 10.
- 1% lỗi tạm thời làm tốn thêm bao nhiêu token và bao nhiêu thời gian?

**D2 (8đ).** Thiết kế đọc dữ liệu và checkpoint:
- Đọc file CSV 4 GB thế nào mà không tràn RAM?
- Khi máy bị tắt, làm sao chạy tiếp mà không embed lại phần đã xong và **không tạo bản ghi trùng lặp**?
- Làm sao để chạy lại pipeline vẫn cho cùng kết quả (reproducible)?

**D3 (8đ).** Xử lý lỗi:
- Thiết kế cơ chế rate limit phía client cho **cả RPM lẫn TPM**.
- Thiết kế chính sách retry.
- Batch 100 chunk bị lỗi `400` vì một chunk xấu: làm sao tìm ra chunk đó? Cần thêm khoảng bao nhiêu request?
- Cần theo dõi những metric nào?
