# Đáp án · Final Tuần 1

> Output của phần A và toàn bộ code phần C đã được chạy kiểm chứng (Python 3.14, 14/14 test pass).

## A. Quick test (24đ, 2đ/câu)

| Câu | Đáp án | Vì sao |
|---|---|---|
| A1 | `[1, 2] [1, 2]` | Giá trị mặc định chỉ được tạo **một lần**, lúc định nghĩa hàm, nên cả hai lời gọi trả về *cùng một* list. `print` tính xong cả hai đối số rồi mới in. |
| A2 | `[[1, 0, 0], [1, 0, 0]]` | `* 2` nhân bản **tham chiếu** tới cùng một list con. |
| A3 | `[2, 2, 2]` | Late binding: `i` được tra lúc lambda *chạy*, khi đó `i = 2`. Sửa: `lambda i=i: i`. |
| A4 | `14 0` | Generator chỉ duyệt được một lần; lần `sum` thứ hai gặp generator đã cạn. |
| A5 | **B** `(4, 2, 3)` | Căn từ phải: `(4,1,3)` với `(_,2,1)`, mỗi chiều có giá trị 1 được kéo giãn ra. |
| A6 | `100` | Slice tạo **view** (dùng chung bộ nhớ với `x`); fancy index `x[[1, 2]]` tạo **copy**. |
| A7 | `{'a': 4, 'b': 2}` | `groupby` mặc định `dropna=True`, nên nhóm `None` bị loại. |
| A8 | **B** (~2 s) | Mỗi `time.sleep(1)` chặn event loop 1 s, hai lần là 2 s. Hai `asyncio.sleep(1)` hết hạn *trong lúc* bị chặn nên không tốn thêm thời gian. |
| A9 | Thread **≥ 8 s** · Process ≈ **2 s + overhead** | GIL chỉ cho một thread chạy bytecode tại một thời điểm, nên thread không tăng tốc được việc CPU-bound. Đo trên máy bạn, thread còn **chậm hơn cả chạy tuần tự** (5.03 s so với 2.11 s) vì các thread tranh nhau GIL. Process thì chạy song song thật. Overhead đến từ việc khởi tạo process (spawn) và pickle dữ liệu; trên Windows overhead này lớn hơn. *(1đ mỗi ý)* |
| A10 | **B** | `dataclass` chặn mutable default ngay lúc định nghĩa class. Dùng `field(default_factory=list)`. |
| A11 | **12** | 3 lần thử của SDK (mặc định `max_retries=2`) × 4 lần thử của decorator. |
| A12 | `abab` | Type hint **không được kiểm tra lúc chạy**; chỉ công cụ như mypy/pyright mới bắt được lỗi này. |

## B. Tính tay (20đ)

**B1 (6đ)**

- **(a)** $10^4 \times 768 \times 4\ \text{B} = 30{,}720{,}000$ B ≈ **30.7 MB**.
- **(b)** Shape `(10000, 10000)`; $10^8 \times 4$ B = **400 MB**.
- **(c)** Mảng trung gian có shape `(10000, 10000, 768)`: $10^8 \times 768 \times 4$ B ≈ **307 GB**.

Viết lại theo công thức:

$$\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2\,a \cdot b$$

Dạng NumPy: `sq = (X**2).sum(1)` rồi `D = sq[:, None] + sq[None, :] - 2 * X @ X.T`. Cách này chỉ cần một ma trận 400 MB, và phép `X @ X.T` dùng BLAS nên rất nhanh. *(2đ mỗi ý)*

**B2 (7đ)**

- **(a)** $600 \times 2 =$ **1200 s** (20 phút).
- **(b)** $600 / 10 = 60$ đợt, mỗi đợt 2 s, tổng **120 s**.
- **(c)** Rate limit cho phép $\lambda = 300/60 = 5$ req/s. Định luật Little: $L = \lambda W = 5 \times 2 =$ **10** request chạy đồng thời. Thời gian ≈ $600/5 = 120$ s, cộng thêm 2 s cho request cuối cùng.
- **(d)** **Không nhanh hơn**, vì rate limit mới là giới hạn thật. 50 request đồng thời sẽ cố gửi khoảng 25 req/s, nên dính `429` hàng loạt; retry lại càng làm chậm thêm. Semaphore chỉ giới hạn *số request đồng thời*, **không giới hạn được tốc độ gửi**; muốn giới hạn tốc độ phải dùng thêm rate limiter (token bucket).

*(Chấm: a, b mỗi câu 1đ; c 3đ; d 2đ.)*

**B3 (7đ)**

- **(a)** $2 \times 3 \times 3 =$ **18** request mỗi lời gọi. Tải lên backend: $20 \times 18 =$ **360 req/s**. *(2đ)*
- **(b)** 5 lần thử thì có 4 lần chờ: 0.5 + 1 + 2 + 4 = 7.5 s. Cộng thời gian của các lần thử $5 \times 2 = 10$ s, tổng tệ nhất là **17.5 s**. *(2đ)*
- **(c)** Chỉ retry ở **một tầng**, là tầng client gọi API. Gateway và service để 1 lần thử, SDK để 3 lần thử.
  - Tải: $20 \times 3 = 60$ req/s, đúng bằng 3 lần lưu lượng ✓.
  - Thời gian: $3 \times 2 + (0.5 + 1) = 7.5$ s ≤ 8 s ✓.
  - Có thể thêm deadline 8 s để chắc chắn không vượt. *(3đ: mỗi điều kiện 1đ, lý do chọn tầng 1đ)*

## C. Code (32đ)

```python
import asyncio
import functools
import random


def retry(*, max_attempts=5, base=0.5, cap=20.0, retry_on=(ConnectionError, TimeoutError),
          sleep=asyncio.sleep, rng=random.random):
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except retry_on:
                    if attempt == max_attempts - 1:
                        raise
                    await sleep(rng() * min(cap, base * 2 ** attempt))
        return wrapper
    return decorator


async def map_bounded(fn, items, limit):
    if limit < 1:
        raise ValueError("limit must be >= 1")
    sem = asyncio.Semaphore(limit)

    async def run(item):
        async with sem:
            return await fn(item)

    return await asyncio.gather(*(run(i) for i in items), return_exceptions=True)


def iter_sse_data(lines):
    buf = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            if buf:
                data = "\n".join(buf)
                buf = []
                if data == "[DONE]":
                    return
                yield data
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            buf.append(value)
```

**C1 (12đ):** 5 test, mỗi test 2đ = 10đ · `ValueError` được raise lúc tạo decorator: 2đ.

**C2 (12đ):** 4 test, mỗi test 2.5đ = 10đ · câu hỏi về 1 triệu item: 2đ.

Ý cần có cho câu hỏi về 1 triệu item:
- `gather` tạo **1 triệu task ngay từ đầu**, và toàn bộ kết quả nằm trong RAM. Bộ nhớ có thể lên tới hàng GB, và vòng lặp sự kiện phải quản lý quá nhiều task.
- Cách sửa: dùng **worker pool**, tức `limit` worker cùng lấy việc từ một `asyncio.Queue`; hoặc xử lý theo từng lô và ghi kết quả ra đĩa ngay khi có, thay vì giữ hết trong RAM.

**C3 (8đ):** 5 test, mỗi test 1.6đ.

**Lỗi hay gặp:**
- C2: tự sắp xếp lại kết quả theo thứ tự hoàn thành, làm sai thứ tự đầu vào.
- C2: dùng `asyncio.wait` rồi bị mất exception.
- C3: dùng `.strip()`, làm mất cả dấu cách thứ hai (test `" two"` sẽ fail).
- C3: vẫn yield event cuối cùng dù chưa có dòng trống kết thúc.
- C3: đọc hết input vào một list trước khi xử lý, làm test lazy fail.

## D. Thiết kế (24đ) — đáp án tham khảo

**D1 (8đ)**

- **Nút cổ chai là TPM.** Tổng token: $10^6 \times 300 = 3 \times 10^8$. Ở mức 1M TPM cần **300 phút = 5 giờ**. Trong khi đó, với batch 100 chỉ có $10^4$ request, ở 3 000 RPM mất 3.3 phút.
- **Batch 100:** mỗi request 30 000 token, nên tối đa $10^6 / 30\,000 = 33.3$ request/phút = 0.556 req/s. Định luật Little: $L = 0.556 \times 1.5 \approx 0.83$, tức **1–2 request đồng thời là đủ**.
- **Batch 10:** mỗi request 3 000 token, tối đa 333 request/phút (vẫn dưới 3 000 RPM), $L \approx 5.56 \times 1.5 \approx$ **8–9 request đồng thời**. Tổng thời gian **vẫn là 5 giờ**.
- **Kết luận:** tăng số request đồng thời không làm nhanh hơn khi TPM là giới hạn.
- **Chi phí của 1% lỗi:** khoảng 100 request lỗi × 30 000 token = 3M token, tức thêm khoảng **3 phút** (giả định trường hợp xấu nhất là request lỗi vẫn bị tính vào TPM).

*Rubric:* xác định đúng nút cổ chai và tính đúng 5 giờ 3đ · tính đúng số request đồng thời cho cả hai cỡ batch 3đ · chi phí retry 2đ.

**D2 (8đ)**

- **Đọc CSV theo luồng:** `pd.read_csv(path, chunksize=10_000)` (hoặc dùng pyarrow), xử lý xong lô nào giải phóng lô đó.
- **ID xác định (deterministic):** `chunk_id = sha256(normalized_text + model + chunker_version)`.
  - **Không** dùng số thứ tự dòng làm ID, vì file nguồn có thể thay đổi.
- **Lưu trữ:**
  - Ràng buộc `UNIQUE(chunk_id, model)` và ghi kiểu **upsert**.
  - Mỗi batch ghi trong **một transaction**, hoặc ghi ra file parquet tạm rồi rename (rename là thao tác nguyên tử).
- **Chạy tiếp sau khi máy tắt:** bỏ qua các `chunk_id` đã có trong DB.
  - Không cần checkpoint dạng "đã chạy tới dòng N".
  - Embedding là hàm thuần, nên chạy lại một chunk cũng an toàn.
- **Reproducible:**
  - Ghim tên model, phiên bản model, số chiều embedding và tham số chunking trong một file config.
  - Dùng lock file cho thư viện.
  - Lưu metadata của mỗi lần chạy vào bảng `runs`.

*Rubric:* đọc theo luồng 2đ · ID xác định + upsert 3đ · resume không trùng lặp 2đ · reproducible 1đ.

**D3 (8đ)**

- **Rate limit phía client:** dùng **hai token bucket**, một cho RPM và một cho TPM.
  - Trước khi gửi, lấy 1 token từ bucket RPM và lấy một lượng bằng số token của batch từ bucket TPM. Số token đếm trước bằng tokenizer.
  - Chạy ở khoảng 90% giới hạn để chừa dư.
  - Điều chỉnh theo các header `x-ratelimit-remaining-*` mà API trả về.
- **Chính sách retry:**
  - Chỉ retry `429`, `5xx` và timeout. Dùng full jitter, base 1 s, cap 60 s, tôn trọng `Retry-After`, tối đa 6 lần thử.
  - Đây là job chạy nền nên deadline có thể dài.
  - Nếu tỉ lệ lỗi vượt 10% thì **dừng pipeline và báo động**, tránh đốt quota.
- **Batch bị `400`:**
  - **Phòng trước tốt hơn chữa:** đếm token từng chunk trước khi gửi; chunk vượt giới hạn thì cắt nhỏ, nên không tốn request nào.
  - **Nếu vẫn xảy ra:** chia đôi batch (bisection), gửi lại từng nửa, lặp lại với nửa bị lỗi. Cần khoảng $\lceil \log_2 100 \rceil = 7$ cấp × 2 request ≈ **14 request**. Chunk xấu được đưa vào **dead-letter queue** để xử lý riêng.
- **Metric cần theo dõi:**
  - Số token/phút so với giới hạn TPM; số chunk/phút; % tiến độ và thời gian dự kiến hoàn thành (ETA).
  - Tỉ lệ `429` và tỉ lệ lỗi theo từng loại; số lần retry.
  - Latency p50/p95; số chunk trong dead-letter queue; chi phí.

*Rubric:* hai bucket 2đ · chính sách retry 2đ · xử lý batch lỗi (có phòng trước và có bisection) 2đ · metric 2đ.
