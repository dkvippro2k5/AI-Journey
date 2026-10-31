# Ngày 3 · Gọi API đồng thời và xử lý lỗi

> **Loại tài liệu: Kỹ thuật.** Trọng tâm là thuật ngữ chuẩn, cơ chế, code và quyết định thiết kế. Toán chỉ dùng để ước lượng nhanh.
> **Hôm nay gồm 3 bài:** **07** Thread, Process & GIL · **08** asyncio & httpx · **09** Retry
> **Thời gian:** đọc ~2h · quiz 45' · bài tập ~2h

---

## 07 · Thread, Process & GIL

> ⏱ Tầng 1: 10' · Tầng 2: 25'

> **📖 Từ mới**
> - **Process:** một chương trình đang chạy, có vùng nhớ riêng.
> - **Thread:** một luồng chạy bên trong process; các thread trong cùng process **dùng chung bộ nhớ**.
> - **CPU-bound:** việc tốn thời gian vì **tính toán**, ví dụ vòng lặp số học hay tokenize hàng triệu câu.
> - **I/O-bound:** việc tốn thời gian vì **chờ**, ví dụ chờ mạng, chờ API trả lời, chờ đọc đĩa.
> - **GIL (Global Interpreter Lock):** "khóa" trong CPython, chỉ cho **một thread chạy code Python tại một thời điểm**.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Hình dung một nhà hàng:
- **Process** là nhiều căn bếp riêng. Mỗi bếp có đầu bếp và dụng cụ riêng, nên nấu được song song thật sự, nhưng tốn chỗ và chuyển đồ giữa các bếp rất mất công.
- **Thread** là nhiều đầu bếp đứng chung một bếp, nhưng chỉ có **một con dao** (GIL). Tại mỗi thời điểm chỉ một người được thái. Nếu việc chính là thái (CPU-bound), thêm người cũng không nhanh hơn. Nếu việc chính là chờ nước sôi (I/O-bound), thì trong lúc người này chờ, người khác cầm dao thái được.
- **asyncio** là **một đầu bếp giỏi xoay xở**: bắc nồi nước lên rồi đi thái rau trong lúc chờ nước sôi, không cần thêm người nào.

**Ba ý cốt lõi:**
1. Gọi API là **I/O-bound**, nên dùng asyncio (hoặc thread nếu thư viện chỉ hỗ trợ kiểu blocking).
2. Tính toán nặng bằng Python thuần là **CPU-bound**, nên dùng process.
3. GIL chỉ chặn việc **chạy code Python song song**. Nó **không** chặn việc **chờ I/O song song**.

### TẦNG 2 · Kiến thức chuẩn

#### 7.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Concurrency vs parallelism** | *Concurrency*: xử lý nhiều việc **xen kẽ** nhau. *Parallelism*: thực thi nhiều việc **cùng một lúc** trên nhiều core | Rob Pike, *Concurrency is not Parallelism* (2012) |
| **GIL** | Mutex trong CPython, đảm bảo tại mỗi thời điểm chỉ một thread thực thi bytecode Python | Python Glossary, *global interpreter lock* |
| **Giải phóng GIL** | Thread **nhả GIL** khi chờ I/O blocking và khi chạy nhiều đoạn code C, ví dụ phần lớn phép toán NumPy | Python C-API docs |
| **Free-threaded build** | Bản build CPython không có GIL, có từ 3.13 và là **tùy chọn**. Bản cài mặc định (như trên máy bạn) **vẫn có GIL** | PEP 703 |
| **Executor** | `ThreadPoolExecutor` / `ProcessPoolExecutor`: giao việc cho một nhóm thread hoặc process có sẵn | PEP 3148, thư viện `concurrent.futures` |
| **Start method** | Cách tạo process con: `spawn` (mặc định trên Windows và macOS), `fork`, `forkserver` | Thư viện `multiprocessing` |
| **Race condition** | Kết quả phụ thuộc vào thứ tự xen kẽ của các thread khi cùng sửa dữ liệu chung | — |

#### 7.2 Cơ chế

**Chọn công cụ nào:**

| Loại việc | Công cụ | Lý do |
|---|---|---|
| Nhiều request mạng (gọi LLM API) | **asyncio** | Một thread, hàng nghìn kết nối, rất ít tốn bộ nhớ |
| I/O dùng thư viện blocking (không có bản async) | **thread**, hoặc `asyncio.to_thread` | Thread nhả GIL trong lúc chờ |
| Tính toán bằng Python thuần | **process** | Mỗi process có GIL riêng |
| Tính toán bằng NumPy hoặc PyTorch | thường **không cần** gì thêm | Thư viện đã tự chạy song song bên trong và nhả GIL |

**Process có chi phí riêng:**
- Dữ liệu gửi sang process con phải được **pickle** (tuần tự hóa).
- Với start method `spawn`, process con **import lại module chính**. Vì vậy trên Windows bắt buộc phải có `if __name__ == "__main__":`, nếu không sẽ báo lỗi hoặc sinh process vô hạn.

```python
from concurrent.futures import ProcessPoolExecutor

def cpu_job(_):                       # phải là hàm cấp module: lambda không pickle được
    return sum(i * i for i in range(10_000_000))

if __name__ == "__main__":            # bắt buộc trên Windows (spawn)
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(cpu_job, range(4)))
```

**Gọi thư viện blocking từ code async** mà không làm đứng event loop:

```python
await asyncio.to_thread(blocking_fn, arg)   # chạy blocking_fn trong một thread riêng
```

#### 7.3 Các con số cần biết

Đo trên máy của bạn: 22 core, Windows, CPython 3.14 có GIL. Mỗi phép đo chạy 4 việc giống nhau.

| Loại việc | Tuần tự | 4 thread | 4 process |
|---|---|---|---|
| CPU-bound (vòng lặp 10 triệu phép nhân) | 2.11 s | **5.03 s** (chậm hơn cả tuần tự!) | **0.91 s** |
| I/O-bound (chờ 0.5 s) | 2.00 s | **0.50 s** | — |

- **Thread chậm hơn cả chạy tuần tự** khi làm việc CPU-bound, vì các thread phải giành nhau GIL.
- **Process không nhanh được đúng 4 lần** ($2.11/4 \approx 0.53$ s): phần chênh khoảng 0.4 s là chi phí `spawn` và pickle.
- **10 000 task asyncio** chiếm khoảng **1.3 KB mỗi task** (đo bằng `tracemalloc`), tổng khoảng 13 MB. Mỗi thread thì có stack riêng, thường cỡ MB bộ nhớ ảo tùy hệ điều hành. Đây là lý do asyncio mở được hàng nghìn kết nối mà thread thì khó làm được.

#### 7.4 Cạm bẫy

1. **Dùng thread cho việc CPU-bound:** không nhanh hơn, có khi còn chậm hơn.
2. **Quên `if __name__ == "__main__":`** khi dùng process trên Windows.
3. **Truyền lambda hoặc hàm lồng nhau** vào `ProcessPoolExecutor`: lỗi pickle.
4. **Nhiều thread cùng sửa một biến** (`counter += 1`): lệnh này **không atomic**, nên kết quả có thể sai. Dùng `threading.Lock`, hoặc tránh dùng trạng thái chung.
5. **Gọi hàm blocking** (`requests.get`, `time.sleep`) **bên trong coroutine:** chặn cả event loop. Dùng thư viện async, hoặc `asyncio.to_thread`.

**Sắc thái hay bị hỏi khi phỏng vấn:** *"Có GIL thì Python không chạy đa luồng được à?"* Không hẳn. Thread vẫn chạy **đồng thời** (concurrent) và vẫn chờ I/O song song được. Chỉ có việc thực thi bytecode là tuần tự. NumPy cũng nhả GIL. Từ 3.13 có thêm bản build free-threaded, không có GIL.

#### 7.5 Đọc thêm

- Python docs: `concurrent.futures`; `multiprocessing`, mục *Contexts and start methods*. PEP 703.
- D. Beazley, *Understanding the Python GIL* (PyCon 2010). R. Pike, *Concurrency is not Parallelism* (2012).

---

## 08 · asyncio & httpx

> ⏱ Tầng 1: 10' · Tầng 2: 35'

> **📖 Từ mới**
> - **Coroutine:** hàm định nghĩa bằng `async def`. Gọi nó thì **chưa chạy gì cả**, chỉ nhận về một object coroutine.
> - **Event loop:** "người điều phối", chuyển qua lại giữa các coroutine mỗi khi một coroutine phải chờ.
> - **Task:** một coroutine đã được giao cho event loop để chạy.
> - **Semaphore:** bộ đếm giới hạn số việc được chạy cùng lúc.
> - **SSE (Server-Sent Events):** định dạng server dùng để gửi dữ liệu về **từng mẩu một**. LLM API dùng nó để stream token.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- Event loop giống **một người phục vụ duy nhất** trong nhà hàng: ghi món cho bàn 1, chuyển vào bếp, rồi **không đứng chờ** mà đi ghi món cho bàn 2.
- `await` nghĩa là "món này đang nấu, tôi đi làm việc khác đã".
- Semaphore giống **số ghế trong phòng chờ**: hết ghế thì khách sau phải đứng đợi.

```python
async def main():
    for _ in range(3):
        await asyncio.sleep(0.2)                                  # tuần tự: ~0.6 s
    await asyncio.gather(*(asyncio.sleep(0.2) for _ in range(3)))  # đồng thời: ~0.2 s
```

Đo thật: **0.61 s** khi chạy tuần tự, **0.21 s** khi dùng `gather`.

**Ba ý cốt lõi:**
1. `await` trong một vòng `for` là **tuần tự**. Muốn chạy đồng thời phải dùng `gather`, `create_task` hoặc `TaskGroup`.
2. Luôn **giới hạn số request đồng thời**, nếu không sẽ bị `429`.
3. **Dùng chung một `AsyncClient`** cho mọi request, và **luôn đặt timeout**.

### TẦNG 2 · Kiến thức chuẩn

#### 8.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **`async` / `await`** | Cú pháp coroutine gốc của Python | PEP 492 |
| **Event loop** | Vòng lặp chạy các task và xử lý I/O; mỗi thread có tối đa một loop | asyncio docs |
| **`asyncio.create_task`** | Lên lịch cho một coroutine chạy **ngay khi** event loop có cơ hội | asyncio docs, *Coroutines and Tasks* |
| **`gather` / `TaskGroup`** | Chạy nhiều task cùng lúc. `TaskGroup` (từ 3.11) theo mô hình *structured concurrency*: khi một task lỗi thì **hủy các task còn lại** và raise `ExceptionGroup` | asyncio docs; PEP 654 |
| **Cancellation** | `task.cancel()` ném `CancelledError` vào task. Từ 3.8, `CancelledError` là lớp con của **`BaseException`**, không phải `Exception` | asyncio docs |
| **`asyncio.timeout`** | Context manager đặt thời hạn (từ 3.11); hết hạn thì raise `TimeoutError` | asyncio docs |
| **Connection pooling** | `httpx.AsyncClient` giữ lại kết nối TCP/TLS (keep-alive) để dùng cho các request sau | httpx docs |
| **`httpx.Timeout`** | Bốn loại timeout: `connect`, `read`, `write`, `pool`. **Mặc định là 5 s** | httpx docs, *Timeouts* |
| **SSE** | MIME `text/event-stream`. Mỗi dòng có dạng `field: value`; **một dòng trống kết thúc một event**; dòng bắt đầu bằng `:` là comment | WHATWG HTML Living Standard, *Server-sent events* |
| **Little's law** | $L = \lambda W$: số việc đang xử lý dở = tốc độ việc đến × thời gian mỗi việc | J. Little, *Operations Research* (1961) |

#### 8.2 Cơ chế

**`gather` khác `TaskGroup` ở đâu** (đã kiểm chứng):

```python
await asyncio.gather(ok(), bad(), return_exceptions=True)   # ['ok', ValueError('bad')] — các task khác vẫn chạy tiếp

async with asyncio.TaskGroup() as tg:                       # bad() lỗi → slow() bị hủy (CancelledError)
    tg.create_task(slow())                                  # → raise ExceptionGroup([ValueError])
    tg.create_task(bad())
```

- Dùng `gather(return_exceptions=True)` khi muốn **lấy kết quả của mọi việc**, kể cả việc lỗi. Ví dụ: gọi nhiều provider rồi so sánh.
- Dùng `TaskGroup` khi các việc **phụ thuộc lẫn nhau**: một việc hỏng thì toàn bộ coi như hỏng.

**Giới hạn số request đồng thời bằng Semaphore** (đã kiểm chứng: 50 request, giới hạn 10, mỗi request 0.1 s → tối đa 10 request chạy cùng lúc, tổng **0.55 s**):

```python
async def fetch_all(client, urls, limit=10):
    sem = asyncio.Semaphore(limit)
    async def one(url):
        async with sem:                     # tối đa `limit` request được chạy cùng lúc
            return (await client.get(url)).status_code
    return await asyncio.gather(*(one(u) for u in urls))
```

**Semaphore không phải là rate limiter.** Semaphore giới hạn **số request đang chạy dở cùng lúc**. Rate limit của nhà cung cấp lại tính theo **số request mỗi phút**. Liên hệ giữa hai đại lượng này là Little's law (mục 8.3).

**Đặt timeout và dùng chung client:**

```python
timeout = httpx.Timeout(60.0, connect=5.0)       # LLM có thể sinh câu trả lời trong vài chục giây
async with httpx.AsyncClient(timeout=timeout) as client:
    ...                                          # dùng chung client cho MỌI request
```

**Stream SSE từ một API kiểu OpenAI.** Đã kiểm chứng: với stream giả lập, kết quả ghép lại được `"Xin chào!"`.

```python
async def stream_text(client, url, payload):
    async with client.stream("POST", url, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue                                   # bỏ qua comment, dòng trống, field khác
            data = line[len("data:"):].strip()
            if data == "[DONE]":                           # quy ước của OpenAI, KHÔNG có trong spec SSE
                break
            delta = json.loads(data)["choices"][0]["delta"].get("content")
            if delta:
                yield delta                                # async generator (bài 04)
```

Phiên bản này **đơn giản hóa**: nó coi mỗi dòng `data:` là một event. Spec SSE đầy đủ còn có các quy tắc khác: event gồm **nhiều dòng `data:`**, chỉ bỏ **một** dấu cách sau dấu `:`, và event cuối thiếu dòng trống thì bị bỏ. Bạn sẽ cài đặt đầy đủ các quy tắc này ở bài final, câu C3.

#### 8.3 Các con số cần biết: dùng Little's law để chọn mức đồng thời

| Rate limit | Latency mỗi request ($W$) | Tốc độ tối đa ($\lambda$) | Số request đồng thời cần có ($L = \lambda W$) |
|---|---|---|---|
| 600 RPM | 1.5 s | 10 req/s | **15** |
| 120 RPM | 3 s | 2 req/s | **6** |
| 60 RPM | 4 s | 1 req/s | **4** |

- **Đặt semaphore nhỏ hơn $L$:** không tận dụng hết rate limit.
- **Đặt lớn hơn $L$:** không nhanh hơn mà chỉ dính `429`.

#### 8.4 Cạm bẫy

1. **Quên `await`:** coroutine không bao giờ chạy, Python chỉ báo `RuntimeWarning: coroutine ... was never awaited`.
2. **`await` lần lượt trong vòng `for`** trong khi ý định là chạy đồng thời.
3. **`gather` 10 000 request mà không có giới hạn:** gây ra một trận bão `429`, và tốn bộ nhớ.
4. **Nuốt mất `CancelledError`** bằng `except BaseException` hoặc `except:` trần: task không thể bị hủy, làm timeout và shutdown bị treo.
5. **Tạo `AsyncClient` mới cho mỗi request:** mất connection pooling, mỗi request lại phải bắt tay TCP và TLS từ đầu.
6. **Để timeout mặc định 5 s của httpx:** các câu trả lời dài của LLM sẽ bị `ReadTimeout`.

**Sắc thái hay bị hỏi khi phỏng vấn:** *"asyncio có chạy song song không?"* Không. Nó là **concurrency trên một thread**. Nó nhanh với I/O vì không phải ngồi chờ, chứ không phải vì dùng nhiều core.

#### 8.5 Đọc thêm

- asyncio docs: *Coroutines and Tasks*. PEP 492, PEP 654.
- httpx docs: *Async Support*, *Timeouts*.
- WHATWG HTML Living Standard, *Server-sent events*.

---

## 09 · Retry, Exponential Backoff & Jitter

> ⏱ Tầng 1: 15 phút · Tầng 2: 25 phút
> **Cần biết trước:** `async def` / `await` (bài 08). Hiểu nôm na, `await` là "chờ việc này xong, trong lúc chờ thì máy làm việc khác".

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Bạn gọi tổng đài ngân hàng và nghe báo máy bận.
- Gọi lại liên tục mỗi giây là cách tệ: vừa mệt, vừa làm tổng đài nghẽn thêm.
- Cách khá hơn: chờ 1 phút rồi gọi lại, sau đó 2 phút, rồi 4 phút. Mỗi lần thất bại thì chờ **lâu gấp đôi**.
- Nhưng nếu 1000 người cùng bị báo bận lúc 9:00 và ai cũng chờ đúng 1 phút, cả 1000 người sẽ cùng gọi lại lúc 9:01.
- Vì vậy mỗi người nên chờ một khoảng **hơi ngẫu nhiên**, để các cuộc gọi lại không dồn vào cùng một lúc.

**Ba ý cốt lõi:**
1. Chỉ thử lại những lỗi **tự hết được** (server quá tải, gửi nhanh quá, timeout). Lỗi do chính mình (sai key, sai định dạng) thì dừng ngay.
2. Chờ **lâu dần**, vì ta không biết sự cố kéo dài bao lâu, và server đang yếu cần được "thở".
3. Chờ **ngẫu nhiên**, để nhiều client không thử lại cùng một khoảnh khắc.

**Ví dụ nhỏ.** Một tổng đài mỗi giây nhận được 2 cuộc gọi. 10 người cùng gọi lúc $t = 0$.

Nếu không chờ ngẫu nhiên, cả nhóm thử lại cùng lúc tại $t = 0, 1, 3, 7, 15$:

| $t$ (s) | Số người gọi | Được nhận | Còn lại |
|---|---|---|---|
| 0 | 10 | 2 | 8 |
| 1 | 8 | 2 | 6 |
| 3 | 6 | 2 | 4 |
| 7 | 4 | 2 | 2 |
| 15 | 2 | 2 | 0 |

Mất 15 giây và tổng cộng 30 cuộc gọi. Ở giữa các mốc trên, tổng đài ngồi không. Nếu chờ ngẫu nhiên, các cuộc gọi được rải đều ra; ước lượng chỉ mất khoảng 7 giây với khoảng 26 cuộc gọi.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 2.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Transient fault** | Lỗi tạm thời, có khả năng tự hết sau một khoảng thời gian ngắn | Azure Architecture Center, *Retry pattern* |
| **Retry policy** | Bộ quy tắc quyết định *có* thử lại hay không và *khi nào* thử lại: số lần thử tối đa, base, cap, deadline, tập lỗi được retry | AWS Builders' Library |
| **Exponential backoff** | Khoảng chờ nhân đôi sau mỗi lần thất bại: `min(cap, base · 2^k)` | IEEE 802.3 (Ethernet) |
| **Jitter** | Thành phần ngẫu nhiên thêm vào khoảng chờ để các client không thử lại đồng loạt | Brooker, AWS Architecture Blog, 2015 |
| **Thundering herd / retry storm** | Nhiều client đồng loạt thử lại cùng lúc, làm hệ thống quá tải | Google SRE Book, ch. 22 |
| **Idempotent** | *"Tác động lên server của nhiều request giống hệt nhau giống như tác động của một request"* | RFC 9110 §9.2.2 |
| **`Retry-After`** | Header server dùng để báo client cần chờ bao lâu; giá trị là số giây hoặc một mốc thời gian (HTTP-date) | RFC 9110 §10.2.3 |
| **`429 Too Many Requests`** | Mã trả về khi client đã gửi quá nhiều request | RFC 6585 |
| **Metastable failure** | Nguyên nhân ban đầu đã hết nhưng hệ thống vẫn không hồi phục, vì chính tải từ retry giữ nó trong trạng thái quá tải | Bronson et al., HotOS 2021 |

**Sắc thái hay bị hỏi khi phỏng vấn:** theo RFC 9110, `POST` **không** phải phương thức idempotent. Mọi LLM API đều gọi bằng `POST`, nhưng việc *sinh văn bản* không gây tác dụng phụ nào ngoài chi phí, nên vẫn retry được. Tool có tác dụng phụ (tạo ticket, trừ tiền) thì phải tự đảm bảo idempotent bằng *idempotency key*.

#### 2.2 Phân loại lỗi khi gọi LLM API

| Retry | Không retry |
|---|---|
| `408`, `429` (rate limit), `500`, `502`, `503`, `504`, `529` (Anthropic overloaded), timeout, connection reset | `400` (kể cả vượt context length), `401`, `403`, `404`, `422`, `429` kèm `insufficient_quota` (OpenAI) |

#### 2.3 Ba chiến lược jitter

Đặt `t = min(cap, base · 2^k)`:

| Chiến lược | Thời gian chờ thực tế | Trung bình | Khi nào dùng |
|---|---|---|---|
| **Full jitter** | ngẫu nhiên trong `[0, t]` | `t/2` | Mặc định, rải đều nhất |
| **Equal jitter** | `t/2` + ngẫu nhiên trong `[0, t/2]` | `3t/4` | Khi cần đảm bảo luôn chờ tối thiểu một khoảng |
| **Decorrelated jitter** | ngẫu nhiên trong `[base, 3 × lần chờ trước]`, có cap | — | Phương án thay thế, được Brooker phân tích |

#### 2.4 Các con số phải ước lượng được

Đây là những phép tính kiểu *back-of-envelope* mà phỏng vấn system design hay hỏi:

| Câu hỏi | Cách tính | Ví dụ |
|---|---|---|
| Tổng thời gian chờ tệ nhất? | Cộng các `min(cap, base·2^k)` | base 0.5, cap 8, 6 lần thử → 0.5+1+2+4+8 = **15.5 s** |
| Có vừa SLA không? | số lần thử × timeout mỗi lần + tổng thời gian chờ | 5 lần × 2 s + 7.5 s = **17.5 s**, vượt SLA 10 s |
| Retry nhiều tầng thì sao? | Nhân số lần thử của các tầng | 3 tầng × 3 lần thử = **27** request/lời gọi |
| Tải lên backend khi nó sập? | lưu lượng × hệ số khuếch đại | 20 req/s × 27 = **540 req/s** |
| Không jitter thì tối đa bao nhiêu client thành công? | số lần thử × số request server nhận trong một khoảnh khắc | 6 × 100 = **600** trên 1000 client |

Dòng cuối có kết quả mô phỏng để đối chiếu: 1000 client, server nhận 100 req/s, trung bình 50 lần chạy.

| | Thành công | Tổng request |
|---|---|---|
| No jitter | 600 | 4500 |
| Full jitter | ~997 | ~3824 |
| Equal jitter | 1000 | ~3557 |

**Nhận xét quan trọng.** Thêm số lần thử *không* cứu được một sự cố kéo dài: vừa lỗi thì lần thử ngay sau gần như chắc chắn cũng lỗi. Thứ cứu được là **thời gian chờ**, và khi sự cố kéo dài quá deadline thì phải **fallback**.

#### 2.5 Thuật toán chuẩn

```
for k in 0 .. max_attempts-1:
    gửi request
    thành công                     → trả kết quả
    lỗi không thuộc tập retry      → raise ngay
    lần thử cuối                   → raise lỗi của lần cuối
    chờ = Retry-After nếu có, ngược lại jitter(min(cap, base·2^k))
    nếu đã trôi + chờ > deadline   → raise TimeoutError   (kiểm tra TRƯỚC khi ngủ)
    await sleep(chờ)
```

#### 2.6 Cài đặt production (đã chạy kiểm chứng)

```python
import asyncio
import random

import httpx

RETRYABLE = {408, 429, 500, 502, 503, 504, 529}


async def post_with_retry(client, url, payload, max_attempts=5, base=0.5, cap=20.0, deadline=60.0):
    loop = asyncio.get_running_loop()
    start = loop.time()                              # đồng hồ monotonic
    for attempt in range(max_attempts):
        last = attempt == max_attempts - 1
        retry_after = None
        try:
            resp = await client.post(url, json=payload)
        except httpx.TransportError:                 # timeout, connection reset/refused
            if last:
                raise
        else:
            if resp.status_code not in RETRYABLE or last:
                resp.raise_for_status()              # 4xx vĩnh viễn → raise; 2xx → trả về
                return resp
            retry_after = resp.headers.get("retry-after")   # giả định Retry-After dạng giây
        delay = float(retry_after) if retry_after else random.uniform(0, min(cap, base * 2 ** attempt))
        if loop.time() - start + delay > deadline:
            raise TimeoutError(f"deadline exceeded after {attempt + 1} attempts")
        await asyncio.sleep(delay)
```

**Các quyết định thiết kế:**
- **Đồng hồ monotonic:** `time.time()` có thể nhảy khi giờ hệ thống được đồng bộ lại qua NTP.
- **Ưu tiên `Retry-After`** khi server gửi về.
- **Kiểm tra deadline trước khi ngủ.**
- **Hết lượt thì raise lại lỗi gốc**, không trả về `None`.
- **SDK `openai`/`anthropic` mặc định đã có `max_retries=2`:** nếu dùng lớp retry riêng thì đặt `max_retries=0` cho SDK.

#### 2.7 Cạm bẫy

1. **`except Exception`:** retry cả lỗi vĩnh viễn và bug của chính mình.
2. **Retry ở nhiều tầng:** số request bị nhân lên. Chỉ nên retry ở **đúng một tầng** (SRE Book, ch. 22).
3. **Coi timeout là thất bại:** request có thể đã được xử lý xong ở server. Tool có tác dụng phụ phải dùng idempotency key.
4. **Retry một stream sau khi đã nhận token đầu tiên:** user sẽ thấy nội dung lặp lại hoặc khác đi. Chỉ tự động retry *trước* token đầu tiên.
5. **`random.seed(hằng số)` trong code service:** mọi replica sinh cùng một dãy jitter.
6. **`time.sleep` trong coroutine:** chặn cả event loop.
7. **Không có retry budget hoặc circuit breaker:** dễ dẫn tới metastable failure (Tuần 7).

#### 2.8 Đọc thêm

- RFC 9110 §9.2.2 (idempotent) và §10.2.3 (`Retry-After`); RFC 6585 (định nghĩa `429`).
- M. Brooker, *Exponential Backoff And Jitter* (AWS Architecture Blog, 2015); *Timeouts, retries, and backoff with jitter* (Amazon Builders' Library).
- Google *SRE Book*, ch. 21–22. Bronson et al., *Metastable Failures in Distributed Systems*, HotOS 2021.

---

## Bài tập hôm nay → `project/multi_caller/caller.py`

Viết hàm:

```python
async def ask(client, provider: ProviderConfig, prompt_id: int, prompt: str, *,
              max_attempts=3, base=0.5, cap=4.0, sleep=asyncio.sleep) -> CallResult
```

Các kiểu `ProviderConfig` và `CallResult` đã có sẵn trong `multi_caller/models.py`.

**Yêu cầu:**
1. **Đọc API key** từ biến môi trường có tên `provider.api_key_env`. Nếu thiếu thì trả về ngay `CallResult` có `error` chứa tên biến đó, **không gửi request nào**.
2. **Gửi request** `POST {base_url}/chat/completions`:
   - header `Authorization: Bearer <key>`;
   - body `{"model": ..., "messages": [{"role": "user", "content": prompt}]}`.
3. **Retry** theo bài 09:
   - Retry các mã `408`, `429`, `5xx`, `529` và lỗi kết nối (`httpx.TransportError`), dùng full jitter.
   - Các mã `4xx` còn lại thì dừng ngay.
   - Thời gian chờ đi qua tham số `sleep` (để test không phải chờ thật).
4. **Đọc kết quả:**
   - `choices[0].message.content` → `text`;
   - `usage.prompt_tokens` → `input_tokens`, `usage.completion_tokens` → `output_tokens`;
   - tính chi phí theo `price_in` / `price_out` (USD cho mỗi 1 triệu token).
5. **Ghi thông tin lần gọi:** `attempts` là số lần đã thử; `latency_s` là tổng thời gian, **kể cả khi lỗi**.
6. **Không raise exception** vì lỗi API. Mọi lỗi đều được trả về qua trường `error`. Đây là một quyết định thiết kế: khi gọi 100 request cùng lúc, một request hỏng không được làm sập cả lô.

**Tự kiểm tra:** đứng ở thư mục `project/`, chạy `python -m pytest -q tests/test_caller.py`. Bộ test dùng `httpx.MockTransport`, nên không cần API key thật và không tốn tiền.

👉 Sau khi đọc xong, làm [ngay-3.quiz.md](ngay-3.quiz.md).
