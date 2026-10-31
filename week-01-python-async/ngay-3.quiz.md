# Quiz · Ngày 3 · Concurrency, asyncio, Retry (bài 07–09)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> **Không chạy code** trong lúc làm. Đáp án ở cuối file.

## Phần 1 — Quick test (16 câu × 1đ)

**Bài 07–08**

1. Có 4 việc CPU-bound viết bằng Python thuần, chạy trên máy nhiều core với CPython có GIL. Cách nào nhanh nhất? A. `ThreadPoolExecutor` · B. `ProcessPoolExecutor` · C. asyncio
2. Đúng/Sai: phép nhân ma trận lớn của NumPy chạy trong một thread sẽ bị GIL chặn hoàn toàn.
3. Đoạn code sau in ra gì? (Bỏ qua các warning.)
   ```python
   async def f(): return 1
   async def main():
       x = f()
       print(type(x).__name__)
   ```
4. `except Exception:` có bắt được `asyncio.CancelledError` không?
5. Timeout mặc định của `httpx.AsyncClient` là bao nhiêu giây?
6. Rate limit là 900 RPM, mỗi request mất 2 s. Cần khoảng bao nhiêu request chạy đồng thời để tận dụng hết rate limit?

**Bài 09**

7. Mã lỗi nào **không** nên retry? A. `429` rate limit · B. `503` · C. `422` · D. `529`
8. Đúng/Sai: nên retry lỗi `429` có code `insufficient_quota` (OpenAI) với backoff.
9. Với base = 1 s, cap = 10 s, không có jitter: lần chờ ứng với $k = 4$ dài bao nhiêu giây?
10. Full jitter với $t_k = 8$ s: kỳ vọng thời gian chờ là bao nhiêu?
11. Equal jitter với $t_k = 8$ s: thời gian chờ nằm trong khoảng nào?
12. Đúng/Sai: "Full jitter làm mất tính chất cấp số nhân vì có thể chờ 0 s."
13. Có 3 tầng, mỗi tầng retry tối đa 4 lần thử. Một request của user tạo ra tối đa bao nhiêu request tới backend?
14. Jitter phát huy tác dụng lớn nhất khi nào? A. lỗi rải rác · B. nhiều client lỗi cùng lúc · C. chỉ có 1 client · D. lỗi `400`
15. Nên dùng đồng hồ nào để đo deadline? A. `time.time()` · B. `time.monotonic()` hoặc `loop.time()`
16. Lỗi xảy ra độc lập với xác suất $p = 0.3$ mỗi lần. Cần tối thiểu bao nhiêu lần thử để xác suất thành công ≥ 99%?

## Phần 2 — Tự luận (4 câu, 30 phút)

**T1 · Tính (4đ).** Một sự cố kéo dài 30 s, base = 0.5 s, không có cap. Tính số lần chờ tối thiểu để tổng thời gian chờ phủ hết sự cố, với 3 kiểu chờ: **cố định**, **tuyến tính**, **cấp số nhân**. Từ kết quả, giải thích vì sao cấp số nhân là lựa chọn mặc định.

**T2 · Code review (3đ).** Tìm ít nhất 4 lỗi trong đoạn code sau:

```python
for i in range(5):
    try:
        return client.chat.completions.create(model=M, messages=msgs)
    except Exception:
        time.sleep(2 ** i)
```

**T3 · Thiết kế (3đ).** Agent gọi tool `charge_customer(amount)` và nhận timeout sau 30 s. Có được tự động retry không? Thiết kế thế nào để không trừ tiền hai lần?

**T4 · Code asyncio (4đ).** Viết `async def first_success(*coros, timeout)` theo kiểu *hedged request*: gửi cùng một câu hỏi tới nhiều provider, lấy kết quả đầu tiên **thành công**. Yêu cầu:
- Một provider lỗi thì chờ các provider còn lại.
- Tất cả đều lỗi thì raise `ExceptionGroup`.
- Quá `timeout` thì raise `TimeoutError`.
- **Trong mọi trường hợp**, các task còn chạy dở phải được hủy và chờ cho tới khi hủy xong.

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Quick test**

1. **B**. Đo trên máy bạn: process 0.91 s, thread 5.03 s, chạy tuần tự 2.11 s.
2. **Sai**. NumPy nhả GIL trong lúc tính toán nặng.
3. `coroutine`. Hàm `f` chưa chạy; Python còn báo thêm warning *never awaited*.
4. **Không**. Từ 3.8, `CancelledError` là lớp con của `BaseException`.
5. **5 s**.
6. $\lambda = 15$ req/s, $W = 2$ s, nên $L = 30$.
7. **C**. `422` là lỗi vĩnh viễn do request sai.
8. **Sai**. Hết quota thì retry không bao giờ thành công.
9. $\min(10, 16) =$ **10 s**.
10. **4 s**.
11. **[4, 8] s**.
12. **Sai**. Kỳ vọng $t_k/2$ vẫn gấp đôi sau mỗi lần thử.
13. $4^3 =$ **64**.
14. **B**.
15. **B**. `time.time()` có thể nhảy khi giờ hệ thống được chỉnh (đồng bộ NTP, hoặc người dùng tự đổi giờ).
16. **4**, vì $0.3^4 = 0.0081 \le 0.01$, trong khi $0.3^3 = 0.027 > 0.01$.

**T1**

- **Cố định:** $0.5n \ge 30$, nên cần **60 lần**.
- **Tuyến tính** (0.5, 1, 1.5, …): $0.5 \cdot \dfrac{n(n+1)}{2} \ge 30$, tức $n(n+1) \ge 120$. Với $n = 10$ được 110, chưa đủ; với $n = 11$ được 132 ✓. Vậy cần **11 lần** (phủ được 33 s).
- **Cấp số nhân:** $0.5\,(2^n - 1) \ge 30$, tức $2^n \ge 61$, nên **$n = 6$** (phủ được 31.5 s).

Với cấp số nhân, số lần thử chỉ tăng theo log của độ dài sự cố. Nhờ vậy ta retry nhanh được khi lỗi ngắn mà vẫn phủ được sự cố dài, và tải dồn lên server cũng giảm theo hàm mũ.

*(Chấm: tính đúng cả 3 trường hợp được 3đ, lập luận được 1đ.)*

**T2**

1. `except Exception` bắt cả lỗi vĩnh viễn và bug của chính mình.
2. Không có jitter.
3. Hết lượt thì trả về `None` mà không báo lỗi, và còn ngủ thêm 16 s sau lần thử cuối.
4. SDK đã tự retry (`max_retries=2`), nên thực tế có thể lên tới 15 request.
5. Bỏ qua header `Retry-After`.
6. Không có deadline tổng.
7. `time.sleep` chặn event loop nếu đoạn code này chạy trong code async.

*(Chấm: chỉ ra đúng 4 lỗi được 3đ.)*

**T3**

**Không** được retry nguyên request như cũ, vì timeout không cho biết tiền đã bị trừ hay chưa. Thiết kế:
- Dùng **idempotency key** = `hash(conversation_id, tool_call_id)`, gửi kèm trong mọi lần thử. Server lưu key này với ràng buộc unique; nếu gặp key đã có thì trả lại kết quả cũ thay vì trừ tiền lần nữa.
- Nếu payment API không hỗ trợ idempotency key: **truy vấn trạng thái giao dịch** trước khi retry, hoặc chuyển cho người xác nhận.
- Ghi audit log cho mọi lần thử.

*(Chấm: có idempotency key được 2đ, xử lý trường hợp API không hỗ trợ được 1đ.)*

**T4** *(đã chạy kiểm chứng với 3 tình huống: một provider lỗi nhanh và một provider thành công chậm; tất cả đều lỗi; quá timeout)*

```python
async def first_success(*coros, timeout):
    tasks = [asyncio.create_task(c) for c in coros]
    try:
        async with asyncio.timeout(timeout):
            pending, errors = set(tasks), []
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    if t.exception() is None:
                        return t.result()
                    errors.append(t.exception())
            raise ExceptionGroup("all providers failed", errors)
    finally:
        for t in tasks:
            t.cancel()                                          # hủy mọi task chưa xong (task đã xong thì cancel không có tác dụng)
        await asyncio.gather(*tasks, return_exceptions=True)    # chờ việc hủy hoàn tất
```

*(Chấm: lấy đúng kết quả thành công đầu tiên được 1đ · tất cả lỗi thì raise `ExceptionGroup` được 1đ · có timeout được 1đ · hủy và chờ hủy xong trong `finally` được 1đ.)*

*Lỗi hay gặp:*
- Dùng `asyncio.wait(..., FIRST_COMPLETED)` đúng một lần rồi trả về luôn, kể cả khi kết quả đầu tiên là lỗi.
- Quên hủy task, để các request vẫn chạy ngầm và tốn tiền.
- Dùng `gather` mà không có `return_exceptions`, nên chỉ cần một lỗi là mất hết.

</details>
