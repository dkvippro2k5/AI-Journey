# Quiz · Ngày 2 · NumPy, Pandas, Code quality (bài 05, 06, 10)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> **Không chạy code** trong lúc làm. Đáp án ở cuối file.

## Phần 1 — Quick test (14 câu × 1đ)

1. `(np.zeros((5, 1)) + np.zeros((1, 4))).shape` là gì?
2. `np.zeros((3,)) + np.zeros((4,))` cho kết quả gì?
3. Đoạn code sau in ra gì?
   ```python
   x = np.arange(10); y = x[::2]; y[:] = 0
   print(x.tolist())
   ```
4. `np.arange(6).reshape(2, 3).sum(axis=1)` cho kết quả gì?
5. Một mảng có shape `(50_000, 1024)`, dtype `float16`. Nó chiếm bao nhiêu MB?
6. Với `x = np.arange(6)`, phép nào trả về **copy**? A. `x[1:3]` · B. `x.reshape(2, 3).T` · C. `x[[0, 1]]` · D. `x.reshape(-1)`
7. `np.array([1, 2.5, True]).dtype` là gì?
8. `pd.Series([1, None, 3]).dtype` là gì?
9. `df` có 10 hàng, index mặc định 0..9. Hỏi `len(df.loc[0:2])` và `len(df.iloc[0:2])` lần lượt bằng bao nhiêu?
10. Bảng trái có key `[1, 2, 2]`, bảng phải có key `[2, 2, 3]`. Inner merge theo key cho ra bao nhiêu hàng?
11. `df` có 6 hàng chia thành 2 nhóm. `df.groupby("g")["v"].transform("mean")` có bao nhiêu phần tử?
12. Đúng/Sai: trong pandas 3, câu lệnh `df[df.a > 0]["b"] = 1` sửa được `df`.
13. `log.info("x=%s", x)` và `log.info(f"x={x}")`: cách nào **chỉ format chuỗi khi log thực sự được ghi**?
14. Đúng/Sai: lỡ commit file `.env` chứa API key, chỉ cần xóa file đó bằng một commit mới là an toàn.

## Phần 2 — Tự luận (4 câu × 4đ, 30 phút)

**T1 · NumPy.** Viết hàm `topk_batch(E, Q, k)` thỏa các yêu cầu sau:
- Đầu vào: `E` shape `(n, d)` là các vector tài liệu, `Q` shape `(q, d)` là các query.
- Trả về chỉ số của `k` tài liệu có cosine similarity cao nhất cho **từng** query, shape `(q, k)`, sắp xếp giảm dần.
- **Không dùng vòng `for`.** Chỉ sắp xếp đầy đủ trên `k` phần tử, không sắp xếp cả `n`.

Gợi ý: dùng `np.argpartition` với `axis=1` và `np.take_along_axis`.

**T2 · Pandas.** Cho bảng log sau:

| provider | latency_s | ok |
|---|---|---|
| a | 1.0 | True |
| a | 2.0 | True |
| a | 3.0 | False |
| a | 10.0 | True |
| b | 0.5 | True |
| b | 0.7 | True |
| b | 0.9 | True |

- (a) Viết code tạo bảng tóm tắt theo từng provider, gồm các cột `n`, `success_rate` và `p95_latency` (dùng `quantile(0.95)`). Tính tay ra kết quả. Pandas mặc định nội suy tuyến tính khi tính quantile.
- (b) Thêm cột `slow`, bằng `True` khi latency của dòng đó lớn hơn p90 **của chính provider đó**. Giải thích vì sao phải dùng `transform` mà không dùng `agg`.

**T3 · Debug.** Báo cáo chi phí tháng này **cao gấp rưỡi** so với hóa đơn thật. Code tính chi phí như sau:

```python
cost = calls.merge(prices, on="model")
cost["usd"] = cost["tokens"] * cost["price"]
```

Bảng `prices` có cột `since` (ngày giá bắt đầu áp dụng), vì một model từng đổi giá. Hãy giải thích nguyên nhân bằng một ví dụ có số, rồi sửa code sao cho lỗi tương tự **không thể xảy ra âm thầm** lần nữa.

**T4 · Reproducibility.** Một bạn trong nhóm báo: *"Notebook của tôi mỗi lần chạy cho ra một kết quả khác nhau."* Liệt kê **5 nguyên nhân** có thể có, kèm cách khắc phục từng nguyên nhân. Ít nhất một nguyên nhân phải liên quan tới thứ tự duyệt của `set`.

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Quick test**

1. `(5, 4)`.
2. `ValueError`: shape 3 và 4 không broadcast được với nhau.
3. `[0, 1, 0, 3, 0, 5, 0, 7, 0, 9]`. Slice có bước nhảy vẫn là view.
4. `[3, 12]`.
5. $5\cdot10^4 \times 1024 \times 2$ B = **102.4 MB**.
6. **C**. Index bằng mảng (advanced indexing) luôn tạo copy.
7. `float64`. Kiểu bool và int được nâng lên float.
8. `float64`, vì `None` thành `NaN`.
9. `3` và `2`. Slice của `loc` lấy cả đầu mút cuối.
10. **4**: key 2 xuất hiện 2 lần ở mỗi bảng, nên có 2 × 2 hàng.
11. **6**: `transform` giữ nguyên số hàng.
12. **Sai**. Đó là chained assignment, không có tác dụng (pandas báo `ChainedAssignmentError`).
13. Cách dùng `%s` (lazy formatting). Với f-string, chuỗi luôn bị format ngay cả khi mức log đang tắt.
14. **Sai**. Key vẫn nằm trong lịch sử git. Phải **thu hồi (revoke) key** ngay; sau đó mới dọn lịch sử nếu cần.

**T1** *(đã chạy kiểm chứng, kết quả trùng với cách dùng `argsort` toàn bộ)*

```python
def topk_batch(E, Q, k):
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    S = Qn @ En.T                                        # (q, n)
    idx = np.argpartition(-S, k, axis=1)[:, :k]          # (q, k), chưa sắp xếp
    part = np.take_along_axis(S, idx, axis=1)            # điểm tương ứng với từng chỉ số
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, axis=1)
```

*Chấm:* đúng shape và đúng thứ tự giảm dần 2đ · không có vòng for 1đ · dùng `argpartition` 1đ. *Lỗi hay gặp:* chuẩn hóa sai axis; quên `keepdims=True` nên broadcast sai.

**T2**

```python
summary = logs.groupby("provider").agg(
    n=("ok", "size"),
    success_rate=("ok", "mean"),
    p95_latency=("latency_s", lambda s: s.quantile(0.95)),
)
logs["slow"] = logs["latency_s"] > logs.groupby("provider")["latency_s"].transform(lambda s: s.quantile(0.9))
```

**(a)** Tính tay quantile (nội suy tuyến tính, vị trí $= 0.95 \times (n-1)$ trên dãy đã sắp xếp):
- Provider a có dãy [1, 2, 3, 10], vị trí $0.95 \times 3 = 2.85$, nên $3 + 0.85 \times (10 - 3) =$ **8.95**.
- Provider b có dãy [0.5, 0.7, 0.9], vị trí $0.95 \times 2 = 1.9$, nên $0.7 + 0.9 \times 0.2 =$ **0.88**.

Bảng kết quả:

| provider | n | success_rate | p95_latency |
|---|---|---|---|
| a | 4 | 0.75 | 8.95 |
| b | 3 | 1.00 | 0.88 |

**(b)** Kết quả cột `slow` là `[F, F, F, T, F, F, T]`. Phải dùng `transform` vì nó trả về **một giá trị cho mỗi dòng** (giá trị p90 của nhóm, được lặp lại cho mọi dòng trong nhóm), nên so sánh được trực tiếp với `latency_s`. `agg` chỉ trả về 1 giá trị cho mỗi nhóm, không khớp số dòng.

*Chấm:* code đúng 2đ · tính tay đúng 1đ · giải thích transform 1đ.

**T3**

*Nguyên nhân:* model `m1` có 2 dòng trong bảng `prices` (giá cũ và giá mới). Merge vì vậy **nhân đôi mọi call của m1**.

*Ví dụ có số:* 3 call (m1: 100 token, m1: 200 token, m2: 300 token) sau khi merge thành **5 hàng**, tổng token thành 900 thay vì 600, tức gấp rưỡi.

*Cách sửa:*

```python
latest = prices.sort_values("since").drop_duplicates("model", keep="last")   # hoặc chọn giá theo ngày gọi
cost = calls.merge(latest, on="model", validate="many_to_one")               # sai cấu trúc là raise MergeError
```

Cách đúng hơn nữa là chọn giá theo **ngày của call** bằng `pd.merge_asof` trên cột thời gian. Điều quan trọng nhất là `validate=`: nó biến một lỗi âm thầm thành một lỗi được báo ra.

*Chấm:* chỉ ra nguyên nhân kèm ví dụ số 2đ · có `validate` 1đ · chọn được đúng một giá cho mỗi call 1đ.

**T4** *(mỗi ý đúng 0.8đ, tối đa 4đ)*

1. **Không cố định seed**, hoặc dùng seed toàn cục bị một hàm khác gọi chen vào. Sửa: dùng `np.random.default_rng(seed)` và truyền `rng` vào hàm.
2. **Thứ tự duyệt `set` chuỗi thay đổi giữa các lần chạy** do hash randomization. Đã kiểm chứng: 6 lần chạy cho ra 6 thứ tự khác nhau. Sửa: dùng `sorted(set_)`, hoặc đặt `PYTHONHASHSEED=0` khi cần debug.
3. **Phiên bản thư viện khác nhau.** Sửa: dùng lock file.
4. **Dữ liệu nguồn đã thay đổi**, hoặc được đọc theo thứ tự khác (ví dụ thứ tự file trả về từ `os.listdir`). Sửa: ghi hash của dữ liệu; sắp xếp danh sách file.
5. **Phép tính không tất định** (GPU, LLM API kể cả với `temperature=0`). Sửa: ghi log lại output; chấp nhận sai số và đánh giá trên nhiều lần chạy.
6. **Thứ tự chạy cell trong notebook không theo trình tự**, còn trạng thái cũ. Sửa: "Restart & Run All".

</details>
