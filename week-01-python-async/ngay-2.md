# Ngày 2 · Dữ liệu với NumPy, Pandas và code chất lượng

> **Loại tài liệu: Kỹ thuật.** Phần toán ở đây chỉ là ước lượng bộ nhớ và tốc độ. Toán của vector và ma trận được học kỹ ở Tuần 2.
> **Hôm nay gồm 3 bài:** **05** NumPy · **06** Pandas · **10** Code quality & reproducibility
> **Thời gian:** đọc ~2h · quiz 45' · bài tập ~2h
> **Phiên bản:** NumPy 2.5, pandas 3.0. Pandas 3 có vài điểm khác hẳn các tài liệu cũ; những chỗ đó được đánh dấu **[pandas 3]**.

---

## 05 · NumPy: mảng, vector hóa, broadcasting

> ⏱ Tầng 1: 10' · Tầng 2: 30'

> **📖 Từ mới**
> - **ndarray:** mảng nhiều chiều của NumPy. Mọi phần tử cùng một kiểu và nằm **liền nhau** trong bộ nhớ.
> - **dtype:** kiểu của phần tử, ví dụ `float32` (4 byte) hay `float64` (8 byte).
> - **shape:** kích thước theo từng chiều, ví dụ `(3, 4)` là 3 hàng × 4 cột.
> - **Vector hóa (vectorization):** viết phép toán trên **cả mảng** thay vì viết vòng `for` cho từng phần tử.
> - **Broadcasting:** NumPy tự "kéo giãn" mảng nhỏ ra cho khớp với mảng lớn khi tính toán.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- Một list Python giống như **tủ hồ sơ**: mỗi ngăn chỉ chứa một tờ giấy ghi "đồ thật nằm ở kho số X". Muốn cộng hai list thì Python phải chạy đi chạy lại giữa tủ và kho cho từng phần tử.
- Một mảng NumPy giống như **một khay trứng**: mọi quả cùng cỡ, xếp sát nhau. Máy chỉ cần "lướt" một lượt qua cả khay, bằng code C chạy rất nhanh.

```python
import numpy as np
x = np.array([1.0, 2.0, 3.0])
print(x * 2 + 1)        # [3. 5. 7.]  ← không có vòng for nào
```

**Ba ý cốt lõi:**
1. Dữ liệu số lớn (embedding, ma trận) thì dùng NumPy, **không** dùng list.
2. Viết phép toán trên cả mảng, tránh vòng `for` trong Python.
3. Hiểu **shape** và **axis** là đã nắm được 80% NumPy.

### TẦNG 2 · Kiến thức chuẩn

#### 5.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **ndarray** | Một khối bộ nhớ liên tục, đi kèm `dtype`, `shape` và `strides` | NumPy docs, *The N-dimensional array* |
| **Strides** | Số byte phải nhảy để sang phần tử kế tiếp theo mỗi chiều | NumPy docs |
| **ufunc** | Hàm tính theo từng phần tử nhưng chạy bằng vòng lặp C (`np.add`, `np.exp`, …) | NumPy docs, *Universal functions* |
| **Broadcasting** | Quy tắc tự khớp các shape khác nhau khi tính toán | NumPy docs, *Broadcasting* |
| **Basic vs advanced indexing** | Slice (`x[1:3]`, `x[:, 0]`) trả về **view**, dùng chung bộ nhớ. Index bằng mảng hoặc mask (`x[[0, 2]]`, `x[x > 0]`) trả về **copy** | NumPy docs, *Indexing on ndarrays* |
| **axis** | Chiều mà phép toán **gom lại**. `sum(axis=0)` cộng dọc theo các hàng nên còn lại một giá trị cho mỗi cột | NumPy docs |

#### 5.2 Cơ chế

**Vì sao NumPy nhanh:**
- Các phần tử nằm liền nhau, nên CPU đọc theo từng khối và dùng được cache.
- Không có object Python cho từng số (nhớ lại: mỗi `float` Python tốn 24 B, bài 01).
- Vòng lặp chạy bằng C, nhiều chỗ dùng lệnh SIMD.
- Phép `@` gọi thư viện **BLAS** đã được tối ưu sẵn.

**Strides.** Ví dụ `x = np.arange(12, dtype=np.float32).reshape(3, 4)`:
- `x.strides == (16, 4)`: sang hàng kế phải nhảy 4 phần tử × 4 B = 16 B; sang cột kế chỉ nhảy 4 B.
- `x.T.strides == (4, 16)`: phép chuyển vị **không sao chép dữ liệu**, chỉ đổi strides. Mảng kết quả vì vậy không còn liên tục theo hàng.

**Quy tắc broadcasting.** So sánh shape **từ phải sang trái**. Hai chiều tương ứng khớp nhau khi chúng **bằng nhau**, hoặc **một trong hai bằng 1**. Chiều bằng 1 sẽ được kéo giãn.

| Shape A | Shape B | Kết quả |
|---|---|---|
| `(2, 3)` | `(3,)` | `(2, 3)` |
| `(4, 1, 3)` | `(2, 1)` | `(4, 2, 3)` |
| `(2, 3)` | `(3, 1)` | **lỗi**: chiều cuối (3 với 1) khớp vì 1 được kéo giãn, nhưng chiều kế tiếp (2 với 3) thì không khớp |
| `(3,)` | `(3, 1)` | `(3, 3)` ← **tai nạn hay gặp** |

**Hai ví dụ về axis:**

```python
x.sum(axis=0)                        # [12. 15. 18. 21.]  → mỗi cột 1 giá trị
x.mean(axis=1, keepdims=True).shape  # (3, 1) → giữ lại chiều để broadcast tiếp
centered = x - x.mean(axis=0)        # trừ trung bình cột (dạng chuẩn hóa feature)
```

#### 5.3 Các con số cần biết

| Phép đo (trên máy tác giả, máy bạn sẽ khác) | Python thuần | NumPy |
|---|---|---|
| Tích vô hướng của 2 vector 1 triệu phần tử | 128 ms | 1.4 ms (**~90×**) |
| Cosine similarity của 1 query với 1 000 vector 384 chiều | 48 ms | 1.7 ms (**~28×**) |

**Ước lượng bộ nhớ:** số phần tử × số byte của dtype. Ví dụ ma trận embedding shape `(10_000, 384)` kiểu `float32` tốn $10^4 \times 384 \times 4$ B ≈ **15.4 MB**; kiểu `float64` thì gấp đôi.

**float32 có giới hạn độ chính xác:** `np.float32(1e8) + 1 - 1e8` cho ra `0.0`, vì float32 chỉ có khoảng 7 chữ số có nghĩa. Embedding thường dùng float32 vì nó đủ chính xác cho việc so sánh và tiết kiệm được một nửa bộ nhớ.

#### 5.4 Code chuẩn: tìm top-k bằng cosine

Đây chính là lõi của semantic search (Tuần 8):

```python
def top_k_cosine(E, q, k=5):
    En = E / np.linalg.norm(E, axis=1, keepdims=True)   # chuẩn hóa từng hàng: (N, d)
    qn = q / np.linalg.norm(q)                          # (d,)
    scores = En @ qn                                    # (N,) cosine similarity
    idx = np.argpartition(-scores, k)[:k]               # O(N): lấy k phần tử lớn nhất, chưa sắp xếp
    idx = idx[np.argsort(-scores[idx])]                 # chỉ sắp xếp k phần tử đó
    return idx, scores[idx]
```

**Lý do thiết kế.** `argpartition` chỉ tốn $O(N)$, trong khi sắp xếp toàn bộ bằng `argsort` tốn $O(N \log N)$. Khi $N$ là hàng triệu và $k = 5$, chênh lệch này rất đáng kể. Kết quả đã được kiểm tra là trùng với cách dùng `argsort` toàn bộ.

#### 5.5 Cạm bẫy

1. **Shape `(n,)` gặp `(n, 1)`:** broadcasting thành `(n, n)` mà không có lỗi nào được báo. Hãy in `.shape` ở mọi bước khi debug.
2. **Sửa trên một view** làm thay đổi cả mảng gốc. Kiểm tra bằng `np.shares_memory(a, b)`.
3. **Tràn số nguyên không báo lỗi:** `np.int32(2e9) + np.int32(2e9)` cho ra `-294967296`.
4. **`np.array([1, "a"])`** thành dtype `<U21`: số `1` bị âm thầm đổi thành chuỗi.
5. **Viết vòng `for` trên mảng NumPy:** còn chậm hơn cả dùng list.

**Sắc thái hay bị hỏi khi phỏng vấn:** *"`x.T` có sao chép dữ liệu không?"* Không, nó chỉ đổi strides. *"Vì sao `x[x > 0] = 0` lại sửa được `x`?"* Vì đó là **phép gán** qua mask (Python gọi `__setitem__`), khác với việc *đọc* `x[x > 0]`, vốn trả về một bản copy.

#### 5.6 Đọc thêm

- NumPy docs: *Broadcasting*, *Indexing on ndarrays*.
- Harris et al., *Array programming with NumPy*, Nature 585 (2020).

---

## 06 · Pandas: bảng dữ liệu

> ⏱ Tầng 1: 10' · Tầng 2: 30'

> **📖 Từ mới**
> - **DataFrame:** một bảng gồm nhiều cột. Mỗi cột là một **Series** có kiểu dữ liệu riêng.
> - **Index:** "nhãn" của các hàng, mặc định là 0, 1, 2, …
> - **NaN / NA:** giá trị bị thiếu.
> - **groupby:** gom các hàng theo một cột rồi tính toán trên từng nhóm.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** DataFrame giống như **một bảng Excel lập trình được**. Mỗi cột là một loại thông tin (id, prompt, category). `groupby` chính là "Pivot table" của Excel: gom theo loại rồi đếm hoặc tính trung bình.

```python
import pandas as pd
df = pd.DataFrame({"provider": ["a", "a", "b"], "latency": [1.0, 3.0, 2.0]})
print(df.groupby("provider")["latency"].mean())   # a: 2.0, b: 2.0
```

**Ba ý cốt lõi:**
1. Pandas thao tác **theo cột**, giống NumPy, nên cũng tránh vòng `for`.
2. Dữ liệu thật luôn bẩn: thiếu giá trị, trùng lặp, dư khoảng trắng. Làm sạch là việc chiếm phần lớn thời gian.
3. `loc` truy cập theo **nhãn**, `iloc` truy cập theo **vị trí**.

### TẦNG 2 · Kiến thức chuẩn

#### 6.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **DataFrame / Series / Index** | Bảng 2 chiều có nhãn; mỗi cột là một Series; các cột dùng chung một Index | pandas docs, *Intro to data structures* |
| **`loc` / `iloc`** | Truy cập theo nhãn / theo vị trí nguyên. Với `loc`, slice **có lấy cả đầu mút cuối** | pandas docs, *Indexing and selecting data* |
| **Split-apply-combine** | Mô hình của `groupby`: tách thành nhóm, áp dụng hàm cho từng nhóm, rồi ghép kết quả | Wickham, *J. Stat. Software* 40 (2011) |
| **Copy-on-Write (CoW)** | **[pandas 3]** Mặc định từ pandas 3.0. Mọi object con lấy ra từ một DataFrame đều hành xử như một bản copy; chỉ khi bị sửa mới thực sự sao chép dữ liệu | pandas PDEP-7 |
| **Kiểu chuỗi mặc định `str`** | **[pandas 3]** Cột văn bản có dtype là `str`, không còn là `object` như trước | pandas PDEP-14 |
| **Nullable dtypes** | `Int64`, `boolean`, … dùng `pd.NA` để biểu diễn giá trị thiếu mà không phải đổi kiểu sang float | pandas docs |

#### 6.2 Cơ chế

**Các thao tác cốt lõi:**

```python
df.loc[df["category"] == "code", "prompt"]          # lọc theo mask, chọn cột
df.groupby("provider").agg(                         # named aggregation
    n=("latency", "size"),
    mean_lat=("latency", "mean"),
    success=("ok", "mean"),                         # trung bình của cột bool = tỉ lệ thành công
)
df["lat_vs_group"] = df.groupby("provider")["latency"].transform("mean")  # giữ nguyên số hàng
left.merge(right, on="k", how="left", validate="one_to_one")              # join kiểu SQL, có kiểm tra
```

**`agg` khác `transform` ở đâu?**
- `agg` trả về **1 hàng cho mỗi nhóm**.
- `transform` trả về kết quả có **đúng số hàng như ban đầu**, nên gán được lại thành một cột mới.

**Copy-on-Write [pandas 3]:**

```python
d = pd.DataFrame({"a": [1, 2]})
col = d["a"]; col.iloc[0] = 100
print(d["a"].tolist())                  # [1, 2]  ← bảng gốc không đổi
d.loc[d["a"] > 1, "a"] = 0              # cách đúng: sửa trực tiếp bằng .loc trên bảng gốc
df[df.a > 1]["b"] = 9                   # chained assignment: KHÔNG có tác dụng,
                                        # và pandas báo ChainedAssignmentError
```

**Giá trị thiếu.** Một cột số nguyên có lẫn `None` sẽ bị đổi thành `float64`, vì `NaN` là một giá trị float. Muốn giữ kiểu nguyên thì dùng `dtype="Int64"`.

**File lớn.** `pd.read_csv(path, chunksize=100_000)` trả về một **iterator** các DataFrame nhỏ. Đây chính là mẫu generator ở bài 04, áp dụng cho bảng.

#### 6.3 Các con số cần biết

| Phép đo (máy tác giả) | Kết quả |
|---|---|
| Tính tổng 100 000 hàng: dùng `iterrows()` | **1539 ms** |
| Tính tổng 100 000 hàng: dùng `df["v"].sum()` | **0.45 ms**, nhanh hơn **~3000×** |
| Cột 300 000 chuỗi chỉ có 3 giá trị khác nhau: dtype `str` | 3.9 MB |
| Cùng cột đó, dtype `category` | 0.3 MB (**~13×** nhỏ hơn) |

#### 6.4 Code chuẩn: làm sạch dữ liệu prompt

Bài tập hôm nay đi theo đúng mẫu này:

```python
def clean(df, max_chars=2000):
    out = df.copy()                                              # không sửa dữ liệu đầu vào
    out["prompt"] = out["prompt"].str.strip()
    out = out[out["prompt"].notna() & (out["prompt"] != "")]
    out["category"] = out["category"].str.strip().str.lower().fillna("general")
    ...
    return out.reset_index(drop=True)
```

**Lý do thiết kế.**
- Hàm làm sạch phải là **hàm thuần**: không sửa đầu vào và trả về bảng mới, nên dễ test và chạy lại được.
- Mỗi bước là **một phép toán theo cột**, không có vòng `for`.
- Thứ tự các bước **ảnh hưởng tới kết quả**. Ví dụ, chuẩn hóa xong rồi mới lọc trùng, nên phải ghi rõ thứ tự trong spec.

#### 6.5 Cạm bẫy

1. **`groupby` mặc định `dropna=True`:** các hàng có key bị thiếu sẽ **biến mất** khỏi kết quả. Dùng `dropna=False` nếu cần giữ chúng.
2. **Merge many-to-many làm nổ số hàng:** key `[1, 1, 2]` merge với `[1, 1, 2]` cho ra **5 hàng**, không phải 3. Dùng `validate="one_to_one"` (sai sẽ raise `MergeError`).
3. **Dùng `iterrows` hay `apply` theo từng hàng** cho những việc vector hóa được.
4. **Chained assignment** `df[mask]["col"] = v`: trong pandas 3 nó không có tác dụng gì.
5. **Đọc file CSV lớn bằng một lệnh duy nhất:** tràn RAM.

**Sắc thái hay bị hỏi khi phỏng vấn:** *"`df[df.x > 0]` trả về view hay copy?"* Với pandas 3 (CoW) thì câu trả lời **luôn là "hành xử như copy"**, và câu hỏi này không còn quan trọng như trước. Nếu người phỏng vấn dùng pandas 1.x hoặc 2.x, hãy nói rõ sự khác biệt giữa các phiên bản.

#### 6.6 Đọc thêm

- pandas docs: *Group by: split-apply-combine*, *Copy-on-Write*.
- H. Wickham, *The Split-Apply-Combine Strategy for Data Analysis*, JSS 2011.

---

## 10 · Code quality & reproducibility

> ⏱ Tầng 1: 5' · Tầng 2: 25'

> **📖 Từ mới**
> - **Virtual environment (venv):** một môi trường Python riêng cho từng project, để thư viện của project này không đụng tới project khác.
> - **Lint:** công cụ tự động tìm lỗi và code xấu.
> - **Unit test:** đoạn code tự động kiểm tra một hàm có chạy đúng không.
> - **Reproducible:** chạy lại thì ra đúng kết quả cũ.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Code chất lượng giống một **căn bếp nhà hàng**:
- **venv** là bếp riêng: dụng cụ của bếp này không lẫn sang bếp khác.
- **Test** là nếm thử trước khi mang món ra.
- **Log** là sổ ghi chép: món nào ra lúc nào, có sự cố gì.
- **Reproducible** là công thức ghi đủ liều lượng, người khác nấu lại vẫn ra đúng vị.

**Ba ý cốt lõi:**
1. Mỗi project có một venv và một danh sách thư viện **có ghim phiên bản**.
2. Không bao giờ để API key nằm trong code.
3. Dùng `logging`, không dùng `print`; viết test cho mọi hàm xử lý dữ liệu.

### TẦNG 2 · Kiến thức chuẩn

#### 10.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **venv** | Môi trường ảo nhẹ, có sẵn trong Python | PEP 405 |
| **`pyproject.toml`** | File cấu hình chuẩn của project: metadata, dependencies, cấu hình công cụ | PEP 518, PEP 621 |
| **Lock file** | Ghim **chính xác** phiên bản của mọi thư viện, kể cả thư viện gián tiếp (uv, pip-tools, poetry) | — |
| **Type hints** | Chú thích kiểu; được mypy/pyright kiểm tra tĩnh, **không được kiểm tra lúc chạy** | PEP 484 |
| **pytest: fixture / parametrize / monkeypatch** | Chuẩn bị dữ liệu dùng chung / chạy một test với nhiều bộ input / thay tạm một hàm hay biến môi trường trong lúc test | pytest docs |
| **Logging levels** | `DEBUG < INFO < WARNING < ERROR < CRITICAL` | Thư viện `logging`, *Logging HOWTO* |
| **Config qua biến môi trường** | Cấu hình (kể cả secret) đặt trong biến môi trường, không viết cứng trong code | *The Twelve-Factor App*, mục III |

#### 10.2 Cơ chế và quy ước

**Cấu trúc project:**

```
project/
├── pyproject.toml        # dependencies + cấu hình ruff/mypy/pytest
├── .env                  # API key thật — CÓ trong .gitignore
├── .env.example          # mẫu, không chứa giá trị thật — được commit
├── multi_caller/         # package
│   ├── __init__.py
│   └── ...
└── tests/
```

**Cài công cụ** (máy bạn chưa có `ruff`, `mypy`):

```
python -m venv .venv
.venv\Scripts\activate                # Windows; Linux/macOS: source .venv/bin/activate
python -m pip install pytest ruff mypy python-dotenv httpx pandas numpy
python -m ruff check . ; python -m ruff format . ; python -m mypy multi_caller
```

**Logging đúng cách:**

```python
import logging
log = logging.getLogger(__name__)          # mỗi module một logger, đặt tên theo module

log.info("processed %d rows", n)           # dùng %: chuỗi chỉ được format khi log thật sự được ghi

# Chỉ cấu hình logging ở ĐIỂM VÀO của chương trình (main/CLI), không cấu hình trong thư viện:
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
```

**Secret:**

```python
import os
from dotenv import load_dotenv
load_dotenv()                               # đọc .env vào biến môi trường (nếu file tồn tại)
api_key = os.environ["API_KEY"]             # thiếu key thì báo KeyError ngay, không chạy âm thầm
```

**Test không gọi API thật.** Dùng `httpx.MockTransport` (Ngày 3) hoặc `monkeypatch`. Test chạy nhanh, không tốn tiền, không phụ thuộc mạng, và cho cùng kết quả mỗi lần chạy.

**Reproducibility:**

| Nguồn ngẫu nhiên | Cách cố định |
|---|---|
| `random` | `random.seed(42)` (chỉ dùng trong script hoặc notebook) |
| NumPy | `rng = np.random.default_rng(42)`; truyền `rng` vào hàm, **không** dùng hàm seed toàn cục (NEP 19) |
| PyTorch | `torch.manual_seed(42)` (Tuần 4) |
| Thư viện | Lock file |
| Cấu hình | Lưu config cùng commit hash vào log của mỗi lần chạy |

**Giới hạn:** phép tính song song trên GPU, và các LLM API kể cả khi `temperature=0`, **không đảm bảo** cho ra kết quả giống hệt nhau từng bit. Reproducible ở đây nghĩa là "cùng quy trình, cùng phiên bản", chứ không phải "cùng từng bit".

#### 10.3 Cạm bẫy

1. **Commit file `.env`** lên git: key bị lộ. Phải thu hồi (revoke) key ngay; xóa file khỏi git là **không đủ**, vì key vẫn nằm trong lịch sử commit.
2. **Gọi `logging.basicConfig` bên trong module thư viện:** ghi đè cấu hình logging của ứng dụng dùng thư viện đó.
3. **Seed toàn cục trong code service** làm mất tác dụng của jitter (bài 09).
4. **Test gọi API thật:** chậm, tốn tiền, lúc qua lúc không (flaky).
5. **Dùng `pip freeze > requirements.txt`** cho cả máy: ghim luôn những thư viện không liên quan. Hãy khai báo dependency trong `pyproject.toml` và dùng lock file.

**Sắc thái hay bị hỏi khi phỏng vấn:** *"Type hint có làm code chậm đi hay an toàn hơn lúc chạy không?"* Không: Python không kiểm tra type hint lúc chạy. Lợi ích nằm ở việc phân tích tĩnh, gợi ý trong IDE, và các thư viện chủ động đọc type hint như pydantic hay FastAPI.

#### 10.4 Đọc thêm

- PEP 8, PEP 484, PEP 621. *Logging HOWTO* (docs.python.org). *The Twelve-Factor App* (12factor.net).

---

## Bài tập hôm nay → `project/multi_caller/prepare.py`

Dữ liệu đầu vào: `project/data/prompts_raw.csv`, gồm 12 dòng có đủ kiểu "bẩn": dư khoảng trắng, prompt rỗng, trùng id, trùng nội dung khác hoa thường, thiếu category.

Viết các hàm sau:
1. **`load_prompts(path)`** — đọc CSV, giữ các cột `id`, `prompt`, `category`.
2. **`clean(df, max_chars=2000)`** — **không được sửa `df` đầu vào**. Làm **đúng theo thứ tự**:
   1. Strip khoảng trắng ở `prompt`, rồi bỏ các prompt rỗng hoặc bị thiếu.
   2. Chuẩn hóa `category`: strip, chuyển chữ thường; nếu thiếu thì gán `"general"`.
   3. Bỏ các dòng trùng `id`, giữ dòng đầu tiên.
   4. Bỏ các dòng trùng prompt **sau khi chuẩn hóa** (casefold, gộp nhiều khoảng trắng thành một), giữ dòng đầu tiên.
   5. Thêm cột `n_chars`, rồi bỏ các prompt dài hơn `max_chars`.
   6. Reset index.
3. **`summarize(df)`** — trả về dict `{"total", "by_category" (các key sắp xếp theo thứ tự chữ cái), "mean_chars" (float)}`.
4. **`write_jsonl(df, path)`** — mỗi dòng là `{"id", "prompt", "category"}`, mã hóa UTF-8, **không escape tiếng Việt** (`ensure_ascii=False`).
5. **CLI** — `python -m multi_caller.prepare data/prompts_raw.csv data/prompts.jsonl [--max-chars N]`, dùng `logging` để in bản tóm tắt.

**Tự kiểm tra:** đứng ở thư mục `project/`, chạy `python -m pytest -q tests/test_prepare.py`.

👉 Sau khi đọc xong, làm [ngay-2.quiz.md](ngay-2.quiz.md).
