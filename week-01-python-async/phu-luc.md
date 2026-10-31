# Phụ lục Tuần 1 · Nền Python còn thiếu: cú pháp, collections, debug, profiling, NumPy/pandas, công cụ

> **Loại tài liệu: Kỹ thuật.** Trọng tâm là thuật ngữ chuẩn, cơ chế, code và quyết định thiết kế. Toán chỉ dùng để ước lượng nhanh.
> **Vì sao có phụ lục này:** khi đối chiếu Tuần 1 với Phase 1 của roadmap ([README.md](../README.md)), các mục sau còn thiếu hoặc mới chỉ được dùng ngầm: slicing và method của chuỗi, `collections`, `functools.reduce`, đọc traceback, `pdb`, `timeit`/`cProfile`, `np.stack`/`np.where`, `concat`/`melt`/`to_datetime`, `tqdm`, config YAML/Hydra, `aiohttp`, Jupyter/Colab.
> **Gồm 6 bài:**
> - **P1** Cú pháp nền: chuỗi, slicing, comprehension, vòng lặp
> - **P2** `collections`, `functools`, `itertools` và độ phức tạp
> - **P3** Debug: đọc traceback, `pdb`
> - **P4** Profiling: `timeit`, `cProfile`, `tracemalloc`
> - **P5** NumPy và pandas bổ sung: ghép mảng, `np.where`, `concat`, `melt`/`pivot`, thời gian
> - **P6** Công cụ thực hành: `tqdm`, config YAML/Hydra, `aiohttp`, Jupyter/Colab
>
> **Thời gian:** đọc ~2h15 · quiz 30' (cuối file). Không có bài tập project riêng; mỗi bài có một đoạn "Tự làm" 10–15'.
> **Môi trường kiểm chứng:** mọi output trong file đã chạy trên Python 3.14.5, NumPy 2.5, pandas 3.0, PyYAML 6.0, tqdm 4.70, aiohttp 3.14 (máy bạn, đang cắm sạc). Hydra/OmegaConf và Jupyter **chưa cài** trên máy bạn, nên các đoạn về chúng chỉ mô tả, không chạy.

---

## P1 · Cú pháp nền: chuỗi, slicing, comprehension, vòng lặp

> ⏱ Tầng 1: 5' · Tầng 2: 20'
> **Cần biết trước:** object và tên (Tuần 1, bài 01).

> **📖 Từ mới**
> - **Slice:** lấy một đoạn liên tiếp của chuỗi/list bằng cú pháp `s[start:stop:step]`.
> - **Nửa mở (half-open):** đoạn `[start, stop)` lấy `start` nhưng **không** lấy `stop`.
> - **Comprehension:** biểu thức tạo list/dict/set trong một dòng: `[f(x) for x in xs if cond]`.
> - **Unicode normalization:** đưa các cách viết khác nhau của cùng một chữ (ví dụ "ệ" viết liền hoặc "e" + 2 dấu ghép) về một dạng duy nhất.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Slice giống như **cắt bánh mì bằng hai nhát dao** đặt ở các *khe giữa* lát bánh, không phải ở lát bánh:

```
 c   h   a   t   b   o   t
0   1   2   3   4   5   6   7      ← vị trí các khe
-7  -6  -5  -4  -3  -2  -1          ← đánh số từ cuối
```

`"chatbot"[1:4]` là cắt ở khe 1 và khe 4, lấy 3 lát giữa: `"hat"`. Vì là khe nên `len(s[a:b]) = b - a`, và `s[:k] + s[k:] == s` với mọi `k`.

**Ba ý cốt lõi:**
1. Slice là **nửa mở** và **không báo lỗi** khi vượt biên (`"chatbot"[10:]` cho `""`), nhưng index đơn thì có (`"chatbot"[10]` → `IndexError`).
2. Chuỗi là **immutable**: mọi method (`strip`, `replace`, `lower`) trả về chuỗi **mới**.
3. Văn bản tiếng Việt có thể trông giống nhau mà **khác byte**: luôn chuẩn hóa NFC trước khi so sánh, đếm, hay khử trùng lặp.

### TẦNG 2 · Kiến thức chuẩn

#### P1.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Sequence slicing** | `a[i:j:k]` chọn các phần tử có chỉ số `i, i+k, …` trước `j`; chỉ số âm tính từ cuối; `k < 0` đi ngược | Python Library Reference, *Common Sequence Operations* |
| **Slice assignment** | `L[i:j] = iterable` thay đoạn đó bằng các phần tử mới; độ dài có thể khác | Language Reference §7.2 |
| **f-string, format spec** | `f"{x:spec}"` theo *Format Specification Mini-Language*; `f"{x=}"` in cả tên biến (từ 3.8) | PEP 498 (2015); Library Reference, `string` |
| **Comprehension** | List/set/dict comprehension và generator expression; có scope riêng, biến lặp không rò ra ngoài | PEP 202 (2000), PEP 274 (2001), PEP 289 (2002) |
| **Extended unpacking** | `first, *rest = xs` | PEP 3132 (2007) |
| **Unicode normalization (NFC/NFD)** | NFC: ghép thành ký tự dựng sẵn; NFD: tách thành chữ gốc + dấu ghép | Unicode Standard Annex #15 |
| **Sort stability** | `sorted`/`list.sort` giữ nguyên thứ tự tương đối của các phần tử có key bằng nhau (Timsort) | Python *Sorting HOWTO* |

#### P1.2 Slicing và method của chuỗi (đã chạy)

```python
w = "chatbot"
w[1:4], w[-3:], w[:-1]     # 'hat'  'bot'  'chatbo'
w[::-1], w[::2]            # 'tobtahc'  'cabt'   ← đảo ngược; lấy cách 1
L = [0, 1, 2, 3, 4, 5]
L[::-2], L[4:1:-1]         # [5, 3, 1]  [4, 3, 2]  ← step âm: start phải LỚN hơn stop
L[1:3] = ["a"]; L          # [0, 'a', 3, 4, 5]    ← thay 2 phần tử bằng 1

"a  b ".split()            # ['a', 'b']            ← không đối số: tách theo MỌI khoảng trắng, bỏ chuỗi rỗng
"a  b ".split(" ")         # ['a', '', 'b', '']    ← có đối số: tách đúng từng dấu cách
"|".join(["a", "b", "c"])  # 'a|b|c'
"abc".find("z")            # -1                    ← còn "abc".index("z") raise ValueError
```

**Format spec hay dùng khi log kết quả thí nghiệm:**

| Biểu thức | Kết quả |
|---|---|
| `f"{3.14159:.2f}"` | `3.14` |
| `f"{1234567:,}"` / `f"{1234567:_}"` | `1,234,567` / `1_234_567` |
| `f"{'ok':>6}|"` / `f"{'ok':<6}|"` | `    ok|` / `ok    |` |
| `f"{0.0731:.1%}"` | `7.3%` |
| `f"{x=}"` / `f"{x=:.1f}"` (với `x = 3.14159`) | `x=3.14159` / `x=3.1` |

#### P1.3 Chuỗi tiếng Việt: một chữ, hai cách mã hóa

"ệ" có thể là **một** code point (U+1EC7, dạng NFC), hoặc **ba** code point: `e` (U+0065) + dấu nặng (U+0323) + dấu mũ (U+0302) (dạng NFD). Văn bản copy từ Word, macOS, hay web thường lẫn cả hai.

```python
import unicodedata
s = "Việt Nam"
nfc = unicodedata.normalize("NFC", s); nfd = unicodedata.normalize("NFD", s)
len(nfc), len(nfd)                        # 8 10
len(nfc.encode()), len(nfd.encode())      # 10 12   ← số byte UTF-8
nfc == nfd                                # False   ← nhìn giống hệt nhau, nhưng so sánh ra False
```

Hệ quả thực tế: khử trùng lặp prompt, đếm token, tra từ điển, hay so khớp keyword trong RAG đều **sai lặng lẽ** nếu không chuẩn hóa. Quy ước: chuẩn hóa NFC ngay ở bước đọc dữ liệu (đúng như hàm `normalize` ở P4).

**Một lỗi gặp thật trên máy bạn.** Khi stdout bị chuyển hướng (pipe, ghi ra file, chạy trong một số terminal), Python trên Windows dùng mã `cp1252` và lệnh `print("là số nguyên tố")` raise `UnicodeEncodeError: 'charmap' codec can't encode character 'ố'`. Cách sửa: chạy `python -X utf8 script.py` hoặc đặt biến môi trường `PYTHONUTF8=1`; và **luôn** mở file bằng `open(path, encoding="utf-8")`.

#### P1.4 Comprehension và vòng lặp

```python
[(i, j) for i in range(3) for j in range(3) if i < j]   # [(0, 1), (0, 2), (1, 2)]
[x if x % 2 else 0 for x in range(6)]                   # [0, 1, 0, 3, 0, 5]  ← if-else ĐỨNG TRƯỚC for: biến đổi
[x for x in range(6) if x % 2]                          # [1, 3, 5]           ← if ĐỨNG SAU for: lọc
{w: len(w) for w in ["ai", "llm", "rag"]}               # {'ai': 2, 'llm': 3, 'rag': 3}
{len(w) for w in ["ai", "llm", "rag"]}                  # {2, 3}
```

**Đọc comprehension lồng nhau:** các `for` đọc **từ trái sang phải giống hệt vòng lặp lồng nhau** viết ra nhiều dòng. Dòng đầu ở trên tương đương:

```python
out = []
for i in range(3):
    for j in range(3):
        if i < j:
            out.append((i, j))
```

**`for … else`:** khối `else` chạy khi vòng lặp **kết thúc bình thường**, tức là **không gặp `break`**. Dùng cho bài toán "tìm; nếu không thấy thì…":

```python
for n in [7, 8]:
    for d in range(2, n):
        if n % d == 0:
            print(n, "chia hết cho", d); break
    else:
        print(n, "là số nguyên tố")
# 7 là số nguyên tố
# 8 chia hết cho 2
```

**Unpacking, sắp xếp, `map`/`filter`/`zip`:**

```python
first, *rest = [1, 2, 3, 4]; *init, last = [1, 2, 3, 4]   # 1 [2, 3, 4]   [1, 2, 3] 4
words = ["rag", "ai", "llm", "mcp", "agent"]
sorted(words, key=len)                       # ['ai', 'rag', 'llm', 'mcp', 'agent']  ← ổn định: rag, llm, mcp giữ thứ tự gốc
sorted(words, key=lambda w: (-len(w), w))    # ['agent', 'llm', 'mcp', 'rag', 'ai']  ← dài trước, cùng độ dài thì theo ABC
list(filter(None, [0, 1, "", "x", None]))    # [1, 'x']  ← filter(None, …) giữ các giá trị "truthy"
list(zip([1, 2, 3], "ab"))                   # [(1, 'a'), (2, 'b')]  ← lặng lẽ cắt theo cái ngắn nhất
list(zip([1, 2, 3], "ab", strict=True))      # ValueError: zip() argument 2 is shorter than argument 1
```

**Số học và giá trị chân lý hay bị hỏi:**

| Biểu thức | Kết quả | Vì sao |
|---|---|---|
| `0.1 + 0.2 == 0.3` | `False` | Số thực nhị phân; so bằng `math.isclose` |
| `round(2.5), round(3.5)` | `2 4` | *Banker's rounding*: làm tròn về số chẵn |
| `7 // -2`, `-7 % 3` | `-4`, `2` | `//` làm tròn **xuống** (về $-\infty$); dấu của `%` theo số chia |
| `10 / 4` | `2.5` | `/` luôn trả về float |
| `bool([]), bool("0"), bool(0.0)` | `False True False` | Chuỗi khác rỗng là truthy, kể cả `"0"` |
| `None or "default"`, `0 or [] or "x"` | `'default'`, `'x'` | `or` trả về **toán hạng** truthy đầu tiên, không phải `True` |

#### P1.5 Cạm bẫy

1. **`split(" ")` thay vì `split()`** khi đếm từ: chuỗi có hai dấu cách liên tiếp sinh ra token rỗng.
2. **Nối chuỗi trong vòng lặp bằng `+=`** cho dữ liệu lớn: dùng `"".join(parts)` (đo ở P2: nhanh hơn ~2.7 lần với 100 000 mảnh).
3. **`zip` không `strict=True`** khi ghép prompt với nhãn: thiếu một dòng là toàn bộ cặp phía sau bị lệch mà không báo lỗi.
4. **Không chuẩn hóa Unicode** trước khi so sánh văn bản tiếng Việt (P1.3).
5. **Dùng `x or default`** khi `0` hoặc `""` là giá trị hợp lệ: `temperature or 0.7` biến `temperature=0` thành `0.7`. Viết `default if x is None else x`.

**Sắc thái phỏng vấn:** *"Đảo ngược chuỗi tiếng Việt bằng `s[::-1]` có đúng không?"* Với NFC thì đúng; với NFD thì **dấu ghép bị tách khỏi chữ gốc** và dính sang chữ khác. Trả lời tốt: chuẩn hóa NFC trước, và nói rõ rằng đơn vị "chữ người đọc thấy" (grapheme cluster) không phải lúc nào cũng là một code point.

#### P1.6 Tự làm (10')

Viết `clean_prompt(s) -> str`: chuẩn hóa NFC, gộp mọi khoảng trắng liên tiếp thành một dấu cách, bỏ khoảng trắng hai đầu, cắt tối đa 200 ký tự. Kiểm tra rằng phiên bản NFC và NFD của `"Việt  Nam "` cho cùng một kết quả.

#### P1.7 Đọc thêm

- Python Library Reference: *Text Sequence Type — str*, *Format Specification Mini-Language*.
- Python *Unicode HOWTO* (docs.python.org).
- Unicode Standard Annex #15, *Unicode Normalization Forms*.

---

## P2 · `collections`, `functools`, `itertools` và độ phức tạp

> ⏱ Tầng 1: 5' · Tầng 2: 20'
> **Cần biết trước:** dict, set, list (P1); closure (Tuần 1, bài 02); generator (Tuần 1, bài 04).

> **📖 Từ mới**
> - **Độ phức tạp (Big-O):** thời gian tăng theo kích thước dữ liệu $n$ như thế nào. $O(1)$: không đổi; $O(n)$: tỉ lệ thuận; $O(n^2)$: gấp đôi dữ liệu thì chậm gấp bốn.
> - **Hash table:** cấu trúc mà `dict` và `set` dùng: tính `hash(key)` để nhảy thẳng tới vị trí lưu, nên tra cứu trung bình $O(1)$.
> - **Memoization:** nhớ kết quả của hàm cho từng đối số để không tính lại.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- **`list`** giống một **dãy ghế xếp hàng**: tìm một người phải đi dọc từ đầu ($O(n)$); người đầu hàng rời đi thì **cả hàng phải dịch lên một ghế** ($O(n)$).
- **`set`/`dict`** giống **tủ đồ có số**: biết số tủ (hash) là mở thẳng ra ($O(1)$).
- **`deque`** giống **hàng đợi có cửa ở cả hai đầu**: vào/ra ở đầu nào cũng nhanh.

**Ví dụ có số (đo trên máy bạn, 100 000 phần tử):** rút hết một list bằng `pop(0)` mất **2 481 ms**; rút hết một `deque` bằng `popleft()` mất **4.7 ms**. Chênh **~520 lần**, chỉ vì chọn sai cấu trúc dữ liệu.

**Ba ý cốt lõi:**
1. Chọn cấu trúc dữ liệu theo **phép toán bạn làm nhiều nhất**: tra cứu → `set`/`dict`; hàng đợi → `deque`; đếm → `Counter`; gom nhóm → `defaultdict(list)`.
2. `functools` cho các công cụ làm việc **với hàm**: `reduce`, `partial`, `lru_cache`.
3. `itertools` cho các công cụ làm việc **với iterator**, lười (lazy) như generator.

### TẦNG 2 · Kiến thức chuẩn

#### P2.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **`Counter`** | `dict` con đếm số lần xuất hiện; key thiếu trả về 0; có `most_common(k)`, phép `+`, `-` (bỏ giá trị ≤ 0) | Library Reference, `collections` |
| **`defaultdict(factory)`** | `dict` con: truy cập key thiếu thì **tạo** `factory()` và **chèn** vào dict | như trên |
| **`deque`** | Hàng đợi hai đầu: `append`/`appendleft`/`pop`/`popleft` đều $O(1)$; `maxlen` tạo cửa sổ trượt | như trên |
| **`namedtuple`** | Tuple có tên trường, immutable, nhẹ hơn class | như trên |
| **`functools.reduce(f, xs, init)`** | Gấp (fold) dãy về một giá trị: `f(f(f(init, x0), x1), x2)` | Library Reference, `functools` |
| **`functools.partial`** | Tạo hàm mới với một số đối số đã cố định | như trên |
| **`functools.lru_cache` / `cache`** | Memoization; `cache` là `lru_cache(maxsize=None)` | như trên |
| **`itertools`** | `chain`, `islice`, `groupby`, `accumulate`, `product`, `combinations`, `batched` (3.12) | Library Reference, `itertools` |
| **Time complexity của kiểu có sẵn** | Bảng chi phí trung bình/xấu nhất của list, dict, set, deque | Python Wiki, *TimeComplexity* |

#### P2.2 Bảng độ phức tạp cần thuộc

| Phép toán | `list` | `deque` | `set` / `dict` |
|---|---|---|---|
| Thêm/bớt ở **cuối** | $O(1)$ (trung bình) | $O(1)$ | — |
| Thêm/bớt ở **đầu** (`insert(0, x)`, `pop(0)`) | $O(n)$ | $O(1)$ | — |
| Truy cập theo index `x[i]` | $O(1)$ | $O(n)$ ở giữa | — |
| `x in c` | $O(n)$ | $O(n)$ | $O(1)$ trung bình |
| Sắp xếp | $O(n \log n)$ | — | — |

**Đo trên máy bạn** (min của 5 lần chạy):

| Thí nghiệm ($n = 100\,000$) | Thời gian | Ghi chú |
|---|---|---|
| Rút hết list bằng `pop(0)` | 2 481 ms | $O(n^2)$ tổng cộng: mỗi lần dịch trung bình $n/2$ con trỏ |
| Rút hết `deque` bằng `popleft()` | 4.7 ms | $O(n)$ tổng cộng |
| Rút hết list bằng `pop()` (từ cuối) | 2.8 ms | $O(n)$ tổng cộng |
| `(n-1) in list` (phần tử cuối) | 815 µs | $O(n)$ |
| `(n-1) in set` | 0.045 µs | $O(1)$; nhanh hơn ~18 000 lần |
| Nối chuỗi bằng `s += "ab"` | 8.2 ms | |
| `"".join([...])` | 3.0 ms | |

**Ước lượng nhanh trước khi đo.** Rút hết list bằng `pop(0)`: tổng số con trỏ phải dịch là $\sum_{k=1}^{n} k \approx n^2/2 = 5 \times 10^9$, mỗi con trỏ 8 byte, tức ~40 GB dữ liệu bị chép trong bộ nhớ. Với tốc độ chép bộ nhớ cỡ 10–20 GB/s, ta dự đoán ~2–4 s. Đo được 2.5 s: khớp bậc độ lớn.

#### P2.3 Code chuẩn (đã chạy)

```python
from collections import Counter, defaultdict, deque, namedtuple
words = "rag llm rag agent llm rag".split()

Counter(words)                         # Counter({'rag': 3, 'llm': 2, 'agent': 1})
Counter("banana").most_common(2)       # [('a', 3), ('n', 2)]
Counter(a=3, b=1) - Counter(a=1, b=2)  # Counter({'a': 2})   ← b = -1 bị bỏ

by_intent = defaultdict(list)          # gom nhóm không cần kiểm tra key
for intent, q in [("faq", "a"), ("bill", "b"), ("faq", "c")]:
    by_intent[intent].append(q)
dict(by_intent)                        # {'faq': ['a', 'c'], 'bill': ['b']}

window = deque(maxlen=3)               # trung bình trượt của 3 latency gần nhất
for x in range(5): window.append(x)
window, sum(window) / len(window)      # deque([2, 3, 4], maxlen=3)  3.0

P = namedtuple("P", "x y"); p = P(1, 2)
p.x, p._asdict()                       # 1  {'x': 1, 'y': 2}
```

```python
from functools import reduce, partial, lru_cache
reduce(lambda acc, x: acc * x, [1, 2, 3, 4], 1)   # 24
reduce(lambda a, b: a + b, [])                    # TypeError: reduce() of empty iterable with no initial value
int2 = partial(int, base=2); int2("1010")         # 10

@lru_cache(maxsize=None)
def fib(n): return n if n < 2 else fib(n - 1) + fib(n - 2)
```

`fib(30)` không cache: **2 692 537 lần gọi**, 153 ms. Có cache: 31 lần tính (`misses=31`), 0.14 ms.

**Cách đọc `reduce` từng bước** với `reduce(lambda acc, x: acc * x, [1, 2, 3, 4], 1)`:

| Bước | `acc` trước | `x` | `acc` sau |
|---|---|---|---|
| 1 | 1 (init) | 1 | 1 |
| 2 | 1 | 2 | 2 |
| 3 | 2 | 3 | 6 |
| 4 | 6 | 4 | 24 |

Trong code Python hiện đại, `sum`, `math.prod`, `max`, `"".join` hoặc một vòng `for` thường dễ đọc hơn `reduce`. Hãy dùng `reduce` khi phép gộp thật sự là một hàm hai ngôi tùy ý, ví dụ gộp nhiều dict config theo thứ tự ưu tiên: `reduce(lambda a, b: {**a, **b}, [defaults, file_cfg, cli_cfg])`.

```python
import itertools as it
data = [("faq", 1), ("bill", 2), ("faq", 3)]
[(k, [v for _, v in g]) for k, g in it.groupby(data, key=lambda r: r[0])]
# [('faq', [1]), ('bill', [2]), ('faq', [3])]   ← chưa sắp xếp: 'faq' bị tách làm hai nhóm!
[(k, [v for _, v in g]) for k, g in it.groupby(sorted(data), key=lambda r: r[0])]
# [('bill', [2]), ('faq', [1, 3])]
list(it.accumulate([1, 2, 3, 4]))       # [1, 3, 6, 10]        ← tổng cộng dồn
list(it.product("ab", [0, 1]))          # [('a', 0), ('a', 1), ('b', 0), ('b', 1)]  ← lưới hyperparameter
list(it.combinations([1, 2, 3], 2))     # [(1, 2), (1, 3), (2, 3)]
list(it.batched(range(7), 3))           # [(0, 1, 2), (3, 4, 5), (6,)]  ← chia batch gửi API
```

#### P2.4 Cạm bẫy

1. **`itertools.groupby` trên dữ liệu chưa sắp xếp:** nó chỉ gom các phần tử **liền kề** có cùng key. Muốn gom toàn cục thì sắp xếp trước, hoặc dùng `defaultdict(list)`.
2. **Đọc `defaultdict` cũng là ghi:** chỉ cần `g["missing"]` là key đó được chèn vào (`len(g)` tăng). Muốn kiểm tra thì dùng `"missing" in g` hoặc `g.get("missing")`.
3. **`lru_cache` trên method hoặc hàm nhận đối số mutable:** đối số phải hashable (list sẽ raise `TypeError`); cache giữ tham chiếu tới `self` nên object không bao giờ được giải phóng.
4. **Dùng `list` làm hàng đợi** (`pop(0)`) trong BFS hay trong một work queue: đổi sang `deque`.
5. **Kiểm tra trùng bằng `x in list`** trong vòng lặp: thành $O(n^2)$ (ví dụ thật ở P4).

**Sắc thái phỏng vấn:** *"`dict` có giữ thứ tự chèn không?"* Có, **đảm bảo từ Python 3.7** theo đặc tả ngôn ngữ (3.6 chỉ là chi tiết cài đặt của CPython). `OrderedDict` vẫn còn ích cho `move_to_end` và so sánh có xét thứ tự, ví dụ khi tự cài LRU cache.

#### P2.5 Tự làm (15')

Cho list các log `(intent, latency_s)`. Viết `summarize(logs)` trả về dict `intent → (count, p50_latency)` dùng `defaultdict(list)`; viết thêm `top_intents(logs, k)` bằng `Counter.most_common`. Viết `moving_avg(xs, w)` bằng `deque(maxlen=w)`.

#### P2.6 Đọc thêm

- Python Library Reference: `collections`, `functools`, `itertools` (mục *Itertools Recipes*).
- Python Wiki, *TimeComplexity*.
- Raymond Hettinger, *Transforming Code into Beautiful, Idiomatic Python* (PyCon US 2013).

---

## P3 · Debug: đọc traceback, `pdb`

> ⏱ Tầng 1: 5' · Tầng 2: 15'
> **Cần biết trước:** exception, `raise … from` (Tuần 1, bài 09); logging (Tuần 1, bài 10).

> **📖 Từ mới**
> - **Traceback:** bản in ra khi có exception không được bắt, liệt kê **chuỗi lời gọi hàm** dẫn tới lỗi.
> - **Frame (khung):** một lời gọi hàm đang chạy, có biến cục bộ riêng. Traceback là danh sách các frame.
> - **Debugger:** công cụ cho phép **dừng** chương trình giữa chừng, xem biến, chạy từng dòng.
> - **Breakpoint:** điểm dừng đặt trước trong code.
> - **Post-mortem:** mở debugger **sau khi** chương trình đã crash, ngay tại frame gây lỗi.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Traceback giống **hóa đơn chuyển khoản qua nhiều trung gian**: dòng trên cùng là nơi tiền xuất phát (`<module>`), mỗi dòng tiếp theo là một trung gian, **dòng cuối là nơi xảy ra sự cố**. Đọc **từ dưới lên**: dòng cuối nói *chuyện gì*, các dòng ngay trên nói *ở đâu*, và đi lên dần để tìm *vì sao* (giá trị sai được tạo ra từ chỗ nào).

**Ba ý cốt lõi:**
1. Đọc traceback **từ dưới lên**: tên exception + thông điệp → dòng code gây lỗi → các hàm gọi nó.
2. Nơi crash **thường không phải** nơi có bug: giá trị sai được tạo ra ở chỗ khác rồi "trôi" tới đây.
3. Dùng `breakpoint()` để **dừng và xem biến**, thay vì rải `print` khắp nơi.

### TẦNG 2 · Kiến thức chuẩn

#### P3.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Traceback** | Ngăn xếp các frame từ điểm vào tới điểm raise; "most recent call last" | Library Reference, `traceback` |
| **Exception chaining** | `raise B from A` đặt `B.__cause__ = A` ("direct cause"); raise trong khối `except` mà không có `from` thì đặt `__context__` ("During handling of the above exception…") | PEP 3134 (2005) |
| **Fine-grained error locations** | Dấu `~~~^^^` chỉ đúng biểu thức con gây lỗi (từ 3.11) | PEP 657 (2021) |
| **`pdb`** | Debugger có sẵn của Python: breakpoint, step, xem biến, đi lên/xuống frame | Library Reference, `pdb` |
| **`breakpoint()`** | Hàm có sẵn, gọi debugger do `PYTHONBREAKPOINT` chỉ định (mặc định `pdb.set_trace`); `PYTHONBREAKPOINT=0` vô hiệu hóa | PEP 553 (2017) |
| **Post-mortem debugging** | `python -m pdb script.py` (tự vào debugger khi crash), `pdb.pm()` sau exception, `%debug` trong Jupyter | Library Reference, `pdb` |

#### P3.2 Đọc một traceback thật, từng bước

Đoạn code tính chi phí gọi API từ log JSON (giá giả định):

```python
import json
PRICES = {"MODEL_ID_A": 0.5, "MODEL_ID_B": 2.0}     # $ / 1M token

def parse(line):
    rec = json.loads(line)
    return rec["model"], rec["tokens"]

def cost(model, tokens):
    return PRICES[model] * tokens / 1_000_000           # dòng 10

def total(lines):
    s = 0.0
    for i, line in enumerate(lines, 1):
        try:
            model, tokens = parse(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"dòng {i} không phải JSON") from e
        s += cost(model, tokens)                         # dòng 19
    return s

total(['{"model": "MODEL_ID_A", "tokens": 1200}',
       '{"model": "MODEL_ID_B", "tokens": "800"}'])      # dòng 25 (qua print)
```

Output thật (rút gọn đường dẫn):

```
Traceback (most recent call last):
  File "prices.py", line 25, in <module>
    print(total(lines))
          ~~~~~^^^^^^^
  File "prices.py", line 19, in total
    s += cost(model, tokens)
         ~~~~^^^^^^^^^^^^^^^
  File "prices.py", line 10, in cost
    return PRICES[model] * tokens / 1_000_000
           ~~~~~~~~~~~~~~^~~~~~~~
TypeError: can't multiply sequence by non-int of type 'float'
```

**Đọc từ dưới lên:**

| Bước | Dòng đọc | Rút ra |
|---|---|---|
| 1 | `TypeError: can't multiply sequence by non-int of type 'float'` | Có phép nhân giữa một **sequence** (chuỗi/list) và một **float** |
| 2 | Dấu `^` dưới `PRICES[model] * tokens` ở dòng 10 | Vế trái `PRICES[model]` là float, nên **`tokens` là sequence**, tức là một chuỗi |
| 3 | Frame `total`, dòng 19 | `tokens` được truyền từ `total`, mà `total` lấy từ `parse` |
| 4 | Không có frame `parse` trong traceback | `parse` đã chạy **thành công**, trả về `"800"` dạng chuỗi. Bug nằm ở **dữ liệu đầu vào** hoặc ở chỗ `parse` không kiểm tra kiểu |

Sửa đúng chỗ: kiểm tra và ép kiểu ngay tại ranh giới dữ liệu (`int(rec["tokens"])`, hoặc dùng schema như pydantic), không phải sửa ở `cost`.

**Exception chaining.** Với dòng log `'{"model": MODEL_ID_B}'` (thiếu ngoặc kép), traceback có **hai khối**:

```
json.decoder.JSONDecodeError: Expecting value: line 1 column 11 (char 10)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  ...
  File "prices.py", line 18, in total
    raise ValueError(f"dòng {i} không phải JSON") from e
ValueError: dòng 2 không phải JSON
```

- "**direct cause**": do `raise … from e`, cố ý gắn nguyên nhân. Đây là cách nên làm: lỗi mới có ngữ cảnh nghiệp vụ ("dòng 2"), lỗi gốc vẫn còn để tra.
- "**During handling of the above exception, another exception occurred**": một lỗi **khác** xảy ra trong khối `except`, thường là bug trong chính code xử lý lỗi.
- `raise … from None`: cố ý **giấu** nguyên nhân (thư viện `json` làm vậy ở dòng cuối của nó).

#### P3.3 `pdb`: các lệnh cần thuộc

| Lệnh | Ý nghĩa |
|---|---|
| `b file.py:10` / `b func` | Đặt breakpoint tại dòng / hàm (`b file.py:10, tokens == "800"` là breakpoint có điều kiện) |
| `c` | Chạy tiếp tới breakpoint kế |
| `n` | Chạy dòng hiện tại, **không** đi vào hàm con (next) |
| `s` | Chạy dòng hiện tại, **đi vào** hàm con (step) |
| `r` | Chạy tới khi hàm hiện tại return |
| `p expr` / `pp expr` | In giá trị / in đẹp |
| `l` / `ll` | Xem code quanh dòng hiện tại / cả hàm |
| `w` | Xem ngăn xếp frame (where) |
| `u` / `d` | Đi lên frame gọi / xuống frame được gọi |
| `interact` | Mở Python shell với biến của frame hiện tại |
| `q` | Thoát |

**Phiên debug thật** cho bug ở P3.2 (`python -m pdb prices.py`, các dòng sau `(Pdb)` là lệnh gõ vào):

```
(Pdb) b prices.py:10
Breakpoint 1 at ...prices.py:10
(Pdb) c
> ...prices.py(10)cost()
-> return PRICES[model] * tokens / 1_000_000
(Pdb) p model, tokens, type(tokens)
('MODEL_ID_A', 1200, <class 'int'>)
(Pdb) c
> ...prices.py(10)cost()
-> return PRICES[model] * tokens / 1_000_000
(Pdb) p model, tokens, type(tokens)
('MODEL_ID_B', '800', <class 'str'>)          ← thấy ngay: tokens là chuỗi
(Pdb) w
  ...prices.py(25)<module>()
-> print(total(lines))
  ...prices.py(19)total()
-> s += cost(model, tokens)
> ...prices.py(10)cost()                      ← dấu > là frame hiện tại
```

**Ba cách vào debugger:**
1. Viết `breakpoint()` vào code ở chỗ nghi ngờ. Nhớ **xóa trước khi commit** (ruff có luật T100 bắt lỗi này).
2. `python -m pdb script.py`: chạy dưới debugger; nếu crash thì tự vào **post-mortem** tại frame lỗi.
3. Trong pytest: `python -m pytest --pdb` vào debugger tại test fail; `python -m pytest -x --pdb` dừng ở lỗi đầu tiên.

#### P3.4 Quy trình debug có hệ thống

1. **Tái hiện** được lỗi bằng một lệnh, tốt nhất là một test fail.
2. **Thu nhỏ** đầu vào tới mức nhỏ nhất vẫn còn lỗi (ở trên: 2 dòng log).
3. **Đặt giả thuyết** từ traceback ("`tokens` là chuỗi"), rồi **kiểm chứng** bằng `p` trong pdb hoặc một `assert`.
4. **Sửa ở gốc** (ranh giới dữ liệu), không sửa ở triệu chứng.
5. **Giữ lại test** tái hiện lỗi để lỗi không quay lại (*regression test*).

Khi không biết lỗi nằm ở đâu trong một chuỗi dài các bước (hoặc các commit), dùng **chia đôi** (bisect): kiểm tra ở giữa, loại một nửa; $n$ bước chỉ cần khoảng $\log_2 n$ lần kiểm tra. `git bisect` làm đúng điều này trên lịch sử commit.

#### P3.5 Cạm bẫy

1. **`except Exception: pass`**: nuốt lỗi, khiến bug xuất hiện ở chỗ khác mà không có traceback. Ít nhất phải `log.exception("…")`, vì hàm này ghi cả traceback.
2. **`raise ValueError(...)` trong `except` mà không `from e`**: vẫn có ngữ cảnh, nhưng thông điệp "During handling…" khiến người đọc tưởng có bug thứ hai.
3. **Dùng `assert` để kiểm tra dữ liệu người dùng:** `python -O` bỏ hết `assert`. Dùng `if … : raise ValueError`.
4. **Quên `breakpoint()` trong code chạy trên server:** tiến trình treo chờ nhập. Đặt `PYTHONBREAKPOINT=0` trong môi trường production.
5. **Debug code async bằng `pdb`:** dừng ở breakpoint là dừng **cả event loop**, nên mọi task khác (và timeout) đều bị ảnh hưởng. Với lỗi đồng thời, ưu tiên log có `task name` và `asyncio` debug mode (`PYTHONASYNCIODEBUG=1`).

**Sắc thái phỏng vấn:** *"Debug một lỗi chỉ xảy ra trên production, không tái hiện được ở máy local?"* Ý phải có: log có cấu trúc kèm request id; ghi đủ ngữ cảnh khi exception (`log.exception`); so sánh khác biệt môi trường (phiên bản thư viện, dữ liệu, cấu hình, tải đồng thời); thu nhỏ đầu vào lấy từ log thành một test tái hiện.

#### P3.6 Tự làm (10')

Chạy đoạn `prices.py` ở trên dưới `python -m pdb`, đặt breakpoint có điều kiện `b prices.py:10, isinstance(tokens, str)`, rồi dùng `u` để đi lên frame `total` và in `i`, `line`. Sau đó sửa `parse` để raise `ValueError` có số dòng khi `tokens` không phải số nguyên.

#### P3.7 Đọc thêm

- Python Library Reference: `pdb`, `traceback`.
- PEP 3134 (exception chaining), PEP 553 (`breakpoint()`), PEP 657 (vị trí lỗi chi tiết).
- Andreas Zeller, *Why Programs Fail* (2nd ed., 2009), chương 1 và 13.

---

## P4 · Profiling: `timeit`, `cProfile`, `tracemalloc`

> ⏱ Tầng 1: 5' · Tầng 2: 20'
> **Cần biết trước:** độ phức tạp (P2); vector hóa NumPy (Tuần 1, bài 05).

> **📖 Từ mới**
> - **Benchmark:** đo thời gian chạy của một đoạn code nhỏ, cô lập.
> - **Profiler:** công cụ đo **mỗi hàm** tốn bao nhiêu thời gian trong cả chương trình.
> - **Hotspot (nút cổ chai):** phần code chiếm phần lớn thời gian chạy.
> - **Wall time:** thời gian thực trôi qua theo đồng hồ, gồm cả lúc chờ I/O.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Bạn muốn về nhà nhanh hơn. Thay xe đạp bằng xe máy (tối ưu 5 phút đi đường) **chẳng giúp gì** nếu bạn mất 40 phút **chờ thang máy**. Phải **đo xem thời gian đi đâu** trước, rồi mới tối ưu đúng chỗ. Profiler chính là cái đồng hồ bấm giờ cho từng chặng.

**Ba ý cốt lõi:**
1. **Đo trước, tối ưu sau.** Trực giác về hotspot thường sai.
2. `timeit` để so **hai cách viết** của một đoạn nhỏ; `cProfile` để tìm **hàm nào** chiếm thời gian trong cả chương trình; `tracemalloc` để tìm **dòng nào** cấp phát bộ nhớ.
3. Tốc độ tăng tối đa bị giới hạn bởi **phần không tối ưu được** (định luật Amdahl).

### TẦNG 2 · Kiến thức chuẩn

#### P4.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **`timeit`** | Chạy một đoạn code `number` lần, lặp lại `repeat` lần; tắt GC trong lúc đo | Library Reference, `timeit` |
| **`time.perf_counter()`** | Đồng hồ có độ phân giải cao nhất, đơn điệu, dùng để đo khoảng thời gian | PEP 418 (2012) |
| **Deterministic profiler (`cProfile`)** | Ghi lại **mọi** lời gọi và return của hàm; có overhead với mỗi lời gọi | Library Reference, *The Python Profilers* |
| **`tottime` / `cumtime`** | Thời gian **trong chính** hàm (không tính hàm con) / thời gian **gồm cả** hàm con | như trên |
| **Sampling profiler** | Lấy mẫu ngăn xếp theo chu kỳ, overhead thấp, gắn được vào tiến trình đang chạy (ví dụ `py-spy`) | py-spy docs |
| **`tracemalloc`** | Theo dõi cấp phát bộ nhớ của Python theo từng dòng code | Library Reference, `tracemalloc` |
| **Amdahl's law** | Tăng tốc phần chiếm tỉ lệ $p$ lên $s$ lần thì tổng thể tăng tốc $\dfrac{1}{(1 - p) + p/s}$ | G. Amdahl, AFIPS 1967 |

#### P4.2 `timeit`: vì sao lấy `min` chứ không lấy trung bình

```python
import timeit, statistics
r = timeit.repeat("sum(range(10_000))", number=1000, repeat=7)
# ms cho 1000 lần, đo trên máy bạn: [178.26, 178.77, 148.48, 153.03, 119.69, 149.06, 184.62]
# min 119.69 · mean 158.84
```

Bảy lần chạy **cùng một code** chênh nhau tới 54%. Phần chênh đến từ các tiến trình khác, từ CPU đổi xung nhịp, từ cache. Những thứ này **chỉ có thể làm chậm đi, không làm nhanh lên**, nên `min` là ước lượng sát nhất cho "chi phí thật của code" (khuyến nghị trong tài liệu `timeit`). Khi muốn biết **trải nghiệm người dùng** (có cả nhiễu), hãy báo cáo phân phối (p50, p95), không chỉ `min`.

Trong Jupyter: `%timeit expr` (một dòng), `%%timeit` (cả cell), `%time` (chạy một lần, in wall time).

#### P4.3 `cProfile` trên một pipeline thật

Pipeline nhỏ: sinh 20 000 prompt tiếng Việt → chuẩn hóa (`normalize`) → khử trùng lặp (`dedupe`) → đếm token. Phiên bản đầu của `dedupe`:

```python
def dedupe(items):
    seen, out = [], []
    for x in items:
        if x not in seen:          # O(len(seen)) mỗi lần
            seen.append(x); out.append(x)
    return out
```

Chạy `python -m cProfile -s cumulative slowpipe.py` (hoặc gọi `cProfile.Profile()` trong code). Output thật, sắp theo `tottime`:

```
         1543934 function calls in 1.498 seconds
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    1.129    1.129    1.133    1.133 slowpipe.py:12(dedupe)
   149612    0.061    0.000    0.164    0.000 random.py:347(choice)
   169612    0.060    0.000    0.095    0.000 random.py:245(_randbelow_with_getrandbits)
    20000    0.035    0.000    0.035    0.000 {method 'sub' of 're.Pattern' objects}
```

**Đọc bảng:**
- `ncalls`: `dedupe` được gọi **1 lần** nhưng chiếm `tottime` 1.129 s, tức thời gian nằm **trong thân hàm** (các phép `x not in seen` không phải lời gọi hàm Python nên không hiện thành dòng riêng).
- `random.choice` được gọi gần 150 000 lần nhưng tổng chỉ 0.16 s. Tối ưu nó gần như vô ích.

**Sửa và đo lại** (bằng `timeit`, không dùng profiler):

```python
def dedupe_fast(items):
    return list(dict.fromkeys(items))     # dict giữ thứ tự chèn → khử trùng lặp O(n), giữ thứ tự
```

| | Thời gian (min của 5 lần) |
|---|---|
| `dedupe` (list) trên 20 000 prompt (19 108 prompt duy nhất) | 1 050 ms |
| `dedupe_fast` | 0.91 ms |
| Cả pipeline, trước khi sửa | 1 114 ms |
| Cả pipeline, sau khi sửa | 131 ms (**nhanh hơn 8.5×**) |

**Ước lượng bằng Amdahl, và một sắc thái quan trọng.** Theo profiler, `dedupe` chiếm $p = 1.129 / 1.498 \approx 0.75$. Nếu `dedupe` nhanh vô hạn ($s \to \infty$) thì tăng tốc tối đa là $1/(1 - 0.75) = 4\times$. Nhưng đo thật được 8.5×. Vì sao? **`cProfile` cộng overhead vào mỗi lời gọi hàm**: phần sinh dữ liệu có ~1.5 triệu lời gọi nhỏ nên bị phóng đại (pipeline đo trong profiler mất 1.50 s, ngoài profiler chỉ 1.11 s). Đo riêng không có profiler, `dedupe` chiếm ~95% thời gian, và giới hạn Amdahl tương ứng là cỡ 18×. Bài học: **dùng profiler để tìm hotspot, dùng `timeit` để đo con số tăng tốc**.

#### P4.4 Bộ nhớ: `tracemalloc`

```python
import tracemalloc
tracemalloc.start()
L = [float(i) for i in range(1_000_000)]
cur, peak = tracemalloc.get_traced_memory()      # cur ≈ 32.4 MB
top = tracemalloc.take_snapshot().statistics("lineno")[0]
# <string>:15: size=30.9 MiB, count=999911, average=32 B   ← dòng tạo list, ~1 triệu object
```

Khớp với Tuần 1, bài 01: mỗi `float` là một object 24 B, cộng con trỏ 8 B trong list, tổng ~32 B mỗi phần tử. Mảng NumPy `float64` cùng kích thước chỉ cần 8 MB.

#### P4.5 Quy trình tối ưu

1. Có **test đúng** trước khi tối ưu (tối ưu xong phải pass y nguyên).
2. Đo **wall time** của cả chương trình trên dữ liệu thật cỡ thật.
3. `cProfile` (hoặc `py-spy` cho tiến trình đang chạy) để tìm hotspot.
4. Sửa theo thứ tự ưu tiên: **thuật toán / cấu trúc dữ liệu** (ở trên: 1 000×) → **vector hóa** NumPy/pandas (Tuần 1, bài 05: ~50×) → **song song/đồng thời** (bài 07–08) → mới tới vi tối ưu cú pháp.
5. Đo lại bằng `timeit`, ghi số trước/sau vào commit hoặc REPORT.

**Đo trên máy bạn ($n$ = 100 000, tính bình phương):** vòng `for` + `append` 8.5 ms; list comprehension 3.9 ms; NumPy `a * a` 0.16 ms. Comprehension nhanh hơn ~2× so với vòng lặp; NumPy nhanh hơn ~50×.

#### P4.6 Cạm bẫy

1. **Đo khi máy chạy pin:** laptop của bạn hạ xung nhịp xuống 1.4 GHz, mọi số đo chậm 3–15 lần. Kiểm tra nguồn điện trước khi benchmark.
2. **Đo một lần duy nhất** bằng `time.time()`: độ phân giải thấp và nhiễu lớn. Dùng `perf_counter` hoặc `timeit.repeat`.
3. **Đo cả phần khởi tạo** (import, sinh dữ liệu) cùng phần muốn so: đưa phần chuẩn bị vào `setup` của `timeit`.
4. **Đo tốc độ code async có I/O** bằng `cProfile`: thời gian chờ mạng không thuộc về hàm nào. Đo wall time từng request, như ở bài 08.
5. **Tối ưu theo `percall` của các hàm gọi nhiều lần** mà không xem tổng `tottime`.

**Sắc thái phỏng vấn:** *"Pipeline tiền xử lý chạy 2 giờ, làm sao tăng tốc?"* Đừng trả lời ngay "dùng multiprocessing". Ý phải có: đo xem thời gian nằm ở đâu (I/O, CPU, hay chờ API); tìm thuật toán $O(n^2)$ ẩn (kiểm tra trùng, `pop(0)`, merge many-to-many); vector hóa; cache kết quả trung gian; rồi mới song song hóa, và ước lượng trần tăng tốc bằng Amdahl.

#### P4.7 Tự làm (15')

Viết hàm `count_vocab(prompts)` theo hai cách (vòng `for` với dict thường, và `Counter` với generator `w for p in prompts for w in p.split()`). Đo bằng `timeit.repeat` trên 20 000 prompt ở trên, báo cáo `min`. Chạy `cProfile` trên cả pipeline sau khi sửa `dedupe` và tìm hotspot mới.

#### P4.8 Đọc thêm

- Python Library Reference: `timeit`, *The Python Profilers*, `tracemalloc`.
- Micha Gorelick & Ian Ozsvald, *High Performance Python* (2nd ed., O'Reilly 2020), chương 2.
- G. Amdahl, *Validity of the single processor approach to achieving large scale computing capabilities*, AFIPS 1967.

---

## P5 · NumPy và pandas bổ sung: ghép mảng, `np.where`, `concat`, `melt`/`pivot`, thời gian

> ⏱ Tầng 1: 5' · Tầng 2: 20'
> **Cần biết trước:** shape, axis, broadcasting (Tuần 1, bài 05); DataFrame, `groupby`, `merge` (Tuần 1, bài 06).

> **📖 Từ mới**
> - **Axis:** chiều của mảng. Với mảng 2 chiều, axis 0 là hàng, axis 1 là cột.
> - **Wide / long format:** bảng "rộng" mỗi metric một cột; bảng "dài" mỗi hàng là một cặp (metric, giá trị).
> - **Timestamp / timezone:** một mốc thời gian, và múi giờ mà mốc đó được diễn giải theo.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- `np.concatenate` giống **nối thêm trang vào cuốn sổ đang có**: số chiều không đổi.
- `np.stack` giống **chồng nhiều cuốn sổ thành một chồng**: thêm một chiều mới ("cuốn thứ mấy").
- `melt` giống **xé một bảng điểm (mỗi môn một cột) thành danh sách các dòng "học sinh, môn, điểm"**; `pivot` ráp ngược lại.

**Ba ý cốt lõi:**
1. `stack` **thêm** một trục mới (mọi mảng phải cùng shape); `concatenate` **nối** theo trục có sẵn.
2. `np.where(cond, a, b)` là `if/else` theo từng phần tử, không cần vòng lặp.
3. Thời gian phải **parse đúng định dạng** và **gắn múi giờ** ngay khi đọc; Việt Nam dùng ngày/tháng, nhiều API trả về UTC.

### TẦNG 2 · Kiến thức chuẩn

#### P5.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **`np.concatenate(arrs, axis)`** | Nối dọc theo trục **có sẵn**; các trục khác phải bằng nhau | NumPy docs, *Joining arrays* |
| **`np.stack(arrs, axis)`** | Nối dọc theo một trục **mới**; mọi mảng phải cùng shape | như trên |
| **`vstack` / `hstack` / `column_stack`** | Tiện ích: nối theo hàng / theo cột; mảng 1 chiều được coi là một hàng (`vstack`) hoặc nối nối tiếp (`hstack`) | như trên |
| **`np.where`** | 1 đối số: trả về **chỉ số** chỗ đúng (như `np.nonzero`); 3 đối số: chọn từng phần tử | NumPy docs |
| **Tidy data** | Mỗi biến một cột, mỗi quan sát một hàng; dạng "long" | H. Wickham, *Tidy Data*, JSS 59 (2014) |
| **`melt` / `pivot` / `pivot_table`** | Rộng → dài / dài → rộng (key phải duy nhất) / dài → rộng có gộp (`aggfunc`) | pandas docs, *Reshaping and pivot tables* |
| **`pd.to_datetime`** | Parse chuỗi/số thành `datetime64`; `format`, `errors="coerce"`, `dayfirst`, `utc`, `unit` | pandas docs |
| **tz-aware vs tz-naive** | Có / không có múi giờ; `tz_localize` gắn múi giờ, `tz_convert` đổi múi giờ | pandas docs, *Time series* |

#### P5.2 Ghép mảng: tính shape bằng tay trước

Với `a = [1, 2, 3]`, `b = [4, 5, 6]` (shape `(3,)`):

| Lệnh | Shape | Kết quả |
|---|---|---|
| `np.stack([a, b])` | `(2, 3)` | `[[1, 2, 3], [4, 5, 6]]` |
| `np.stack([a, b], axis=1)` | `(3, 2)` | `[[1, 4], [2, 5], [3, 6]]` |
| `np.vstack([a, b])` | `(2, 3)` | như `stack` axis 0 |
| `np.hstack([a, b])` / `np.concatenate([a, b])` | `(6,)` | `[1, 2, 3, 4, 5, 6]` |
| `np.column_stack([a, b])` | `(3, 2)` | như `stack` axis 1 |

Với hai batch embedding `X`, `Y` cùng shape `(32, 768)`:

| Lệnh | Shape | Dùng khi |
|---|---|---|
| `np.concatenate([X, Y])` = `np.vstack` | `(64, 768)` | gộp hai batch thành một batch lớn hơn |
| `np.concatenate([X, Y], axis=1)` = `np.hstack` | `(32, 1536)` | nối đặc trưng của cùng một mẫu (ví dụ embedding câu hỏi ‖ embedding câu trả lời) |
| `np.stack([X, Y])` | `(2, 32, 768)` | giữ "nguồn" như một trục riêng |
| `np.stack([X, Y], axis=-1)` | `(32, 768, 2)` | |

**Quy tắc nhẩm:** `stack` của $k$ mảng shape $S$ cho shape $S$ chèn thêm $k$ ở vị trí `axis`; `concatenate` cộng kích thước ở trục `axis`. `np.stack` với hai mảng khác shape raise `ValueError: all input arrays must have the same shape`.

#### P5.3 `np.where` và chọn theo điều kiện

```python
s = np.array([0.9, 0.2, 0.55, 0.4])
np.where(s > 0.5)                 # (array([0, 2]),)          ← 1 đối số: tuple các mảng CHỈ SỐ
np.where(s > 0.5, 1, 0)           # [1 0 1 0]                 ← 3 đối số: chọn từng phần tử
np.where(s > 0.5, "bad", "ok")    # ['bad' 'ok' 'bad' 'ok']
np.select([s > 0.8, s > 0.5], ["high", "mid"], default="low")   # ['high' 'low' 'mid' 'low'] ← nhiều nhánh, điều kiện đầu tiên đúng thắng
M = np.array([[1, -2], [-3, 4]])
np.where(M < 0)                   # (array([0, 1]), array([1, 0]))  ← (hàng, cột) của các phần tử âm
np.argwhere(M < 0)                # [[0 1] [1 0]]                   ← cùng thông tin, mỗi hàng là một tọa độ
lat = np.array([1.2, np.nan, 3.0])
np.where(np.isnan(lat), np.nanmedian(lat), lat)   # [1.2 2.1 3. ]   ← điền giá trị thiếu
```

`np.where(cond, a, b)` **tính cả `a` và `b` trên toàn mảng** rồi mới chọn. Vì vậy `np.where(x > 0, np.log(x), 0)` vẫn sinh cảnh báo `log` của số âm. Cách an toàn: `np.log(x, out=np.zeros_like(x), where=x > 0)`.

#### P5.4 pandas: `concat`, `melt`/`pivot`

```python
d1 = pd.DataFrame({"id": [1, 2], "lat": [1.0, 2.0]})
d2 = pd.DataFrame({"id": [3], "lat": [5.0], "model": ["x"]})
pd.concat([d1, d2])                     # index 0, 1, 0 (bị trùng!); cột model của d1 thành NaN
pd.concat([d1, d2], ignore_index=True)  # index 0, 1, 2
pd.concat([d1, d2], keys=["thg9", "thg10"]).index.tolist()
# [('thg9', 0), ('thg9', 1), ('thg10', 0)]   ← MultiIndex ghi nguồn của từng hàng
```

`concat` **xếp chồng** (union các cột, thiếu thì NaN); `merge` **ghép theo key** (Tuần 1, bài 06). Gộp log của 30 file ngày: đọc từng file vào list rồi `pd.concat(list, ignore_index=True)` **một lần**. Đừng `concat` trong vòng lặp, vì mỗi lần lại chép toàn bộ bảng, tổng chi phí thành $O(n^2)$ (cùng lỗi với `pop(0)` ở P2).

```python
wide = pd.DataFrame({"model": ["A", "B"], "acc": [0.81, 0.78], "f1": [0.70, 0.74]})
long = wide.melt(id_vars="model", var_name="metric", value_name="value")
#   model metric  value
# 0     A    acc   0.81
# 1     B    acc   0.78
# 2     A     f1   0.70
# 3     B     f1   0.74
long.pivot(index="model", columns="metric", values="value")   # quay lại dạng rộng
```

Vì sao cần dạng dài: vẽ biểu đồ nhiều metric (seaborn/plotly nhận dạng dài), `groupby("metric")`, và lưu kết quả thí nghiệm khi số metric thay đổi giữa các lần chạy. Nếu dạng dài có **key trùng** (cùng model, cùng metric hai lần), `pivot` raise `ValueError: Index contains duplicate entries, cannot reshape`; dùng `pivot_table(..., aggfunc="mean")` để gộp.

#### P5.5 Thời gian: parse, múi giờ, gom theo ngày

```python
raw = pd.Series(["2026-10-27 08:15:00", "2026-10-27 23:40:00", "không rõ", "2026-10-28 00:05:00"])
pd.to_datetime(raw)          # ValueError: time data "không rõ" doesn't match format ...
t = pd.to_datetime(raw, format="%Y-%m-%d %H:%M:%S", errors="coerce")
t.dtype                      # datetime64[us]              ← pandas 3 mặc định độ phân giải micro giây
t.tolist()                   # [..., NaT, ...]             ← giá trị lỗi thành NaT (Not a Time)
t.dt.hour.tolist()           # [8.0, 23.0, nan, 0.0]
t.dt.day_name().tolist()     # ['Tuesday', 'Tuesday', nan, 'Wednesday']
t[1] - t[0]                  # Timedelta('0 days 15:25:00') → .total_seconds() = 55500.0
```

**Ngày/tháng hay tháng/ngày.** Chuỗi `"03/04/2026"`: `dayfirst=True` cho **2026-04-03** (kiểu Việt Nam), `format="%m/%d/%Y"` cho **2026-03-04** (kiểu Mỹ). Cả hai đều parse "thành công", nên lỗi này **không bao giờ tự báo**. Luôn chỉ định `format` rõ ràng.

**Múi giờ.** Log của API thường ở UTC; báo cáo cho người Việt cần giờ Việt Nam (UTC+7):

```python
utc = pd.to_datetime(pd.Series(["2026-10-27T17:30:00Z"]))   # dtype datetime64[us, UTC]
utc.dt.tz_convert("Asia/Ho_Chi_Minh")[0]                   # 2026-10-28 00:30:00+07:00  ← sang ngày hôm sau!
naive = pd.Series(pd.to_datetime(["2026-10-27 08:00"]))
naive.dt.tz_localize("Asia/Ho_Chi_Minh")[0]                 # 2026-10-27 08:00:00+07:00  ← GẮN múi giờ, không đổi giờ
```

Ví dụ trên cho thấy: một sự kiện lúc 17:30 UTC ngày 27 được tính vào **ngày 28** theo giờ Việt Nam. Nếu thống kê "số 👎 theo ngày" mà quên đổi múi giờ, mọi sự kiện từ 17:00 đến 24:00 UTC bị xếp sai ngày. Đặc trưng `hour` của project Tuần 3 cũng phải là giờ **địa phương**.

**Gom theo khoảng thời gian:**

```python
ts = pd.DataFrame({"t": t, "lat": [1.0, 2.0, 9.0, 3.0]}).dropna(subset=["t"])
ts.set_index("t").resample("D")["lat"].agg(["count", "mean"])
#             count  mean
# 2026-10-27      2   1.5
# 2026-10-28      1   3.0
```

`resample` cần index là thời gian; tần suất viết thường trong pandas 3 (`"h"`, `"D"`, `"min"`). `resample` tạo cả các khoảng **không có dữ liệu** (count 0), còn `groupby(t.dt.date)` thì bỏ qua chúng; với dashboard, các khoảng trống đó là thông tin.

#### P5.6 Cạm bẫy

1. **`np.stack` khi muốn `concatenate`** (hoặc ngược lại): shape sai một trục, và lỗi chỉ hiện ra vài bước sau.
2. **`pd.concat` không `ignore_index=True`**: index trùng, rồi `df.loc[0]` trả về **nhiều hàng**.
3. **Parse ngày không có `format`**: nhầm ngày/tháng lặng lẽ; chậm hơn nhiều trên dữ liệu lớn.
4. **Trộn tz-aware với tz-naive**: so sánh hoặc trừ hai loại raise `TypeError`. Quy ước: lưu UTC, chỉ đổi sang giờ địa phương khi hiển thị hoặc khi tạo đặc trưng.
5. **`np.where(cond, f(x), g(x))` với `f` không xác định ở một số chỗ**: vẫn tính trên toàn mảng.

**Sắc thái phỏng vấn:** *"Bạn có bảng log 50 triệu dòng, cần thống kê latency p95 theo giờ cho từng model. Làm thế nào?"* Ý phải có: parse thời gian một lần với `format` rõ ràng; đổi múi giờ; `groupby([model, t.dt.floor("h")])["lat"].quantile(0.95)`; đọc theo chunk hoặc dùng định dạng cột (Parquet) nếu không đủ RAM.

#### P5.7 Tự làm (10')

Cho `results` dạng dài (`run_id, seed, metric, value`) từ 3 seed × 4 metric. Dùng `pivot_table` để có bảng `run_id × metric` với giá trị trung bình qua seed, rồi `melt` ngược lại và kiểm tra số hàng.

#### P5.8 Đọc thêm

- NumPy docs: *Array manipulation routines* (Joining arrays), `numpy.where`.
- pandas docs: *Reshaping and pivot tables*, *Time series / date functionality*.
- H. Wickham, *Tidy Data*, Journal of Statistical Software 59 (2014).

---

## P6 · Công cụ thực hành: `tqdm`, config YAML/Hydra, `aiohttp`, Jupyter/Colab

> ⏱ Tầng 1: 5' · Tầng 2: 20'
> **Cần biết trước:** asyncio, Semaphore, Little's law (Tuần 1, bài 08); cấu hình qua biến môi trường (Tuần 1, bài 10).

> **📖 Từ mới**
> - **Progress bar:** thanh tiến độ, cho biết đã chạy bao nhiêu phần trăm và còn bao lâu.
> - **Config file:** file (YAML, TOML, JSON) chứa các tham số của thí nghiệm, tách khỏi code.
> - **YAML:** định dạng config dựa trên thụt lề, dễ đọc với người, nhưng có nhiều quy tắc đoán kiểu ngầm.
> - **Notebook:** tài liệu gồm các cell code và văn bản, chạy từng cell trên một **kernel** (tiến trình Python) giữ trạng thái giữa các cell.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- `tqdm` là **thanh tải phim**: không làm phim tải nhanh hơn, nhưng cho biết nên chờ hay nên đi pha cà phê.
- File config là **tờ công thức nấu ăn để riêng**: đổi lượng muối không phải sửa "cách nấu" (code).
- Notebook là **giấy nháp**: tuyệt vời để thử, nhưng thứ tự bạn viết trên giấy không phải thứ tự bạn đã tính.

**Ba ý cốt lõi:**
1. Thí nghiệm phải **tái lập được từ config + seed**, không phải từ trí nhớ về các cell đã chạy.
2. YAML **đoán kiểu dữ liệu** theo những quy tắc bất ngờ: `1e-3` thành **chuỗi**, `NO` thành `False`.
3. `aiohttp` và `httpx` giải cùng một bài toán (HTTP async); chọn một, và **dùng lại một session** cho mọi request.

### TẦNG 2 · Kiến thức chuẩn

#### P6.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **`tqdm`** | Bọc một iterable để in tiến độ, tốc độ (it/s), thời gian còn lại; có bản cho asyncio (`tqdm.asyncio`) | tqdm docs |
| **YAML 1.1 / 1.2** | Hai phiên bản đặc tả; PyYAML theo **1.1**, có quy tắc đoán kiểu rộng (`yes`/`no`/`on`/`off` là bool, số có `:` là hệ 60) | yaml.org, *YAML 1.1* (2005), *YAML 1.2* (2009) |
| **`yaml.safe_load`** | Chỉ dựng các kiểu cơ bản (dict, list, str, số, bool, null); `yaml.load` với `Loader` không an toàn có thể **chạy code tùy ý** | PyYAML docs; CVE-2017-18342 |
| **OmegaConf / Hydra** | OmegaConf: config phân cấp có nội suy `${...}` và kiểm tra kiểu; Hydra: ghép config từ nhiều file, ghi đè từ dòng lệnh, tự tạo thư mục output cho mỗi lần chạy | Hydra docs (Meta, 2019–) |
| **`aiohttp.ClientSession`** | Client HTTP async, giữ connection pool; nên tạo **một** session cho cả ứng dụng | aiohttp docs, *Client Quickstart* |
| **Jupyter kernel, magic** | Tiến trình giữ trạng thái; lệnh `%` (dòng) và `%%` (cell) như `%timeit`, `%debug`, `%autoreload` | IPython docs, *Built-in magic commands* |

#### P6.2 `tqdm`

```python
from tqdm import tqdm
for batch in tqdm(loader, desc="epoch 1", unit="batch"):
    loss = train_step(batch)
    # tqdm.write(...) thay cho print bên trong vòng lặp, để không làm vỡ thanh tiến độ

pbar = tqdm(total=n_requests)             # khi không có iterable: tự cập nhật
pbar.update(1); pbar.set_postfix(loss=f"{loss:.3f}")
```

**Overhead (đo trên máy bạn):** vòng lặp rỗng 1 triệu lần: 15 ns/vòng; bọc `tqdm`: 286 ns/vòng, tức thêm ~270 ns mỗi vòng. Với một bước train (vài ms) hay một request API (vài trăm ms), con số này không đáng kể. Với vòng lặp hàng triệu phần tử rất nhẹ thì thời gian tăng gần 20×; khi đó đặt `tqdm` ở vòng **ngoài** (theo batch hoặc theo file).

Với asyncio, `await tqdm.asyncio.tqdm.gather(*coros)` thay cho `asyncio.gather` để có thanh tiến độ khi chạy hàng loạt request (đo: 100 request mất 0.621 s, so với 0.599 s khi dùng `gather` thường).

Khi ghi log ra file hoặc chạy trong CI, thanh tiến độ thành hàng nghìn dòng rác: dùng `disable=not sys.stderr.isatty()` hoặc biến môi trường `TQDM_DISABLE=1`.

#### P6.3 Config YAML: những quy tắc đoán kiểu bất ngờ

File `config.yaml` sau được đọc bằng `yaml.safe_load` (PyYAML 6.0, đã chạy):

| Dòng YAML | Python nhận được | Có đúng ý không? |
|---|---|---|
| `lr: 1e-3` | `'1e-3'` (**str**) | ❌ YAML 1.1 bắt buộc có dấu chấm: phải viết `1.0e-3` |
| `lr2: 1.0e-3` | `0.001` (float) | ✅ |
| `use_rag: yes` | `True` | ✅ nhưng dễ nhầm |
| `country: NO` | `False` | ❌ "Norway problem": `NO` là bool |
| `version: 3.10` | `3.1` (float) | ❌ phải viết `"3.10"` |
| `seed: 010` | `8` | ❌ số có số 0 đứng đầu được đọc theo hệ bát phân |
| `time: 12:30` | `750` (int) | ❌ hệ 60: $12 \times 60 + 30$ |
| `model: MODEL_ID` | `'MODEL_ID'` | ✅ |

Lỗi `lr: 1e-3` đặc biệt nguy hiểm: `optimizer = SGD(params, lr=cfg["lr"])` có thể raise lỗi kiểu ở chỗ xa, hoặc tệ hơn, một phép so sánh chuỗi chạy "được" mà sai. **Cách phòng:** (1) quote mọi giá trị không phải số rõ ràng; (2) **validate** config ngay khi đọc bằng dataclass (Tuần 1, bài 03) hoặc pydantic, ép kiểu và báo lỗi sớm:

```python
from dataclasses import dataclass, fields
import yaml

@dataclass(frozen=True)
class TrainConfig:
    model: str
    lr: float
    epochs: int = 10
    seed: int = 0

def load_config(path, overrides=()):
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    for item in overrides:                       # ghi đè kiểu Hydra: "lr=0.01"
        k, v = item.split("=", 1); raw[k] = yaml.safe_load(v)
    types = {f.name: f.type for f in fields(TrainConfig)}
    unknown = set(raw) - set(types)
    if unknown:
        raise ValueError(f"khóa không hợp lệ: {sorted(unknown)}")   # bắt lỗi gõ sai tên ("lerning_rate")
    return TrainConfig(**{k: types[k](v) for k, v in raw.items()})  # float("1e-3") = 0.001 ✓
```

**An toàn.** `yaml.load(text)` không có `Loader` raise `TypeError` trong PyYAML 6. Với chuỗi `!!python/object/apply:os.system [...]`, `yaml.safe_load` raise `ConstructorError` (đã chạy), còn `yaml.load(..., Loader=yaml.UnsafeLoader)` sẽ **thực thi lệnh**. Không bao giờ đọc YAML từ nguồn không tin cậy bằng loader không an toàn.

**Hydra (mô tả, chưa cài trên máy bạn).** Khi số thí nghiệm lớn, Hydra cung cấp sẵn những gì `load_config` ở trên tự làm, cộng thêm:
- ghép config từ nhiều nhóm: `conf/model/small.yaml`, `conf/optim/adamw.yaml`, chọn bằng `python train.py model=small optim=adamw`;
- ghi đè bất kỳ khóa nào từ dòng lệnh: `optim.lr=3e-4` (cú pháp ghi đè của Hydra đọc `3e-4` là float);
- `--multirun` để quét lưới tham số: `python train.py -m optim.lr=1e-3,3e-4 seed=0,1,2` chạy 6 lần;
- mỗi lần chạy có thư mục output riêng, chứa bản sao config đã dùng, nhờ đó tái lập được.

#### P6.4 `aiohttp` so với `httpx`

Cùng bài toán ở Tuần 1, bài 08 (gọi nhiều request đồng thời có giới hạn), viết bằng `aiohttp`:

```python
import asyncio, aiohttp

async def ask(session, sem, url, q):
    async with sem:
        async with session.post(url, json={"q": q}) as resp:   # phải đọc body TRONG khối async with
            resp.raise_for_status()
            return (await resp.json())["answer"]

async def main(url, questions, limit=10):
    sem = asyncio.Semaphore(limit)
    timeout = aiohttp.ClientTimeout(total=30, connect=5)       # aiohttp mặc định total=300 s
    async with aiohttp.ClientSession(timeout=timeout) as session:   # MỘT session cho mọi request
        return await asyncio.gather(*(ask(session, sem, url, q) for q in questions))
```

**Đo trên máy bạn** (server `aiohttp` chạy local, mỗi request chờ 50 ms; 100 request, Semaphore 10):

| Cách gọi | Wall time |
|---|---|
| Ước lượng bằng Little's law: $100 / 10 \times 50$ ms | 0.50 s |
| `aiohttp`, một session | 0.60 s |
| `httpx.AsyncClient`, một client | 0.64 s |
| `aiohttp`, **tạo session mới cho mỗi request** | 0.67 s |

Trên localhost, tạo session mới mỗi lần chỉ chậm hơn ~12%, vì mở kết nối TCP local gần như miễn phí. Với API thật qua HTTPS, mỗi kết nối mới tốn thêm 1–2 round-trip cho bắt tay TCP + TLS (ví dụ 2 × 30 ms), và chênh lệch sẽ lớn hơn nhiều. Phần này **chưa đo** được vì máy không gọi ra mạng trong bài.

| | `httpx` | `aiohttp` |
|---|---|---|
| Sync + async | Cả hai, API giống `requests` | Chỉ async |
| HTTP/2 | Có (tùy chọn) | Không |
| Server | Không | Có (`aiohttp.web`) |
| Timeout mặc định | 5 s | 300 s tổng |
| Dùng trong project này | Tuần 1 (`multi_caller`), Tuần 6 | Khi thư viện khác yêu cầu, hoặc cần server nhỏ |

Kết luận cho dự án: tiếp tục dùng `httpx` (đã có test); biết đọc code `aiohttp` vì nhiều SDK và ví dụ cũ dùng nó.

#### P6.5 Jupyter và Colab/Kaggle

**Mô hình chạy.** Notebook gửi từng cell tới **kernel**. Biến tồn tại tới khi kernel restart, **bất kể bạn đã xóa hay sửa cell định nghĩa nó**. Số `In [7]` cho biết thứ tự đã chạy thật. Notebook chạy "được" trên máy bạn nhưng lỗi khi người khác chạy từ đầu, vì nó phụ thuộc vào một cell đã bị xóa: đây là lỗi phổ biến nhất. Trước khi chia sẻ, luôn chạy **Restart & Run All**.

| Magic | Tác dụng |
|---|---|
| `%timeit expr` / `%%timeit` | Benchmark một dòng / một cell (P4) |
| `%time expr` | Chạy một lần, in wall time |
| `%prun stmt` | `cProfile` cho một câu lệnh |
| `%debug` | Post-mortem `pdb` tại exception vừa xảy ra (P3) |
| `%load_ext autoreload` + `%autoreload 2` | Tự import lại module `.py` khi file thay đổi |
| `%pip install pkg` | Cài vào **đúng** môi trường của kernel (`!pip` có thể cài nhầm môi trường) |
| `!cmd` | Chạy lệnh shell, ví dụ `!nvidia-smi` |

**Quy ước làm việc cho kế hoạch 16 tuần:** notebook để **khám phá và vẽ biểu đồ**; logic được dùng lại phải nằm trong package `.py` có test (như mọi project từ Tuần 1); notebook chỉ `import` từ package, cùng với `%autoreload 2`.

**Colab và Kaggle** (bạn không có GPU, nên đây là nơi chạy các tuần 11–12):
- **Bật GPU:** Colab: *Runtime → Change runtime type → GPU*; Kaggle: *Settings → Accelerator*. Kiểm tra bằng `!nvidia-smi` và `torch.cuda.is_available()`.
- **Bộ nhớ không bền:** mọi file trong máy ảo **mất** khi phiên kết thúc. Colab: `from google.colab import drive; drive.mount("/content/drive")` để ghi checkpoint lên Drive. Kaggle: chỉ `/kaggle/working` được giữ lại khi *Save Version*.
- **Giới hạn thời gian và hạn mức:** gói miễn phí có giới hạn thời lượng phiên, thời gian rảnh trước khi bị ngắt, và hạn mức GPU mỗi tuần (Kaggle). Các con số **thay đổi theo thời gian**, nên hãy kiểm tra trang trợ giúp hiện tại. Thiết kế huấn luyện để **lưu checkpoint định kỳ và chạy tiếp được** (*resume*), không giả định phiên chạy liền một mạch.
- **Secret:** không dán API key vào cell. Dùng *Secrets* của Colab/Kaggle và đọc qua biến môi trường (Tuần 1, bài 10).

#### P6.6 Cạm bẫy

1. **`lr: 1e-3` trong YAML** thành chuỗi (P6.3).
2. **`yaml.load` với loader không an toàn** trên file từ bên ngoài: thực thi code tùy ý.
3. **Tạo `ClientSession`/`AsyncClient` mới cho mỗi request**: mất connection pooling, và dễ cạn file descriptor khi tải cao.
4. **Đọc body sau khi đã thoát khối `async with session.post(...)`** trong `aiohttp`: kết nối đã được trả về pool, body có thể không đọc được.
5. **Notebook phụ thuộc thứ tự chạy ẩn**: luôn Restart & Run All trước khi báo cáo kết quả.
6. **Huấn luyện trên Colab không có checkpoint**: phiên bị ngắt là mất hết.

**Sắc thái phỏng vấn:** *"Làm sao để một thí nghiệm ML tái lập được?"* Ý phải có: config được version (Git) và được **lưu kèm output**; seed cho mọi nguồn ngẫu nhiên; ghi phiên bản thư viện (`pip freeze`/lockfile); dữ liệu có version hoặc hash; code chạy từ đầu bằng một lệnh, không dựa vào trạng thái notebook.

#### P6.7 Tự làm (15')

Viết `config.yaml` cho project `quality` (Tuần 3) gồm `model`, `lr`, `max_iter`, `seed`, và cố ý viết `lr: 1e-3`. Viết `load_config` như ở P6.3, kiểm tra rằng nó trả về `lr == 0.001` (float), và raise `ValueError` khi có khóa gõ sai. Thêm ghi đè `seed=1` từ dòng lệnh.

#### P6.8 Đọc thêm

- tqdm docs (github.com/tqdm/tqdm); PyYAML docs; *The YAML Specification* 1.1 và 1.2.
- Hydra docs (hydra.cc): *Getting started*, *Override grammar*.
- aiohttp docs: *Client Quickstart*, *Client Reference*.
- J. Pimentel et al., *A Large-Scale Study About Quality and Reproducibility of Jupyter Notebooks*, MSR 2019. Trong ~860 000 notebook, chỉ ~24% chạy được không lỗi, và ~4% cho lại đúng kết quả.

---

## Quiz phụ lục Tuần 1

> ⏱ **30 phút** · 10 câu nhanh (1đ) + 2 câu tự luận (5đ) · **20 điểm** · Đạt khi ≥ 14
> Làm trên giấy, **không chạy code**. Với câu đoán output: ghi chính xác những gì được in ra; nếu báo lỗi thì ghi tên exception.

**Phần 1 — Câu nhanh**

1. `s = "transformer"`. In ra gì: `s[3:7]`, `s[-4:]`, `s[::3]`?
2. `"  rag  llm ".split(" ")` cho bao nhiêu phần tử?
3. `[x * 2 if x > 1 else -1 for x in range(4) if x != 2]` cho ra gì?
4. `len(unicodedata.normalize("NFD", "Hà Nội"))` bằng bao nhiêu? (Gợi ý: "à" và "ộ" tách thành mấy code point?)
5. `from collections import Counter; (Counter("aab") + Counter("bbc")).most_common(1)` cho ra gì?
6. Một work queue dùng `list` với `pop(0)` chạy 2 s với 50 000 job. Ước lượng thời gian với 200 000 job (giả sử phần còn lại không đáng kể). Nên đổi sang cấu trúc nào?
7. Traceback kết thúc bằng hai khối nối với nhau bởi dòng "During handling of the above exception, another exception occurred". Dòng này cho bạn biết gì, và nó khác "direct cause" ở đâu?
8. `cProfile` báo hàm `f` có `ncalls=1`, `tottime=0.02`, `cumtime=3.1`. Hotspot nằm trong `f` hay trong các hàm mà `f` gọi?
9. `np.stack([A, B, C], axis=1)` với `A`, `B`, `C` shape `(4, 5)` cho shape gì? `np.concatenate([A, B, C], axis=1)` cho shape gì?
10. YAML `patience: 05`, `dropout: .1`, `name: off`. PyYAML `safe_load` cho ra ba giá trị Python nào? (Gợi ý: YAML 1.1 coi `off` là bool.)

**Phần 2 — Tự luận**

**T1 · Debug và tối ưu (5đ).** Một hàm gộp log:

```python
def merge_logs(paths):
    df = pd.DataFrame()
    for p in paths:
        part = pd.read_csv(p)
        part["ts"] = pd.to_datetime(part["ts"])          # chuỗi dạng "27/10/2026 08:15"
        df = pd.concat([df, part])
    df["date"] = df["ts"].dt.date
    return df.groupby("date")["latency"].mean()
```

- (a) Chỉ ra **ba** vấn đề (đúng đắn hoặc hiệu năng), mỗi vấn đề nói rõ hậu quả.
- (b) Viết lại hàm. Log được ghi theo giờ UTC, còn báo cáo theo ngày giờ Việt Nam.
- (c) Với 365 file, mỗi file 100 000 dòng, ước lượng bậc độ lớn số hàng bị sao chép bởi bản gốc so với bản sửa.

**T2 · Config và tái lập (5đ).** Đồng nghiệp gửi `config.yaml`:

```yaml
model: MODEL_ID
lr: 3e-4
warmup_steps: 1_000
eval_every: 1:00
use_cache: no
```

- (a) Với PyYAML `safe_load`, mỗi khóa nhận giá trị gì, kiểu gì? Khóa nào không đúng ý người viết? (Gợi ý: YAML 1.1 của PyYAML chấp nhận dấu `_` trong số nguyên.)
- (b) Đề xuất cách đọc config phát hiện được các lỗi này **ngay lúc khởi động** (mô tả hoặc viết code ngắn).
- (c) Nêu 3 thứ phải lưu kèm kết quả của mỗi lần chạy để người khác tái lập được.

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Phần 1**

1. `nsfo`, `rmer`, `tnoe`. (`s[3:7]` là các chỉ số 3–6: n, s, f, o; `s[::3]` là các chỉ số 0, 3, 6, 9: t, n, o, e.)
2. **6**: `['', '', 'rag', '', 'llm', '']`.
3. `[-1, -1, 6]`. `x = 0` → −1; `x = 1` → −1; `x = 2` bị lọc; `x = 3` → 6.
4. **9**. NFC có 6 ký tự ("H", "à", " ", "N", "ộ", "i"). NFD tách "à" thành 2 code point (a + dấu huyền) và "ộ" thành 3 (o + dấu nặng + dấu mũ): $6 + 1 + 2 = 9$.
5. `[('b', 3)]`. Tổng là a: 2, b: 1 + 2 = 3, c: 1.
6. $O(n^2)$: tăng 4 lần $n$ thì thời gian tăng 16 lần, tức **~32 s**. Đổi sang `collections.deque` + `popleft()`.
7. Có một exception **thứ hai** xảy ra **trong khối `except`** đang xử lý exception đầu (thường là bug trong code xử lý lỗi, hoặc raise lại mà không có `from`). "Direct cause" (`raise … from e`) là **cố ý** gắn nguyên nhân.
8. Trong các **hàm con**: `cumtime` lớn nhưng `tottime` nhỏ, nên thân `f` gần như không tốn thời gian.
9. `(4, 3, 5)` và `(4, 15)`.
10. `5` (int), `0.1` (float), `False` (bool).

**T1 (5đ)**

(a) 2đ, mỗi ý đúng 2/3đ, tối đa 2đ:
- `pd.concat` **trong vòng lặp**: mỗi lần chép lại toàn bộ `df`, tổng chi phí $O(k^2)$ theo số file. Thêm nữa, không có `ignore_index=True` nên index bị trùng.
- `pd.to_datetime` **không có `format`** với chuỗi "27/10/2026": tùy phiên bản, hoặc raise, hoặc parse nhầm ngày/tháng với những ngày ≤ 12. Lỗi lặng lẽ.
- **Không xử lý múi giờ**: log UTC bị gom theo ngày UTC, nên sự kiện từ 17:00–24:00 UTC (00:00–07:00 giờ Việt Nam ngày hôm sau) bị xếp sai ngày.
- (Cũng chấp nhận:) không `errors="coerce"`, nên một dòng hỏng làm hỏng cả lần chạy; `groupby(date)` bỏ qua các ngày không có dữ liệu.

(b) 2đ:

```python
def merge_logs(paths):
    parts = [pd.read_csv(p) for p in paths]
    df = pd.concat(parts, ignore_index=True)                      # concat MỘT lần
    ts = pd.to_datetime(df["ts"], format="%d/%m/%Y %H:%M", errors="coerce")
    df["ts"] = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Ho_Chi_Minh")
    df = df.dropna(subset=["ts"])
    return df.set_index("ts").resample("D")["latency"].mean()     # có cả ngày trống
```

(c) 1đ. Bản gốc: lần lặp thứ $k$ chép ~$k \times 10^5$ hàng, tổng $\approx 10^5 \times 365^2 / 2 \approx 6.7 \times 10^9$ hàng. Bản sửa: chép một lần $3.65 \times 10^7$ hàng. Chênh khoảng **180 lần** ($\approx 365/2$).

**T2 (5đ)**

(a) 2đ:
- `model: 'MODEL_ID'` (str) ✓
- `lr: '3e-4'` (**str**) ✗: YAML 1.1 cần dấu chấm, phải viết `3.0e-4`.
- `warmup_steps: 1000` (int) ✓: PyYAML chấp nhận `_` trong số.
- `eval_every: 60` (int) ✗: hệ 60, $1 \times 60 + 0$.
- `use_cache: False` (bool): đúng ý nếu người viết muốn bool, nhưng dễ nhầm. Nên viết `false`.

(b) 2đ: đọc bằng `safe_load` rồi validate ngay bằng dataclass/pydantic có khai báo kiểu: ép `float(lr)`, từ chối khóa lạ, từ chối giá trị không đúng kiểu (`eval_every` phải là chuỗi khoảng thời gian, hoặc là số bước ghi rõ đơn vị). Log lại config **sau khi** parse lúc khởi động. Cho điểm đầy đủ nếu có: kiểu được khai báo + kiểm tra lúc khởi động + báo lỗi rõ ràng.

(c) 1đ: bất kỳ 3 trong số: config đã parse; seed; commit hash của code; phiên bản thư viện (lockfile); hash hoặc version của dữ liệu; lệnh chạy.

</details>
