# Quiz · Ngày 1 · Python cốt lõi (bài 01–04)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> **Không chạy code** trong lúc làm, đoán output bằng đầu. Đáp án ở cuối file.

## Phần 1 — Quick test (14 câu, 1đ/câu; riêng câu 14 là 2đ)

Với các câu đoán output: ghi chính xác những gì được in ra. Nếu code báo lỗi, ghi tên exception.

1. `a = [1, 2]; b = a[:]; b.append(3); print(a)`
2. `x = (1, [2]); x[1].append(3); print(x)`
3. Object nào **hashable**? A. `[1, 2]` · B. `{"a": 1}` · C. `(1, 2)` · D. `(1, [2])`
4. `print([] is [], [] == [])`
5. `def f(*args, **kwargs): return len(args), sorted(kwargs)` — hỏi `f(1, 2, a=3, b=4)` trả về gì?
6. `def g(a, *, b): ...` — hỏi `g(1, 2)` cho kết quả gì?
7. Đoạn code dưới đây in ra gì, theo thứ tự nào, và in vào lúc nào? (Chú ý: code không hề gọi `h()`.)
   ```python
   def d1(f): print("d1"); return f
   def d2(f): print("d2"); return f
   @d1
   @d2
   def h(): pass
   ```
8. Đoạn code dưới đây in ra gì?
   ```python
   def outer():
       x = 1
       def inner(): return x
       x = 2
       return inner
   print(outer()())
   ```
9. Với `@dataclass class Pt: x: int; y: int` (không có `frozen`): `Pt(1, 2) == Pt(1, 2)` cho kết quả gì? Còn `{Pt(1, 2)}` thì sao?
10. `class X: pass`, `class Y(X): pass`, `class Z(X): pass`, `class W(Z, Y): pass`. Hãy viết `W.__mro__` dưới dạng danh sách tên class.
11. Đúng/Sai: Python kiểm tra `Protocol` lúc chạy và raise lỗi nếu object thiếu method.
12. `it = iter([1, 2, 3]); print(list(zip(it, it)))`
13. `g = (i for i in range(3)); print(2 in g, list(g))`
14. *(2đ)* Lấy 1 triệu phần tử. Với list comprehension và với generator expression, `sys.getsizeof` của mỗi loại cỡ bao nhiêu? Loại nào tăng theo kích thước dữ liệu?

## Phần 2 — Tự luận (4 câu, 30 phút)

**T1 · Code review (4đ).** Đoạn code dưới đây có bug. Bug đó gây ra hậu quả gì sau 3 lần gọi `build_request`? Sửa lại cho đúng.

```python
DEFAULTS = {"temperature": 0.2, "stop": []}

def build_request(prompt, overrides={}):
    req = DEFAULTS.copy()
    req.update(overrides)
    req["stop"].append("\n\n")
    req["prompt"] = prompt
    return req
```

**T2 · Code (4đ).** Viết decorator `memoize` cho hàm đồng bộ: cache kết quả theo các đối số. Yêu cầu:
- Đúng với cả đối số vị trí lẫn đối số tên.
- Hai lời gọi `f(a=1, b=2)` và `f(b=2, a=1)` phải dùng chung một mục cache.

Sau đó nêu **2 giới hạn** của cách làm này, và cho biết thư viện chuẩn có sẵn công cụ nào thay thế.

**T3 · Thiết kế (3đ).** Gateway của bạn có 2 loại provider:
- (a) provider do team bạn tự viết;
- (b) plugin do team khác viết, bọc quanh SDK của họ, và bạn **không sửa được** code đó.

Chọn ABC hay Protocol cho từng loại và giải thích. `ChatMessage` và `CallResult` nên là dataclass `frozen` hay không? Vì sao?

**T4 · Ước lượng (4đ).** File JSONL nặng 2 GB, có 4 triệu dòng, mỗi dòng khoảng 500 ký tự ASCII. Ước lượng RAM cần dùng khi:
- (a) đọc bằng `fh.readlines()`;
- (b) dùng pipeline gồm generator `read_jsonl` và `batched(..., 64)`.

Biết `sys.getsizeof("")` = 41 B và mỗi con trỏ trong list là 8 B.

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Quick test**

1. `[1, 2]`. Slice `[:]` tạo một list mới (shallow copy).
2. `(1, [2, 3])`. Tuple là immutable, nhưng list nằm bên trong nó thì vẫn sửa được.
3. **C**. Đáp án D có chứa list, nên `hash` sẽ raise `TypeError`.
4. `False True`. Đây là hai object khác nhau nhưng có giá trị bằng nhau.
5. `(2, ['a', 'b'])`.
6. `TypeError`, vì `b` là keyword-only.
7. In `d2` rồi đến `d1`, ngay **lúc định nghĩa** hàm, dù `h` chưa hề được gọi. Decorator được áp từ dưới lên trên.
8. `2`. Closure tra giá trị của `x` lúc `inner` *chạy*, khi đó `x` đã là 2 (late binding).
9. `True`; còn `{Pt(1, 2)}` raise `TypeError` (unhashable), vì `eq=True` mà không `frozen` thì Python đặt `__hash__ = None`.
10. `['W', 'Z', 'Y', 'X', 'object']`.
11. **Sai**. Protocol chỉ được kiểm tra tĩnh bởi mypy/pyright, trừ khi dùng `@runtime_checkable` (và kể cả khi đó, Python cũng chỉ kiểm tra method có tồn tại hay không).
12. `[(1, 2)]`. Cả hai đối số của `zip` là **cùng một** iterator; số `3` bị lấy ra nhưng không có cặp nên bị bỏ.
13. `True []`. Phép `in` đã duyệt qua 0, 1, 2, nên generator đã cạn.
14. List ≈ 8.4 MB và tăng theo kích thước dữ liệu; generator ≈ 208 B, cố định.

**T1**

*Hậu quả:* `DEFAULTS.copy()` chỉ là shallow copy, nên `req["stop"]` **chính là** `DEFAULTS["stop"]`. Sau 3 lần gọi, `DEFAULTS["stop"] == ['\n\n', '\n\n', '\n\n']`, và mỗi request sau lại có thêm một chuỗi dừng. Đây là bug "rò trạng thái" (state leak) giữa các request, rất khó phát hiện. *(2đ)*

*Cách sửa (2đ):*

```python
import copy

def build_request(prompt, overrides=None):
    req = copy.deepcopy(DEFAULTS)
    req.update(overrides or {})
    req["stop"] = [*req["stop"], "\n\n"]
    req["prompt"] = prompt
    return req
```

Ngoài ra, `overrides={}` là mutable default. Code hiện tại chưa sửa vào nó nên chưa gây bug, nhưng đây là một quả mìn chờ nổ.

**T2**

```python
import functools

def memoize(fn):
    cache = {}
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = fn(*args, **kwargs)
        return cache[key]
    return wrapper
```

Dùng `frozenset` cho kwargs nên thứ tự truyền tên không ảnh hưởng tới key.

*Giới hạn:*
1. Đối số phải **hashable**; truyền list hay dict vào sẽ gây `TypeError`.
2. Cache **tăng mãi**, không giới hạn, nên bộ nhớ rò rỉ dần.
3. Ngoài ra: `f(1)` và `f(a=1)` bị coi là hai key khác nhau; cache không an toàn khi dùng từ nhiều thread.

*Công cụ thay thế:* `functools.lru_cache(maxsize=...)`, hoặc `functools.cache`.

*(Chấm: code đúng 3đ, nêu đủ 2 giới hạn 1đ)*

**T3**

- (a) Provider tự viết: **ABC**. Bạn sở hữu cả hierarchy, và thiếu method thì bị chặn ngay lúc tạo instance.
- (b) Plugin của team khác: **Protocol**. Không ép họ phải kế thừa class của bạn; chỉ cần có đúng `async def complete(...)` là được, và mypy kiểm tra giúp.
- `ChatMessage`: **frozen**. Tin nhắn đã gửi đi không nên bị sửa, và frozen thì hashable, nên dùng được làm cache key.
- `CallResult`: **không frozen**. Kết quả cần được điền dần trong quá trình gọi (latency, danh sách lỗi), và dùng `field(default_factory=list)` cho danh sách lỗi.

*(1đ mỗi ý đúng, tối đa 3đ)*

**T4**

- **(a) `readlines()`:** khoảng $4\cdot10^6 \times (500 + 41)$ B ≈ **2.16 GB** cho các chuỗi, cộng $4\cdot10^6 \times 8$ B = 32 MB cho con trỏ của list. Tổng ≈ **2.2 GB**. Nếu parse tiếp thành list các dict thì còn tăng thêm vài GB (mỗi dict ≥ 184 B, cộng các object khóa và giá trị).
- **(b) Generator:** tại mỗi thời điểm chỉ có một lô 64 dòng trong bộ nhớ, khoảng $64 \times (541 + \text{dict}) \approx$ **dưới 100 KB**, cố định bất kể file lớn cỡ nào.

*(Chấm: (a) 2đ, (b) 2đ. Chấp nhận sai số trong khoảng ±30% nếu cách lập luận đúng.)*

</details>
