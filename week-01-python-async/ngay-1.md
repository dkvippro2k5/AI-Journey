# Ngày 1 · Python cốt lõi

> **Loại tài liệu: Kỹ thuật.** Trọng tâm là thuật ngữ chuẩn, cơ chế, code và quyết định thiết kế. Toán chỉ dùng để ước lượng nhanh.
> **Hôm nay gồm 4 bài:**
> - **01** Data model: object, tên và bộ nhớ
> - **02** Function, closure, decorator
> - **03** OOP & dataclass
> - **04** Iterator & generator
>
> **Thời gian:** đọc ~2h30 · quiz 45' · bài tập ~2h
> **Môi trường kiểm chứng:** mọi đoạn code và output trong file đã chạy trên Python 3.14.

---

## 01 · Data model: object, tên và bộ nhớ

> ⏱ Tầng 1: 10' · Tầng 2: 25'

> **📖 Từ mới**
> - **Object:** mọi giá trị trong Python (số, chuỗi, list, hàm, class) đều là object nằm trong bộ nhớ.
> - **Tên (biến):** một cái nhãn trỏ tới object. Tên không phải là "cái hộp chứa giá trị".
> - **Mutable / immutable:** đổi được / không đổi được nội dung sau khi tạo.
> - **Hashable:** có giá trị băm (hash) không đổi trong suốt vòng đời, nên dùng được làm key của dict hoặc phần tử của set.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- Object là một **cái hộp**, còn tên biến là **tờ nhãn dán** lên hộp.
- `b = a` **không** tạo hộp mới. Nó chỉ dán thêm nhãn `b` lên đúng cái hộp đang có nhãn `a`.
- Có loại hộp mở ra thay đồ bên trong được: **mutable**, như `list`, `dict`, `set`.
- Có loại hộp bị niêm phong: **immutable**, như `int`, `str`, `tuple`. Muốn "đổi" thì phải lấy một hộp mới rồi dán nhãn sang hộp đó.

```python
a = [1, 2]; b = a; b.append(3)
print(a)        # [1, 2, 3]  ← cùng một hộp, sửa qua b thì a cũng thấy
x = 5; y = x; y += 1
print(x, y)     # 5 6        ← int bị niêm phong: y += 1 tạo hộp mới, x vẫn là 5
```

**Ba ý cốt lõi:**
1. Gán (`=`) là **dán nhãn**, không phải sao chép.
2. Sửa một object mutable thì **mọi tên** trỏ tới nó đều thấy thay đổi.
3. Muốn có bản riêng thì phải **sao chép rõ ràng** (copy).

### TẦNG 2 · Kiến thức chuẩn

#### 1.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Object: identity, type, value** | Mọi object có *identity* (không đổi; `id()`, so sánh bằng `is`), *type* (quyết định các phép toán được phép), và *value* | Python Language Reference §3.1 |
| **Name binding** | Câu lệnh gán gắn (bind) một tên với một object | Language Reference §4.2 |
| **Mutable / immutable** | Value có đổi được sau khi tạo hay không | Language Reference §3.1 |
| **Aliasing** | Nhiều tên cùng trỏ tới một object | — |
| **Call by sharing** (Python FAQ gọi là *pass by assignment*) | Hàm nhận **tham chiếu tới cùng object**. Sửa object bên trong hàm thì bên ngoài thấy; gán lại tên bên trong hàm thì bên ngoài không thấy | Python FAQ, *Programming* |
| **Hashable** | Có `__hash__` không đổi trong suốt vòng đời và có `__eq__`. Các kiểu immutable có sẵn đều hashable | Python Glossary |
| **Shallow / deep copy** | `copy.copy` chỉ sao chép lớp ngoài cùng; `copy.deepcopy` sao chép đệ quy cả các object con | Thư viện `copy` |
| **Reference counting + GC** | CPython giải phóng object khi số tham chiếu tới nó về 0; bộ thu gom vòng (`gc`) xử lý các tham chiếu vòng | Thư viện `gc` (chi tiết cài đặt của CPython) |

#### 1.2 Cơ chế

- **`is` và `==`.** `is` so sánh *identity* (có phải cùng một object không). `==` so sánh *value* (gọi `__eq__`). PEP 8 quy định chỉ dùng `is` cho singleton như `None`. Việc `256 is 256` cho ra `True` là do CPython cache sẵn các số nguyên nhỏ; đây là **chi tiết cài đặt**, không được dựa vào.
- **`+=` phụ thuộc vào kiểu.**
  - Với `list`, `+=` gọi `__iadd__` và **sửa tại chỗ**, vẫn là object cũ.
  - Với `int`, `str`, `tuple`, `+=` **tạo object mới** rồi gán lại tên.
- **Truyền tham số (call by sharing):**

```python
def add_item(lst): lst.append("x")     # sửa object → bên ngoài thấy
def rebind(lst):   lst = ["new"]       # gán lại tên cục bộ → bên ngoài không thấy
l1 = []; add_item(l1); rebind(l1)
print(l1)   # ['x']
```

#### 1.3 Các con số cần biết

Đo bằng `sys.getsizeof` và `tracemalloc` trên CPython 3.14, 64-bit:

| Thứ | Kích thước |
|---|---|
| một `int` (ví dụ 1000) | 28 B |
| một `float` | 24 B |
| list rỗng | 56 B |
| mỗi phần tử của list | 8 B, vì list chỉ chứa **con trỏ** tới object |
| list 1 triệu float (con trỏ + các object float) | **≈ 32 MB** |
| mảng NumPy `float64` 1 triệu phần tử | **8 MB** |

Ta tốn gấp khoảng 4 lần bộ nhớ khi dùng list, vì mỗi số là một object riêng có header đi kèm. Đây là lý do số liệu lớn phải dùng NumPy (bài 05).

#### 1.4 Code chuẩn

```python
import copy

def append_to(item, bucket=None):        # mặc định None, không dùng []
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket

print(append_to(1), append_to(2))        # [1] [2]

cfg = {"model": "MODEL_ID", "params": {"temperature": 0.2}}
shallow = copy.copy(cfg)
deep = copy.deepcopy(cfg)
shallow["params"]["temperature"] = 0.9
print(cfg["params"]["temperature"])      # 0.9 ← shallow copy dùng chung dict con
print(deep["params"]["temperature"])     # 0.2
```

**Lý do thiết kế.** Giá trị mặc định của tham số được tính **một lần, lúc định nghĩa hàm** (Python Tutorial §4.9.1). Vì vậy dùng `None` làm giá trị canh (sentinel) và tạo list mới bên trong hàm.

#### 1.5 Cạm bẫy

1. **Mutable default** (`def f(x, items=[])`): mọi lời gọi dùng chung một list.
2. **`[[0] * 3] * 2`**: hai hàng là **cùng một list**. Dùng `[[0] * 3 for _ in range(2)]`.
3. **Xóa phần tử trong lúc duyệt chính list đó.** `[1, 2, 2, 3]` sau khi xóa các số `2` lại thành `[1, 2, 3]`, vì phần tử bị nhảy cóc. Hãy duyệt trên một bản sao, hoặc tạo list mới bằng comprehension.
4. **Shallow copy một config lồng nhau** rồi sửa: config gốc bị sửa theo.
5. **Dùng `is` để so sánh số hoặc chuỗi.** Kết quả lúc đúng lúc sai tùy chi tiết cài đặt.

**Sắc thái hay bị hỏi khi phỏng vấn:** `t = ([],)` rồi `t[0] += [1]` sẽ **raise `TypeError`**, nhưng sau đó `t` đã thành `([1],)`. Lý do: `+=` làm hai bước. Bước 1 là `list.__iadd__`, sửa list tại chỗ và thành công. Bước 2 là gán lại vào `t[0]`, bị cấm vì tuple là immutable.

#### 1.6 Đọc thêm

- Python Language Reference §3 *Data model*.
- Ned Batchelder, *Facts and myths about Python names and values* (PyCon 2015).

---

## 02 · Function, closure, decorator

> ⏱ Tầng 1: 10' · Tầng 2: 25'

> **📖 Từ mới**
> - **First-class function:** hàm cũng là một object: gán được cho biến, truyền được vào hàm khác, trả về được từ hàm khác.
> - **Closure:** hàm "mang theo" các biến của nơi nó được tạo ra.
> - **Decorator:** hàm nhận một hàm và trả về một hàm mới đã được "bọc" thêm hành vi.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- **Decorator giống như bọc quà.** Món quà (hàm gốc) giữ nguyên; lớp giấy bọc thêm việc trước và sau, ví dụ bấm giờ, ghi log, thử lại khi lỗi. Ta không phải sửa vào ruột hàm.
- **Closure giống chiếc ba lô.** Hàm con được tạo ra bên trong hàm mẹ sẽ mang theo các biến của hàm mẹ, kể cả khi hàm mẹ đã chạy xong.

```python
def make_counter():
    count = 0
    def inc():
        nonlocal count      # "count là biến trong ba lô, không phải biến mới"
        count += 1
        return count
    return inc

c = make_counter(); c()
print(c())   # 2   ← count vẫn sống sau khi make_counter đã kết thúc
```

**Ba ý cốt lõi:**
1. Hàm là object, nên có thể truyền hàm vào hàm khác.
2. Closure giữ lại biến của nơi nó được sinh ra.
3. `@deco` đặt trên `def f` chỉ là cách viết tắt của `f = deco(f)`.

### TẦNG 2 · Kiến thức chuẩn

#### 2.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Positional-only `/`** | Các tham số đứng trước `/` chỉ truyền được theo vị trí | PEP 570 |
| **Keyword-only `*`** | Các tham số đứng sau `*` (hoặc sau `*args`) bắt buộc truyền theo tên | PEP 3102 |
| **`*args`, `**kwargs`** | Gom các đối số vị trí thừa thành tuple; gom các đối số tên thừa thành dict | Language Reference §8.7 |
| **LEGB** | Thứ tự tra tên: Local → Enclosing → Global → Built-in | Language Reference §4.2.2 |
| **Free variable / cell** | Biến của hàm bao ngoài mà hàm con dùng tới; được lưu trong *cell* để vẫn sống sau khi hàm ngoài kết thúc | Language Reference §4.2 |
| **`nonlocal`** | Khai báo rằng một tên thuộc scope bao ngoài, cho phép gán lại nó | PEP 3104 |
| **Decorator** | `@d` đặt trên `def f` tương đương `f = d(f)`, thực thi **ngay lúc định nghĩa hàm** | PEP 318 |
| **`functools.wraps`** | Sao chép `__name__`, `__doc__`, … từ hàm gốc sang wrapper và thêm thuộc tính `__wrapped__` | Thư viện `functools` |

#### 2.2 Cơ chế

**Kiểu tham số:**

```python
def f(a, b, /, c, *, d): ...
f(1, 2, c=3, d=4)        # OK
f(a=1, b=2, c=3, d=4)    # TypeError: a, b là positional-only
```

Các SDK AI hay dùng keyword-only cho tham số cấu hình, ví dụ `client.chat(model=..., messages=...)`. Lý do: tránh truyền nhầm thứ tự khi có rất nhiều tham số.

**Late binding.** Closure tra giá trị của biến **lúc hàm con chạy**, không phải lúc hàm con được tạo. `[lambda: i for i in range(3)]` vì vậy cho ra toàn `2`. Cách sửa: `lambda i=i: i`, hoặc dùng `functools.partial`.

**Ba dạng decorator:**

```python
import functools, time

# (1) Decorator cơ bản
def timed(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:                                   # vẫn in thời gian kể cả khi fn raise lỗi
            print(f"{fn.__name__} took {time.perf_counter() - start:.3f}s")
    return wrapper

# (2) Decorator factory: decorator có tham số → thêm một lớp hàm bọc ngoài
def log_calls(prefix):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            print(f"{prefix} {fn.__name__}{args}")
            return fn(*args, **kwargs)
        return wrapper
    return decorator

@log_calls("[CALL]")
def mul(a, b): return a * b
mul(3, 4)            # in ra: [CALL] mul(3, 4)

# (3) Decorator đăng ký (registry): chạy lúc import, trả nguyên hàm gốc
TOOLS = []
def register(fn):
    TOOLS.append(fn.__name__)
    return fn
```

Dạng (3) chính là cách nhiều framework agent đăng ký "tool" cho LLM, ví dụ các decorator kiểu `@tool` (Tuần 7, 10).

#### 2.3 Decorator cho hàm async: khác ở đâu

Gọi một hàm `async def` **chưa chạy gì cả**. Nó chỉ trả về một object *coroutine* (bài 08). Vì vậy dùng `timed` ở trên cho một hàm async sẽ đo sai:

```python
@timed
async def call_api():
    await asyncio.sleep(0.05)
asyncio.run(call_api())     # in ra "call_api took 0.000s"  ← SAI: chỉ đo lúc tạo coroutine
```

Wrapper đúng cho hàm async phải là `async def wrapper` và `await fn(...)` bên trong. Để một decorator dùng được cho cả hai loại hàm, kiểm tra bằng `inspect.iscoroutinefunction(fn)` rồi trả về wrapper tương ứng. Đây là bài tập hôm nay.

#### 2.4 Cạm bẫy

1. **Quên `functools.wraps`:** mọi hàm sau khi bọc đều tên là `wrapper`, làm log, debug và các framework dựa vào tên hàm bị sai.
2. **Decorator đồng bộ bọc hàm async:** đo sai, hoặc retry không bắt được lỗi, vì lỗi chỉ xảy ra lúc `await`.
3. **`UnboundLocalError`:** viết `total += 1` trong hàm con mà không khai báo `nonlocal total`. Python coi `total` là biến cục bộ chưa có giá trị.
4. **Late binding trong vòng lặp** khi tạo callback.
5. **Decorator làm việc nặng lúc import** (ví dụ mở kết nối): làm chậm việc khởi động và khó test.

**Sắc thái hay bị hỏi khi phỏng vấn:** decorator chạy **lúc định nghĩa hàm**, tức lúc import module, chứ không phải lúc gọi hàm. Câu hỏi hay gặp: *"Đoạn code này in gì, và in vào lúc nào?"*

#### 2.5 Đọc thêm

- PEP 318 (decorators), PEP 3104 (`nonlocal`), PEP 570 (positional-only).
- Tài liệu thư viện `functools`.

---

## 03 · OOP & dataclass

> ⏱ Tầng 1: 10' · Tầng 2: 30'

> **📖 Từ mới**
> - **Class / instance:** khuôn / sản phẩm được đúc từ khuôn.
> - **Method:** hàm gắn với class. `self` là instance đang gọi method đó.
> - **Dunder method:** method có tên dạng `__xxx__`, để Python tự gọi khi cần. Ví dụ `len(obj)` sẽ gọi `obj.__len__()`.
> - **Interface:** "hợp đồng" quy định một class phải có những method nào.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- **Class là khuôn bánh, instance là chiếc bánh.**
- **Abstract base class (ABC) là bản hợp đồng.** Ví dụ: "mọi nhà cung cấp LLM phải có method `complete`". Ai ký hợp đồng (kế thừa ABC) mà thiếu method đó thì không đúc ra được instance nào.
- **Dataclass là thư ký.** Bạn chỉ cần khai báo các trường dữ liệu; thư ký tự viết `__init__`, `__repr__`, `__eq__` cho bạn.

**Ba ý cốt lõi:**
1. Class gom **dữ liệu** và **hành vi** lại một chỗ.
2. Dùng **interface** (ABC hoặc Protocol) để phần còn lại của hệ thống không phụ thuộc vào một nhà cung cấp cụ thể. Gateway ở Tuần 6 đứng vững được là nhờ điều này.
3. Dữ liệu thuần (kết quả gọi API, một tin nhắn) thì dùng **dataclass**.

### TẦNG 2 · Kiến thức chuẩn

#### 3.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Class / instance attribute** | Thuộc tính của class được **mọi instance dùng chung**; thuộc tính của instance nằm trong `obj.__dict__` của riêng instance đó | Python Tutorial §9.3 |
| **MRO** (Method Resolution Order) | Thứ tự Python tìm method khi có đa kế thừa; tính bằng thuật toán **C3 linearization**; xem bằng `Cls.__mro__` | *The Python 2.3 MRO* (docs.python.org) |
| **`super()`** | Gọi method của lớp **kế tiếp trong MRO**, không nhất thiết là lớp cha trực tiếp | Thư viện built-in |
| **Special (dunder) methods** | Các hook để Python gọi: `__repr__`, `__eq__`, `__hash__`, `__len__`, `__iter__`, `__getitem__`, … | Language Reference §3.3 |
| **`@property` / `@classmethod` / `@staticmethod`** | Thuộc tính tính toán / constructor thay thế (`from_dict`) / hàm tiện ích đặt trong namespace của class | Thư viện built-in |
| **ABC** | Kiểu danh nghĩa (*nominal typing*): class phải **kế thừa** ABC và cài đủ mọi `@abstractmethod` thì mới tạo instance được | PEP 3119, thư viện `abc` |
| **Protocol** | Kiểu cấu trúc (*structural typing*, "duck typing có kiểm tra"): class chỉ cần **có đúng method** là được, không cần kế thừa | PEP 544 |
| **Dataclass** | Decorator tự sinh `__init__`, `__repr__`, `__eq__` từ các trường đã khai báo | PEP 557 |

#### 3.2 Cơ chế

**Interface cho nhà cung cấp LLM.** Đây là mẫu sẽ dùng lại suốt khóa:

```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def complete(self, prompt: str) -> str: ...

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r})"

class FakeProvider(LLMProvider):
    name = "fake"
    async def complete(self, prompt: str) -> str:
        return prompt.upper()

LLMProvider()   # TypeError: Can't instantiate abstract class LLMProvider
                #            without an implementation for abstract method 'complete'
```

**ABC hay Protocol?**
- **ABC:** dùng khi bạn **sở hữu** cả hierarchy và muốn bị chặn lỗi ngay lúc tạo object.
- **Protocol:** dùng khi muốn nhận **bất kỳ object nào có đúng method**, kể cả class của thư viện khác không kế thừa gì từ bạn. Protocol được kiểm tra bởi mypy/pyright, **không kiểm tra lúc chạy**, trừ khi dùng `@runtime_checkable`.

**Dataclass:**

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

@dataclass
class CallResult:
    provider: str
    text: str = ""
    latency_s: float = 0.0
    errors: list[str] = field(default_factory=list)   # mỗi instance có list riêng

    @property
    def ok(self) -> bool:
        return not self.errors

    @classmethod
    def from_dict(cls, d: dict) -> "CallResult":
        return cls(**d)
```

**Quy tắc về hash của dataclass:**
- `eq=True` (mặc định) mà không có `frozen=True` thì Python đặt `__hash__ = None`, nên instance **không hashable** (`unhashable type`).
- Với `frozen=True`, instance hashable và gán lại trường sẽ raise `FrozenInstanceError`.
- Lý do: object đổi được giá trị mà vẫn hashable thì có thể "biến mất" khỏi dict hoặc set, vì hash của nó không còn khớp với vị trí lưu.

**`slots=True`.** Đo với 100 000 instance có 2 trường: không dùng slots **12.8 MB**, dùng `@dataclass(slots=True)` còn **8.8 MB**, giảm khoảng 30%. Lý do: không còn `__dict__` riêng cho từng instance.

**Dataclass không kiểm tra kiểu dữ liệu.** `CallResult(provider=123)` vẫn chạy bình thường. Dữ liệu đi vào từ bên ngoài (JSON từ API, input của user) thì cần **pydantic** để validate (Tuần 6). Dữ liệu nội bộ thì dùng dataclass.

**MRO và `super()`:**

```python
class A:
    def hi(self): return "A"
class B(A):
    def hi(self): return "B>" + super().hi()
class C(A):
    def hi(self): return "C>" + super().hi()
class D(B, C):
    def hi(self): return "D>" + super().hi()

print([k.__name__ for k in D.__mro__])   # ['D', 'B', 'C', 'A', 'object']
print(D().hi())                          # D>B>C>A   ← super() trong B gọi C, không phải A
```

#### 3.3 Cạm bẫy

1. **Class attribute là mutable** (`history = []` ở cấp class): mọi instance dùng chung một list.
2. **Tự định nghĩa `__eq__` mà quên `__hash__`:** object trở nên unhashable.
3. **Quên gọi `super().__init__()`** trong lớp con.
4. **Kế thừa quá sâu.** Hãy ưu tiên *composition*: object chứa object khác (GoF, *Design Patterns*).
5. **Dùng dataclass cho dữ liệu từ bên ngoài** mà tưởng nó đã được validate.

**Sắc thái hay bị hỏi khi phỏng vấn:** vì sao `super()` trong `B` lại gọi tới `C`? Vì `super()` đi theo **MRO của instance thực tế** (`D`), chứ không theo lớp cha được viết trong code của `B`.

#### 3.4 Đọc thêm

- PEP 557 (dataclasses), PEP 544 (Protocol), PEP 3119 (ABC).
- *The Python 2.3 Method Resolution Order* (docs.python.org).

---

## 04 · Iterator & generator

> ⏱ Tầng 1: 10' · Tầng 2: 25'

> **📖 Từ mới**
> - **Iterable:** thứ duyệt được bằng `for`, như list, chuỗi, file, dict.
> - **Iterator:** "con trỏ" đang duyệt dở; mỗi lần gọi `next()` nhả ra một phần tử.
> - **Generator:** cách viết iterator nhanh nhất, bằng một hàm có `yield`.
> - **Lazy:** chỉ tính khi được hỏi tới.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- **List giống như đổ đầy một bể nước** rồi mới dùng: tốn chỗ chứa toàn bộ.
- **Generator giống như vòi nước:** vặn tới đâu chảy tới đó, không cần bể chứa.
- Đổi lại, vòi chỉ chảy **một lần**. Nước đã chảy qua thì không lấy lại được.

```python
squares = (x * x for x in range(4))
print(sum(squares), sum(squares))   # 14 0   ← lần thứ hai vòi đã cạn
```

**Ba ý cốt lõi:**
1. Generator **không giữ toàn bộ dữ liệu**, nên đọc được file 10 GB mà RAM không tăng.
2. Generator **chỉ duyệt được một lần**.
3. Stream câu trả lời của LLM chính là một generator: từng token đến dần, và user thấy chữ đầu tiên ngay (Ngày 3).

### TẦNG 2 · Kiến thức chuẩn

#### 4.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Iterator protocol** | Iterable có `__iter__()` trả về một iterator; iterator có `__next__()` trả về phần tử tiếp theo và raise `StopIteration` khi hết | Library Reference, *Iterator Types* |
| **Generator function** | Hàm có `yield`; khi gọi, nó trả về một *generator object* và chưa chạy dòng code nào | PEP 255 |
| **Generator expression** | `(expr for x in it)`, phiên bản lazy của list comprehension | PEP 289 |
| **`yield from`** | Ủy quyền cho một iterable hoặc generator con | PEP 380 |
| **Async generator** | `async def` có `yield`, được duyệt bằng `async for` | PEP 525 |
| **`itertools`** | `islice`, `chain`, `count`, `batched` (từ 3.12), … | Thư viện `itertools` |

#### 4.2 Cơ chế

Vòng `for` thực chất là:

```python
it = iter(iterable)           # gọi __iter__
while True:
    try:
        x = next(it)          # gọi __next__
    except StopIteration:
        break
    ...                       # thân vòng lặp
```

Generator **tạm dừng tại `yield`** và giữ nguyên toàn bộ biến cục bộ. Lần `next()` tiếp theo chạy tiếp từ đúng chỗ đó. Lệnh `return value` bên trong generator sẽ kết thúc nó, và `value` nằm trong `StopIteration.value`.

**Vòng đời và dọn dẹp.** `gen.close()` ném `GeneratorExit` vào chỗ generator đang dừng, nên khối `finally` hoặc `with` bên trong vẫn chạy. Nhưng nếu bạn bỏ dở một generator mà **không** close, file mở bên trong nó sẽ **chưa được đóng** cho tới khi generator bị garbage collect.

#### 4.3 Các con số cần biết

| | Kích thước (đo bằng `sys.getsizeof`) |
|---|---|
| `[x * x for x in range(1_000_000)]` | **≈ 8.4 MB**, và tăng theo kích thước dữ liệu |
| `(x * x for x in range(1_000_000))` | **208 B**, cố định bất kể dữ liệu lớn cỡ nào |

#### 4.4 Code chuẩn: pipeline lazy

```python
import json, logging

log = logging.getLogger(__name__)

def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("skip bad line %d", lineno)

def batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch        # đừng quên lô cuối chưa đầy

for batch in batched(read_jsonl("prompts.jsonl"), 32):
    ...                    # tại mỗi thời điểm, RAM chỉ chứa đúng 1 lô
```

**Lý do thiết kế.** Mỗi tầng của pipeline (đọc → lọc → chia lô) là một generator, nên dữ liệu chảy qua **từng phần tử một**. Bộ nhớ chỉ phụ thuộc vào kích thước một lô, không phụ thuộc kích thước file. Python 3.12 có sẵn `itertools.batched`, trả về các tuple. Tự viết hàm này là để hiểu cơ chế.

#### 4.5 Cạm bẫy

1. **Duyệt generator lần thứ hai:** không có gì, cũng không có lỗi nào báo cho bạn biết.
2. **Gọi `len()` trên generator:** `TypeError`.
3. **Vô tình "đổ đầy bể"** bằng `list(gen)`, `sorted(gen)` hoặc in thử ra để debug, làm mất ưu điểm lazy.
4. **Quên `yield` lô cuối** trong `batched`.
5. **Generator mở file bị bỏ dở:** file chưa được đóng.

**Sắc thái hay bị hỏi khi phỏng vấn:** *"Vì sao stream giảm độ trễ mà user cảm nhận?"* Vì user thấy token đầu tiên sau **TTFT** (time to first token) thay vì phải chờ toàn bộ câu trả lời. Tổng thời gian sinh không đổi, nhưng thời gian chờ cảm nhận giảm mạnh.

#### 4.6 Đọc thêm

- PEP 255, 289, 380, 525.
- David Beazley, *Generator Tricks for Systems Programmers* (PyCon 2008).

---

## Bài tập hôm nay → `project/multi_caller/utils.py`

Viết 3 hàm sau. Chúng sẽ được dùng lại trong project ngày 4.

1. **`timed`** — decorator dùng được cho **cả hàm thường lẫn hàm async**.
   - Ghi log ở mức `INFO` qua logger `multi_caller`, theo dạng `"<tên hàm> took <x.xxx>s"`.
   - Vẫn ghi log khi hàm raise lỗi.
   - Giữ nguyên metadata của hàm gốc.
2. **`read_jsonl(path)`** — generator lazy.
   - Bỏ qua dòng trống.
   - Bỏ qua dòng JSON hỏng và ghi log `WARNING`.
3. **`batched(iterable, n)`** — generator trả về các **list** dài tối đa `n` phần tử.
   - Không dùng `itertools.batched`.
   - `n < 1` thì raise `ValueError`.

**Tự kiểm tra:** đứng ở thư mục `project/`, chạy `python -m pytest -q tests/test_utils.py`. Lời giải tham khảo nằm ở `project/_reference/`; chỉ mở khi đã làm xong, hoặc kẹt quá 1 giờ.

👉 Sau khi đọc xong, làm [ngay-1.quiz.md](ngay-1.quiz.md).
