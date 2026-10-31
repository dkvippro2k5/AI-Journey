# Ngày 1 · PyTorch và MLP: tensor, autograd, vòng lặp huấn luyện, mạng nhiều lớp

> **Loại tài liệu: Lý thuyết (bản nhẹ).** Công thức đầy đủ và ví dụ tính tay. Chứng minh chỉ còn phác thảo vài dòng ("vì sao").
> **Hôm nay gồm 3 bài:**
> - **01** Tensor và autograd
> - **02** `nn.Module`, loss, optimizer và vòng lặp huấn luyện chuẩn
> - **03** MLP: neuron, activation, đếm tham số
>
> **Thời gian:** đọc ~2h30 · quiz 45' · bài tập ~2h
> **Dữ liệu xuyên suốt tuần:** vẫn là log chatbot của Tuần 3 (`qnet/data.py` giống hệt `quality/data.py`), nhãn `bad = 1` nếu người dùng bấm 👎 (khoảng 15.5%). Tuần này ta thay HistGradientBoosting bằng **mạng neural** và xem nó thắng hay thua. Mọi con số "đo trên máy bạn" chạy bằng PyTorch 2.14 (CPU, không GPU).

---

## 01 · Tensor và autograd

> ⏱ Tầng 1: 10' · Tầng 2: 35'
> **Cần biết trước:** NumPy và broadcasting (Tuần 1, bài 05); chain rule và backpropagation (Tuần 2, bài 05).

> **📖 Từ mới**
> - **Tensor:** mảng nhiều chiều, giống `np.ndarray`, nhưng chạy được trên GPU và **tự ghi nhớ** nó được tính ra từ đâu.
> - **Shape / dtype / device:** kích thước các chiều / kiểu số (`float32`, `int64`, …) / nơi chứa dữ liệu (`cpu`, `cuda`).
> - **Autograd:** bộ phận của PyTorch tự tính gradient bằng cách ghi lại mọi phép toán rồi chạy chain rule ngược lại.
> - **Đồ thị tính toán (computational graph):** các phép toán đã thực hiện, nối với nhau theo luồng dữ liệu (Tuần 2, bài 05).
> - **Leaf tensor:** tensor do bạn tạo trực tiếp (thường là tham số), không phải kết quả của phép toán nào. Gradient chỉ được **lưu lại** ở leaf.
> - **`requires_grad`:** cờ báo "hãy theo dõi tensor này để tính gradient".

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Một kế toán ghi **sổ nhật ký** mọi giao dịch: "lấy A nhân 2", "cộng với B". Cuối tháng, giám đốc hỏi "nếu A tăng 1 đồng thì lợi nhuận đổi bao nhiêu?". Kế toán chỉ cần **lật sổ từ dưới lên**, nhân các hệ số lại. Autograd chính là người kế toán đó:
- **Forward:** bạn tính bình thường; PyTorch lặng lẽ ghi sổ (mỗi tensor kết quả có `grad_fn` trỏ về phép toán đã tạo ra nó).
- **`loss.backward()`:** PyTorch lật sổ ngược, áp chain rule, và **cộng** gradient vào `.grad` của từng leaf.

**Ví dụ có số.** Một mạng tí hon: $h = \mathrm{ReLU}(w_1 x + b_1)$, $\hat y = w_2 h + b_2$, $L = (\hat y - y)^2$, với $x = 2$, $y = 1$, $w_1 = 0.5$, $b_1 = -0.5$, $w_2 = 3$, $b_2 = 0$.
- Forward: $w_1 x + b_1 = 0.5$, nên $h = 0.5$; $\hat y = 1.5$; $L = 0.25$.
- Backward: $\partial L/\partial \hat y = 2(\hat y - y) = 1$. Từ đó $\partial L/\partial w_2 = 1 \cdot h = 0.5$, $\partial L/\partial b_2 = 1$, $\partial L/\partial h = w_2 = 3$. ReLU đang "mở" (đầu vào $0.5 > 0$) nên cho gradient đi qua nguyên vẹn: $\partial L/\partial w_1 = 3 \cdot x = 6$, $\partial L/\partial b_1 = 3$.

`torch.autograd` cho đúng $(6,\ 3,\ 0.5,\ 1)$ (đã chạy kiểm chứng).

**Ba ý cốt lõi:**
1. Tensor = mảng NumPy + GPU + **lịch sử phép tính**.
2. `backward()` tính gradient của **một số vô hướng** (loss) theo **mọi** leaf có `requires_grad=True`, chỉ trong một lần đi ngược.
3. Gradient được **cộng dồn** vào `.grad`, không ghi đè. Vì vậy mỗi bước huấn luyện phải xóa gradient cũ.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 1.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Tensor** | Mảng $n$ chiều có `shape`, `dtype`, `device`, và **strides** (số phần tử phải nhảy để sang chỉ số kế tiếp ở mỗi chiều) | PyTorch docs, *Tensor Attributes* |
| **View** | Tensor mới **dùng chung bộ nhớ** với tensor gốc, chỉ khác shape/strides (`view`, `t()`, slicing). Sửa view là sửa gốc | PyTorch docs, *Tensor Views* |
| **Contiguous** | Các phần tử nằm liền nhau theo thứ tự hàng. `t()` cho ra tensor **không** contiguous, nên `view` sẽ lỗi; `reshape` thì tự sao chép khi cần | — |
| **Define-by-run (dynamic graph)** | Đồ thị được dựng **trong lúc** code Python chạy, nên `if`/`for` bình thường đều dùng được | Paszke et al., *PyTorch*, NeurIPS 2019 |
| **Reverse-mode AD** | Một lần đi ngược cho gradient của một đầu ra vô hướng theo mọi đầu vào; về bản chất là chuỗi tích **vector–Jacobian** | Baydin et al., JMLR 2018 (Tuần 2, bài 05) |
| **Leaf / `grad_fn`** | Leaf: `grad_fn is None`, gradient được lưu vào `.grad`. Tensor trung gian có `grad_fn` (ví dụ `MulBackward0`), gradient của nó không được giữ lại | PyTorch docs, *Autograd mechanics* |
| **`torch.no_grad()` / `inference_mode()`** | Tắt việc ghi đồ thị: kết quả có `requires_grad=False`, tiết kiệm bộ nhớ và thời gian. `inference_mode` còn chặt hơn (tensor tạo ra không bao giờ dùng lại được cho autograd) | PyTorch docs |
| **`detach()`** | Trả về tensor **dùng chung dữ liệu** nhưng tách khỏi đồ thị | — |

#### 1.2 Cơ chế

**Cộng dồn gradient.** Gọi `(3 * w).backward()` ba lần liên tiếp thì `w.grad` bằng **9**, không phải 3 (đã chạy kiểm chứng). Lý do thiết kế: khi một tham số được dùng ở nhiều chỗ, gradient từ các nhánh phải **cộng** lại (Tuần 2, Mệnh đề 5.1); PyTorch làm việc cộng đó ngay trên `.grad`. Hệ quả: dùng được **gradient accumulation** (bài 06), nhưng bạn phải tự `zero_grad()` mỗi bước.

**Chỉ backward được từ một số vô hướng.** `y.backward()` với `y` có 2 phần tử báo lỗi `grad can be implicitly created only for scalar outputs`. Muốn backward từ vector $y$ thì phải truyền vào vector $v$, và PyTorch sẽ tính $v^\top J$. Trong huấn luyện, loss luôn là trung bình nên luôn là vô hướng.

**Bộ nhớ của autograd.** Mỗi phép toán giữ lại những gì nó cần cho backward (ví dụ ReLU giữ mặt nạ vị trí nào dương). Đó là **activation memory**: tỉ lệ với batch size × độ sâu, và thường chiếm phần lớn bộ nhớ khi huấn luyện (Tuần 11: gradient checkpointing).

#### 1.3 Ví dụ tính tay: bảng forward/backward

Dùng lại mạng tí hon ở tầng 1.

| Nút | Forward | Local gradient | Gradient của $L$ (upstream × local) |
|---|---|---|---|
| $L = (\hat y - y)^2$ | $0.25$ | $2(\hat y - y) = 1$ | $\partial L/\partial \hat y = 1$ |
| $\hat y = w_2 h + b_2$ | $1.5$ | $\partial \hat y/\partial w_2 = h$, $\partial \hat y/\partial h = w_2$ | $\partial L/\partial w_2 = 0.5$, $\partial L/\partial b_2 = 1$, $\partial L/\partial h = 3$ |
| $h = \mathrm{ReLU}(a)$ | $0.5$ | $\mathbb 1[a > 0] = 1$ | $\partial L/\partial a = 3$ |
| $a = w_1 x + b_1$ | $0.5$ | $\partial a/\partial w_1 = x = 2$ | $\partial L/\partial w_1 = 6$, $\partial L/\partial b_1 = 3$ |

**Nếu đổi $b_1 = -1.5$:** $a = -0.5 < 0$, nên $h = 0$ và ReLU "đóng". Khi đó gradient theo $w_1$ và $b_1$ đều bằng **0**: hai tham số này không học được gì từ mẫu này. Hiện tượng **dead ReLU** (bài 03) bắt đầu từ đây.

#### 1.4 Các con số cần biết

**Kiểu số** (đã chạy kiểm chứng bằng `torch.finfo`):

| dtype | Byte/phần tử | Giá trị lớn nhất | Độ chính xác tương đối (eps) | Dùng khi |
|---|---|---|---|---|
| `float64` | 8 | $1.8 \times 10^{308}$ | $2.2 \times 10^{-16}$ | Kiểm tra gradient, khoa học |
| `float32` | 4 | $3.4 \times 10^{38}$ | $1.2 \times 10^{-7}$ | **Mặc định** khi huấn luyện |
| `bfloat16` | 2 | $3.4 \times 10^{38}$ | $7.8 \times 10^{-3}$ | Huấn luyện/suy luận LLM trên GPU đời mới: cùng **dải** giá trị với fp32, kém chính xác hơn |
| `float16` | 2 | **65 504** | $9.8 \times 10^{-4}$ | GPU cũ (T4, V100); dễ tràn số nên cần *loss scaling* (bài 06) |

**Ước lượng nhanh:** một model 7 tỉ tham số chiếm $7 \times 10^9 \times 2$ byte $= 14$ GB ở bf16, và 28 GB ở fp32, mới chỉ tính trọng số.

**Tốc độ trên máy bạn** (CPU 22 core, **cắm sạc**; khi chạy pin, cùng phép nhân 2048×2048 fp32 chỉ đạt 33 GFLOP/s):

| Phép toán | Thời gian | Thông lượng |
|---|---|---|
| Nhân ma trận 512×512 bằng vòng lặp Python (mỗi phần tử một tích vô hướng NumPy) | ~0.67 s | — |
| Cùng phép đó bằng `A @ B` | **1.06 ms** | 253 GFLOP/s |
| `A @ B` 2048×2048, float32 | 35.5 ms | 484 GFLOP/s |
| `A @ B` 2048×2048, float64 | 74.6 ms | 230 GFLOP/s |
| `A @ B` 2048×2048, **bfloat16** | **30.3 s** | ~1 GFLOP/s |

Hai bài học từ bảng này:
- **Vectorize** nhanh hơn vòng lặp khoảng 600 lần (điều đã thấy ở Tuần 1, bài 05).
- **bf16 trên CPU này chậm hơn fp32 khoảng 850 lần**, vì CPU không có lệnh phần cứng cho bf16 nên phải giả lập. "Dùng số ít bit cho nhanh" chỉ đúng khi **phần cứng hỗ trợ** (GPU Ampere trở lên, hoặc CPU có AMX).

FLOP của phép nhân ma trận $(m \times k)(k \times n)$ là $2mkn$: mỗi phần tử kết quả cần $k$ phép nhân và $k$ phép cộng.

#### 1.5 Code chuẩn

```python
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"        # máy bạn: "cpu"; Kaggle: "cuda"

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])                      # float32 mặc định
a = torch.from_numpy(np.ones(3))                                # GIỮ dtype float64 của NumPy, DÙNG CHUNG bộ nhớ
a = a.float().to(device)                                        # đổi dtype và chuyển thiết bị (tạo bản sao)

w = torch.tensor(0.5, requires_grad=True)                       # leaf: tham số
loss = (w * 2 - 3) ** 2
loss.backward()                                                 # w.grad = 2·(2w - 3)·2 = -8
print(w.grad)                                                   # tensor(-8.)

with torch.no_grad():                                           # cập nhật tham số KHÔNG được ghi vào đồ thị
    w -= 0.1 * w.grad
w.grad = None                                                   # xóa gradient trước bước sau

value = loss.item()                                             # lấy số Python để log, ngắt khỏi đồ thị
arr = w.detach().cpu().numpy()                                  # sang NumPy: detach → cpu → numpy
```

#### 1.6 Cạm bẫy

Mọi thông báo lỗi dưới đây đã chạy kiểm chứng.

1. **Trộn dtype.** `torch.from_numpy` giữ nguyên `float64`; đưa vào `nn.Linear` (float32) sẽ báo `mat1 and mat2 must have the same dtype, but got Double and Float`. Sửa bằng `.float()`, hoặc `np.asarray(..., dtype=np.float32)` ngay từ đầu.
2. **Cộng dồn loss tensor để log:** `total += loss` giữ lại **đồ thị của mọi batch**, khiến bộ nhớ tăng dần theo từng bước. Dùng `total += loss.item()`.
3. **Sửa tại chỗ (in-place) một tensor mà backward cần:** `b = a.sigmoid(); b.mul_(2)` rồi backward sẽ báo `one of the variables needed for gradient computation has been modified by an inplace operation`.
4. **`.numpy()` trên tensor đang có `requires_grad`** báo lỗi; phải `.detach()` trước. Tensor trên GPU còn phải `.cpu()`.
5. **`view` trên tensor không contiguous** (ví dụ sau `.t()`) báo lỗi; dùng `reshape`.
6. **`from_numpy` dùng chung bộ nhớ:** sửa mảng NumPy thì tensor cũng đổi theo.
7. **Tensor và model khác device:** `RuntimeError: Expected all tensors to be on the same device`. Dữ liệu và model đều phải `.to(device)`.

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"`detach()` khác `torch.no_grad()` thế nào?"* `detach` tách **một tensor** khỏi đồ thị; `no_grad` tắt việc ghi đồ thị cho **mọi phép toán** trong khối lệnh.
- *"Vì sao phải `zero_grad()`?"* Vì `.grad` cộng dồn (mục 1.2).
- *"Vì sao huấn luyện tốn bộ nhớ hơn suy luận nhiều?"* Phải giữ activation cho backward, cộng với gradient và trạng thái optimizer (bài 06).

#### 1.7 Đọc thêm

- PyTorch docs: *Autograd mechanics*; *Tensor Views*; tutorial *A Gentle Introduction to torch.autograd*.
- A. Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS 2019.
- A. Karpathy, *micrograd* (GitHub): một autograd vô hướng khoảng 100 dòng, đáng đọc trong 30 phút.

---

## 02 · `nn.Module`, loss, optimizer và vòng lặp huấn luyện chuẩn

> ⏱ Tầng 1: 10' · Tầng 2: 40'
> **Cần biết trước:** bài 01; SGD và Adam (Tuần 2, bài 06–07); BCE và cross-entropy (Tuần 2, bài 05 và 10).

> **📖 Từ mới**
> - **`nn.Module`:** lớp cơ sở cho mọi "khối" mạng. Nó tự gom tham số của các khối con.
> - **Parameter / buffer:** tensor **được học** (trọng số) / tensor **không được học** nhưng là một phần trạng thái của model (ví dụ running mean của BatchNorm, bài 05).
> - **`state_dict`:** dict tên → tensor chứa toàn bộ parameter và buffer. Đây là thứ được lưu thành checkpoint.
> - **Batch (mini-batch):** nhóm mẫu xử lý cùng lúc. **Step:** một lần cập nhật tham số. **Epoch:** một lượt đi qua toàn bộ tập train.
> - **`Dataset` / `DataLoader`:** "lấy mẫu thứ $i$" / "gom mẫu thành batch, xáo trộn, chạy song song".
> - **Learning rate scheduler:** quy tắc thay đổi learning rate theo thời gian.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Tập nấu một món theo công thức:
1. **Nấu thử** một nồi nhỏ (forward trên một batch).
2. **Nếm** và so với vị chuẩn (tính loss).
3. **Suy ra** nên thêm muối hay bớt đường, và bao nhiêu (backward).
4. **Điều chỉnh** công thức một chút (`optimizer.step()`).
5. **Rửa thìa nếm** trước lần nếm sau (`optimizer.zero_grad()`). Quên rửa thì vị lần trước lẫn vào lần sau.

Nấu hết một lượt nguyên liệu là một epoch. Thỉnh thoảng nhờ **người ngoài** nếm (validation), vì người nấu nếm mãi sẽ quen vị.

**Năm dòng mà mọi project PyTorch đều có:**

```python
for xb, yb in loader:
    optimizer.zero_grad()            # 5. rửa thìa
    loss = loss_fn(model(xb), yb)    # 1–2. nấu thử, nếm
    loss.backward()                  # 3. suy ra hướng sửa
    optimizer.step()                 # 4. sửa công thức
```

**Ba ý cốt lõi:**
1. Model là một `nn.Module`; loss nhận **logit**; optimizer giữ danh sách tham số và trạng thái riêng (ví dụ $m, v$ của Adam).
2. `model.train()` khi huấn luyện, `model.eval()` + `torch.no_grad()` khi đánh giá. Dropout và BatchNorm cư xử **khác nhau** ở hai chế độ.
3. Lưu **`state_dict`** của cả model **và** optimizer; dùng validation để biết lúc nào dừng.

---

### TẦNG 2 · Kiến thức chuẩn

#### 2.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **`nn.Module`** | Gán `self.lin = nn.Linear(...)` trong `__init__` là đủ để đăng ký module con; `parameters()` duyệt đệ quy. `forward` định nghĩa phép tính; **gọi `model(x)`**, đừng gọi `model.forward(x)` (để hook chạy) | PyTorch docs, *torch.nn* |
| **`register_buffer`** | Đăng ký tensor không học (nằm trong `state_dict`, đi theo `.to(device)`) | — |
| **`train()` / `eval()`** | Bật/tắt cờ `self.training` cho mọi module con. **Không** tắt autograd; cần thêm `no_grad` | — |
| **`BCEWithLogitsLoss`** | $\ell = \log(1 + e^{z}) - yz$, nhận **logit** $z$; ổn định số học hơn `sigmoid` + `BCELoss` | Tuần 2, mục 5.4 |
| **`CrossEntropyLoss`** | Nhận logit $(B, K)$ và nhãn **chỉ số lớp** $(B,)$; tự làm log-softmax bên trong | PyTorch docs |
| **Mini-batch SGD** | Gradient trên batch $B$ mẫu là ước lượng không chệch của gradient toàn tập, với phương sai tỉ lệ $\approx 1/B$ | Bottou et al., *SIAM Review* 2018 |
| **AdamW** | Adam với **weight decay tách rời** (bài 06) | Loshchilov & Hutter, ICLR 2019 |
| **Warmup + cosine decay** | Tăng LR tuyến tính trong vài trăm bước đầu, rồi giảm theo nửa chu kỳ cos. Mặc định cho Transformer | Goyal et al. 2017; Loshchilov & Hutter, ICLR 2017 (SGDR) |

#### 2.2 Công thức cần nhớ

**Số bước mỗi epoch** $= \lceil n / B \rceil$. Project: $n_\text{fit} = 3\,840$, $B = 128$, nên có **30 bước/epoch**.

**Cosine decay:** $\eta_t = \eta_{\min} + \tfrac12(\eta_0 - \eta_{\min})\big(1 + \cos(\pi t/T)\big)$. **Warmup tuyến tính** $W$ bước: $\eta_t = \eta_0 \cdot (t+1)/W$ (một biến thể phổ biến; PyTorch `LinearLR` dùng hệ số bắt đầu `start_factor`).

**Loss của "model hằng số":** đoán $p = \bar y$ cho mọi mẫu thì BCE $= -[\bar y\ln\bar y + (1 - \bar y)\ln(1 - \bar y)] = H(\bar y)$. Với $\bar y = 0.155$, ta được $\approx 0.432$ (đo trên tập val). Model nào có val loss **không** thấp hơn 0.432 thì chưa học được gì.

#### 2.3 Ví dụ tính tay: lịch learning rate

$\eta_0 = 0.1$ (đã chạy kiểm chứng bằng `torch.optim.lr_scheduler`):

| Bước | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `StepLR(step=3, γ=0.5)` | 0.1 | 0.1 | 0.1 | 0.05 | 0.05 | 0.05 | 0.025 | 0.025 | | | |
| `CosineAnnealingLR(T=10)` | 0.1 | 0.0976 | 0.0905 | 0.0794 | 0.0655 | **0.05** | 0.0345 | 0.0206 | 0.0095 | 0.0024 | 0 |
| Warmup 3 bước (`start_factor=0.25`) + cosine 7 bước | 0.025 | 0.05 | 0.075 | **0.1** | 0.0950 | 0.0812 | 0.0611 | 0.0389 | 0.0188 | 0.0050 | 0 |

Kiểm tra một ô: cosine tại $t = 5$, $T = 10$ cho $\tfrac12 \cdot 0.1 \cdot (1 + \cos(\pi/2)) = 0.05$.

**Vì sao cần warmup?** Ở những bước đầu, Adam ước lượng $v$ (bình phương gradient) từ rất ít mẫu nên còn nhiễu, và các tham số khởi tạo ngẫu nhiên cho gradient lớn và lộn xộn. Bước đi nhỏ lúc đầu giúp tránh "văng" ra vùng xấu. Với Transformer, thiếu warmup thường làm huấn luyện phân kỳ (Liu et al., *On the Variance of the Adaptive Learning Rate*, ICLR 2020).

#### 2.4 Đo trên máy bạn: các lỗi kinh điển của vòng lặp

Tất cả dùng `TabMLP` của project (2 lớp ẩn 64, AdamW, lr $10^{-3}$, batch 128).

**(a) Quên `zero_grad()`.** Val loss theo epoch:

| Epoch | 1 | 2 | 3 | 5 | 8 | 10 | 15 |
|---|---|---|---|---|---|---|---|
| Có `zero_grad` | 0.462 | 0.412 | 0.385 | 0.367 | 0.360 | 0.363 | **0.353** |
| Quên `zero_grad` | 0.586 | 0.484 | 0.470 | 0.400 | 0.447 | 0.487 | 0.479 |

Khi quên, gradient của mọi batch trước cộng dồn vào, nên mỗi bước đi theo "tổng lịch sử" chứ không phải batch hiện tại: loss **giật lên giật xuống** và không bao giờ xuống tới 0.353. Lưu ý là **code vẫn chạy, không báo lỗi**.

**(b) Quên `model.eval()`.** Một model có dropout 0.3, dự đoán trên tập test:
- Ở chế độ `eval()`: AUC **0.784**.
- Ở chế độ `train()`: AUC 0.766. Hai lần dự đoán **cùng một mẫu** chênh nhau tới **0.58** xác suất, vì mỗi lần dropout tắt một tập neuron khác nhau.

**(c) Batch size** (lr giữ nguyên $10^{-3}$, early stopping patience 10):

| Batch | Bước/epoch | Thời gian/epoch | Epoch tốt nhất | AUC test |
|---|---|---|---|---|
| 16 | 240 | 0.40 s | 10 | 0.782 |
| 64 | 60 | 0.14 s | 12 | 0.781 |
| 128 | 30 | 0.09 s | 18 | 0.780 |
| 512 | 8 | 0.08 s | 45 | 0.781 |
| 3 840 (cả tập, lr $10^{-2}$) | 1 | 0.06 s | 17 | **0.733** |

- Batch nhỏ thì mỗi epoch chậm hơn (nhiều bước Python, trên CPU), nhưng cần **ít epoch** hơn.
- Batch toàn tập chỉ được **1 bước mỗi epoch**: dừng ở epoch 27 nghĩa là model mới được cập nhật **27 lần**. Thứ quan trọng là **số bước cập nhật**, không phải số epoch.

**(d) Learning rate** (batch 128):

| lr | $10^{-4}$ | $10^{-3}$ | $10^{-2}$ | $10^{-1}$ |
|---|---|---|---|---|
| Epoch tốt nhất | 159 | 18 | 10 | 3 |
| Val loss tốt nhất | 0.353 | **0.352** | 0.360 | 0.371 |
| AUC test | 0.777 | 0.780 | 0.780 | 0.748 |

lr nhỏ: tốt nhưng chậm gấp 9 lần. lr quá lớn: nhảy qua lại quanh điểm tốt và dừng sớm ở một chỗ kém.

#### 2.5 Code chuẩn

```python
from torch.utils.data import DataLoader, TensorDataset

train_loader = DataLoader(TensorDataset(X_num, X_cat, y), batch_size=128, shuffle=True,
                          generator=torch.Generator().manual_seed(0))   # xáo trộn tái lập được
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()                                  # nhận LOGIT

for epoch in range(max_epochs):
    model.train()
    for *xs, yb in train_loader:
        opt.zero_grad()
        loss = loss_fn(model(*xs), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)          # tùy chọn (bài 04)
        opt.step()
    model.eval()
    with torch.no_grad():
        val = sum(loss_fn(model(*xs), yb).item() * len(yb) for *xs, yb in val_loader) / len(val_loader.dataset)
    # early stopping + lưu checkpoint khi val cải thiện: bài tập hôm nay

torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch}, "ckpt.pt")
state = torch.load("ckpt.pt", weights_only=True)                        # weights_only=True là mặc định từ 2.6
model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
```

**Dataset tự viết** (khi dữ liệu không nằm sẵn trong RAM, ví dụ ảnh trên đĩa): cài `__len__` và `__getitem__(i)` trả về một mẫu. `DataLoader(num_workers=4)` sẽ gọi `__getitem__` song song ở 4 process (trên Windows phải đặt code trong `if __name__ == "__main__":`).

#### 2.6 Cạm bẫy

1. **Quên `zero_grad()`**: mục 2.4(a). Code chạy bình thường nên rất khó phát hiện.
2. **Quên `eval()`** khi validate/predict: mục 2.4(b).
3. **Đưa xác suất vào `BCEWithLogitsLoss`/`CrossEntropyLoss`**: áp sigmoid/softmax hai lần (Tuần 2, mục 5.5).
4. **Sai shape nhãn**: logit $(B, 1)$ với nhãn $(B,)$ sẽ broadcast thành $(B, B)$ với `MSELoss` (có cảnh báo), còn `BCEWithLogitsLoss` thì báo lỗi. Hãy `squeeze(-1)` logit.
5. **`torch.save(model)` cả object**: file phụ thuộc vào đường dẫn import của class, dễ hỏng khi đổi code. Luôn lưu `state_dict`.
6. **Chỉ lưu model, quên optimizer**: khi resume, Adam mất $m, v$ nên những bước đầu đi như "mới khởi động" và loss nhảy lên.
7. **Loss trung bình theo batch rồi lấy trung bình các batch**: batch cuối nhỏ hơn bị tính nặng ngang batch đầy. Nên nhân với `len(yb)` rồi chia cho tổng số mẫu.

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"`model.eval()` có tắt gradient không?"* Không; nó chỉ đổi hành vi của dropout/BatchNorm. Cần thêm `torch.no_grad()`.
- *"Tăng batch size gấp $k$ lần thì learning rate nên thế nào?"* Quy tắc kinh nghiệm: tăng lr tuyến tính theo $k$, kèm warmup (Goyal et al. 2017). Quy tắc này ngừng đúng khi batch quá lớn.
- *"Model không học (loss đứng yên) thì kiểm tra gì trước?"* Thử **overfit một batch nhỏ** (khoảng 10 mẫu): nếu loss không về gần 0 thì có bug (nhãn sai, quên `zero_grad`, lr sai, loss nhận sai đầu vào).

#### 2.7 Đọc thêm

- PyTorch tutorial: *Learn the Basics* (Datasets & DataLoaders, Optimization loop, Save and Load).
- A. Karpathy, *A Recipe for Training Neural Networks* (blog, 2019): đặc biệt mục "overfit one batch".
- P. Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, 2017.
- I. Loshchilov & F. Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, ICLR 2017.

---

## 03 · MLP: neuron, activation, đếm tham số

> ⏱ Tầng 1: 15' · Tầng 2: 40'
> **Cần biết trước:** logistic regression (Tuần 3, bài 05); bài 01–02.

> **📖 Từ mới**
> - **Neuron:** tính tổng có trọng số của đầu vào, cộng bias, rồi đưa qua một hàm phi tuyến: $a = \phi(w^\top x + b)$.
> - **Activation function $\phi$:** hàm phi tuyến đó (ReLU, sigmoid, …).
> - **Lớp (layer) fully-connected / `Linear`:** nhiều neuron cùng nhìn vào một đầu vào: $h = \phi(Wx + b)$.
> - **MLP (multi-layer perceptron):** chồng nhiều lớp Linear + activation lên nhau. **Lớp ẩn (hidden layer):** lớp nằm giữa đầu vào và đầu ra.
> - **Embedding:** bảng tra cứu biến mỗi giá trị phân loại (ví dụ `intent = "billing"`) thành một vector số **được học**.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Một công ty duyệt hồ sơ vay:
- **Nhân viên lớp 1** mỗi người nhìn toàn bộ hồ sơ nhưng chỉ chú ý một khía cạnh ("thu nhập có ổn định không?", "nợ có cao không?"), rồi cho một điểm. Điểm âm được làm tròn về 0, nghĩa là "không có ý kiến" (ReLU).
- **Nhân viên lớp 2** đọc **điểm của lớp 1** chứ không đọc hồ sơ gốc, và tổ hợp chúng lại ("thu nhập ổn định **nhưng** nợ cao").
- **Giám đốc** đọc ý kiến lớp 2 và ra một con số cuối cùng.

Không ai được dặn trước nên chú ý điều gì: tất cả **tự học** từ dữ liệu.

**Vì sao cần phi tuyến?** Nếu bỏ activation, hai lớp tuyến tính gộp lại vẫn chỉ là một lớp tuyến tính: $W_2(W_1 x) = (W_2 W_1)x$. Chồng 100 lớp tuyến tính cũng không mạnh hơn logistic regression.

**Ví dụ có số: XOR.** Logistic regression **không** giải được XOR (không có đường thẳng nào tách được (0,1), (1,0) khỏi (0,0), (1,1)). Một MLP 2 neuron ẩn thì giải được: $h_1 = \mathrm{ReLU}(x_1 + x_2)$, $h_2 = \mathrm{ReLU}(x_1 + x_2 - 1)$, $\hat y = h_1 - 2h_2$.

| $x_1, x_2$ | $h_1$ | $h_2$ | $\hat y$ | XOR |
|---|---|---|---|---|
| 0, 0 | 0 | 0 | 0 | 0 |
| 0, 1 | 1 | 0 | 1 | 1 |
| 1, 0 | 1 | 0 | 1 | 1 |
| 1, 1 | 2 | 1 | **0** | 0 |

$h_2$ chỉ "bật" khi cả hai đầu vào bằng 1, và trừ đi đúng phần thừa. Lớp ẩn đã tạo ra một **đặc trưng mới** ("cả hai cùng bật") mà đầu vào không có sẵn.

**Ba ý cốt lõi:**
1. MLP = chồng các lớp $\phi(Wx + b)$; phi tuyến là bắt buộc.
2. Lớp ẩn **tự tạo đặc trưng**. Đó là khác biệt với Tuần 3, nơi bạn phải tự thêm `is_small_tech`.
3. Số tham số của một lớp Linear là $d_\text{in} \cdot d_\text{out} + d_\text{out}$; FLOP của forward xấp xỉ $2 \times$ số trọng số cho mỗi mẫu.

---

### TẦNG 2 · Kiến thức chuẩn

#### 3.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Perceptron** | $\hat y = \mathbb 1[w^\top x + b > 0]$, học bằng quy tắc cập nhật khi đoán sai | Rosenblatt, *Psychological Review* 1958 |
| **Giới hạn XOR** | Một perceptron đơn lẻ chỉ tách được dữ liệu **tách tuyến tính** | Minsky & Papert, *Perceptrons* (1969) |
| **MLP / feedforward network** | $h^{(l)} = \phi(W^{(l)}h^{(l-1)} + b^{(l)})$, $l = 1..L$; lớp cuối không có $\phi$ (cho ra logit) | Goodfellow ch. 6 |
| **Universal approximation theorem** | MLP **một lớp ẩn** với số neuron đủ lớn xấp xỉ được mọi hàm liên tục trên miền compact tới sai số tùy ý. Định lý chỉ nói **tồn tại**, không nói học được hay cần bao nhiêu neuron | Cybenko 1989 (sigmoid); Hornik 1991; Leshno et al. 1993 (mọi $\phi$ không phải đa thức) |
| **Entity embedding** | Biến phân loại $K$ giá trị → bảng $K \times d$ được học cùng mạng; tương đương one-hot nhân với một ma trận | Guo & Berkhahn, arXiv 2016 |

**Các activation** (giá trị và đạo hàm đã chạy kiểm chứng bằng autograd):

| Hàm | Công thức | Giá trị tại $-2, -1, 0, 1, 2$ | Đạo hàm tại $-2, -1, 0, 1, 2$ | Ghi chú |
|---|---|---|---|---|
| Sigmoid | $\sigma(x) = 1/(1 + e^{-x})$ | 0.119, 0.269, 0.5, 0.731, 0.881 | 0.105, 0.197, **0.25**, 0.197, 0.105 | Đạo hàm $\le 1/4$ → vanishing gradient (bài 04). Dùng ở **cổng** (LSTM, bài 08) |
| Tanh | $\tanh x$ | −0.964, −0.762, 0, 0.762, 0.964 | 0.071, 0.420, **1**, 0.420, 0.071 | Tâm 0; bão hòa ở hai đầu |
| **ReLU** | $\max(0, x)$ | 0, 0, 0, 1, 2 | 0, 0, 0*, 1, 1 | Rẻ, không bão hòa phía dương. *PyTorch quy ước đạo hàm tại 0 là 0. Nair & Hinton 2010 |
| **GELU** | $x\,\Phi(x)$ ($\Phi$: CDF chuẩn) | −0.045, −0.159, 0, 0.841, 1.955 | −0.085, −0.083, 0.5, 1.083, 1.085 | Bản "mềm" của ReLU; dùng trong BERT, GPT. Hendrycks & Gimpel 2016 |
| **SiLU / Swish** | $x\,\sigma(x)$ | −0.238, −0.269, 0, 0.731, 1.762 | −0.091, 0.072, 0.5, 0.928, 1.091 | Ramachandran et al. 2017 |
| **SwiGLU** | $(\mathrm{SiLU}(xW_1) \odot xW_3)\,W_2$ | — | — | **Cổng** nhân từng phần tử; FFN của LLaMA, PaLM. Shazeer 2020 |

#### 3.2 Đếm tham số và FLOP

**Một lớp Linear** $d_\text{in} \to d_\text{out}$: $d_\text{in} d_\text{out}$ trọng số + $d_\text{out}$ bias. **FLOP forward** cho mỗi mẫu $\approx 2\,d_\text{in} d_\text{out}$ (một nhân, một cộng). **Backward** tốn khoảng gấp đôi forward (gradient theo đầu vào và theo trọng số), nên **một bước huấn luyện ≈ 6 × số tham số FLOP mỗi mẫu**: đây là gốc của công thức $6ND$ trong scaling law (Kaplan et al. 2020).

**Ví dụ tính tay: `TabMLP` của project.** Đầu vào gồm 7 cột số + 2 cột chỉ báo thiếu (`retrieval_score`, `latency_s`) = 9 số; `intent` có 5 giá trị + 1 chỉ số "lạ" = 6, `model` có 3 + 1 = 4; embedding 4 chiều.

| Thành phần | Phép tính | Tham số |
|---|---|---|
| Embedding `intent` | $6 \times 4$ | 24 |
| Embedding `model` | $4 \times 4$ | 16 |
| Đầu vào lớp 1 | $9 + 4 + 4 = 17$ chiều | — |
| Linear $17 \to 64$ | $17 \cdot 64 + 64$ | 1 152 |
| Linear $64 \to 64$ | $64 \cdot 64 + 64$ | 4 160 |
| Head $64 \to 1$ | $64 + 1$ | 65 |
| **Tổng** | | **5 417** |

FLOP forward mỗi mẫu $\approx 2(17 \cdot 64 + 64 \cdot 64 + 64) \approx 10\,500$. So với HGB của Tuần 3 (200 cây × 7 lá): model nhỏ, rẻ.

**FFN của Transformer** với $d = 4\,096$ (con số sẽ gặp lại ở Tuần 5):
- FFN cổ điển $d \to 4d \to d$: $2 \cdot 4d^2 = 134\,217\,728$ trọng số.
- SwiGLU có **3** ma trận, nên chọn chiều ẩn $\tfrac{8}{3}d \approx 10\,922$ để giữ số tham số gần bằng: $3 \cdot d \cdot 10\,922 = 134\,209\,536$. LLaMA làm tròn lên 11 008 (bội số của 256).

#### 3.3 Đo trên máy bạn: chiều rộng và universal approximation

Khớp $y = \sin(2\pi x)$ trên $[0, 1]$ (200 điểm) bằng MLP **một lớp ẩn** tanh, Adam, 3 000 bước:

| Số neuron ẩn | 1 | 2 | 4 | 16 | 64 |
|---|---|---|---|---|---|
| Tham số | 4 | 7 | 13 | 49 | 193 |
| MSE | 0.0751 | 0.0040 | 0.0022 | 0.00002 | 0.00001 |

Một neuron tanh chỉ vẽ được một "bậc thang mềm", nên không thể uốn thành hình sin. Thêm neuron thì xấp xỉ tốt dần, đúng như định lý. Nhưng định lý **không** nói gì về dữ liệu **mới**: MSE ở đây là MSE trên chính tập train.

**Sâu hay rộng?** Về lý thuyết, có những hàm mà mạng sâu biểu diễn với số neuron ít hơn **theo hàm mũ** so với mạng nông (Telgarsky, COLT 2016). Thực tế: mạng sâu dùng tham số hiệu quả hơn, nhưng **khó huấn luyện hơn**. Đó là nội dung Ngày 2.

#### 3.4 Biến phân loại: one-hot hay embedding?

- One-hot $K$ giá trị rồi nhân $W \in \mathbb R^{K \times h}$ **chính là** tra hàng của $W$. Embedding chỉ là cách làm phép đó mà không cần tạo vector one-hot.
- Embedding $d$ chiều với $d \ll K$ hợp với biến có **nhiều** giá trị (mã tỉnh, mã sản phẩm, token ở Tuần 5). Các giá trị giống nhau sẽ có vector gần nhau.
- **Chỉ số 0 = "không biết"** cho giá trị lạ hoặc thiếu ở dữ liệu mới. Đây là phiên bản PyTorch của `OneHotEncoder(handle_unknown="ignore")` ở Tuần 3.

#### 3.5 Code chuẩn

```python
from torch import nn

class TabMLP(nn.Module):
    def __init__(self, n_num, cards, emb_dim=4, hidden=(64, 64), dropout=0.1):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(c, emb_dim) for c in cards)   # ModuleList, KHÔNG phải list Python
        d, layers = n_num + emb_dim * len(cards), []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)                                          # KHÔNG có sigmoid: trả về logit

    def forward(self, x_num, x_cat):
        e = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.head(self.body(torch.cat([x_num, *e], dim=1))).squeeze(-1)

print(sum(p.numel() for p in TabMLP(9, [6, 4]).parameters()))               # 5417
```

#### 3.6 Cạm bẫy

1. **Dùng `list` Python thay cho `nn.ModuleList`:** các module con không được đăng ký, nên `parameters()` bỏ sót chúng. Optimizer không bao giờ cập nhật chúng, và `.to(device)` cũng không chuyển chúng.
2. **Thêm sigmoid ở lớp cuối** rồi dùng `BCEWithLogitsLoss`: sigmoid hai lần.
3. **Không chuẩn hóa đầu vào số:** `answer_tokens` cỡ 150, còn `retrieval_score` cỡ 0.7. Gradient bị chi phối bởi cột có thang lớn (giống regularize mà không chuẩn hóa, Tuần 3, bài 02). Project chuẩn hóa bằng mean/std của **tập train**.
4. **Dead ReLU:** neuron có $w^\top x + b < 0$ với **mọi** mẫu sẽ cho gradient bằng 0 mãi mãi. Thường do lr quá lớn hoặc bias khởi tạo âm. Theo dõi tỉ lệ activation bằng 0; cân nhắc Leaky ReLU hoặc GELU.
5. **Hiểu sai universal approximation:** "MLP xấp xỉ được mọi hàm" **không** có nghĩa "MLP luôn tốt nhất". Trên dữ liệu bảng, cây thường thắng (project hôm nay sẽ đo).

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"Vì sao ReLU thay thế sigmoid trong lớp ẩn?"* Đạo hàm bằng 1 ở phía dương (không bão hòa, giảm vanishing gradient), tính rẻ, và cho activation thưa.
- *"GELU khác ReLU chỗ nào?"* Trơn, có giá trị âm nhỏ quanh 0, và gradient khác 0 với $x$ hơi âm. Thực nghiệm tốt hơn một chút cho Transformer.
- *"SwiGLU vì sao có 3 ma trận?"* Một nhánh qua SiLU làm **cổng**, một nhánh tuyến tính làm **giá trị**, nhân từng phần tử rồi chiếu về. Chiều ẩn giảm còn $\tfrac{8}{3}d$ để tổng số tham số không đổi.

#### 3.7 Đọc thêm

- Goodfellow, Bengio, Courville, *Deep Learning*, ch. 6 (Deep Feedforward Networks).
- G. Cybenko, *Approximation by Superpositions of a Sigmoidal Function*, Math. Control Signals Systems 1989.
- D. Hendrycks & K. Gimpel, *Gaussian Error Linear Units (GELUs)*, arXiv 2016.
- N. Shazeer, *GLU Variants Improve Transformer*, arXiv 2020.
- C. Guo & F. Berkhahn, *Entity Embeddings of Categorical Variables*, arXiv 2016.

---

## Bài tập Ngày 1 · `qnet/loop.py` (~2h)

Mảnh đầu tiên của project ([project.md](project.md)): vòng lặp huấn luyện **dùng lại cho mọi model** trong tuần. Lệnh test: `python -m pytest -q tests/test_loop.py` (8 test).

**Quy ước chung:** mỗi batch là một tuple tensor; **phần tử cuối là nhãn**, các phần tử còn lại là đầu vào, được truyền vào model theo dạng `model(*xs)`.

| Hàm | Yêu cầu |
|---|---|
| `set_seed(seed)` | Gọi `random.seed`, `np.random.seed`, `torch.manual_seed` |
| `make_loader(*tensors, batch_size=128, shuffle=False, seed=0)` | `DataLoader(TensorDataset(*tensors), ...)`. Khi `shuffle=True` thì truyền `generator=torch.Generator().manual_seed(seed)` |
| `train_one_epoch(model, loader, loss_fn, optimizer, *, clip_norm=None)` | `model.train()`; mỗi batch: `zero_grad` → forward → loss → `backward` → nếu có `clip_norm` thì `nn.utils.clip_grad_norm_` → `step`. Trả về **loss trung bình theo mẫu**: $\sum_b \text{loss}_b \cdot |b| / n$ |
| `evaluate(model, loader, loss_fn)` | `model.eval()` + `torch.no_grad()`; loss trung bình theo mẫu |
| `predict(model, *inputs, batch_size=1024)` | Ở chế độ eval, không gradient; chạy theo từng batch rồi `torch.cat` các đầu ra |
| `fit(model, train_loader, val_loader, loss_fn, optimizer, *, max_epochs=100, patience=10, min_delta=1e-4, clip_norm=None, scheduler=None, ckpt_path=None)` | Mỗi epoch: train → val → `scheduler.step()` nếu có. **Cải thiện** khi `val < best_val - min_delta`: ghi lại `best_val`, `best_epoch` (đếm từ 0), **bản sao sâu** của `state_dict`, và lưu checkpoint nếu có `ckpt_path`. Dừng khi `patience` epoch **liên tiếp** không cải thiện. Cuối cùng **nạp lại trọng số tốt nhất**. Trả về dict `train_loss` (list), `val_loss` (list), `best_epoch`, `best_val`, `stopped_epoch` (epoch cuối cùng đã chạy) |
| `save_checkpoint(path, model, optimizer=None, **extra)` | `torch.save` một dict gồm `"model"`, `"optimizer"` (chỉ khi có), và các khóa trong `extra` (ví dụ `epoch`, `best_val`) |
| `load_checkpoint(path, model, optimizer=None)` | `torch.load(path, weights_only=True)`; nạp vào model (và optimizer nếu có); trả về dict đã đọc |
| `count_params(model, trainable_only=True)` | Tổng `numel()`; nếu `trainable_only` thì chỉ đếm tham số có `requires_grad` |

**Gợi ý:**
- `copy.deepcopy(model.state_dict())`. Nếu chỉ gán `best = model.state_dict()` thì bạn giữ **tham chiếu** tới chính các tensor đang được cập nhật, và "trọng số tốt nhất" sẽ thay đổi theo.
- Test `test_fit_early_stopping_and_restores_best` dùng `lr=0`: val loss không đổi, nên phải dừng đúng ở epoch `patience` với `best_epoch = 0`.
