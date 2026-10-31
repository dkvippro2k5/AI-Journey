# Ngày 2 · Giải tích & Tối ưu hóa

> **Loại tài liệu: Lý thuyết.** Đầy đủ ký hiệu, định nghĩa, mệnh đề kèm chứng minh, và bài tính tay.
> **Hôm nay gồm 4 bài:**
> - **04** Đạo hàm, gradient, Jacobian, Hessian
> - **05** Chain rule, đồ thị tính toán và backpropagation
> - **06** Gradient descent, tính lồi và learning rate
> - **07** Momentum, RMSProp, Adam và lịch learning rate
>
> **Thời gian:** đọc ~3h · quiz 45' · bài tập ~2h
> **Môi trường kiểm chứng:** NumPy 2.5, PyTorch 2.14 (CPU). Mọi con số đo được trong file đều đã chạy thật.

---

## 04 · Đạo hàm, gradient, Jacobian, Hessian

> ⏱ Tầng 1: 10' · Tầng 2: 35'
> **Cần biết trước:** đạo hàm một biến ở mức THPT; tích vô hướng và Cauchy–Schwarz (bài 01); trị riêng (bài 03).

> **📖 Từ mới**
> - **Đạo hàm:** tốc độ thay đổi của $f$ khi $x$ thay đổi một chút; về hình học là độ dốc của tiếp tuyến.
> - **Đạo hàm riêng $\partial f/\partial x_i$:** đạo hàm theo một biến, giữ nguyên các biến còn lại.
> - **Gradient $\nabla f$:** vector gom mọi đạo hàm riêng. Nó chỉ hướng $f$ **tăng nhanh nhất**.
> - **Jacobian:** khi hàm trả về **nhiều** giá trị, Jacobian là bảng mọi đạo hàm riêng của mọi đầu ra theo mọi đầu vào.
> - **Hessian:** bảng các đạo hàm bậc hai; mô tả **độ cong** của $f$.
> - **Điểm dừng (critical point):** nơi gradient bằng 0. Có thể là cực tiểu, cực đại, hoặc **điểm yên ngựa** (saddle point): theo hướng này thì là đáy, theo hướng khác thì là đỉnh.
> - **Sai phân (finite difference):** xấp xỉ đạo hàm bằng $\big(f(x + h) - f(x)\big)/h$ với $h$ nhỏ.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Bạn đứng trên sườn đồi, mắt bị bịt.
- Đặt chân thử về phía Bắc thấy cao lên 3 cm; về phía Đông thấy cao lên 4 cm. Hai con số đó là **hai đạo hàm riêng**.
- Gom lại thành mũi tên $(3, 4)$: đó là **gradient**, chỉ hướng **leo dốc nhanh nhất**, với độ dốc bằng độ dài mũi tên là 5.
- Muốn xuống núi thì đi theo hướng **ngược** gradient. Đây là toàn bộ ý tưởng của việc huấn luyện model (bài 06).
- **Hessian** cho biết mặt đất cong thế nào: lòng chảo (cực tiểu), đỉnh núi (cực đại), hay **yên ngựa** (dọc lưng ngựa là đáy, ngang lưng ngựa là đỉnh).

**Ba ý cốt lõi:**
1. Gradient là vector cùng shape với tham số, chỉ hướng tăng nhanh nhất.
2. Huấn luyện = đi ngược gradient của loss.
3. Gradient = 0 **chưa chắc** là cực tiểu; phải xem độ cong (Hessian).

**Ví dụ nhỏ.** $f(x, y) = x^2 + 3xy + y^2$ tại $(1, 2)$: $\partial f/\partial x = 2x + 3y = 8$, $\partial f/\partial y = 3x + 2y = 7$, nên $\nabla f = (8, 7)$.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 4.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Gradient** | $\nabla f(x) = \left(\dfrac{\partial f}{\partial x_1}, \dots, \dfrac{\partial f}{\partial x_d}\right)^\top \in \mathbb{R}^d$ — **cùng shape với $x$** | MML §5.2 |
| **Directional derivative** | $D_u f(x) = \nabla f(x)^\top u$, với $\|u\| = 1$ | MML §5.2 |
| **Jacobian** | Với $f: \mathbb{R}^n \to \mathbb{R}^m$: $J \in \mathbb{R}^{m \times n}$, $J_{ij} = \partial f_i/\partial x_j$ | MML §5.3 |
| **Hessian** | $H_{ij} = \partial^2 f/\partial x_i \partial x_j$. Đối xứng nếu $f$ khả vi liên tục cấp 2 (định lý Schwarz) | MML §5.7 |
| **Taylor bậc 2** | $f(x + \delta) \approx f(x) + \nabla f(x)^\top\delta + \tfrac{1}{2}\delta^\top H(x)\,\delta$ | Goodfellow §4.3 |
| **Critical point** | $\nabla f(x) = 0$. Với $H$ tại đó: mọi $\lambda_i > 0$ → cực tiểu chặt; mọi $\lambda_i < 0$ → cực đại chặt; có cả dấu dương lẫn âm → **saddle point** | Goodfellow §4.3.1 |
| **Convex function** | $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$, $\forall \lambda \in [0, 1]$. Với $f \in C^2$: lồi $\Leftrightarrow H(x)$ PSD với mọi $x$ | Boyd & Vandenberghe §3.1 |

**Các đạo hàm phải thuộc:**

| $f$ | $f'$ | Ghi chú |
|---|---|---|
| $x^n$, $e^x$, $\ln x$ | $nx^{n-1}$, $e^x$, $1/x$ | |
| $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ | $\sigma(z)\big(1 - \sigma(z)\big)$ | Mệnh đề 4.2 |
| $\tanh z$ | $1 - \tanh^2 z$ | |
| $\mathrm{ReLU}(z) = \max(0, z)$ | $\mathbb{1}[z > 0]$ | Không khả vi tại 0; PyTorch quy ước đạo hàm tại 0 là 0 |
| $\mathrm{softmax}(z)_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$ | $\dfrac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$, tức $J = \mathrm{diag}(p) - pp^\top$ | Mệnh đề 4.3 |

**Đạo hàm ma trận** (chứng minh bằng cách viết ra từng tọa độ):
- $\nabla_\theta (a^\top \theta) = a$
- $\nabla_\theta (\theta^\top A \theta) = (A + A^\top)\theta$, bằng $2A\theta$ nếu $A$ đối xứng
- $\nabla_\theta \|X\theta - y\|^2 = 2X^\top(X\theta - y)$. *Chứng minh:* $\|X\theta - y\|^2 = \theta^\top X^\top X\theta - 2y^\top X\theta + y^\top y$; áp dụng hai dòng trên. Cho gradient bằng 0, ta được lại **phương trình chuẩn** của bài 02.

#### 4.2 Mệnh đề và chứng minh

**Mệnh đề 4.1 (gradient là hướng dốc nhất).** Trong mọi hướng đơn vị $u$, đạo hàm theo hướng $D_u f = \nabla f^\top u$ lớn nhất khi $u = \nabla f/\|\nabla f\|$ và nhỏ nhất khi $u = -\nabla f/\|\nabla f\|$.

*Chứng minh.* Theo Cauchy–Schwarz, $-\|\nabla f\| \le \nabla f^\top u \le \|\nabla f\|\,\|u\| = \|\nabla f\|$; hai dấu bằng xảy ra khi $u$ cùng phương với $\nabla f$. ∎

**Mệnh đề 4.2.** $\sigma'(z) = \sigma(z)(1 - \sigma(z)) \le 1/4$.

*Chứng minh.* $\sigma(z) = (1 + e^{-z})^{-1}$, nên $\sigma'(z) = \dfrac{e^{-z}}{(1 + e^{-z})^2} = \dfrac{1}{1 + e^{-z}} \cdot \dfrac{e^{-z}}{1 + e^{-z}} = \sigma(1 - \sigma)$. Hàm $s(1 - s)$ trên $[0, 1]$ đạt cực đại $1/4$ tại $s = 1/2$, tức $z = 0$. ∎

*Hệ quả:* đi qua $k$ lớp sigmoid, gradient bị nhân với tối đa $(1/4)^k$; với $k = 10$ còn $9.5 \times 10^{-7}$. Đây là **vanishing gradient**, lý do mạng sâu dùng ReLU và residual connection (Tuần 4).

**Mệnh đề 4.3 (Jacobian của softmax).** $\dfrac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$. Mỗi hàng của $J$ có tổng bằng 0.

*Chứng minh.* $p_i = e^{z_i}/S$ với $S = \sum_k e^{z_k}$. Theo quy tắc thương: $\dfrac{\partial p_i}{\partial z_j} = \dfrac{\delta_{ij}e^{z_i}S - e^{z_i}e^{z_j}}{S^2} = p_i\delta_{ij} - p_ip_j$. Tổng hàng $i$ bằng $p_i - p_i\sum_j p_j = 0$. Điều này hợp lý: cộng cùng một hằng số vào mọi $z_j$ thì softmax không đổi. ∎

**Mệnh đề 4.4 (sai số của sai phân).** Với $f$ đủ trơn:
- Sai phân tiến: $\dfrac{f(x + h) - f(x)}{h} = f'(x) + \dfrac{h}{2}f''(x) + O(h^2)$, sai số $O(h)$.
- Sai phân trung tâm: $\dfrac{f(x + h) - f(x - h)}{2h} = f'(x) + \dfrac{h^2}{6}f'''(x) + O(h^4)$, sai số $O(h^2)$.

*Chứng minh.* Khai triển Taylor $f(x \pm h) = f \pm hf' + \tfrac{h^2}{2}f'' \pm \tfrac{h^3}{6}f''' + \dots$. Lấy hiệu thì các số hạng bậc chẵn triệt tiêu. ∎

Nhưng $h$ không thể nhỏ tùy ý: sai số làm tròn của float64 cỡ $\varepsilon |f|/h$ với $\varepsilon \approx 2.2 \times 10^{-16}$, **tăng** khi $h$ giảm. Đo trên máy bạn với $f = \sigma$ tại $x = 1$ ($f'$ đúng $= 0.196612$):

| $h$ | Sai số sai phân tiến | Sai số sai phân trung tâm |
|---|---|---|
| $10^{-1}$ | $4.6 \times 10^{-3}$ | $5.9 \times 10^{-5}$ |
| $10^{-3}$ | $4.5 \times 10^{-5}$ | $5.9 \times 10^{-9}$ |
| $10^{-5}$ | $4.5 \times 10^{-7}$ | $\mathbf{1.8 \times 10^{-12}}$ |
| $10^{-8}$ | $6.5 \times 10^{-9}$ | $9.5 \times 10^{-10}$ |
| $10^{-11}$ | $2.5 \times 10^{-6}$ | $3.0 \times 10^{-6}$ |

Sai phân trung tâm tốt nhất ở $h \approx 10^{-5}$–$10^{-6}$ (lý thuyết: $h \sim \varepsilon^{1/3}$). Nhỏ hơn nữa thì sai số làm tròn lấn át.

#### 4.3 Ví dụ tính tay

**(a) Điểm yên ngựa.** $f(x, y) = x^2 + 3xy + y^2$ có $\nabla f = (2x + 3y,\ 3x + 2y)$ bằng 0 tại gốc tọa độ. $H = \begin{bmatrix}2 & 3\\3 & 2\end{bmatrix}$ có trị riêng $2 \pm 3 = 5$ và $-1$, trái dấu, nên gốc là **saddle point**. Kiểm tra: dọc hướng $(1, 1)$ thì $f = 5t^2$ (đi lên); dọc hướng $(1, -1)$ thì $f = -t^2$ (đi xuống).

**(b) Softmax.** $z = (1, 2, 3)$, nên $p = (0.090,\ 0.245,\ 0.665)$. Hàng đầu của $J$: $(p_1(1 - p_1),\ -p_1p_2,\ -p_1p_3) = (0.0819,\ -0.0220,\ -0.0599)$, tổng bằng 0 ✓.

#### 4.4 Code chuẩn: kiểm tra gradient (gradient check)

```python
def numerical_grad(f, theta, eps=1e-6):
    theta = np.array(theta, dtype=float)          # float64, bản sao riêng
    g = np.zeros_like(theta)
    for i in range(theta.size):
        old = theta.flat[i]
        theta.flat[i] = old + eps; fp = f(theta)
        theta.flat[i] = old - eps; fm = f(theta)
        theta.flat[i] = old
        g.flat[i] = (fp - fm) / (2 * eps)
    return g

rel = np.linalg.norm(g_analytic - g_num) / (np.linalg.norm(g_analytic) + np.linalg.norm(g_num))
# float64: rel < 1e-7 là tốt; > 1e-4 gần như chắc chắn có bug (CS231n)
```

#### 4.5 Cạm bẫy

1. **Gradient check bằng float32.** Sai số làm tròn lớn gấp khoảng $10^9$ lần so với float64; kết quả kiểm tra vô nghĩa. Luôn check bằng float64.
2. **Check tại điểm gãy** (ReLU tại 0, `abs`, `max`): sai phân "nhìn" cả hai phía nên không khớp với đạo hàm một phía. Chọn điểm ngẫu nhiên, tránh điểm gãy.
3. **Nhầm hằng số.** $\frac{1}{n}\|r\|^2$ có gradient $\frac{2}{n}X^\top r$, còn $\frac{1}{2n}\|r\|^2$ có gradient $\frac{1}{n}X^\top r$. Tuần này thống nhất dùng $\frac{1}{2n}$.
4. **$\nabla f = 0$ chưa chắc là cực tiểu.** Có thể là saddle point; với mạng neural nhiều chiều, điều này còn **thường gặp hơn** cực tiểu (bài 06).

**Sắc thái hay bị hỏi khi phỏng vấn:** *"Vì sao đi theo $-\nabla f$?"* Mệnh đề 4.1: trong mọi hướng có cùng độ dài bước, đó là hướng làm $f$ giảm nhanh nhất **theo xấp xỉ bậc nhất**. Nó không phải hướng tốt nhất toàn cục; Newton's method dùng thêm Hessian để sửa hướng.

#### 4.6 Đọc thêm

- MML ch. 5 (Vector Calculus).
- Goodfellow et al., *Deep Learning*, §4.3.
- K. B. Petersen & M. S. Pedersen, *The Matrix Cookbook* (tra cứu đạo hàm ma trận).
- CS231n notes, *Gradient checks*.

---

## 05 · Chain rule, đồ thị tính toán và backpropagation

> ⏱ Tầng 1: 15' · Tầng 2: 40'
> **Cần biết trước:** đạo hàm của sigmoid và softmax (bài 04).

> **📖 Từ mới**
> - **Hàm hợp:** đầu ra của hàm này là đầu vào của hàm kia, ví dụ $f(g(x))$.
> - **Chain rule (quy tắc dây chuyền):** đạo hàm của hàm hợp bằng **tích** các đạo hàm thành phần.
> - **Đồ thị tính toán:** vẽ phép tính thành các nút (phép toán) nối bằng mũi tên (dữ liệu chảy qua).
> - **Forward pass:** tính từ đầu vào tới loss. **Backward pass:** tính gradient đi ngược từ loss về các tham số.
> - **Backpropagation:** thuật toán backward pass dùng chain rule; tên gọi trong deep learning của **reverse-mode automatic differentiation**.
> - **Local gradient:** đạo hàm của một nút theo đầu vào của chính nó. **Upstream gradient:** gradient của loss theo đầu ra của nút đó, nhận từ phía sau truyền về.
> - **Logit:** đầu ra thô của model **trước** sigmoid/softmax.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Một dây chuyền ba trạm: bột → bánh → hộp bánh → doanh thu.
- Thêm 1 kg bột thì được thêm 10 cái bánh; thêm 1 cái bánh thì được thêm 0.5 hộp; thêm 1 hộp thì doanh thu tăng 40 nghìn.
- Vậy thêm 1 kg bột thì doanh thu tăng $10 \times 0.5 \times 40 = 200$ nghìn. **Nhân các hệ số dọc dây chuyền**: đó là chain rule.
- Backprop làm phép nhân đó **từ cuối về đầu**: trạm cuối báo "1 hộp = 40", trạm giữa nhân với 0.5 rồi báo "1 bánh = 20", trạm đầu nhân với 10: "1 kg bột = 200". Mỗi trạm chỉ cần biết **hệ số của riêng nó** và con số nhận từ phía sau.

**Ba ý cốt lõi:**
1. Gradient tại một nút = upstream gradient × local gradient.
2. Đi ngược **một lần** là có gradient theo **mọi** tham số; nhờ vậy huấn luyện được model hàng tỉ tham số.
3. Nếu một biến được dùng ở nhiều nơi, gradient từ các nơi đó **cộng lại**.

**Ví dụ nhỏ** (ví dụ kinh điển của CS231n). $f = (x + y) \cdot z$ với $x = 2$, $y = -3$, $z = 4$.
- Forward: $q = x + y = -1$, $f = qz = -4$.
- Backward: $\partial f/\partial z = q = -1$; $\partial f/\partial q = z = 4$; phép cộng có local gradient bằng 1 cho mỗi đầu vào, nên $\partial f/\partial x = \partial f/\partial y = 4 \cdot 1 = 4$.

---

### TẦNG 2 · Kiến thức chuẩn

#### 5.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Chain rule (một biến)** | $(f \circ g)'(x) = f'(g(x))\,g'(x)$ | MML §5.2.2 |
| **Chain rule (nhiều biến)** | Với $y = g(x)$, $L = f(y)$: $\dfrac{\partial L}{\partial x_i} = \sum_j \dfrac{\partial L}{\partial y_j}\dfrac{\partial y_j}{\partial x_i}$, tức $\nabla_x L = J_g(x)^\top\,\nabla_y L$ | MML §5.3 |
| **Computational graph** | Đồ thị có hướng không chu trình (DAG): mỗi nút là một phép toán sơ cấp | Goodfellow §6.5.1 |
| **Forward-mode AD** | Tính tích Jacobian–vector $J v$ cùng chiều với forward pass; một lần chạy cho đạo hàm theo **một hướng đầu vào** | Baydin et al., JMLR 2018 |
| **Reverse-mode AD** | Tính tích vector–Jacobian $v^\top J$ đi ngược; một lần chạy cho gradient của **một đầu ra vô hướng** theo **mọi** đầu vào | Baydin et al., JMLR 2018 |
| **Backpropagation** | Reverse-mode AD áp dụng cho mạng neural | Rumelhart, Hinton, Williams, *Nature* 1986 (ý tưởng từ Linnainmaa 1970) |
| **Binary cross-entropy (BCE)** | $\ell(p, y) = -[y\ln p + (1 - y)\ln(1 - p)]$, với $y \in \{0, 1\}$. Lý do chọn hàm này: Ngày 3, bài 10 | — |
| **Cross-entropy (đa lớp)** | $\ell(p, y) = -\sum_k y_k \ln p_k$, với $y$ one-hot | — |

#### 5.2 Mệnh đề và chứng minh

**Mệnh đề 5.1 (cộng gradient ở nút rẽ nhánh).** Nếu $x$ đi vào nhiều nút $y_1, \dots, y_m$ thì $\dfrac{\partial L}{\partial x} = \sum_j \dfrac{\partial L}{\partial y_j}\dfrac{\partial y_j}{\partial x}$. Đây chính là chain rule nhiều biến. *Hệ quả thực tế:* PyTorch **cộng dồn** gradient vào `.grad`, nên phải gọi `optimizer.zero_grad()` trước mỗi lần backward.

**Mệnh đề 5.2 (chi phí của reverse mode).** Với $f: \mathbb{R}^n \to \mathbb{R}$ (loss vô hướng, $n$ tham số):
- Reverse mode cho cả $\nabla f$ với chi phí cỡ **một hằng số nhỏ nhân với chi phí forward** (cỡ vài lần), **không phụ thuộc $n$** (*cheap gradient principle*, Griewank & Walther, *Evaluating Derivatives*).
- Forward mode cần $n$ lần chạy, mỗi lần một hướng.
- Với $n = 7 \times 10^9$ (một LLM 7B), forward mode cần 7 tỉ lần chạy; reverse mode cần **một**.
- Cái giá: reverse mode phải **lưu mọi giá trị trung gian** của forward pass để dùng khi đi ngược (activation memory). Đây là lý do *gradient checkpointing* tồn tại (Tuần 11).

**Mệnh đề 5.3 (sigmoid + BCE).** Với $z = w^\top x + b$, $p = \sigma(z)$, $\ell = -[y\ln p + (1 - y)\ln(1 - p)]$:

$$\frac{\partial \ell}{\partial z} = p - y, \qquad \nabla_w \ell = (p - y)\,x, \qquad \frac{\partial \ell}{\partial b} = p - y.$$

*Chứng minh.* $\dfrac{\partial \ell}{\partial p} = -\dfrac{y}{p} + \dfrac{1 - y}{1 - p}$; theo Mệnh đề 4.2, $\dfrac{\partial p}{\partial z} = p(1 - p)$. Nhân lại: $-y(1 - p) + (1 - y)p = p - y$. Cuối cùng $\partial z/\partial w = x$, $\partial z/\partial b = 1$. ∎

Với cả batch $X \in \mathbb{R}^{n \times d}$ và loss trung bình: $\nabla_\theta L = \dfrac{1}{n}X^\top\big(\sigma(X\theta) - y\big)$.

**Mệnh đề 5.4 (softmax + cross-entropy).** Với $p = \mathrm{softmax}(z)$ và $y$ one-hot: $\nabla_z \ell = p - y$.

*Chứng minh.* $\dfrac{\partial \ell}{\partial z_j} = -\sum_k \dfrac{y_k}{p_k}\dfrac{\partial p_k}{\partial z_j} = -\sum_k \dfrac{y_k}{p_k}\,p_k(\delta_{kj} - p_j) = -y_j + p_j\sum_k y_k = p_j - y_j$, vì $\sum_k y_k = 1$. ∎

Cả hai mệnh đề cho cùng một dạng rất gọn: **"dự đoán trừ nhãn"**. Đó là lý do các framework gộp softmax và cross-entropy vào **một** phép toán nhận logit (`nn.CrossEntropyLoss`, `BCEWithLogitsLoss`): gradient đơn giản và tính ổn định số học tốt hơn (mục 5.5).

**Mệnh đề 5.5 (MSE).** $L(\theta) = \dfrac{1}{2n}\|X\theta - y\|^2$ có $\nabla L = \dfrac{1}{n}X^\top(X\theta - y)$ và Hessian $H = \dfrac{1}{n}X^\top X$ (hằng số). *Chứng minh:* từ công thức đạo hàm ma trận ở mục 4.1. ∎ Đây là lý do hàm `hessian_eigs_mse` của Ngày 1 tính trị riêng của $X^\top X/n$.

#### 5.3 Ví dụ tính tay: một neuron logistic

$x = (1, 2)$, $w = (0.5, -0.25)$, $b = 0.1$, nhãn $y = 1$.

| Bước | Forward | Backward |
|---|---|---|
| $z = w^\top x + b$ | $0.5 - 0.5 + 0.1 = 0.1$ | $\partial \ell/\partial z = p - y = -0.47502$ |
| $p = \sigma(z)$ | $0.52498$ | (đã gộp vào dòng trên nhờ Mệnh đề 5.3) |
| $\ell = -\ln p$ | $0.64440$ | — |
| $\nabla_w \ell = (p - y)\,x$ | — | $(-0.47502,\ -0.95004)$ |
| $\partial \ell/\partial b$ | — | $-0.47502$ |

Đã kiểm chứng ba cách cho cùng kết quả tới 10 chữ số: tính tay, sai phân trung tâm, và `torch.autograd`:

```python
w = torch.tensor([0.5, -0.25], dtype=torch.float64, requires_grad=True)
b = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
loss = F.binary_cross_entropy_with_logits(w @ torch.tensor([1., 2.], dtype=torch.float64) + b,
                                          torch.tensor(1., dtype=torch.float64))
loss.backward()
print(loss.item(), w.grad, b.grad)   # 0.64440  [-0.47502, -0.95004]  -0.47502
```

Ý nghĩa của dấu: gradient âm, nên bước $w \leftarrow w - \alpha\nabla_w\ell$ **tăng** $w$, làm $z$ và $p$ tăng, kéo dự đoán về gần nhãn 1. Đúng như mong đợi.

**Softmax + CE:** $z = (1, 2, 3)$, nhãn đúng là lớp 0. $\ell = -\ln 0.090 = 2.408$; $\nabla_z\ell = p - y = (-0.910,\ 0.245,\ 0.665)$: đẩy logit đúng lên, đẩy hai logit sai xuống.

#### 5.4 Code chuẩn: loss kèm gradient, ổn định số học

```python
def logistic_loss(theta, X, y):
    z = X @ theta
    # BCE viết theo logit: -[y ln σ(z) + (1-y) ln(1-σ(z))] = log(1 + e^z) - y·z
    loss = np.mean(np.logaddexp(0.0, z) - y * z)        # logaddexp(0, z) = log(1 + e^z), không tràn số
    grad = X.T @ (sigmoid(z) - y) / len(y)              # Mệnh đề 5.3
    return loss, grad
```

Đẳng thức trong comment: $\ln\sigma(z) = -\ln(1 + e^{-z})$ và $\ln(1 - \sigma(z)) = -\ln(1 + e^{z})$. Do đó $\ell = y\ln(1 + e^{-z}) + (1 - y)\ln(1 + e^z)$. Thay $\ln(1 + e^{-z}) = \ln(1 + e^z) - z$ là ra $\ln(1 + e^z) - yz$.

#### 5.5 Cạm bẫy

1. **Tính `log(sigmoid(z))` trực tiếp.** Đo trên máy bạn: với $z = -800$, `np.log(1/(1+np.exp(800)))` cho **`-inf`** (kèm cảnh báo overflow), trong khi đáp án đúng là $-800$. `-np.logaddexp(0, 800)` cho đúng $-800$.
2. **Quên `zero_grad()`** trong PyTorch: gradient của các bước trước cộng dồn vào (Mệnh đề 5.1).
3. **Đưa xác suất (sau softmax) vào `nn.CrossEntropyLoss`.** Hàm này nhận **logit**; đưa xác suất vào là áp softmax hai lần, model vẫn chạy nhưng học sai.
4. **Gradient khác shape với tham số.** `X.T @ r` cho shape `(d,)`; nếu lỡ dùng `r @ X.T` hay quên chia $n$, gradient check sẽ bắt được.

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"Chứng minh gradient của softmax cross-entropy là $p - y$."* Câu hỏi kinh điển; phải viết được Mệnh đề 5.4 trong 2 phút.
- *"Vì sao deep learning dùng reverse mode?"* Một đầu ra vô hướng (loss), hàng tỉ đầu vào (tham số): Mệnh đề 5.2.
- *"Backprop tốn bộ nhớ ở đâu?"* Lưu activation của forward pass.

#### 5.6 Đọc thêm

- Goodfellow §6.5 (Back-Propagation and Other Differentiation Algorithms).
- A. G. Baydin et al., *Automatic Differentiation in Machine Learning: a Survey*, JMLR 18 (2018).
- A. Karpathy, *micrograd* (GitHub) và video *The spelled-out intro to neural networks and backpropagation*.
- CS231n notes, *Backpropagation, Intuitions*.

---

## 06 · Gradient descent, tính lồi và learning rate

> ⏱ Tầng 1: 15' · Tầng 2: 45'
> **Cần biết trước:** trị riêng (bài 03), gradient và Hessian (bài 04), gradient của MSE (bài 05).

> **📖 Từ mới**
> - **Loss (hàm mất mát) / hàm mục tiêu:** con số đo model sai bao nhiêu; huấn luyện là làm nó nhỏ đi.
> - **Gradient descent (GD):** lặp lại: tính gradient, bước một đoạn ngược hướng gradient.
> - **Learning rate $\alpha$:** độ dài bước.
> - **Hội tụ / phân kỳ:** tiến dần tới đáy / càng chạy càng xa đáy (loss tăng vọt, có khi thành `nan`).
> - **Hàm lồi:** hàm hình "cái bát": chỉ có **một** đáy.
> - **SGD (stochastic GD):** mỗi bước chỉ dùng gradient tính trên **một phần nhỏ** dữ liệu (mini-batch).
> - **Epoch:** một lượt đi qua toàn bộ dữ liệu. **Step / iteration:** một lần cập nhật tham số.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Xuống núi trong sương mù dày, chỉ cảm nhận được độ dốc ngay dưới chân.
- Mỗi bước đi theo hướng dốc xuống nhất. **Bước quá ngắn**: tới đáy rất lâu. **Bước quá dài**: nhảy qua đáy sang sườn bên kia, có khi còn **cao hơn** chỗ cũ; cứ thế càng nhảy càng cao.
- Nếu thung lũng **hẹp và dài**, sườn hai bên rất dốc nhưng đáy thung lũng gần như phẳng: bước dài thì văng khỏi sườn, bước ngắn thì đi dọc đáy mãi không tới. Đó là bài toán có **số điều kiện lớn**.

**Ba ý cốt lõi:**
1. Learning rate có **giới hạn trên** do chỗ cong nhất của loss quyết định: vượt quá là phân kỳ.
2. Số bước cần thiết tỉ lệ với **tỉ số giữa độ cong lớn nhất và nhỏ nhất** ($\kappa$). Chuẩn hóa đặc trưng làm $\kappa$ nhỏ lại.
3. SGD dùng gradient "có nhiễu" nhưng rẻ hơn rất nhiều; nhiễu giảm khi batch lớn hơn.

**Ví dụ nhỏ.** $L(\theta) = 2\theta^2$, nên $L'(\theta) = 4\theta$. Bắt đầu từ $\theta_0 = 1$, bước $\theta \leftarrow \theta - \alpha \cdot 4\theta = (1 - 4\alpha)\,\theta$:

| $\alpha$ | Hệ số $1 - 4\alpha$ | $\theta_0 \dots \theta_4$ | Kết luận |
|---|---|---|---|
| 0.1 | 0.6 | 1, 0.6, 0.36, 0.216, 0.13 | hội tụ đều |
| 0.25 | 0 | 1, 0, 0, 0, 0 | tới đáy sau **1 bước** |
| 0.4 | −0.6 | 1, −0.6, 0.36, −0.216, 0.13 | hội tụ nhưng dao động |
| 0.5 | −1 | 1, −1, 1, −1, 1 | dao động mãi |
| 0.6 | −1.4 | 1, −1.4, 1.96, −2.74, 3.84 | **phân kỳ** |

Ngưỡng là $\alpha = 2/4 = 0.5$, tức $2/\text{độ cong}$.

---

### TẦNG 2 · Kiến thức chuẩn

#### 6.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Gradient descent** | $\theta_{t+1} = \theta_t - \alpha\nabla L(\theta_t)$ | Cauchy (1847); Boyd & Vandenberghe §9.3 |
| **$L$-smooth** | $\|\nabla f(x) - \nabla f(y)\| \le L\|x - y\|$. Với $f \in C^2$: $\lambda_{\max}(H) \le L$ ở mọi điểm | Nesterov, *Lectures on Convex Optimization* §2.1 |
| **$\mu$-strongly convex** | $\lambda_{\min}(H) \ge \mu > 0$ ở mọi điểm | như trên |
| **Condition number** | $\kappa = L/\mu$. Với loss bậc hai thì bằng $\lambda_{\max}/\lambda_{\min}$ của Hessian | — |
| **Global / local minimum** | $f(x^\star) \le f(x)$ với mọi $x$ / với mọi $x$ trong một lân cận | Goodfellow §4.3 |
| **SGD** | $\theta_{t+1} = \theta_t - \alpha\,\hat g_t$ với $\hat g_t$ là ước lượng không chệch của $\nabla L$ | Robbins & Monro (1951) |
| **Mini-batch gradient** | $\hat g_B = \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla\ell_i(\theta)$ | Goodfellow §8.1.3 |

#### 6.2 Mệnh đề và chứng minh

**Mệnh đề 6.1 (một bước nhỏ ngược gradient luôn làm loss giảm).** Nếu $\nabla L(\theta) \ne 0$ thì với $\alpha > 0$ đủ nhỏ, $L(\theta - \alpha\nabla L) < L(\theta)$.

*Chứng minh.* Taylor bậc 1: $L(\theta - \alpha\nabla L) = L(\theta) - \alpha\|\nabla L\|^2 + O(\alpha^2)$. Với $\alpha$ đủ nhỏ, số hạng $-\alpha\|\nabla L\|^2 < 0$ lấn át $O(\alpha^2)$. ∎

**Mệnh đề 6.2 (hàm lồi: cực tiểu địa phương là toàn cục).**

*Chứng minh.* Giả sử $x$ là cực tiểu địa phương nhưng có $y$ với $f(y) < f(x)$. Với $\lambda \in (0, 1]$ nhỏ tùy ý, điểm $z = (1 - \lambda)x + \lambda y$ nằm trong lân cận của $x$, và theo tính lồi $f(z) \le (1 - \lambda)f(x) + \lambda f(y) < f(x)$. Mâu thuẫn với giả thiết $x$ là cực tiểu địa phương. ∎

MSE của linear regression ($H = X^\top X/n$ là PSD) và BCE của logistic regression đều **lồi**. Loss của mạng neural thì **không lồi**.

**Mệnh đề 6.3 (GD trên hàm bậc hai: định lý trung tâm của bài).** Cho $L(\theta) = \tfrac{1}{2}(\theta - \theta^\star)^\top H(\theta - \theta^\star) + c$, $H$ đối xứng xác định dương có trị riêng $\mu = \lambda_{\min} \le \dots \le \lambda_{\max} = L$. Khi đó:
1. GD hội tụ từ **mọi** điểm khởi đầu $\Leftrightarrow 0 < \alpha < 2/L$.
2. Sai số co lại mỗi bước theo hệ số $\rho(\alpha) = \max\big(\lvert 1 - \alpha\mu\rvert,\ \lvert 1 - \alpha L\rvert\big)$.
3. $\alpha = 1/L$ cho $\rho = 1 - 1/\kappa$. Giá trị $\alpha$ tốt nhất là $\alpha^\star = 2/(L + \mu)$, cho $\rho^\star = \dfrac{\kappa - 1}{\kappa + 1}$.

*Chứng minh.* $\nabla L = H(\theta - \theta^\star)$. Đặt $e_t = \theta_t - \theta^\star$, ta có $e_{t+1} = (I - \alpha H)\,e_t$. Viết $H = Q\Lambda Q^\top$ và đổi biến $c_t = Q^\top e_t$: mỗi tọa độ tiến hóa độc lập, $c_{t+1, i} = (1 - \alpha\lambda_i)\,c_{t,i}$. Tọa độ $i$ tiến về 0 với mọi giá trị đầu $\Leftrightarrow \lvert 1 - \alpha\lambda_i\rvert < 1 \Leftrightarrow 0 < \alpha < 2/\lambda_i$; điều kiện chặt nhất là với $\lambda_{\max} = L$. Hàm $\lvert 1 - \alpha\lambda\rvert$ theo $\lambda$ là hàm lồi nên trên $[\mu, L]$ đạt max ở hai đầu mút, cho ra (2). Thay $\alpha = 1/L$: $\max(1 - \mu/L,\ 0) = 1 - 1/\kappa$. Để cực tiểu hóa (2), cân bằng hai đầu $1 - \alpha\mu = \alpha L - 1$, được $\alpha^\star = 2/(L + \mu)$. ∎

*Vì sao định lý cho hàm bậc hai lại quan trọng?* (i) MSE của linear regression **chính là** hàm bậc hai, với $H = X^\top X/n$. (ii) Gần một cực tiểu, **mọi** loss trơn đều xấp xỉ bậc hai (Taylor), nên định lý mô tả hành vi cục bộ của cả mạng neural.

**Hệ quả 6.4 (số bước).** Để sai số giảm $1/\varepsilon$ lần cần $t \approx \dfrac{\ln(1/\varepsilon)}{-\ln\rho}$ bước. Với $\alpha = 1/L$: $t \approx \kappa\ln(1/\varepsilon)$, **tỉ lệ tuyến tính với $\kappa$**.

| $\kappa$ | Số bước để giảm $10^3$ lần, $\alpha = 1/L$ | Với $\alpha^\star = 2/(L + \mu)$ |
|---|---|---|
| 10 | 66 | 35 |
| 100 | 688 | 346 |
| 1 000 | 6 905 | 3 454 |

Đã chạy GD thật trên $H = \mathrm{diag}(1, 100)$ từ $(1, 1)$, dừng khi $\|\theta\| < 10^{-3}$: **688 bước** với $\alpha = 1/L$ (khớp bảng) và **363 bước** với $\alpha^\star$ (hơn 346 một chút vì $\|\theta_0\| = \sqrt{2}$ chứ không phải 1).

**Mệnh đề 6.5 (chuẩn hóa đặc trưng giảm $\kappa$).** Đo trên máy bạn: hai đặc trưng, một có độ lệch chuẩn 1, một có độ lệch chuẩn 100 và trung bình 50, cộng thêm cột bias:

| Dữ liệu | Trị riêng của $X^\top X/n$ | $\kappa$ |
|---|---|---|
| Thô | $0.79,\ 0.98,\ 12\,880$ | **16 318** |
| Sau `standardize` | $0.94,\ 1.00,\ 1.06$ | **1.12** |

Với cùng $\alpha = 1/L$, số bước giảm từ cỡ $10^5$ xuống **vài bước** (bạn sẽ đo chính xác ở project, thí nghiệm `scaling`). Lưu ý cả **trung bình** cũng gây hại: cột có trung bình 50 gần như song song với cột bias, tạo ra trị riêng rất lớn.

**Mệnh đề 6.6 (gradient của mini-batch).** Lấy $B$ mẫu độc lập, đồng xác suất (có hoàn lại) từ $n$ mẫu. Khi đó $\mathbb{E}[\hat g_B] = \nabla L$ (không chệch) và $\mathrm{Var}[\hat g_B] = \sigma^2/B$, với $\sigma^2$ là phương sai của gradient từng mẫu.

*Chứng minh.* Tuyến tính của kỳ vọng; phương sai của trung bình $B$ biến độc lập cùng phân phối bằng phương sai một biến chia $B$ (Ngày 3, bài 09). ∎

Đo trên máy bạn (hồi quy 5 chiều, 4 000 lần lấy mẫu mỗi cỡ batch): tổng phương sai của $\hat g_B$ là **181.9** ($B = 1$), **23.2** ($B = 8$), **2.85** ($B = 64$). Đúng tỉ lệ $1/B$: $181.9/8 = 22.7$ và $181.9/64 = 2.84$.

*Hệ quả:* độ lệch chuẩn của nhiễu chỉ giảm theo $1/\sqrt{B}$. Tăng batch 4 lần thì chi phí mỗi bước gấp 4 nhưng nhiễu chỉ giảm một nửa. Goyal et al. (2017) đưa ra quy tắc thực nghiệm *linear scaling rule*: nhân $B$ lên $k$ lần thì nhân learning rate lên $k$ lần, kèm warmup (bài 07).

**Về saddle point trong không gian nhiều chiều.** Tại một điểm dừng ngẫu nhiên, để là cực tiểu thì **cả $d$ trị riêng** của Hessian phải dương. Với $d$ hàng triệu, điều đó rất hiếm, nên phần lớn các điểm dừng là saddle point (Dauphin et al., NeurIPS 2014). GD chậm lại gần saddle point vì gradient nhỏ; nhiễu của SGD giúp thoát ra.

#### 6.3 Ví dụ tính tay: $\kappa$ trong hai chiều

$L(\theta) = \tfrac{1}{2}(\theta_1^2 + 10\,\theta_2^2)$, nên $H = \mathrm{diag}(1, 10)$, $L = 10$, $\mu = 1$, $\kappa = 10$. Chọn $\alpha = 1/L = 0.1$, xuất phát từ $(1, 1)$:
- Tọa độ 2 nhân với $1 - 0.1 \cdot 10 = 0$: tới đáy sau **1 bước**.
- Tọa độ 1 nhân với $1 - 0.1 \cdot 1 = 0.9$ mỗi bước: sau 10 bước còn $0.9^{10} = 0.349$ (đã chạy: khớp).
- Muốn tọa độ 1 nhanh hơn thì phải tăng $\alpha$, nhưng $\alpha \ge 0.2 = 2/L$ thì tọa độ 2 phân kỳ. **Chiều cong nhất quyết định $\alpha$; chiều phẳng nhất quyết định số bước.**

#### 6.4 Các con số cần biết

| Thông tin | Giá trị |
|---|---|
| Learning rate mặc định của `torch.optim.Adam` | $10^{-3}$ |
| GPT-3 (Brown et al. 2020, Table 2.1): learning rate cực đại | $6.0 \times 10^{-4}$ (model 125M) giảm dần tới $0.6 \times 10^{-4}$ (model 175B): **model càng lớn, lr càng nhỏ** |
| Ngưỡng ổn định của GD trên hàm bậc hai | $\alpha < 2/\lambda_{\max}(H)$ |
| Số bước của GD với $\alpha = 1/L$ | $\approx \kappa\ln(1/\varepsilon)$ |

#### 6.5 Cạm bẫy

1. **Loss thành `nan` sau vài bước.** Nghi phạm số 1 là learning rate quá lớn so với $2/\lambda_{\max}$. Giảm $\alpha$ 3–10 lần rồi thử lại.
2. **Không chuẩn hóa đặc trưng.** $\kappa$ khổng lồ, GD "chạy mãi không xong".
3. **Không xáo trộn dữ liệu mỗi epoch.** Thứ tự cố định (ví dụ sắp theo nhãn) làm gradient bị chệch theo từng đoạn.
4. **Theo dõi loss của mini-batch** để quyết định dừng. Loss đó nhiễu; hãy đo loss trên toàn bộ tập (hoặc tập validation) sau mỗi epoch.
5. **So sánh hai optimizer với cùng một learning rate.** Mỗi optimizer có thang learning rate riêng; phải tune riêng cho từng cái (bài 07).

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"Vì sao chuẩn hóa đặc trưng làm GD nhanh hơn?"* Trả lời bằng Mệnh đề 6.3: số bước $\propto \kappa$, và chuẩn hóa làm $\kappa$ gần 1.
- *"Cực tiểu địa phương có phải vấn đề lớn của deep learning không?"* Ít hơn ta tưởng: điểm dừng chủ yếu là saddle point, và nhiều cực tiểu địa phương có loss gần như nhau.

#### 6.6 Đọc thêm

- Boyd & Vandenberghe, *Convex Optimization*, ch. 3 (hàm lồi) và §9.3 (GD).
- Goodfellow ch. 8 (Optimization for Training Deep Models).
- L. Bottou, F. Curtis, J. Nocedal, *Optimization Methods for Large-Scale Machine Learning*, SIAM Review 2018.
- P. Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, 2017.

---

## 07 · Momentum, RMSProp, Adam và lịch learning rate

> ⏱ Tầng 1: 10' · Tầng 2: 40'
> **Cần biết trước:** Mệnh đề 6.3 (GD trên hàm bậc hai).

> **📖 Từ mới**
> - **Trung bình động lũy thừa (EMA):** trung bình có trọng số, giá trị càng gần đây trọng số càng lớn: $m \leftarrow \beta m + (1 - \beta)g$.
> - **Momentum (quán tính):** cộng dồn các gradient cũ để bước đi có "đà".
> - **Adaptive learning rate:** mỗi tham số có độ dài bước riêng, tự điều chỉnh theo độ lớn gradient của chính nó.
> - **Bias correction:** hiệu chỉnh vì EMA khởi tạo bằng 0 nên các giá trị đầu bị kéo về gần 0.
> - **Weight decay:** mỗi bước kéo trọng số về gần 0 một chút, để model không "học thuộc".
> - **Warmup:** tăng dần learning rate trong các bước đầu. **Cosine schedule:** giảm learning rate theo hình nửa chu kỳ cos.
> - **Gradient clipping:** nếu gradient quá dài thì co lại về một độ dài tối đa.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- **Momentum:** một hòn bi lăn trong thung lũng hẹp. GD thường thì dội qua dội lại giữa hai sườn. Hòn bi có quán tính: các cú dội ngang **triệt tiêu nhau**, còn chuyển động dọc thung lũng **cộng dồn**, nên nó lăn nhanh dần về đích.
- **RMSProp / Adam:** mỗi tham số có một hộp số riêng. Tham số nào gradient luôn lớn thì **về số thấp** (bước nhỏ lại); tham số nào gradient nhỏ thì **lên số** (bước to ra).
- **Warmup và cosine:** lúc khởi động xe, chưa biết đường, nên nhấn ga từ từ; khi gần đích, nhả ga dần để dừng đúng chỗ.
- **Clipping:** giới hạn tốc độ tối đa, phòng khi một cú gradient bất thường làm xe văng khỏi đường.

**Ba ý cốt lõi:**
1. Momentum = EMA của gradient: khuếch đại hướng **ổn định**, triệt tiêu hướng **dao động**.
2. Adam = momentum + chia cho căn của EMA của $g^2$: bước của mỗi tham số **xấp xỉ $\alpha$**, gần như không phụ thuộc độ lớn gradient.
3. Learning rate không cố định mà **đi theo lịch**: warmup, rồi giảm dần.

**Ví dụ nhỏ.** Momentum với $\beta = 0.9$, gradient luôn bằng 1: $v = 1,\ 1.9,\ 2.71,\ \dots \to 10$. Về lâu dài, bước đi **dài gấp 10 lần** GD thường.

---

### TẦNG 2 · Kiến thức chuẩn

#### 7.1 Thuật ngữ chuẩn

| Thuật ngữ | Cập nhật | Nguồn |
|---|---|---|
| **Momentum (heavy ball)** | $v \leftarrow \beta v + g$; $\theta \leftarrow \theta - \alpha v$ (cách viết của PyTorch) | Polyak (1964) |
| **Nesterov momentum** | Tính gradient tại điểm "nhìn trước" $\theta - \alpha\beta v$ | Nesterov (1983); Sutskever et al. (ICML 2013) |
| **AdaGrad** | $s \leftarrow s + g^2$; $\theta \leftarrow \theta - \alpha g/(\sqrt{s} + \epsilon)$. $s$ chỉ tăng, nên bước nhỏ dần về 0 | Duchi et al., JMLR 2011 |
| **RMSProp** | $s \leftarrow \rho s + (1 - \rho)g^2$; $\theta \leftarrow \theta - \alpha g/(\sqrt{s} + \epsilon)$ | Hinton, Coursera *Neural Networks for ML*, lecture 6e (2012) — chưa từng được công bố thành bài báo |
| **Adam** | $m \leftarrow \beta_1 m + (1 - \beta_1)g$; $v \leftarrow \beta_2 v + (1 - \beta_2)g^2$; $\hat m = m/(1 - \beta_1^t)$; $\hat v = v/(1 - \beta_2^t)$; $\theta \leftarrow \theta - \alpha\hat m/(\sqrt{\hat v} + \epsilon)$. Mặc định $\alpha = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ | Kingma & Ba, ICLR 2015, Algorithm 1 |
| **AdamW** | Như Adam, cộng thêm $-\alpha\lambda\theta$ **tách riêng** khỏi phần thích nghi | Loshchilov & Hutter, ICLR 2019 |
| **Cosine annealing** | $\alpha_t = \alpha_{\min} + \tfrac{1}{2}(\alpha_{\max} - \alpha_{\min})\big(1 + \cos(\pi t/T)\big)$ | Loshchilov & Hutter, ICLR 2017 (SGDR) |
| **Linear warmup** | Tăng tuyến tính từ gần 0 lên $\alpha_{\max}$ trong $T_w$ bước đầu | Goyal et al. 2017 |
| **Clipping by global norm** | $g \leftarrow g \cdot \min\big(1,\ c/\|g\|\big)$ | Pascanu, Mikolov, Bengio, ICML 2013 |

#### 7.2 Mệnh đề và chứng minh

**Mệnh đề 7.1 (EMA và bias correction).** Với $m_0 = 0$ và $m_t = \beta m_{t-1} + (1 - \beta)g_t$:

$$m_t = (1 - \beta)\sum_{k=1}^{t}\beta^{t-k}g_k, \qquad \text{tổng các trọng số} = 1 - \beta^t.$$

Nếu $g_k \equiv g$ (hoặc $\mathbb{E}[g_k] = g$ với mọi $k$) thì $\mathbb{E}[m_t] = (1 - \beta^t)\,g$. Vì vậy $\hat m_t = m_t/(1 - \beta^t)$ là ước lượng **không chệch**.

*Chứng minh.* Quy nạp cho công thức tổng. Tổng trọng số là tổng cấp số nhân: $(1 - \beta)(1 + \beta + \dots + \beta^{t-1}) = 1 - \beta^t$. ∎

Với $\beta_2 = 0.999$, ở bước $t = 1$ thì $1 - \beta_2^t = 0.001$: **không hiệu chỉnh** thì $v$ nhỏ hơn giá trị thật 1000 lần.

**Mệnh đề 7.2 (độ dài bước hiệu dụng của momentum).** Với gradient không đổi $g$: $v_t = \dfrac{1 - \beta^t}{1 - \beta}\,g \to \dfrac{g}{1 - \beta}$. Bước dài nhất là $\dfrac{\alpha}{1 - \beta}\,g$: **gấp 10 lần** với $\beta = 0.9$, gấp 100 lần với $\beta = 0.99$.

Trên hàm bậc hai, với $\alpha, \beta$ chọn tối ưu, heavy ball có tốc độ $\dfrac{\sqrt\kappa - 1}{\sqrt\kappa + 1}$ thay vì $\dfrac{\kappa - 1}{\kappa + 1}$ (Polyak 1964). Số bước tỉ lệ với $\sqrt\kappa$ thay vì $\kappa$. Với $\kappa = 100$: hệ số co $0.818$ so với $0.980$.

**Mệnh đề 7.3 (bước đầu tiên của Adam).** Ở $t = 1$: $\hat m_1 = g_1$ và $\hat v_1 = g_1^2$, nên

$$\Delta\theta_1 = -\alpha\frac{g_1}{\lvert g_1\rvert + \epsilon} \approx -\alpha\,\mathrm{sign}(g_1).$$

Mỗi tọa độ dịch đúng khoảng $\alpha$, **bất kể** gradient là 0.001 hay 1000.

*Hệ quả.*
1. Adam **bất biến với việc nhân loss cho một hằng số** (bỏ qua $\epsilon$): $g \to cg$ làm $\hat m \to c\hat m$ và $\sqrt{\hat v} \to \lvert c\rvert\sqrt{\hat v}$.
2. Mỗi bước, $\lvert\Delta\theta_i\rvert$ thường không vượt quá cỡ $\alpha$ (Kingma & Ba, §2.1). Muốn một tham số đi được quãng đường $D$ thì cần **ít nhất khoảng $D/\alpha$ bước**. Ở project, bạn sẽ thấy Adam với $\alpha = 0.01$ chậm hơn SGD chính vì lý do này.
3. RMSProp **không có** bias correction: bước đầu là $\alpha g/\sqrt{(1 - \rho)g^2} = \alpha/\sqrt{1 - \rho} \approx 3.16\alpha$ với $\rho = 0.9$. Bước khởi đầu bị lớn bất thường.

**Mệnh đề 7.4 (L2 regularization ≠ weight decay khi dùng Adam).**
- Với SGD: thêm $\tfrac{\lambda}{2}\|\theta\|^2$ vào loss cho bước $\theta - \alpha(g + \lambda\theta) = (1 - \alpha\lambda)\theta - \alpha g$, **giống hệt** weight decay.
- Với Adam: số hạng $\lambda\theta$ đi qua phép chia cho $\sqrt{\hat v}$, nên tham số có lịch sử gradient lớn bị phạt **ít hơn**. Loshchilov & Hutter (2019) đề xuất **AdamW**: cộng $\alpha\lambda\theta$ ngoài phần thích nghi. AdamW là mặc định khi huấn luyện Transformer.

#### 7.3 Ví dụ tính tay: hai bước Adam

$\alpha = 0.01$, $g_1 = (0.1,\ 10)$, $g_2 = (0.1,\ -10)$ (tọa độ 2 **đổi dấu**). Đã chạy kiểm chứng:

| $t$ | $m$ | $v$ | $\hat m$ | $\hat v$ | $\hat m/\sqrt{\hat v}$ |
|---|---|---|---|---|---|
| 1 | $(0.01,\ 1)$ | $(10^{-5},\ 0.1)$ | $(0.1,\ 10)$ | $(0.01,\ 100)$ | $(1,\ 1)$ |
| 2 | $(0.019,\ -0.1)$ | $(\approx 2 \times 10^{-5},\ 0.1999)$ | $(0.1,\ -0.526)$ | $(0.01,\ 100)$ | $(1,\ -0.053)$ |

- Bước 1: cả hai tọa độ dịch đúng $0.01$, dù gradient chênh nhau 100 lần.
- Bước 2: tọa độ 1 (gradient ổn định) vẫn dịch $0.01$. Tọa độ 2 (gradient đổi dấu) chỉ dịch $0.00053$, nhỏ hơn **19 lần**: momentum tự động **giảm chấn** hướng dao động.

**So sánh trên $H = \mathrm{diag}(1, 100)$** ($\kappa = 100$), xuất phát từ $(1, 1)$, dừng khi $\|\theta\| < 10^{-3}$. Đã chạy:

| Optimizer | Số bước |
|---|---|
| GD, $\alpha = 1/L$ | 688 |
| GD, $\alpha^\star = 2/(L + \mu)$ | 363 |
| Momentum, $\alpha = 1/L$, $\beta = 0.9$ | **126** |
| Momentum, $\alpha, \beta$ tối ưu theo Polyak | **58** |
| Adam, $\alpha = 0.01$ | 276 |
| Adam, $\alpha = 0.1$ | 82 |

#### 7.4 Các con số cần biết

| Thông tin | Giá trị |
|---|---|
| Bộ nhớ thêm của Adam | 2 bản sao tham số ($m$, $v$). Với fp32: **8 byte/tham số**. Model 7B: $7 \times 10^9 \times 8$ B = **56 GB**, chỉ riêng cho optimizer. Lý do LoRA và optimizer 8-bit ra đời (Tuần 11) |
| Cấu hình GPT-3 (Brown et al. 2020) | Adam $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$; clip global norm 1.0; cosine decay; warmup tuyến tính; weight decay 0.1 |
| Cosine với $\alpha_{\max} = 10^{-3}$, $\alpha_{\min} = 0$, $T = 1000$ | $t = 0, 250, 500, 750, 1000$: $10^{-3},\ 8.54 \times 10^{-4},\ 5 \times 10^{-4},\ 1.46 \times 10^{-4},\ 0$ |
| Clip $g = (3, 4)$ với $c = 1$ | $(0.6,\ 0.8)$: giữ hướng, độ dài còn 1 |

**Vì sao cần warmup?** Ở những bước đầu, $\hat v$ được ước lượng từ rất ít mẫu nên phương sai lớn, trong khi trọng số còn ngẫu nhiên và gradient có thể rất lớn. Learning rate lớn ngay từ đầu dễ làm huấn luyện mất ổn định. Liu et al. (ICLR 2020, RAdam) phân tích chính hiện tượng này.

#### 7.5 Code chuẩn

```python
class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, theta, grad):
        if self.m is None:
            self.m, self.v = np.zeros_like(grad), np.zeros_like(grad)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad**2
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)   # trả về mảng mới, không sửa theta
```

#### 7.6 Cạm bẫy

1. **Dùng L2 trong loss với Adam rồi tưởng là weight decay.** Dùng AdamW (Mệnh đề 7.4).
2. **Không lưu trạng thái optimizer khi checkpoint.** Khi chạy tiếp, $m$, $v$, $t$ bị reset, bias correction chạy lại từ đầu và loss nhảy vọt. Phải lưu `optimizer.state_dict()`.
3. **Lịch learning rate đếm theo epoch trong khi code đếm theo step** (hoặc ngược lại). Warmup 1000 "bước" có thể vô tình thành 1000 epoch.
4. **Clip từng tham số riêng lẻ** thay vì clip **global norm**: làm đổi hướng của gradient tổng.
5. **$\epsilon$ quá nhỏ khi huấn luyện fp16.** $10^{-8}$ bị làm tròn về 0 trong fp16; các framework xử lý bằng cách giữ trạng thái optimizer ở fp32.

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"Adam hay SGD?"* Adam hội tụ nhanh, ít phải tune, là mặc định cho Transformer. SGD + momentum đôi khi tổng quát hóa tốt hơn trên bài toán thị giác (Wilson et al., NeurIPS 2017).
- *"Vì sao Adam cần bias correction?"* Mệnh đề 7.1.
- *"Adam tốn bao nhiêu bộ nhớ?"* 2 trạng thái cho mỗi tham số (mục 7.4).

#### 7.7 Đọc thêm

- D. Kingma & J. Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015.
- I. Loshchilov & F. Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019; *SGDR*, ICLR 2017.
- G. Goh, *Why Momentum Really Works*, Distill 2017 (có hình tương tác rất trực quan).
- S. Ruder, *An Overview of Gradient Descent Optimization Algorithms*, arXiv:1609.04747.

---

## Bài tập Ngày 2 · `optlab/grad.py` và `optlab/optim.py` (~2h)

Test: `python -m pytest -q tests/test_grad.py tests/test_optim.py` (19 test).

**`optlab/grad.py`**

| Hàm | Yêu cầu |
|---|---|
| `numerical_grad(f, theta, eps=1e-6)` | Sai phân **trung tâm**, trả về mảng cùng shape với `theta`; **không được làm thay đổi** `theta` đầu vào |
| `rel_error(a, b)` | $\|a - b\|/\max(10^{-12},\ \|a\| + \|b\|)$ |
| `sigmoid(z)` | Không overflow với $\lvert z\rvert$ lớn: với $z \ge 0$ dùng $1/(1 + e^{-z})$, với $z < 0$ dùng $e^z/(1 + e^z)$ |
| `mse_loss(theta, X, y, l2=0.0) -> (loss, grad)` | $L = \frac{1}{2n}\|X\theta - y\|^2 + \frac{\ell_2}{2}\|\theta\|^2$ |
| `logistic_loss(theta, X, y, l2=0.0) -> (loss, grad)` | BCE trung bình viết theo logit bằng `np.logaddexp` (mục 5.4), cộng $\frac{\ell_2}{2}\|\theta\|^2$ |

**`optlab/optim.py`**

| Thành phần | Yêu cầu |
|---|---|
| `SGD(lr)`, `Momentum(lr, beta=0.9)`, `RMSProp(lr, rho=0.9, eps=1e-8)`, `Adam(lr, beta1=0.9, beta2=0.999, eps=1e-8)` | Mỗi class có `step(theta, grad) -> theta_mới`, **trả về mảng mới**, không sửa `theta` hay `grad`. Trạng thái ($v$, $s$, $m$, $t$) khởi tạo ở lần `step` đầu. Công thức đúng như bảng 7.1; `Adam` có thuộc tính `t` |
| `make_optimizer(name, lr)` | `"sgd" / "momentum" / "rmsprop" / "adam"`; tên khác thì raise `ValueError` |
| `clip_by_norm(grad, max_norm)` | Mục 7.1; luôn trả về **mảng mới** |
| `cosine_lr(step, total, base_lr, warmup=0, min_lr=0.0)` | `step < warmup`: trả về `base_lr * (step + 1) / warmup`. Sau đó cosine từ `base_lr` (tại `step = warmup`) về `min_lr` (tại `step = total`); `step > total` thì giữ `min_lr` |

**Gợi ý:** các test của optimizer dùng đúng các ví dụ tính tay của bài 07 (momentum với gradient bằng 1; hai bước Adam ở mục 7.3). Nếu test sai, hãy in ra $m$, $v$ từng bước và so với bảng.
