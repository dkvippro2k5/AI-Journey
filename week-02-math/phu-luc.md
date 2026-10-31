# Phụ lục Tuần 2 · Tối ưu có ràng buộc, lịch learning rate, plateau

> **Loại tài liệu: Lý thuyết (bản nhẹ).** Có công thức và ví dụ tính tay đầy đủ; **không có chứng minh**, chỉ phác thảo "vì sao" vài dòng.
> **Vì sao có phụ lục này:** Phase 2 của roadmap có ba mục mà Tuần 2 chưa dạy: *constrained vs unconstrained optimization*, *step decay* (bài 07 mới có cosine và warmup), và *plateau regions*.
> **Gồm 3 bài:**
> - **P1** Tối ưu có ràng buộc: nhân tử Lagrange, điều kiện KKT
> - **P2** Lịch learning rate: step decay, exponential, ReduceLROnPlateau, và vì sao phải giảm learning rate
> - **P3** Plateau: vì sao loss đứng yên rồi đột ngột giảm
>
> **Thời gian:** đọc ~1h30 · quiz 30' (cuối file).
> **Nhận lại từ Tuần 2:** gradient và Hessian (bài 04), GD và learning rate (bài 06), Adam và cosine/warmup (bài 07), PCA là bài toán tối ưu (bài 03), ridge/lasso (bài 10). Mọi con số trong file đã chạy bằng NumPy 2.5, SciPy 1.18 (`minimize(method="SLSQP")`), scikit-learn 1.9 và PyTorch 2.14 trên máy bạn.

---

## P1 · Tối ưu có ràng buộc: nhân tử Lagrange, điều kiện KKT

> ⏱ Tầng 1: 10' · Tầng 2: 25'
> **Cần biết trước:** gradient chỉ hướng tăng nhanh nhất và vuông góc với đường đồng mức (Tuần 2, bài 04).

> **📖 Từ mới**
> - **Ràng buộc (constraint):** điều kiện mà nghiệm bắt buộc phải thỏa. Ràng buộc **đẳng thức** $g(x) = c$ (phải nằm đúng trên một đường/mặt), ràng buộc **bất đẳng thức** $h(x) \le c$ (phải nằm trong một vùng).
> - **Miền khả thi (feasible set):** tập các điểm thỏa mọi ràng buộc.
> - **Đường đồng mức:** tập các điểm có cùng giá trị $f$, giống đường đồng mức trên bản đồ địa hình.
> - **Nhân tử Lagrange $\lambda$:** con số đo "độ căng" của ràng buộc: nới ràng buộc thêm một chút thì giá trị tối ưu thay đổi bao nhiêu.
> - **Ràng buộc active / inactive:** tại nghiệm, ràng buộc bất đẳng thức đang "chạm biên" ($h = c$) hay còn dư ($h < c$).

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Bạn muốn xuống chỗ **thấp nhất** của một thung lũng, nhưng bắt buộc phải **đi trên một con đường** cắt ngang thung lũng. Đi dọc đường, bạn thấy độ cao giảm dần, rồi tăng lại. Điểm thấp nhất **trên đường** là chỗ con đường **chạm tiếp xúc** với một đường đồng mức: tại đó, đi tiếp theo hướng nào dọc đường cũng không thấp hơn được nữa.

**Ví dụ có số, đi từng bước.** Tìm điểm trên đường thẳng $x + 2y = 5$ gần gốc tọa độ nhất, tức là cực tiểu $f(x, y) = x^2 + y^2$.

*Cách 1: thế biến.* Từ ràng buộc, $x = 5 - 2y$. Thay vào: $f = (5 - 2y)^2 + y^2 = 5y^2 - 20y + 25$. Đạo hàm $10y - 20 = 0$ nên $y = 2$, $x = 1$, và $f = 5$.

*Cách 2: nhìn gradient.* Con đường có hướng đi dọc là $d = (2, -1)$ (thử lại: $(x, y) + t(2, -1)$ vẫn thỏa $x + 2y = 5$). Tại một điểm trên đường, nếu đi theo $d$ làm $f$ thay đổi, thì luôn đi được theo $d$ hoặc $-d$ để **giảm** $f$. Mức thay đổi là $\nabla f \cdot d$:

| Điểm trên đường | $\nabla f = (2x, 2y)$ | $\nabla f \cdot d$ | Kết luận |
|---|---|---|---|
| $(3, 1)$ | $(6, 2)$ | $12 - 2 = 10 > 0$ | Đi theo $-d$ thì $f$ giảm, nên **chưa** tối ưu |
| $(5, 0)$ | $(10, 0)$ | $20 > 0$ | Chưa tối ưu |
| $(1, 2)$ | $(2, 4)$ | $4 - 4 = 0$ | Không còn hướng nào dọc đường để giảm: **tối ưu** |

Tại $(1, 2)$: $\nabla f = (2, 4)$ và gradient của ràng buộc $\nabla g = (1, 2)$ **cùng phương**: $\nabla f = 2\,\nabla g$. Con số $\lambda = 2$ đó chính là **nhân tử Lagrange**.

**$\lambda$ có nghĩa gì?** Nới ràng buộc thành $x + 2y = 5.1$: nghiệm mới là $(1.02,\ 2.04)$ và $f^\star = 5.202$ (đã chạy). Giá trị tối ưu tăng $0.202 \approx \lambda \times 0.1 = 0.2$. Vậy $\lambda$ là **"giá bóng" (shadow price)**: nới ràng buộc một đơn vị thì giá trị tối ưu thay đổi khoảng $\lambda$ đơn vị.

**Ba ý cốt lõi:**
1. Tại nghiệm có ràng buộc đẳng thức, $\nabla f$ **song song** với $\nabla g$: $\nabla f = \lambda \nabla g$.
2. Với ràng buộc bất đẳng thức, hoặc ràng buộc **không chạm** ($\lambda = 0$, nghiệm như không có ràng buộc), hoặc **chạm biên** ($\lambda > 0$). Đây là điều kiện KKT.
3. Nhiều kỹ thuật ML **chính là** bài toán có ràng buộc viết dưới dạng phạt: ridge, SVM, PCA, softmax (max entropy), và cả hàm mục tiêu RLHF ở Tuần 11.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### P1.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Bài toán có ràng buộc** | $\min_x f(x)$ với $g_i(x) = 0$ và $h_j(x) \le 0$ | Boyd & Vandenberghe, *Convex Optimization* (2004), §4.1 |
| **Lagrangian** | $\mathcal L(x, \lambda, \mu) = f(x) + \sum_i \lambda_i g_i(x) + \sum_j \mu_j h_j(x)$ | Lagrange (1788); Boyd §5.1 |
| **Điều kiện KKT** | (1) dừng: $\nabla_x \mathcal L = 0$; (2) khả thi nguyên thủy: $g_i = 0$, $h_j \le 0$; (3) khả thi đối ngẫu: $\mu_j \ge 0$; (4) **bù (complementary slackness)**: $\mu_j h_j(x) = 0$ | Karush (1939); Kuhn & Tucker (1951); Boyd §5.5.3 |
| **Shadow price / sensitivity** | $\partial f^\star / \partial c = \lambda$ khi nới ràng buộc $g(x) = c$ | Boyd §5.6 |
| **Projected gradient descent** | Bước GD rồi **chiếu** về miền khả thi: $x \leftarrow \Pi_C(x - \alpha \nabla f)$ | Boyd & Vandenberghe; Nocedal & Wright, *Numerical Optimization* (2006), ch. 16 |
| **Penalty method** | Thay ràng buộc bằng số hạng phạt trong hàm mục tiêu: $f(x) + \rho \cdot \text{vi phạm}(x)$ | Nocedal & Wright, ch. 17 |

Quy ước dấu: tài liệu này viết ràng buộc bất đẳng thức dạng $h(x) \le 0$, khi đó $\mu \ge 0$. Nhiều sách đổi dấu; điều cần nhớ là **ở nghiệm, $-\nabla f$ chỉ ra ngoài miền khả thi** (muốn đi tiếp xuống dốc nhưng bị biên chặn lại).

**Vì sao (phác thảo).** Ở nghiệm $x^\star$, mọi hướng $d$ đi được dọc ràng buộc (tức $\nabla g \cdot d = 0$) phải có $\nabla f \cdot d = 0$; nếu không thì đi theo $d$ hoặc $-d$ sẽ giảm $f$. Vậy $\nabla f$ vuông góc với **mọi** hướng vuông góc với $\nabla g$, nên $\nabla f$ song song với $\nabla g$. Với bất đẳng thức, nếu biên chưa chạm thì xung quanh $x^\star$ là miền tự do, nên $\nabla f = 0$ và $\mu = 0$.

#### P1.2 Công thức làm bài: các bước giải

**Ràng buộc đẳng thức** $\min f(x)$ với $g(x) = c$:
1. Viết $\nabla f(x) = \lambda \nabla g(x)$, và $g(x) = c$.
2. Giải hệ ($n + 1$ phương trình, $n + 1$ ẩn).
3. Nếu có nhiều nghiệm, so giá trị $f$.

**Ràng buộc bất đẳng thức** $\min f(x)$ với $h(x) \le c$: thử **hai trường hợp**:
1. *Inactive* ($\mu = 0$): giải $\nabla f = 0$. Nếu nghiệm thỏa $h \le c$ thì xong.
2. *Active* ($h = c$): giải như đẳng thức với $\nabla f + \mu \nabla h = 0$. Kiểm tra $\mu \ge 0$; nếu $\mu < 0$ thì nghiệm này không hợp lệ.

#### P1.3 Ví dụ tính tay

**(a) KKT, hai trường hợp.** $\min (x - 3)^2 + (y - 2)^2$ với $x + y \le k$.

- *Thử inactive:* $\nabla f = 0$ cho $(3, 2)$, tổng $3 + 2 = 5$.
  - Với $k = 6$: $5 \le 6$, thỏa. **Nghiệm $(3, 2)$, $\mu = 0$, $f^\star = 0$.** Ràng buộc không có tác dụng.
  - Với $k = 4$: $5 > 4$, vi phạm. Phải thử active.
- *Active với $k = 4$:* $\nabla f + \mu \nabla h = 0$:
  - $2(x - 3) + \mu = 0$ và $2(y - 2) + \mu = 0$, nên $x - 3 = y - 2$, tức $x = y + 1$.
  - Thay vào $x + y = 4$: $2y + 1 = 4$, nên $y = 1.5$ và $x = 2.5$.
  - $\mu = 2(3 - 2.5) = 1 \ge 0$ ✓. $f^\star = 0.25 + 0.25 = 0.5$.

Đã kiểm chứng bằng SLSQP: $k = 4$ cho $(2.5,\ 1.5)$, $f = 0.5$; $k = 6$ cho $(3,\ 2)$, $f = 0$.

**(b) Max entropy cho ra softmax.** Ba trạng thái $x \in \{0, 1, 2\}$. Tìm phân phối $p$ có entropy $H(p) = -\sum p_i \ln p_i$ lớn nhất.
- Chỉ với ràng buộc $\sum p_i = 1$: điều kiện dừng $-\ln p_i - 1 = \lambda$ cho mọi $i$, nên mọi $p_i$ bằng nhau: **phân phối đều** $(1/3, 1/3, 1/3)$, $H = \ln 3 = 1.0986$.
- Thêm ràng buộc trung bình $\sum p_i x_i = m$: điều kiện dừng thành $-\ln p_i - 1 = \lambda + \beta' x_i$, nên $p_i \propto e^{\beta x_i}$. Đó chính là **dạng softmax** (phân phối Gibbs). Chọn $\beta = \ln 2$: $p \propto (1, 2, 4)$, tức $p = (1/7, 2/7, 4/7)$, trung bình $m = (0 + 2 + 8)/7 = 10/7$. SLSQP với ràng buộc $m = 10/7$ cho đúng $(0.14286,\ 0.28571,\ 0.57143)$.

Ý nghĩa: softmax là phân phối **"ít giả định nhất"** khớp với một ràng buộc về giá trị trung bình của điểm số (Jaynes 1957).

**(c) PCA.** $\max w^\top C w$ với $\lVert w\rVert^2 = 1$, $C = \begin{bmatrix}2 & 1\\1 & 2\end{bmatrix}$. Điều kiện dừng: $2Cw = 2\lambda w$, tức **$Cw = \lambda w$**: $w$ là vector riêng, và $w^\top C w = \lambda$. Muốn lớn nhất thì chọn trị riêng lớn nhất: $\lambda = 3$, $w = (1, 1)/\sqrt 2$. Đây là Mệnh đề 3.3 ở Tuần 2, bài 03, nhìn dưới góc Lagrange; SLSQP cho $(0.70711,\ 0.70711)$, giá trị 3.0.

**(d) SVM: complementary slackness chọn ra support vector.** Dữ liệu 1 chiều: lớp $-1$ tại $x = -2, -1$; lớp $+1$ tại $x = 1, 3$. Hard-margin SVM giải $\min \tfrac12 w^2$ với $y_i(wx_i + b) \ge 1$.
- Biên đẹp nhất nằm giữa hai điểm gần nhau nhất của hai lớp ($-1$ và $1$): $b = 0$. Ràng buộc tại $x = 1$: $w \cdot 1 \ge 1$, nên $w = 1$ (nhỏ nhất có thể).
- Kiểm tra $y_i f(x_i)$: $(2, 1, 1, 3)$. Chỉ $x = -1$ và $x = 1$ **chạm biên** (bằng 1).
- Complementary slackness: điểm không chạm biên có $\alpha_i = 0$. Điều kiện dừng $w = \sum \alpha_i y_i x_i$ và $\sum \alpha_i y_i = 0$ cho $\alpha_{-1} = \alpha_{1} = 0.5$.

`SVC(kernel="linear", C=1e6)` cho `coef_ = [[1.]]`, `intercept_ = [-0.]`, support vector $\{-1, 1\}$, `dual_coef_` (tức $\alpha_i y_i$) $= [-0.5,\ 0.5]$: khớp. **Chỉ các support vector quyết định đường biên**; xóa điểm $x = 3$ thì model không đổi.

#### P1.4 Các bài toán ML là bài toán có ràng buộc

| Bài toán | Dạng ràng buộc | Dạng phạt (Lagrangian) | Ghi chú |
|---|---|---|---|
| **Ridge** (Tuần 2, bài 10) | $\min \lVert Xw - y\rVert^2$ với $\lVert w\rVert^2 \le t$ | $\lVert Xw - y\rVert^2 + \alpha\lVert w\rVert^2$ | $\alpha$ chính là $\mu$ của KKT; $t$ nhỏ ↔ $\alpha$ lớn |
| **Lasso** | $\lVert w\rVert_1 \le t$ | $+\ \alpha\lVert w\rVert_1$ | Miền $L_1$ là hình thoi có **góc nhọn trên các trục**; đường đồng mức thường chạm ở góc, nên nhiều $w_j = 0$ |
| **SVM** | $y_i(w^\top x_i + b) \ge 1$ | Dual theo $\alpha_i \ge 0$ | Support vector là điểm có $\alpha_i > 0$ |
| **PCA** | $\lVert w\rVert = 1$ | $w^\top Cw - \lambda(w^\top w - 1)$ | $\lambda$ là trị riêng |
| **Softmax / max entropy** | $\sum p = 1$, $\mathbb E[x] = m$ | | $p \propto e^{\beta x}$ |
| **RLHF** (Tuần 11) | $\max \mathbb E[r]$ với $\mathrm{KL}(\pi \,\Vert\, \pi_\text{ref}) \le \epsilon$ | $\mathbb E[r] - \beta\,\mathrm{KL}$ | $\beta$ là nhân tử; DPO bắt đầu từ chính dạng này |
| **Gradient clipping** (Tuần 4, bài 04) | bước cập nhật có $\lVert \Delta\rVert \le c$ | | Chính là phép chiếu lên quả cầu bán kính $c$ |

**Kiểm chứng ridge ↔ ràng buộc (đã chạy).** $A$ là ma trận $50 \times 5$ ngẫu nhiên, nghiệm OLS có chuẩn 3.76. Giải $\min \lVert Aw - b\rVert^2$ với $\lVert w\rVert \le 1$ bằng **projected GD** (2 000 bước, mỗi bước GD rồi chiếu $w \leftarrow w \cdot \min(1, 1/\lVert w\rVert)$):
- Projected GD và SLSQP cho **cùng** $w = (0.686,\ -0.594,\ 0.301,\ -0.140,\ 0.257)$, với $\lVert w\rVert = 1$ (ràng buộc active).
- Tìm $\alpha$ của ridge sao cho $\lVert w_\text{ridge}\rVert = 1$: $\alpha = 89.54$, và $w_\text{ridge}$ **trùng** với nghiệm trên.
- Tính $\mu$ từ điều kiện dừng $\nabla f + 2\mu w = 0$ cho từng tọa độ: cả 5 tọa độ đều cho $\mu = 89.544$. Nhân tử KKT **bằng đúng** hệ số phạt của ridge.

#### P1.5 Code chuẩn

```python
import numpy as np
from scipy.optimize import minimize

# (1) Bộ giải tổng quát cho bài nhỏ: SLSQP. "ineq" nghĩa là fun(x) >= 0.
res = minimize(lambda v: (v[0] - 3)**2 + (v[1] - 2)**2, x0=[0, 0], method="SLSQP",
               constraints=[{"type": "ineq", "fun": lambda v: 4 - v[0] - v[1]}])
res.x, res.fun        # [2.5 1.5] 0.5

# (2) Projected GD: khi phép chiếu rẻ (hộp, quả cầu, simplex) thì đây là cách chuẩn cho bài lớn.
def project_ball(w, r):
    n = np.linalg.norm(w)
    return w if n <= r else w * (r / n)

def projected_gd(grad, w0, project, lr, steps):
    w = w0.copy()
    for _ in range(steps):
        w = project(w - lr * grad(w))
    return w

# ràng buộc hộp lo <= w <= hi: phép chiếu là np.clip(w, lo, hi)
```

**Lý do thiết kế.** SLSQP tiện cho bài vài chục biến và để **kiểm tra** đáp án tính tay. Với model nhiều tham số, người ta dùng (i) dạng phạt (ridge, weight decay, hệ số KL), (ii) projected GD khi phép chiếu rẻ, hoặc (iii) **đổi biến** để khử ràng buộc: muốn $p$ là phân phối thì tham số hóa $p = \mathrm{softmax}(z)$; muốn $\sigma > 0$ thì học $\log \sigma$.

#### P1.6 Cạm bẫy

1. **Quên kiểm tra $\mu \ge 0$** trong trường hợp active: nghiệm tìm được có thể là điểm mà ràng buộc đang "kéo ngược", không phải nghiệm.
2. **Chỉ thử trường hợp active:** nếu nghiệm không ràng buộc đã khả thi thì đáp án là nó, với $\mu = 0$.
3. **Nhầm quy ước dấu** của `scipy`: `"ineq"` nghĩa là `fun(x) >= 0`, ngược với cách viết $h(x) \le 0$ trong sách.
4. **Nghĩ rằng ridge với $\alpha$ cố định là "ràng buộc $\lVert w\rVert \le t$ với $t$ cố định":** cặp $(\alpha, t)$ tương ứng **phụ thuộc dữ liệu**. Đổi dữ liệu thì cùng $\alpha$ cho $t$ khác.
5. **Điều kiện Lagrange chỉ là điều kiện cần.** Với bài không lồi (như PCA ở trên), mọi vector riêng đều thỏa điều kiện dừng; phải so giá trị $f$ để chọn.

**Sắc thái phỏng vấn:**
- *"Vì sao lasso cho nghiệm thưa còn ridge thì không?"* Vẽ bằng lời: miền $L_1$ là hình thoi có góc nằm trên trục; đường đồng mức elip của loss thường chạm hình thoi **ở góc**, tức nhiều tọa độ bằng 0. Miền $L_2$ là hình tròn trơn, không có góc, nên điểm chạm thường có mọi tọa độ khác 0.
- *"Hệ số $\beta$ trong hàm mục tiêu RLHF có ý nghĩa gì?"* Nó là nhân tử Lagrange của ràng buộc "không đi xa model gốc quá $\epsilon$ theo KL". $\beta$ lớn ↔ $\epsilon$ nhỏ.

#### P1.7 Đọc thêm

- S. Boyd & L. Vandenberghe, *Convex Optimization* (2004), ch. 5 (duality, KKT). Bản PDF miễn phí trên trang của tác giả.
- Deisenroth, Faisal, Ong, *Mathematics for Machine Learning* (2020), §7.2 (constrained optimization).
- C. Bishop, *Pattern Recognition and Machine Learning* (2006), Appendix E (Lagrange multipliers), §7.1 (SVM).
- E. T. Jaynes, *Information Theory and Statistical Mechanics*, Physical Review 106 (1957).

---

## P2 · Lịch learning rate: step decay, exponential, ReduceLROnPlateau

> ⏱ Tầng 1: 10' · Tầng 2: 20'
> **Cần biết trước:** SGD và nhiễu của minibatch (Tuần 2, bài 06); cosine và warmup (Tuần 2, bài 07).

> **📖 Từ mới**
> - **Learning rate schedule:** quy tắc thay đổi learning rate $\alpha_t$ theo bước hoặc epoch.
> - **Step decay:** giữ $\alpha$ không đổi rồi **cắt bậc thang** (thường chia 10) tại vài mốc.
> - **ReduceLROnPlateau:** giảm $\alpha$ khi một metric (thường là val loss) **ngừng cải thiện** trong một số epoch.
> - **Noise floor (sàn nhiễu):** mức loss mà SGD với $\alpha$ cố định dao động quanh đó mãi, không xuống thấp hơn được.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Bạn tìm một chiếc chìa khóa rơi trong sân bằng cách **bước theo hướng có tiếng leng keng**, nhưng tai bị ù (nhiễu). Bước dài thì đi nhanh tới gần, nhưng sau đó **cứ bước qua bước lại quanh** chiếc chìa khóa. Muốn nhặt được thì phải **bước ngắn lại**. Learning rate lớn giúp đi nhanh lúc đầu; learning rate nhỏ giúp đứng yên đúng chỗ lúc cuối. Schedule là quy tắc "khi nào thì bước ngắn lại".

**Ví dụ có số, đi từng bước: vì sao $\alpha$ cố định không xuống tới đáy.** Xét bài 1 chiều $f(x) = \tfrac12 x^2$ (đáy tại $x = 0$), gradient đo bị nhiễu: $g = x + \xi$, với $\xi$ có trung bình 0, phương sai 1. Một bước SGD:

$$x_{t+1} = x_t - \alpha(x_t + \xi_t) = (1 - \alpha)\,x_t - \alpha\,\xi_t.$$

Mỗi bước có **hai lực** tác động lên phương sai $V = \mathrm{Var}(x)$:
- **Kéo về đáy:** nhân $x$ với $(1 - \alpha)$, nên phương sai nhân với $(1 - \alpha)^2$.
- **Đẩy ra do nhiễu:** cộng thêm $\alpha^2 \cdot 1$.

Khi hai lực cân bằng, $V$ không đổi: $V = (1 - \alpha)^2 V + \alpha^2$. Thay số với $\alpha = 0.1$:
- $(1 - 0.1)^2 = 0.81$, nên $V - 0.81V = 0.01$, tức $0.19\,V = 0.01$, và $V = 0.0526$.
- Loss trung bình: $\mathbb E[\tfrac12 x^2] = V/2 = 0.0263$. Đây là **sàn nhiễu**: chạy bao lâu cũng chỉ dao động quanh mức này.

Với $\alpha = 0.01$: $0.0199\,V = 0.0001$, nên $V = 0.00503$ và sàn là $0.00251$. **Giảm $\alpha$ 10 lần thì sàn giảm khoảng 10 lần.** Mô phỏng 400 000 bước trên máy bạn: 0.02625 và 0.00253, khớp với tính tay.

Nhưng $\alpha$ nhỏ thì **đi chậm** lúc đầu. Muốn có cả hai, hãy dùng $\alpha$ lớn lúc đầu rồi giảm dần: đó là toàn bộ lý do tồn tại của learning rate schedule.

**Ba ý cốt lõi:**
1. Với SGD, $\alpha$ cố định để lại một **sàn nhiễu tỉ lệ với $\alpha$**; giảm $\alpha$ về cuối là để hạ sàn này.
2. Các schedule khác nhau ở **khi nào** và **giảm bao nhiêu**: theo mốc định trước (step, cosine), hay theo phản hồi từ dữ liệu (plateau).
3. Schedule tốt thường **quan trọng ngang** việc chọn optimizer.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### P2.1 Thuật ngữ và công thức

| Schedule | Công thức ($t$ là epoch hoặc step) | PyTorch | Nguồn / nơi dùng |
|---|---|---|---|
| **Step decay** | $\alpha_t = \alpha_0\,\gamma^{\lfloor t/s\rfloor}$ | `StepLR(step_size=s, gamma=γ)` | ResNet: chia 10 khi lỗi chững lại (He et al., CVPR 2016); ImageNet 90 epoch: chia 10 tại epoch 30, 60, 80 (Goyal et al. 2017) |
| **Multi-step** | Nhân $\gamma$ tại các mốc $m_1 < m_2 < \dots$ | `MultiStepLR(milestones, gamma)` | như trên |
| **Exponential** | $\alpha_t = \alpha_0\,\gamma^t$ | `ExponentialLR(gamma)` | |
| **Cosine** | $\alpha_{\min} + \tfrac12(\alpha_0 - \alpha_{\min})(1 + \cos(\pi t/T))$ | `CosineAnnealingLR(T_max, eta_min)` | Loshchilov & Hutter, ICLR 2017 |
| **Linear warmup** | Tăng tuyến tính tới $\alpha_0$ trong $T_w$ bước | `LinearLR` + `SequentialLR` | Goyal et al. 2017; Transformer (Vaswani et al. 2017) |
| **ReduceLROnPlateau** | Nhân `factor` khi metric không cải thiện quá `patience` epoch | `ReduceLROnPlateau(mode, factor, patience, threshold)` | Keras/PyTorch; dùng nhiều khi fine-tune |
| **Warmup–stable–decay (WSD)** | Warmup → giữ hằng số → giảm nhanh ở ~10% cuối | — | Dùng cho pretraining LLM khi chưa biết trước tổng số bước (Hu et al., *MiniCPM*, 2024) |

#### P2.2 Ví dụ tính tay (đã đối chiếu với scheduler của PyTorch)

**Step decay**, $\alpha_0 = 0.1$, $\gamma = 0.1$, $s = 30$. Công thức: $\alpha_t = 0.1 \times 0.1^{\lfloor t/30\rfloor}$.

| $t$ | $\lfloor t/30\rfloor$ | $\alpha_t$ |
|---|---|---|
| 0, 29 | 0 | 0.1 |
| 30, 45, 59 | 1 | 0.01 |
| 60 | 2 | 0.001 |
| 90 | 3 | 0.0001 |

**Exponential**, $\gamma = 0.95$: $\alpha_{10} = 0.1 \times 0.95^{10} = 0.0599$; $\alpha_{20} = 0.0358$. Muốn biết sau bao nhiêu epoch thì $\alpha$ giảm một nửa: $0.95^t = 0.5$, nên $t = \ln 0.5 / \ln 0.95 \approx 13.5$.

**Cosine**, $\alpha_0 = 0.1$, $\alpha_{\min} = 0.001$, $T = 100$: $t = 25 \to 0.0855$; $t = 50 \to 0.0505$; $t = 75 \to 0.0155$; $t = 100 \to 0.001$.

**Warmup 5 epoch rồi cosine** (`LinearLR(start_factor=0.1, total_iters=5)`): các epoch 0–5 có $\alpha$ = 0.01, 0.028, 0.046, 0.064, 0.082, 0.1. Mỗi epoch tăng $(0.1 - 0.01)/5 = 0.018$.

**ReduceLROnPlateau, lần theo từng epoch.** `mode="min"`, `factor=0.5`, `patience=2`, `threshold=0.01` (tương đối: chỉ tính là "cải thiện" nếu $v < \text{best} \times (1 - 0.01)$). $\alpha$ ban đầu 0.1. Val loss theo epoch: `1.00, 0.80, 0.70, 0.69, 0.695, 0.70, 0.69, 0.689, 0.70, 0.70, 0.71`.

| Epoch | val | Ngưỡng $\text{best} \times 0.99$ | Cải thiện? | `bad` | Hành động | $\alpha$ cho epoch sau |
|---|---|---|---|---|---|---|
| 0 | 1.00 | — | có (lần đầu) | 0 | best = 1.00 | 0.1 |
| 1 | 0.80 | 0.990 | có | 0 | best = 0.80 | 0.1 |
| 2 | 0.70 | 0.792 | có | 0 | best = 0.70 | 0.1 |
| 3 | 0.69 | 0.693 | có | 0 | best = 0.69 | 0.1 |
| 4 | 0.695 | 0.6831 | không | 1 | | 0.1 |
| 5 | 0.70 | 0.6831 | không | 2 | | 0.1 |
| 6 | 0.69 | 0.6831 | không | 3 > 2 | **giảm**, `bad` = 0 | **0.05** |
| 7 | 0.689 | 0.6831 | không | 1 | | 0.05 |
| 8 | 0.70 | 0.6831 | không | 2 | | 0.05 |
| 9 | 0.70 | 0.6831 | không | 3 > 2 | **giảm** | **0.025** |

Đối chiếu `ReduceLROnPlateau` của PyTorch: $\alpha$ trước mỗi epoch là `[0.1 × 7, 0.05 × 3, 0.025, …]`, khớp. Với `threshold=1e-4` thì epoch 7 (0.689 < 0.69 × 0.9999) **được tính** là cải thiện, nên lần giảm thứ hai lùi tới sau epoch 10. Cả `threshold` lẫn `patience` đều quyết định độ nhạy.

#### P2.3 Đo trên máy bạn: so sánh các schedule

Hồi quy tuyến tính 20 chiều, 8 000 mẫu, SGD minibatch 32, 30 epoch; đo **excess loss** $L(w) - L(w^\star)$ ở cuối mỗi epoch, trung bình 5 seed.

| Schedule | Epoch 1 | Epoch 10 | Epoch 20 | Epoch 30 |
|---|---|---|---|---|
| Hằng số $\alpha = 0.05$ | $5.4 \times 10^{-3}$ | $7.3 \times 10^{-3}$ | $6.4 \times 10^{-3}$ | $7.4 \times 10^{-3}$ |
| Hằng số $\alpha = 0.005$ | $8.8 \times 10^{-1}$ | $1.0 \times 10^{-4}$ | $7.7 \times 10^{-5}$ | $1.0 \times 10^{-4}$ |
| Step ($0.05$, chia 10 mỗi 10 epoch) | $5.4 \times 10^{-3}$ | $7.3 \times 10^{-3}$ | $7.7 \times 10^{-5}$ | $\mathbf{6.1 \times 10^{-6}}$ |
| Cosine ($0.05 \to 0$) | $5.4 \times 10^{-3}$ | $5.2 \times 10^{-3}$ | $7.4 \times 10^{-4}$ | $\mathbf{4.9 \times 10^{-6}}$ |
| Plateau ($0.05$, factor 0.5, patience 2) | $5.4 \times 10^{-3}$ | $1.9 \times 10^{-3}$ | $1.5 \times 10^{-4}$ | $2.1 \times 10^{-5}$ |

**Đọc bảng:**
- $\alpha = 0.05$ cố định: xuống rất nhanh ở epoch 1, rồi **dừng ở sàn nhiễu** ~$7 \times 10^{-3}$ suốt 29 epoch còn lại.
- $\alpha = 0.005$ cố định: epoch 1 còn rất tệ (0.88), nhưng sàn thấp hơn nhiều.
- Step decay **lấy được cả hai**: ngay sau mốc epoch 10, loss rơi xuống bằng sàn của $\alpha = 0.005$; sau mốc thứ hai, xuống thêm hơn 10 lần.
- Cosine giảm êm hơn và cho kết quả cuối tương đương step. Plateau tự động, nhưng phụ thuộc `patience` và tới epoch 30 chưa giảm đủ sâu.

(Ghi chú trung thực: mô hình 1 chiều ở Tầng 1 dự đoán "sàn tỉ lệ với $\alpha$". Ở đây, vì đo ở **cuối epoch** với dữ liệu được xáo lại mỗi epoch, hai sàn chênh nhau ~70 lần chứ không phải 10 lần. Chiều hướng giống nhau, còn tỉ lệ chính xác phụ thuộc cách lấy mẫu.)

#### P2.4 Code chuẩn

```python
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
sched = StepLR(opt, step_size=30, gamma=0.1)            # schedule theo EPOCH

for epoch in range(90):
    for xb, yb in loader:
        opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    sched.step()                                         # SAU vòng lặp batch, mỗi epoch một lần

# theo phản hồi: phải truyền metric
plateau = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, threshold=0.01, min_lr=1e-6)
for epoch in range(n_epochs):
    train_one_epoch(...)
    val = evaluate(...)
    plateau.step(val)

# lưu checkpoint: lưu CẢ trạng thái scheduler
torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch}, path)
```

**Chọn schedule thế nào:**

| Tình huống | Gợi ý | Vì sao |
|---|---|---|
| Biết trước số epoch/step, huấn luyện từ đầu | Warmup + cosine (hoặc linear decay) | Mặc định của Transformer và LLM; ít tham số cần chỉnh |
| Tái hiện công thức CNN kinh điển | Multi-step chia 10 | Để so sánh được với paper |
| Fine-tune, không chắc cần bao nhiêu epoch | ReduceLROnPlateau + early stopping | Tự thích nghi theo val loss |
| Pretraining dài, muốn dừng/tiếp tục linh hoạt | WSD | Pha "stable" kéo dài tùy ý; chỉ cần pha decay ngắn ở cuối |
| Adam/AdamW | Vẫn nên có warmup + decay | Adam tự chỉnh **theo từng tọa độ** nhưng không giảm **mức nhiễu chung**; schedule vẫn có ích |

#### P2.5 Cạm bẫy

1. **Gọi `sched.step()` theo step trong khi schedule được thiết kế theo epoch** (hoặc ngược lại): `StepLR(step_size=30)` gọi mỗi batch sẽ chia 10 sau **30 batch**. Lỗi "1000 bước thành 1000 epoch" ở Tuần 2, bài 07 là cùng loại.
2. **Gọi `sched.step()` trước `opt.step()`**: PyTorch cảnh báo, và giá trị đầu tiên của schedule bị bỏ qua.
3. **Quên truyền metric** cho `ReduceLROnPlateau.step(val)`: raise lỗi. Truyền **train** loss thay vì val loss thì scheduler phản ứng với overfitting theo hướng sai.
4. **Resume từ checkpoint mà không khôi phục scheduler:** $\alpha$ nhảy về giá trị ban đầu giữa chừng, loss vọt lên.
5. **`patience` quá ngắn với val metric nhiễu:** cắt $\alpha$ quá sớm. Dùng `threshold` hợp lý, `patience` vài epoch, và `min_lr` để không giảm về 0.

**Sắc thái phỏng vấn:** *"Vì sao không cứ để learning rate nhỏ từ đầu cho an toàn?"* Vì (i) đi chậm ở pha đầu (bảng trên: 0.88 so với 0.0054 sau một epoch); (ii) $\alpha$ lớn kèm nhiễu giúp thoát vùng phẳng và saddle point (P3), và có bằng chứng thực nghiệm rằng nó dẫn tới nghiệm tổng quát hóa tốt hơn. Lịch "lớn trước, nhỏ sau" lấy được cả hai lợi ích.

#### P2.6 Đọc thêm

- K. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 (§3.4: lịch chia 10).
- P. Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, 2017.
- I. Loshchilov & F. Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, ICLR 2017.
- PyTorch docs: `torch.optim` → *How to adjust learning rate*.
- L. Bottou, F. Curtis, J. Nocedal, *Optimization Methods for Large-Scale Machine Learning*, SIAM Review 2018, §4 (SGD với bước cố định và bước giảm dần).

---

## P3 · Plateau: vì sao loss đứng yên rồi đột ngột giảm

> ⏱ Tầng 1: 10' · Tầng 2: 20'
> **Cần biết trước:** saddle point và Hessian (Tuần 2, bài 04 và 06); đạo hàm của sigmoid $\le 1/4$ (Tuần 2, bài 05); Adam (Tuần 2, bài 07).

> **📖 Từ mới**
> - **Plateau (cao nguyên):** vùng mà loss **gần như bằng phẳng**: gradient rất nhỏ, nhưng loss **còn cao**, chưa phải đáy.
> - **Bão hòa (saturation):** activation như sigmoid/tanh nhận đầu vào lớn nên đầu ra gần 0 hoặc 1, và đạo hàm gần 0.
> - **Đối xứng (symmetry):** các neuron/tham số bắt đầu giống hệt nhau nên nhận gradient giống hệt nhau và không "tách" ra được.
> - **Thoát (escape):** rời khỏi vùng phẳng để đi xuống tiếp.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Bạn xuống núi trong **sương mù dày**, chỉ cảm nhận được độ dốc ngay dưới chân. Gặp một **cao nguyên bằng phẳng**: dốc gần như bằng 0, nên mỗi bước (tỉ lệ với độ dốc) rất ngắn. Bạn dễ tưởng mình **đã tới đáy**, nhưng thực ra vẫn đang ở trên cao. Đi đủ lâu tới mép cao nguyên, dốc đột ngột xuất hiện, và bạn tụt xuống rất nhanh.

**Ví dụ có số, đi từng bước.** Mạng tuyến tính "sâu" nhỏ nhất có thể: hai tham số nhân với nhau, $\hat y = a \cdot b$, cần học $\hat y = 1$. Loss $L = \tfrac12(1 - ab)^2$. Khởi tạo nhỏ $a = b = 0.001$, GD với $\alpha = 0.1$.

Gradient: $\partial L / \partial a = -(1 - ab)\,b$. Khi $a, b$ còn nhỏ, $1 - ab \approx 1$, nên $\partial L/\partial a \approx -b$. Một bước GD: $a \leftarrow a + 0.1\,b$. Vì $a = b$ (đối xứng), ta có $a \leftarrow 1.1\,a$: **mỗi bước chỉ nhân $a$ với 1.1**.

| Bước | $a \approx 0.001 \times 1.1^t$ | $ab$ | Loss $\tfrac12(1 - ab)^2$ |
|---|---|---|---|
| 0 | 0.001 | $10^{-6}$ | 0.5000 |
| 24 | 0.0099 | 0.0001 | 0.4999 |
| 48 | 0.097 | 0.0094 | 0.4907 |
| 60 | 0.30 | 0.093 | ≈ 0.41 |
| 72 | 0.96 | — | (pha cuối, xấp xỉ $1 - ab \approx 1$ không còn đúng) |

Loss **đo thật** trên máy bạn: bước 0: 0.500; bước 20: 0.500; bước 40: 0.498; bước 60: 0.418; bước 65: 0.321; bước 70: 0.180; bước 75: 0.061; bước 80: 0.012. **69% số bước** nằm trên cao nguyên (loss > 0.45), rồi loss tụt từ 0.42 xuống 0.01 chỉ trong 20 bước.

Cao nguyên dài bao lâu? Cần $a$ tăng từ $\varepsilon$ tới cỡ 1, mỗi bước nhân $(1 + \alpha)$:

$$t_\text{plateau} \approx \frac{\ln(1/\varepsilon)}{\ln(1 + \alpha)}.$$

Với $\varepsilon = 0.001$, $\alpha = 0.1$: $6.908 / 0.0953 = 72.5$ bước, khớp với bảng. Khởi tạo **nhỏ đi 10 lần** thì cao nguyên dài thêm $\ln 10 / \ln 1.1 = 24.2$ bước. Đo thật (số bước tới khi loss < 0.005):

| $\varepsilon$ | 0.1 | 0.01 | 0.001 | 0.0001 | $10^{-6}$ |
|---|---|---|---|---|---|
| Số bước đo được | 34 | 59 | 83 | 107 | 155 |
| Chênh so với cột trước | | +25 | +24 | +24 | +48 (2 bậc) |

Mỗi bậc 10 thêm đúng ~24 bước như dự đoán. Phần chênh cố định ~10 bước so với công thức là pha cuối, khi xấp xỉ $1 - ab \approx 1$ không còn đúng.

**Ba ý cốt lõi:**
1. Plateau là vùng **gradient nhỏ mà loss còn cao**. Nó **không** phải là đã hội tụ.
2. Nguyên nhân hay gặp: khởi tạo quá nhỏ hoặc đối xứng (gần một saddle point), activation bão hòa, learning rate quá nhỏ.
3. Momentum và đặc biệt Adam thoát plateau nhanh hơn GD nhiều, vì chúng không để bước đi tỉ lệ thuận với độ lớn của gradient.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### P3.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Plateau** | Vùng có $\lVert\nabla L\rVert$ rất nhỏ trên một khoảng rộng, trong khi $L$ còn xa giá trị tối ưu | Goodfellow, Bengio, Courville, *Deep Learning* (2016), §8.2.3 |
| **Saddle point trong không gian nhiều chiều** | Phần lớn điểm dừng của loss mạng neural là saddle point, và xung quanh chúng thường là các plateau | Dauphin et al., NeurIPS 2014 |
| **Động lực học của mạng tuyến tính sâu** | Với khởi tạo nhỏ, loss có các plateau xen kẽ những pha giảm đột ngột; thời gian plateau $\propto \ln(1/\varepsilon)$ | Saxe, McClelland, Ganguli, ICLR 2014 |
| **Saturation** | $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ giảm theo $\approx e^{-\lvert z\rvert}$ khi $\lvert z\rvert$ lớn | LeCun et al., *Efficient BackProp* (1998) |
| **Symmetry breaking** | Khởi tạo **ngẫu nhiên** để các neuron khác nhau; khởi tạo bằng nhau thì chúng mãi giống nhau | Goodfellow et al. §8.4 |

#### P3.2 Các nguyên nhân và cách nhận ra

| Nguyên nhân | Cơ chế | Con số | Dấu hiệu |
|---|---|---|---|
| **Khởi tạo nhỏ / gần saddle** | Tín hiệu và gradient đều là **tích** của các tham số nhỏ | Ví dụ ở Tầng 1: thời gian plateau $\approx \ln(1/\varepsilon)/\ln(1 + \alpha)$ | Loss đứng đúng ở mức "đoán hằng số"; norm của trọng số tăng dần theo hàm mũ |
| **Activation bão hòa** | $\sigma'(z) \approx e^{-\lvert z\rvert}$ | $\sigma'(0) = 0.25$; $\sigma'(5) = 0.0066$ (nhỏ hơn 38 lần); $\sigma'(8) = 0.000335$ (nhỏ hơn **746 lần**) | Tỉ lệ activation sát 0 hoặc 1 cao |
| **Dead ReLU** | $z < 0$ với mọi mẫu nên gradient bằng 0 vĩnh viễn | — | Tỉ lệ neuron có đầu ra luôn bằng 0 |
| **Learning rate quá nhỏ** | Bước nhỏ ở mọi nơi | — | Tăng $\alpha$ 3–10 lần thì loss giảm ngay |
| **Điều kiện xấu** (Tuần 2, bài 06) | Thung lũng hẹp và dài: đi ngang rất chậm | $\kappa$ bước | Loss giảm đều nhưng rất chậm, gradient không nhỏ |

**Mức loss "đoán hằng số" để đối chiếu.** Phân loại 2 lớp cân bằng: $\ln 2 = 0.693$; $K$ lớp: $\ln K$ (10 lớp: 2.303). Nếu train loss đứng đúng ở các mức này thì model **chưa học được gì**, chứ không phải đã hội tụ. Tuần 4, bài 04: `TabMLP` khởi tạo std 0.01 đứng ở 0.694 trong epoch đầu; MLP 10 lớp sigmoid với SGD kẹt ở accuracy 10% (đúng mức đoán bừa cho 10 lớp).

#### P3.3 Đo trên máy bạn: optimizer nào thoát nhanh hơn

Cùng bài $L = \tfrac12(1 - ab)^2$; số bước để loss < 0.005:

| Optimizer | $\varepsilon = 10^{-2}$ | $10^{-3}$ | $10^{-6}$ |
|---|---|---|---|
| GD, $\alpha = 0.1$ | 59 | 83 | 155 |
| Momentum 0.9, $\alpha = 0.1$ | 19 | 33 | 58 |
| Momentum 0.9, $\alpha = 0.01$ | 83 | 120 | 230 |
| Adam, $\alpha = 0.01$ | 81 | 82 | **82** |
| Adam, $\alpha = 0.1$ | 10 | 11 | **11** |

**Giải thích từng dòng:**
- **GD:** bước tỉ lệ với gradient, mà gradient tỉ lệ với $a$, nên $a$ tăng theo cấp số nhân với công bội $1 + \alpha$. Thời gian tăng theo $\ln(1/\varepsilon)$.
- **Momentum:** vận tốc tích lũy các gradient cùng hướng, nên công bội hiệu dụng lớn hơn. Nhanh hơn GD ~2.5–3 lần, nhưng thời gian **vẫn tăng** theo $\ln(1/\varepsilon)$.
- **Adam:** bước cập nhật là $\hat m / \sqrt{\hat v}$. Khi gradient đều một hướng thì tỉ số này ≈ 1, **bất kể gradient lớn hay nhỏ**. Mỗi bước $a$ tăng khoảng $\alpha$ theo **cấp số cộng**, nên thời gian ≈ (khoảng cần đi)/$\alpha$ và **gần như không phụ thuộc $\varepsilon$**: $\varepsilon = 10^{-2}$ hay $10^{-6}$ đều mất 81–82 bước với $\alpha = 0.01$.

Đây cũng là lý do ở Tuần 4, bài 04, Adam vẫn học được với khởi tạo std 0.01, còn SGD thì kẹt: Adam **tự phóng to** gradient nhỏ.

#### P3.4 Chẩn đoán và xử lý

Khi loss đứng yên, đi theo thứ tự:
1. **So với mức "đoán hằng số"** ($\ln 2$, $\ln K$, hoặc phương sai của $y$ với hồi quy). Bằng mức đó tức là chưa học.
2. **Log chuẩn gradient theo từng lớp** (Tuần 4, bài 04). Rất nhỏ ở mọi lớp: plateau hoặc bão hòa. Chỉ nhỏ ở các lớp đầu: vanishing gradient.
3. **Tăng thử learning rate** 3–10 lần trong vài trăm bước (hoặc chạy *LR range test*). Loss giảm ngay thì nguyên nhân là $\alpha$ quá nhỏ.
4. **Kiểm tra khởi tạo** (He/Xavier, không khởi tạo hằng số), **activation** (ReLU/GELU thay sigmoid ở lớp ẩn), **chuẩn hóa** đầu vào và dùng norm layer.
5. **Đổi optimizer:** Adam/AdamW với warmup.
6. Nếu đã làm hết mà val loss đứng yên trong khi train loss vẫn giảm: đó **không phải** plateau mà là **overfitting**. Dùng early stopping, regularization (Tuần 4, bài 06).

**Plateau trong lúc fine-tune.** Val loss đứng yên vài epoch rồi giảm tiếp là chuyện thường. Vì vậy `ReduceLROnPlateau` và early stopping đều có `patience`: đừng kết luận sau một epoch.

#### P3.5 Code chuẩn: theo dõi để nhận ra plateau

```python
import math, torch

def grad_norms(model):
    """Chuẩn L2 của gradient từng tham số: log mỗi vài trăm bước."""
    return {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}

def is_stuck(train_losses, n_classes, window=50, tol=1e-3):
    """Loss gần như không đổi trong `window` bước VÀ còn sát mức đoán đều ln(K)."""
    if len(train_losses) < window:
        return False
    recent = train_losses[-window:]
    flat = max(recent) - min(recent) < tol * max(1.0, abs(recent[-1]))
    return flat and abs(recent[-1] - math.log(n_classes)) < 0.05
```

#### P3.6 Cạm bẫy

1. **Dừng huấn luyện vì "loss không giảm nữa"** khi đang ở plateau: kiểm tra mức đoán hằng số và chuẩn gradient trước.
2. **Khởi tạo mọi trọng số bằng 0** (hoặc bằng nhau): đối xứng không bao giờ bị phá. Đây không phải plateau mà là kẹt vĩnh viễn.
3. **Giảm learning rate khi gặp plateau do khởi tạo nhỏ:** làm plateau **dài hơn** (bảng P3.3: momentum với $\alpha = 0.01$ chậm hơn GD với $\alpha = 0.1$). `ReduceLROnPlateau` có thể phản ứng sai trong pha đầu, nên hãy kết hợp warmup và `patience` đủ dài.
4. **Nhầm plateau với overfitting:** plateau thì **train** loss đứng yên; overfitting thì train loss giảm còn val loss tăng.

**Sắc thái phỏng vấn:** *"Loss của bạn đứng ở 0.693 suốt 3 epoch đầu. Bạn làm gì?"* Ý phải có: 0.693 $= \ln 2$, tức model đang dự đoán 0.5 cho mọi mẫu; kiểm tra nhãn và dữ liệu (đầu vào có bị toàn 0? nhãn có bị xáo trộn?); log chuẩn gradient; kiểm tra khởi tạo, activation, learning rate; thử overfit một batch nhỏ (nếu không overfit nổi 32 mẫu thì có bug).

#### P3.7 Đọc thêm

- A. Saxe, J. McClelland, S. Ganguli, *Exact solutions to the nonlinear dynamics of learning in deep linear neural networks*, ICLR 2014.
- Y. Dauphin et al., *Identifying and attacking the saddle point problem in high-dimensional non-convex optimization*, NeurIPS 2014.
- Goodfellow, Bengio, Courville, *Deep Learning* (2016), §8.2 (*Challenges in Neural Network Optimization*).
- L. Smith, *Cyclical Learning Rates for Training Neural Networks*, WACV 2017 (LR range test).

---

## Quiz phụ lục Tuần 2

> ⏱ **30 phút** · 10 câu nhanh (1đ) + 2 câu tự luận (5đ) · **20 điểm** · Đạt khi ≥ 14
> Làm trên giấy, được dùng máy tính cầm tay. **Không có câu chứng minh.**

**Phần 1 — Câu nhanh**

1. $\min x^2 + y^2$ với $x + y = 4$. Tìm nghiệm, $f^\star$, và $\lambda$. Ước lượng $f^\star$ khi ràng buộc thành $x + y = 4.1$ mà không giải lại.
2. Tại một điểm trên ràng buộc $g(x, y) = c$, $\nabla f = (3, 1)$ và $\nabla g = (1, 2)$. Điểm này có thể là nghiệm tối ưu không? Vì sao?
3. Tại nghiệm của một bài có ràng buộc $h(x) \le 0$, ta có $h(x^\star) = -0.3$. Nhân tử $\mu$ bằng bao nhiêu?
4. $\min (x - 1)^2$ với $x \ge 2$. Tìm $x^\star$ và $\mu$.
5. Trong SVM, những điểm dữ liệu nào có $\alpha_i > 0$? Xóa một điểm có $\alpha_i = 0$ thì đường biên thay đổi thế nào?
6. `StepLR(step_size=10, gamma=0.5)` với $\alpha_0 = 0.05$. Learning rate ở epoch 25 (đếm từ 0)?
7. `ExponentialLR(gamma=0.9)` từ $\alpha_0 = 0.1$. Sau bao nhiêu epoch thì $\alpha$ lần đầu **nhỏ hơn** 0.01?
8. SGD 1 chiều trên $\tfrac12 x^2$ với nhiễu gradient phương sai 1 và $\alpha = 0.2$. Tính sàn nhiễu $\mathbb E[\tfrac12 x^2]$.
9. Bài $\tfrac12(1 - ab)^2$, khởi tạo $a = b = 10^{-4}$, GD với $\alpha = 0.05$. Ước lượng độ dài plateau theo công thức.
10. $\sigma'(4)$ bằng bao nhiêu, và nhỏ hơn $\sigma'(0)$ bao nhiêu lần?

**Phần 2 — Tự luận**

**T1 · KKT tính tay (5đ).** $\min (x - 2)^2 + (y - 1)^2$ với $x + 2y \le c$.
- (a) Với $c = 5$: giải và cho biết $\mu$.
- (b) Với $c = 2$: giải đầy đủ theo các bước (thử inactive, rồi active), kiểm tra $\mu \ge 0$, tính $f^\star$.
- (c) Dùng $\mu$ ở (b) để ước lượng $f^\star$ khi $c = 2.1$. So với nghiệm đúng (gợi ý: lặp lại (b) với $c = 2.1$).

**T2 · Chọn và lần theo schedule (5đ).** Fine-tune một classifier, `ReduceLROnPlateau(mode="min", factor=0.1, patience=1, threshold=0.01)`, $\alpha_0 = 10^{-3}$. Val loss các epoch 0–9: `0.90, 0.70, 0.65, 0.648, 0.66, 0.60, 0.598, 0.597, 0.59, 0.50`.
- (a) Lập bảng như P2.2 và cho biết $\alpha$ dùng ở **mỗi** epoch 0–9.
- (b) Nhận xét: cấu hình này có hợp lý không, khi val loss vẫn đang giảm tới 0.50 ở epoch 9? Đề xuất sửa 2 tham số.
- (c) Một đồng nghiệp dùng `StepLR(step_size=30)` nhưng gọi `sched.step()` sau **mỗi batch**, với 500 batch/epoch. Sau 1 epoch, $\alpha$ đã bị nhân $\gamma$ bao nhiêu lần?

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Phần 1**

1. $(2, 2)$, $f^\star = 8$. $\nabla f = (4, 4) = \lambda(1, 1)$ nên **$\lambda = 4$**. Ước lượng $f^\star(4.1) \approx 8 + 4 \times 0.1 = 8.4$ (đúng: $4.1^2/2 = 8.405$).
2. **Không.** Ở nghiệm phải có $\nabla f = \lambda \nabla g$, mà $(3, 1)$ không cùng phương với $(1, 2)$ (vì $3/1 \ne 1/2$).
3. **$\mu = 0$** (ràng buộc inactive; complementary slackness $\mu h = 0$).
4. Không ràng buộc thì $x = 1$, vi phạm $x \ge 2$, nên ràng buộc active: **$x^\star = 2$**. Viết ràng buộc dạng $h = 2 - x \le 0$: $2(x - 1) + \mu \cdot (-1) = 0$ nên **$\mu = 2$** $\ge 0$ ✓.
5. Các **support vector**: điểm nằm đúng trên lề ($y_i f(x_i) = 1$) hoặc vi phạm lề (soft-margin). Xóa điểm có $\alpha_i = 0$: **không đổi**.
6. $0.05 \times 0.5^{\lfloor 25/10\rfloor} = 0.05 \times 0.25 =$ **0.0125**.
7. $0.9^t < 0.1$ khi $t > \ln 0.1/\ln 0.9 = 21.85$, nên **$t = 22$** ($\alpha_{21} = 0.01094$, $\alpha_{22} = 0.00985$).
8. $V = \alpha/(2 - \alpha) = 0.2/1.8 = 0.111$, sàn $= V/2 =$ **0.0556**.
9. $\ln(10^4)/\ln 1.05 = 9.210/0.0488 \approx$ **189 bước** (mô phỏng: 210 bước để loss < 0.005; ~20 bước chênh là pha cuối).
10. $\sigma(4) = 0.9820$, $\sigma'(4) = 0.9820 \times 0.0180 =$ **0.0177**; nhỏ hơn $0.25$ khoảng **14 lần**.

**T1 (5đ)**

(a) 1đ. Không ràng buộc: $(2, 1)$, $2 + 2 \times 1 = 4 \le 5$ ✓. Nghiệm $(2, 1)$, $f^\star = 0$, **$\mu = 0$**.

(b) 3đ.
- Thử inactive: $(2, 1)$ có $x + 2y = 4 > 2$, vi phạm (0.5đ).
- Active: $2(x - 2) + \mu = 0$ và $2(y - 1) + 2\mu = 0$, nên $x = 2 - \mu/2$, $y = 1 - \mu$ (1đ).
- Thay vào $x + 2y = 2$: $2 - \mu/2 + 2 - 2\mu = 4 - 2.5\mu = 2$, nên **$\mu = 0.8$** $\ge 0$ ✓ (1đ).
- $x = 1.6$, $y = 0.2$, $f^\star = 0.16 + 0.64 =$ **0.8** (0.5đ). (SLSQP: $(1.6,\ 0.2)$, $f = 0.8$.)

(c) 1đ. Nới $c$ thêm 0.1 thì $f^\star$ **giảm** khoảng $\mu \times 0.1 = 0.08$, còn ~0.72. Đúng: $\mu = (4 - 2.1)/2.5 = 0.76$, $(x, y) = (1.62,\ 0.24)$, $f^\star = 0.722$, tức giảm 0.078. (Dấu: với $h \le c$, nới rộng miền khả thi chỉ có thể làm $f^\star$ giảm.)

**T2 (5đ)**

(a) 3đ. Cải thiện khi $v < \text{best} \times 0.99$; giảm $\alpha$ khi `bad` > 1.

| Epoch | val | Ngưỡng | Cải thiện? | `bad` | $\alpha$ dùng ở epoch này |
|---|---|---|---|---|---|
| 0 | 0.90 | — | có | 0 | $10^{-3}$ |
| 1 | 0.70 | 0.891 | có | 0 | $10^{-3}$ |
| 2 | 0.65 | 0.693 | có | 0 | $10^{-3}$ |
| 3 | 0.648 | 0.6435 | không | 1 | $10^{-3}$ |
| 4 | 0.66 | 0.6435 | không | 2 → **giảm** | $10^{-3}$ |
| 5 | 0.60 | 0.6435 | có | 0 | $10^{-4}$ |
| 6 | 0.598 | 0.594 | không | 1 | $10^{-4}$ |
| 7 | 0.597 | 0.594 | không | 2 → **giảm** | $10^{-4}$ |
| 8 | 0.59 | 0.594 | có | 0 | $10^{-5}$ |
| 9 | 0.50 | 0.5841 | có | 0 | $10^{-5}$ |

(Đã đối chiếu với PyTorch: `[1e-3]*5 + [1e-4]*3 + [1e-5]*2`.)

(b) 1.5đ. Không hợp lý: `patience=1` với `factor=0.1` cắt $\alpha$ **100 lần** chỉ sau 8 epoch, trong khi val loss vẫn đang giảm mạnh. Sửa: tăng `patience` lên 3–5; dùng `factor` 0.3–0.5; có thể giảm `threshold`, thêm `min_lr`.

(c) 0.5đ. $\lfloor 500/30 \rfloor =$ **16 lần**, tức $\alpha$ đã bị nhân $\gamma^{16}$ (với $\gamma = 0.1$ là $10^{-16}$ lần: huấn luyện gần như dừng hẳn).

</details>
