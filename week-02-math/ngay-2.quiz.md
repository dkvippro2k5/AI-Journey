# Quiz · Ngày 2 · Giải tích & Tối ưu (bài 04–07)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> Làm trên giấy, được dùng máy tính cầm tay. **Không chạy code.** Đáp án ở cuối file.

## Phần 1 — Quick test (14 câu, câu 14 được 2đ, các câu khác 1đ)

1. Đạo hàm của $f(x) = \ln(1 + e^x)$ là hàm nào quen thuộc?
2. Tính $\sigma(0)$ và $\sigma'(0)$.
3. Tính gradient của $f(x, y) = x^2y + y^3$ tại $(1, 2)$.
4. Tại một điểm dừng, Hessian là $\begin{bmatrix}4 & 1\\1 & -2\end{bmatrix}$. Đó là cực tiểu, cực đại hay saddle point? Giải thích bằng một phép tính.
5. Sai số của sai phân tiến và sai phân trung tâm có bậc là bao nhiêu theo $h$?
6. $f = x \cdot y + x$ với $x = 3$, $y = -2$. Tính $\partial f/\partial x$.
7. Softmax cho $p = (0.2,\ 0.5,\ 0.3)$, nhãn đúng là lớp thứ 2 (chỉ số 1). Tính loss cross-entropy và $\nabla_z\ell$.
8. Một neuron logistic có $p = 0.8$, nhãn $y = 0$, đầu vào $x = (1, -2)$. Tính $\nabla_w\ell$.
9. Hessian của một loss bậc hai có trị riêng $0.5$ và $50$. Learning rate lớn nhất để GD còn hội tụ là bao nhiêu? $\kappa$ bằng bao nhiêu?
10. Với $\kappa = 50$ và $\alpha = 1/L$, cần khoảng bao nhiêu bước để sai số giảm 1000 lần?
11. Tăng batch size từ 16 lên 256 thì độ lệch chuẩn của nhiễu gradient thay đổi bao nhiêu lần?
12. Momentum với $\beta = 0.99$ và gradient không đổi: về lâu dài, bước đi dài gấp bao nhiêu lần GD thường?
13. Adam, $\alpha = 0.001$, gradient đầu tiên $g_1 = (-0.001,\ 50)$. Tính $\Delta\theta_1$ (bỏ qua $\epsilon$).
14. *(2đ)* Dùng `cosine_lr` của project với `base_lr=3e-4`, `warmup=100`, `total=1100`, `min_lr=0`. Tính learning rate tại các bước 0, 49, 600, 1100.

## Phần 2 — Tự luận (4 câu, 30 phút)

**T1 · Chứng minh (4đ).** Với $p = \mathrm{softmax}(z)$ và $\ell = -\sum_k y_k\ln p_k$ ($y$ one-hot):
- (a) Chứng minh $\partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$.
- (b) Suy ra $\nabla_z\ell = p - y$.
- (c) Chứng minh tổng các phần tử của $\nabla_z\ell$ bằng 0, và giải thích ý nghĩa của điều đó.

**T2 · Backprop tính tay (4đ).** Mạng 2 lớp với đầu vào vô hướng:

$$a = w_1x + b_1,\quad h = \mathrm{ReLU}(a),\quad \hat y = w_2h + b_2,\quad L = \tfrac{1}{2}(\hat y - y)^2,$$

với $x = 2$, $y = 2$, $w_1 = 0.5$, $b_1 = -0.5$, $w_2 = 3$, $b_2 = 1$.
- (a) Tính forward: $a, h, \hat y, L$.
- (b) Tính gradient của $L$ theo $w_1, b_1, w_2, b_2$.
- (c) Làm một bước SGD với $\alpha = 0.1$, rồi tính lại $L$. Loss tăng hay giảm? Điều gì đã xảy ra với neuron ẩn, và gradient theo $w_1$ ở bước tiếp theo bằng bao nhiêu?

**T3 · Phân tích GD (3đ).** $L(\theta) = \tfrac{1}{2}(\theta_1^2 + 25\,\theta_2^2)$.
- (a) Với những $\alpha$ nào thì GD hội tụ từ mọi điểm xuất phát?
- (b) Tính $\alpha^\star$ và hệ số co $\rho^\star$ tương ứng.
- (c) Đổi biến $u_2 = 5\theta_2$ (tương đương chuẩn hóa đặc trưng). Viết lại $L$ theo $(\theta_1, u_2)$. Giá trị $\alpha$ nào cho phép tới cực tiểu sau đúng **một** bước?

**T4 · Code (4đ).** Viết class `AdamW(lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)` có `step(theta, grad) -> theta_mới`, áp dụng weight decay tách rời như Loshchilov & Hutter. Sau đó tính tay, với $\theta = (1, -2)$, $g = (0, 0)$, $\alpha = 0.01$, $\lambda = 0.1$:
- (a) $\theta$ sau một bước `AdamW`.
- (b) $\theta$ sau một bước `Adam` thường, khi đưa L2 vào loss (tức gradient thành $g + \lambda\theta$).

Giải thích vì sao hai kết quả khác nhau nhiều như vậy.

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Quick test**

1. $f'(x) = \dfrac{e^x}{1 + e^x} = \sigma(x)$. ($f$ là hàm *softplus*.)
2. $\sigma(0) = 0.5$; $\sigma'(0) = 0.5 \cdot 0.5 = 0.25$.
3. $\nabla f = (2xy,\ x^2 + 3y^2) = (4,\ 13)$.
4. $\det H = -8 - 1 = -9 < 0$, nên hai trị riêng **trái dấu**: **saddle point**. (Trị riêng: $1 \pm \sqrt{10} \approx 4.16$ và $-2.16$.)
5. Sai phân tiến $O(h)$; sai phân trung tâm $O(h^2)$.
6. $x$ đi vào hai nhánh, nên gradient cộng lại: $y + 1 = -1$.
7. $\ell = -\ln 0.5 = 0.693$; $\nabla_z\ell = p - y = (0.2,\ -0.5,\ 0.3)$.
8. $(p - y)\,x = 0.8 \cdot (1, -2) = (0.8,\ -1.6)$.
9. $\alpha < 2/50 =$ **0.04**; $\kappa = 50/0.5 =$ **100**.
10. $\ln(1000)/(-\ln 0.98) \approx$ **342** (xấp xỉ $\kappa\ln 1000 \approx 345$ đều được điểm).
11. Phương sai giảm 16 lần, nên độ lệch chuẩn giảm $\sqrt{16} =$ **4** lần.
12. $1/(1 - 0.99) =$ **100** lần.
13. $\Delta\theta_1 \approx -\alpha\,\mathrm{sign}(g_1) = (+0.001,\ -0.001)$. Độ lớn của gradient không quan trọng.
14. $3 \times 10^{-6}$ (tức $3 \times 10^{-4} \cdot 1/100$); $1.5 \times 10^{-4}$ (tức $\cdot 50/100$); $1.5 \times 10^{-4}$ (tiến độ $500/1000 = 0.5$, $\cos(\pi/2) = 0$); $0$. *(0.5đ mỗi giá trị)*

**T1**

(a) Mệnh đề 4.3 (quy tắc thương). *(1.5đ)*

(b) Mệnh đề 5.4. Bước then chốt là $\sum_k y_k = 1$. *(1.5đ)*

(c) $\sum_j(p_j - y_j) = 1 - 1 = 0$. Ý nghĩa: cộng cùng một hằng số $c$ vào mọi logit thì softmax **không đổi**, nên loss cũng không đổi; đạo hàm của loss theo hướng $(1, 1, \dots, 1)$ phải bằng 0. Hệ quả thực tế: có thể trừ $\max_j z_j$ khỏi mọi logit để tránh tràn số (Ngày 3, bài 11) mà không làm thay đổi gì. *(1đ)*

**T2** *(đã chạy kiểm chứng bằng `torch.autograd`)*

(a) $a = 0.5 \cdot 2 - 0.5 = 0.5$; $h = 0.5$; $\hat y = 3 \cdot 0.5 + 1 = 2.5$; $L = \tfrac{1}{2}(0.5)^2 = 0.125$. *(1đ)*

(b) Đi ngược:

| Đại lượng | Giá trị |
|---|---|
| $\partial L/\partial\hat y = \hat y - y$ | 0.5 |
| $\partial L/\partial w_2 = 0.5 \cdot h$ | **0.25** |
| $\partial L/\partial b_2$ | **0.5** |
| $\partial L/\partial h = 0.5 \cdot w_2$ | 1.5 |
| $\partial L/\partial a = 1.5 \cdot \mathbb{1}[a > 0]$ | 1.5 |
| $\partial L/\partial w_1 = 1.5 \cdot x$ | **3** |
| $\partial L/\partial b_1$ | **1.5** |

*(1.5đ)*

(c) Sau một bước: $w_1 = 0.5 - 0.3 = 0.2$; $b_1 = -0.5 - 0.15 = -0.65$; $w_2 = 2.975$; $b_2 = 0.95$.
- Forward lại: $a = 0.4 - 0.65 = -0.25 < 0$, nên $h = 0$, $\hat y = 0.95$, $L = \tfrac{1}{2}(1.05)^2 \approx$ **0.551**.
- Loss **tăng** từ 0.125 lên 0.551.
- Neuron ẩn bị **"chết"** (dead ReLU): $a < 0$ nên local gradient của ReLU bằng 0. Ở bước tiếp theo, $\partial L/\partial w_1 = \partial L/\partial b_1 =$ **0**; chỉ còn $b_2$ được cập nhật.
- Nguyên nhân: learning rate quá lớn so với độ cong theo hướng $w_1$ (gradient bằng 3), làm $a$ vọt qua 0. *(1.5đ)*

**T3**

(a) $H = \mathrm{diag}(1, 25)$, nên $L_{\text{smooth}} = 25$: GD hội tụ khi $0 < \alpha < 2/25 =$ **0.08**. *(1đ)*

(b) $\alpha^\star = 2/(25 + 1) \approx 0.0769$; $\rho^\star = (25 - 1)/(25 + 1) = 24/26 \approx 0.923$. *(1đ)*

(c) $25\theta_2^2 = u_2^2$, nên $L = \tfrac{1}{2}(\theta_1^2 + u_2^2)$ có $H = I$ và $\kappa = 1$. Với $\alpha = 1$: $\theta - 1 \cdot \theta = 0$, tới cực tiểu sau **một bước**. *(1đ)*

**T4** *(đã chạy kiểm chứng: khớp `torch.optim.AdamW` tới 10 chữ số sau 5 bước với gradient ngẫu nhiên)*

```python
class AdamW:
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, beta1, beta2, eps, weight_decay
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
        return theta - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * theta)   # decay KHÔNG chia √v̂
```

*(Chấm: code đúng được 2đ)*

(a) $g = 0$ nên phần Adam bằng 0; $\theta \leftarrow \theta - 0.01 \cdot 0.1 \cdot \theta = 0.999\,\theta =$ **$(0.999,\ -1.998)$**. *(0.5đ)*

(b) Gradient là $\lambda\theta = (0.1,\ -0.2)$. Bước đầu của Adam là $-\alpha\,\mathrm{sign}(\cdot)$, nên $\theta = (1 - 0.01,\ -2 + 0.01) =$ **$(0.99,\ -1.99)$**. *(0.5đ)*

Vì sao khác: với Adam + L2, số hạng phạt bị **chuẩn hóa** bởi $\sqrt{\hat v}$, nên mỗi bước kéo **mọi** tham số về 0 đúng $\alpha$, bất kể $\lambda$ lớn hay nhỏ: ở đây mạnh gấp 10 lần (tọa độ 1) và gấp 5 lần (tọa độ 2) so với ý định. Khi gradient thật khác 0, ngược lại, phần phạt bị "pha loãng" ở những tham số có gradient lớn. AdamW tách weight decay ra để nó luôn đúng bằng $\alpha\lambda\theta$ (Mệnh đề 7.4). *(1đ)*

</details>
