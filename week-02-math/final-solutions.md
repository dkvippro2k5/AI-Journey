# Đáp án · Final Tuần 2

> Mọi con số của phần A, B đã được chạy kiểm chứng bằng NumPy. Code phần C đã chạy: 9/9 test pass.

## A. Quick test (24đ, 2đ/câu)

| Câu | Đáp án | Vì sao |
|---|---|---|
| A1 | $4/9 \approx 0.444$ | $x^\top y = 2 - 2 + 4 = 4$; $\|x\| = \|y\| = 3$ |
| A2 | Vô nghiệm; nghiệm bình phương tối thiểu **không duy nhất** | Hạng bằng 1 (cột 2 = 2 × cột 1). $b = (1, 3)$ không cùng phương với $(1, 2)$, nên không nằm trong không gian cột. Hạt nhân khác $\{0\}$ nên có vô số nghiệm bình phương tối thiểu; `lstsq` trả về nghiệm có chuẩn nhỏ nhất $(0.28,\ 0.56)$. *(1đ mỗi ý)* |
| A3 | **2** | $(4 + 3)/10 = 70\%$ |
| A4 | $0.09$ | $\sigma(1 - \sigma) = 0.9 \times 0.1$ |
| A5 | **18** | $f = u^2$ với $u = xy$: $\partial f/\partial x = 2u \cdot y = 2 \cdot 3 \cdot 3$ |
| A6 | **Phân kỳ**; thành phần ứng với $\lambda = 8$, hệ số $1 - 0.3 \cdot 8 = -1.4$ | $2/\lambda_{\max} = 0.25 < 0.3$. Thành phần $\lambda = 2$ vẫn co với hệ số $0.4$ |
| A7 | $\approx 0.01$ | $1 - 0.999^{10} = 0.00996$: $v$ bị kéo về 0 khoảng 100 lần |
| A8 | **0.2** | $\alpha g/(1 - \beta) = 0.01 \cdot 2/0.1$ |
| A9 | $\approx 0.296$ | $0.04/(0.04 + 0.095)$ |
| A10 | $\approx 0.030$ | $1.96\sqrt{0.21/900} = 1.96 \times 0.0153$ |
| A11 | MLE **0**; MAP $1/10 =$ **0.1** | $(k + 1)/(n + 2)$ |
| A12 | **1 bit**; **$+\infty$** | $D_{\mathrm{KL}}(p\,\|\,q) = 1 \cdot \log_2(1/0.5)$. Theo chiều ngược lại, $q$ đặt xác suất $0.5$ vào chỗ $p = 0$ nên KL vô hạn: ví dụ rõ nhất cho tính không đối xứng |

## B. Tính tay (20đ)

**B1 (7đ)**

(a) $\sum x^2 = 10$, $\sum y^2 = 4$, $\sum xy = 6$, nên $C = \dfrac{1}{3}\begin{bmatrix}10 & 6\\6 & 4\end{bmatrix} = \begin{bmatrix}3.333 & 2\\2 & 1.333\end{bmatrix}$.
- $\mathrm{tr} = 4.667$, $\det = 4.444 - 4 = 0.444$.
- $\lambda = \dfrac{4.667 \pm \sqrt{4.667^2 - 4 \cdot 0.444}}{2} = \dfrac{4.667 \pm 4.472}{2}$, tức $\lambda_1 = 4.569$ và $\lambda_2 = 0.097$.
- Tỉ lệ phương sai của PC1 $= 4.569/4.667 =$ **0.979**. *(3đ)*

(b) $(C - \lambda_1 I)v = 0$: $(3.333 - 4.569)v_1 + 2v_2 = 0$, nên $v_2 = 0.618\,v_1$. Chuẩn hóa: $v = (0.851,\ 0.526)$. Hệ số góc **0.618** (đúng bằng $(\sqrt5 - 1)/2$). *(1.5đ)*

(c) $\theta = \sum xy/\sum x^2 = 6/10 =$ **0.6**. *(1đ)*

(d) Hồi quy cực tiểu tổng bình phương khoảng cách **theo chiều dọc** (sai số chỉ nằm ở $y$). PCA cực tiểu tổng bình phương khoảng cách **vuông góc** tới đường thẳng (Eckart–Young cho hạng 1), coi $x$ và $y$ bình đẳng. Kiểm tra thêm: hồi quy ngược $x$ theo $y$ cho hệ số góc $1/1.5 = 0.667$ trong mặt phẳng $(x, y)$; PCA nằm **giữa** hai đường hồi quy ($0.6 < 0.618 < 0.667$). *(1.5đ)*

**B2 (7đ)**

(a) $e^{(2,1,0)} = (7.389,\ 2.718,\ 1)$, tổng $11.107$, nên $p = (0.665,\ 0.245,\ 0.090)$. Loss $= -\ln 0.090 =$ **2.408**. $\nabla_z\ell = p - y = (0.665,\ 0.245,\ -0.910)$. *(3đ)*

(b) $L = 4.5$, $\mu = 0.5$, $\kappa = 9$.
- $\alpha_{\max} = 2/4.5 = 0.444$.
- $\alpha^\star = 2/5 = 0.4$; $\rho^\star = 8/10 = 0.8$.
- Số bước: $\ln 10^6/(-\ln 0.8) = 13.82/0.223 \approx$ **62**. *(3đ)*

(c) $\sqrt\kappa = 3$, hệ số co $2/4 = 0.5$: $13.82/0.693 \approx$ **20** bước. *(1đ)*

**B3 (6đ)**

(a) $\mathrm{SE} = \sqrt{0.85 \cdot 0.15/1000} = 0.0113$; khoảng tin cậy $0.85 \pm 0.022$, tức $[0.828,\ 0.872]$. *(1.5đ)*

(b) $H(p, q) = -(0.6\ln 0.5 + 0.3\ln 0.25 + 0.1\ln 0.25) = 0.416 + 0.416 + 0.139 =$ **0.970** nat. $D_{\mathrm{KL}} = 0.970 - 0.898 =$ **0.072** nat. *(2đ)*

(c) Mean NLL $= (\ln 2 + \ln 4 + \ln 8 + \ln 2)/4 = 1.213$ nat, nên $\mathrm{PPL} = e^{1.213} =$ **3.364**.

Chứng minh: $\exp\left(-\frac{1}{T}\sum\ln p_t\right) = \left(\prod p_t\right)^{-1/T}$, tức nghịch đảo của trung bình nhân. Kiểm tra: $(0.5 \cdot 0.25 \cdot 0.125 \cdot 0.5)^{-1/4} = 128^{1/4} = 3.364$ ✓. *(2.5đ)*

## C. Code (32đ)

Chấm theo test: **C1** 3 test (12đ) · **C2** 3 test (12đ) · **C3** 3 test (6đ) + 2 câu trả lời (2đ).

```python
import numpy as np


# ---------- C1 ----------
def pca_power(X, k, iters=1000, tol=1e-10, seed=0):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if not 1 <= k <= d:
        raise ValueError("k phải trong [1, d]")
    Xc = X - X.mean(axis=0)                            # tạo mảng mới, không sửa X
    C = Xc.T @ Xc / (n - 1)
    rng = np.random.default_rng(seed)
    comps, vals = [], []
    for _ in range(k):
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
        lam = float(v @ C @ v)
        for _ in range(iters):
            w = C @ v
            nw = np.linalg.norm(w)
            if nw == 0:
                lam = 0.0
                break
            v = w / nw
            lam = float(v @ C @ v)                     # Rayleigh quotient
            if np.linalg.norm(C @ v - lam * v) <= tol * max(1.0, abs(lam)):
                break
        v = v * np.sign(v[np.argmax(np.abs(v))])       # quy ước dấu
        comps.append(v)
        vals.append(lam)
        C = C - lam * np.outer(v, v)                   # deflation
    return np.array(comps), np.array(vals)


# ---------- C2 ----------
def softmax_ce_loss(W, X, y, l2=0.0):
    Z = X @ W
    m = Z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(Z - m).sum(axis=1))  # log-sum-exp ổn định
    n = len(y)
    loss = float(np.mean(lse - Z[np.arange(n), y])) + 0.5 * l2 * float(np.sum(W * W))
    P = np.exp(Z - lse[:, None])                       # softmax
    P[np.arange(n), y] -= 1                            # P − Y
    return loss, X.T @ P / n + l2 * W


def fit_softmax(X, y, K, *, lr=0.5, epochs=300, l2=1e-3):
    W = np.zeros((X.shape[1], K))
    for _ in range(epochs):
        _, g = softmax_ce_loss(W, X, y, l2)
        W = W - lr * g
    return W


def predict_softmax(W, X):
    return np.argmax(X @ W, axis=1)


# ---------- C3 ----------
def bootstrap_ci(x, stat=np.mean, n_boot=2000, level=0.95, seed=0):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("cần ít nhất 2 mẫu")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))         # lấy mẫu lại có hoàn lại
    stats = np.array([stat(x[row]) for row in idx])
    a = (1 - level) / 2
    lo, hi = np.quantile(stats, [a, 1 - a])
    return float(lo), float(hi)


def paired_bootstrap_diff(a, b, n_boot=2000, level=0.95, seed=0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a và b phải cùng shape")
    d = b - a                                          # lấy mẫu lại hiệu = cùng chỉ số cho a và b
    return float(d.mean()), *bootstrap_ci(d, np.mean, n_boot, level, seed)
```

**Hai câu trả lời của C3:**
1. Với cùng một câu hỏi, kết quả của A và B **tương quan dương**: câu dễ thì cả hai cùng đúng, câu khó thì cả hai cùng sai. $\mathrm{Var}(\bar b - \bar a) = \mathrm{Var}(\bar a) + \mathrm{Var}(\bar b) - 2\,\mathrm{Cov}$. Lấy mẫu theo cặp giữ lại số hạng $-2\,\mathrm{Cov}$; lấy mẫu độc lập thì bỏ mất nó, nên phương sai bị **phóng đại**.
2. Dùng bootstrap khi thống kê không có công thức SE đơn giản (median, p95 latency, recall@k trung bình theo nhóm, F1), khi $n$ nhỏ, hoặc khi phân phối rất lệch. Với tỉ lệ đơn giản và $n$ lớn, Wald/Wilson cho kết quả gần như bootstrap mà rẻ hơn nhiều.

**Lỗi hay gặp:**
- `pca_power` sửa thẳng vào `X` bằng `X -= X.mean(0)`.
- Dừng power iteration theo "$\lambda$ thôi thay đổi" (xem quiz Ngày 1, T4).
- Quên chia $n$ trong gradient của softmax.
- `paired_bootstrap_diff` lấy hai bộ chỉ số khác nhau cho `a` và `b`.

## D. Thiết kế (24đ)

**D1 (8đ)**

(a) $5 \times 10^6 \times 1024 \times 4$ B = **20.5 GB**. Với $k = 256$: **5.1 GB**. *(1đ)*

(b) Quy trình: *(3đ)*
1. Lấy một **mẫu ngẫu nhiên** khoảng $10^5$ vector. Phương sai và trục chính ước lượng từ mẫu đã rất ổn định (bài 09), không cần dùng toàn bộ 5 triệu vector.
2. Tính `mean` (1024 chiều) và SVD của mẫu đã trừ `mean`. Chi phí $O(nd^2) \approx 10^5 \times 1024^2 \approx 10^{11}$ phép tính, tức cỡ vài chục giây trên CPU.
3. **Lưu lại** `mean` và `components` (ma trận $256 \times 1024$), kèm phiên bản của model embedding.
4. Áp dụng **cùng một phép biến đổi** cho mọi tài liệu và mọi câu truy vấn: $z = (x - \text{mean})\,\cdot$ `components.T`, rồi chuẩn hóa $z$.
5. Khi đổi model embedding thì phải fit lại PCA và tính lại toàn bộ index.

(c) *(2đ)*
- PCA cắt bỏ các chiều cuối, nên độ dài vector sau khi chiếu **khác nhau** giữa các vector, phụ thuộc vào lượng "năng lượng" nằm ở các chiều bị cắt. Nếu không chuẩn hóa lại thì tích vô hướng không còn là cosine (Mệnh đề 1.3), và thứ hạng bị lệch về phía các vector dài.
- Nếu lúc truy vấn quên trừ `mean`, mọi câu truy vấn bị dịch đi cùng một vector $\text{mean}\cdot$ `components.T`, cosine bị lệch một cách có hệ thống, và recall giảm mà không báo lỗi nào.

(d) Dùng tỉ lệ phương sai tích lũy (ví dụ ≥ 90–95%) để **khoanh vùng** $k$, rồi **quyết định bằng metric thật**: đo recall@10 trên bộ đánh giá với vài giá trị $k$ (128, 256, 384), và chọn $k$ nhỏ nhất mà recall giảm không đáng kể (dùng kiểm định ghép cặp như D2). Phương sai lớn không đảm bảo hữu ích cho việc phân biệt văn bản liên quan (bài 03, cạm bẫy 5). *(2đ)*

**D2 (8đ)**

(a) $n = 1.96^2 \cdot 0.7 \cdot 0.3/0.02^2 \approx$ **2 017** câu hỏi. *(1.5đ)*

(b) *(3đ)*
- Chạy **cả hai model trên cùng bộ câu hỏi** (thiết kế ghép cặp).
- Kiểm định **McNemar** trên các câu bất đồng, hoặc paired bootstrap cho hiệu recall@10.
- Báo cáo: recall của từng model kèm khoảng tin cậy; **hiệu** kèm khoảng tin cậy 95%; số câu bất đồng $b$, $c$; p-value.
- Thiết kế ghép cặp loại bỏ được độ khó riêng của từng câu (quiz Ngày 3, T3: cùng dữ liệu, kiểm định độc lập cho $p = 0.18$, còn McNemar cho $p = 0.03$).

(c) Các câu sinh từ cùng một đoạn văn **tương quan** với nhau (dữ liệu không i.i.d.), nên SE thật lớn hơn công thức và khoảng tin cậy bị hẹp một cách giả tạo. Cách xử lý: bootstrap **theo đoạn văn** (lấy mẫu lại cả cụm), hoặc dùng clustered standard errors (Miller 2024). Trong báo cáo, nêu rõ cả số câu hỏi lẫn số đoạn văn. *(1.5đ)*

(d) Không đáng tin: chọn cấu hình tốt nhất trong 20 cấu hình trên **cùng** bộ đánh giá làm kết quả bị **thiên lệch lạc quan** (vấn đề so sánh bội / overfitting vào tập đánh giá). Cách làm đúng: chia bộ đánh giá thành phần **dev** (dùng để chọn cấu hình) và phần **test** giữ kín (chỉ đo một lần cho cấu hình cuối). Nếu vẫn muốn kiểm định nhiều cấu hình trên cùng tập, dùng hiệu chỉnh Bonferroni hoặc Holm. *(2đ)*

**D3 (8đ)**

(a)–(b) Giả thuyết và thí nghiệm kiểm tra: *(4đ — 0.5đ cho mỗi giả thuyết đúng, 0.5đ cho mỗi thí nghiệm phù hợp, tối đa 4 cặp)*

| # | Giả thuyết | Vi phạm điều gì | Thí nghiệm kiểm tra |
|---|---|---|---|
| 1 | **Logit nổ tung vì đặc trưng chưa chuẩn hóa.** Adam dịch mỗi trọng số khoảng `lr` mỗi bước (Mệnh đề 7.3), nên với đặc trưng cỡ $10^4$, logit thay đổi khoảng $10^{-2} \times 10^4 = 100$ **mỗi bước** | Bài 06 (chuẩn hóa, $\kappa$) và Mệnh đề 7.3 | Log `max|z|` mỗi bước; lặp lại thí nghiệm sau khi `standardize` |
| 2 | **Softmax tự viết bị tràn số:** `exp(z)` tràn khi $z > 88.7$ (float32) hoặc $> 709.8$ (float64), cho `inf/inf = nan`; một xác suất bị làm tròn về 0 thì `log(0) = -inf` | Mệnh đề 11.5, cạm bẫy 5.5 | Đưa batch gây `nan` qua bản `logsumexp`; kiểm tra `np.isinf` trên `exp(z)` |
| 3 | **Learning rate quá lớn**, không warmup: giai đoạn đầu dao động mạnh | Mệnh đề 6.3, mục 7.4 (warmup) | Quét `lr` ∈ {1e-2, 3e-3, 1e-3}; so sánh có và không có warmup |
| 4 | **Không clipping:** một batch có giá trị ngoại lai tạo gradient khổng lồ | Mục 7.1 (clipping) | Log `‖g‖` mỗi bước; tìm bước có đỉnh ngay trước khi ra `nan` |
| 5 | Dữ liệu có sẵn `nan`/`inf` | — | `np.isfinite(X).all()` |
| 6 | Gradient tự viết bị sai | Bài 04, gradient check | `numerical_grad` trên một batch nhỏ, kiểm tra `rel_error < 1e-7` |

(c) Cấu hình đã sửa: *(2đ)*
- `standardize` đặc trưng (lưu mean/std để dùng lúc suy luận): giảm $\kappa$, logit ở mức vừa phải.
- Dùng loss tính thẳng từ logit (log-softmax/`logsumexp`): không còn tràn số.
- AdamW với `lr` khoảng 1e-3, warmup tuyến tính khoảng 5% số bước rồi giảm theo cosine.
- Clip global norm 1.0; weight decay nhỏ (prior Gauss, Mệnh đề 10.5).

(d) Cần theo dõi: *(2đ)*
- loss train/validation (trên toàn tập, không chỉ mini-batch);
- `‖g‖` toàn cục và tỉ lệ số bước bị clip;
- `max|logit|` và entropy của dự đoán (entropy sụp về 0 nghĩa là model quá tự tin);
- learning rate thực tế theo lịch;
- số lượng `nan`/`inf`;
- tỉ lệ $\|\Delta\theta\|/\|\theta\|$ (cỡ $10^{-3}$ là hợp lý; lớn hơn nhiều là dấu hiệu lr quá lớn).
