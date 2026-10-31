# Ngày 1 · Đại số tuyến tính

> **Loại tài liệu: Lý thuyết.** Đầy đủ ký hiệu, định nghĩa, mệnh đề kèm chứng minh, và bài tính tay. Đề thi chấm theo tầng 2.
> **Hôm nay gồm 3 bài:**
> - **01** Vector, chuẩn, tích vô hướng, cosine
> - **02** Ma trận, hạng và bình phương tối thiểu
> - **03** Trị riêng, SVD và PCA
>
> **Thời gian:** đọc ~2h45 · quiz 45' · bài tập ~2h
> **Ký hiệu chung của tuần:** chữ thường đậm hoặc thường ($x$, $\theta$) là vector cột; chữ hoa ($A$, $X$) là ma trận; $X \in \mathbb{R}^{n \times d}$ là $n$ mẫu, mỗi mẫu $d$ đặc trưng; $x^\top y$ là tích vô hướng.
> **Môi trường kiểm chứng:** mọi con số trong file đã chạy trên NumPy 2.5.

---

## 01 · Vector, chuẩn, tích vô hướng, cosine

> ⏱ Tầng 1: 10' · Tầng 2: 35'
> **Cần biết trước:** mảng NumPy, phép `@` (Tuần 1, bài 05).

> **📖 Từ mới**
> - **Vector:** một danh sách số có thứ tự, ví dụ $(3, 4)$. Về hình học, nó là mũi tên từ gốc tọa độ tới điểm $(3, 4)$.
> - **Chiều (dimension):** số phần tử của vector. Embedding của nhiều model văn bản có 768 hoặc 1024 chiều.
> - **Chuẩn (norm):** "độ dài" của vector.
> - **Tích vô hướng (dot product):** nhân từng cặp phần tử rồi cộng lại: $x^\top y = \sum_i x_i y_i$.
> - **Vector đơn vị:** vector có độ dài 1. **Chuẩn hóa** là chia vector cho độ dài của nó.
> - **Trực giao (orthogonal):** hai vector vuông góc, tích vô hướng bằng 0.
> - **Embedding:** vector số mà một model gán cho một đoạn văn bản (hoặc ảnh). Hai đoạn có nghĩa gần nhau thì vector gần nhau.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Mô tả mỗi món ăn bằng hai con số: (độ cay, độ ngọt).
- Phở bò là $(3, 4)$, bún bò là $(4, 3)$. Hai mũi tên này **gần cùng hướng**, nên hai món "giống vị".
- Một bát phở cỡ lớn có thể là $(6, 8)$: **dài gấp đôi** nhưng **cùng hướng** với $(3, 4)$. Về khẩu vị nó vẫn là phở.
- Muốn so "giống vị" mà không quan tâm bát to hay nhỏ, ta so **góc** giữa hai mũi tên chứ không so độ dài. Đó là **cosine similarity**.

**Ba ý cốt lõi:**
1. Vector vừa là danh sách số, vừa là mũi tên có **độ dài** và **hướng**.
2. Tích vô hướng = độ dài × độ dài × cos(góc). Chia cho hai độ dài thì còn lại cos(góc): **1** là cùng hướng, **0** là vuông góc (không liên quan), **−1** là ngược hướng.
3. Tìm kiếm ngữ nghĩa (RAG, Tuần 8) chính là: biến câu hỏi và tài liệu thành vector rồi tìm các vector có cosine lớn nhất.

**Ví dụ nhỏ.** $a = (3, 4)$, $b = (4, 3)$:
- $a^\top b = 3 \cdot 4 + 4 \cdot 3 = 24$
- $\|a\| = \sqrt{9 + 16} = 5$, $\|b\| = 5$
- $\cos(a, b) = 24 / (5 \cdot 5) = 0.96$: gần như cùng hướng.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 1.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Vector space** $\mathbb{R}^d$ | Tập các vector $d$ chiều, đóng với phép cộng và nhân vô hướng | Strang, *Intro to Linear Algebra*, ch. 1 |
| **Norm** $\|\cdot\|$ | Hàm thỏa 3 tiên đề: $\|x\| \ge 0$ và $\|x\| = 0 \Leftrightarrow x = 0$; $\|\alpha x\| = \lvert\alpha\rvert\,\|x\|$; $\|x + y\| \le \|x\| + \|y\|$ (bất đẳng thức tam giác) | MML (Deisenroth et al.) §3.1 |
| **$L_p$ norm** | $\|x\|_p = \left(\sum_i \lvert x_i\rvert^p\right)^{1/p}$, với $p \ge 1$. $\|x\|_\infty = \max_i \lvert x_i\rvert$ | Goodfellow et al., *Deep Learning* §2.5 |
| **Inner product** | Với $\mathbb{R}^d$: $\langle x, y\rangle = x^\top y = \sum_i x_i y_i$; và $\|x\|_2 = \sqrt{x^\top x}$ | MML §3.2 |
| **Cosine similarity** | $\cos(x, y) = \dfrac{x^\top y}{\|x\|_2\,\|y\|_2}$, chỉ định nghĩa khi $x, y \ne 0$ | — |
| **Orthogonal** | $x^\top y = 0$. **Orthonormal**: trực giao và mỗi vector có độ dài 1 | MML §3.4 |
| **Projection** | Hình chiếu của $x$ lên đường thẳng theo hướng $y$: $\mathrm{proj}_y x = \dfrac{x^\top y}{y^\top y}\, y$ | Strang §4.2 |

#### 1.2 Mệnh đề và chứng minh

**Mệnh đề 1.1 (Cauchy–Schwarz).** Với mọi $x, y \in \mathbb{R}^d$: $\lvert x^\top y\rvert \le \|x\|\,\|y\|$. Dấu bằng xảy ra khi và chỉ khi $x, y$ cùng phương.

*Chứng minh.* Nếu $y = 0$ thì hiển nhiên. Với $y \ne 0$, xét hàm bậc hai theo $t$:

$$g(t) = \|x - t y\|^2 = \|x\|^2 - 2t\, x^\top y + t^2 \|y\|^2 \ge 0 \quad \forall t.$$

Một đa thức bậc hai không âm với mọi $t$ thì có biệt thức $\le 0$: $4(x^\top y)^2 - 4\|x\|^2\|y\|^2 \le 0$. Dấu bằng $\Leftrightarrow$ tồn tại $t$ để $g(t) = 0$, tức $x = ty$. ∎

*Hệ quả:* $\cos(x, y) \in [-1, 1]$, nên cosine luôn là một "góc" hợp lệ. Bất đẳng thức tam giác của $L_2$ cũng suy ra từ đây: $\|x+y\|^2 = \|x\|^2 + 2x^\top y + \|y\|^2 \le (\|x\| + \|y\|)^2$.

**Mệnh đề 1.2 (ý nghĩa hình học).** $x^\top y = \|x\|\,\|y\|\cos\varphi$, với $\varphi$ là góc giữa hai vector.

*Chứng minh.* Định lý cosine trong tam giác có hai cạnh $x$, $y$: $\|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2\|x\|\|y\|\cos\varphi$. Khai triển vế trái: $\|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2x^\top y$. So sánh hai vế. ∎

**Mệnh đề 1.3 (cosine và khoảng cách trên vector đơn vị).** Nếu $\|u\| = \|v\| = 1$ thì

$$\|u - v\|^2 = 2 - 2\cos(u, v).$$

*Chứng minh.* $\|u - v\|^2 = \|u\|^2 + \|v\|^2 - 2u^\top v = 2 - 2u^\top v$, và $u^\top v = \cos(u, v)$ vì hai độ dài bằng 1. ∎

*Hệ quả quan trọng cho RAG:* sau khi chuẩn hóa, **xếp hạng theo cosine, theo tích vô hướng và theo khoảng cách L2 cho cùng một kết quả** ($\|u-v\|$ là hàm giảm của cosine). Vì vậy các vector database thường yêu cầu chuẩn hóa embedding rồi dùng tích vô hướng, phép tính rẻ nhất.

**Mệnh đề 1.4 (hình chiếu).** Phần dư $r = x - \mathrm{proj}_y x$ trực giao với $y$.

*Chứng minh.* $y^\top r = y^\top x - \dfrac{x^\top y}{y^\top y}\, y^\top y = 0$. ∎

Hình chiếu là điểm trên đường thẳng **gần $x$ nhất**. Ý này sẽ quay lại ở bài 02, bình phương tối thiểu.

#### 1.3 Ví dụ tính tay

Với $a = (3, 4)$, $b = (4, 3)$:

| Đại lượng | Phép tính | Kết quả |
|---|---|---|
| $\|a\|_1$ | $3 + 4$ | 7 |
| $\|a\|_2$ | $\sqrt{9 + 16}$ | 5 |
| $\|a\|_\infty$ | $\max(3, 4)$ | 4 |
| $a^\top b$ | $12 + 12$ | 24 |
| $\cos(a, b)$ | $24 / 25$ | 0.96 |
| $\mathrm{proj}_b\, a$ | $(24/25)(4, 3)$ | $(3.84,\ 2.88)$ |
| Phần dư $r$ | $(3, 4) - (3.84, 2.88)$ | $(-0.84,\ 1.12)$; kiểm tra $b^\top r = -3.36 + 3.36 = 0$ ✓ |
| $\|u - v\|^2$ với $u = a/5$, $v = b/5$ | $(−0.2)^2 + 0.2^2$ | $0.08 = 2 - 2 \cdot 0.96$ ✓ |

Thứ tự luôn có: $\|x\|_\infty \le \|x\|_2 \le \|x\|_1$ (ở đây $4 \le 5 \le 7$).

#### 1.4 Các con số cần biết: chiều cao thì vector ngẫu nhiên gần như vuông góc

Lấy ngẫu nhiên 1000 cặp vector đơn vị (hướng phân bố đều) rồi đo cosine. Đã chạy trên máy của bạn:

| Số chiều $d$ | Độ lệch chuẩn của cosine | $1/\sqrt{d}$ | Tỉ lệ cặp có $\lvert\cos\rvert > 0.1$ |
|---|---|---|---|
| 2 | 0.715 | 0.707 | 93.9% |
| 10 | 0.303 | 0.316 | 71.8% |
| 100 | 0.098 | 0.100 | 31.5% |
| 768 | 0.036 | 0.036 | **0.3%** |

Cosine của hai vector ngẫu nhiên có trung bình 0 và độ lệch chuẩn xấp xỉ $1/\sqrt{d}$. Ở 768 chiều, gần như mọi cặp ngẫu nhiên đều có cosine trong khoảng $\pm 0.1$.

Lưu ý: embedding thật **không** phân bố đều theo mọi hướng (hiện tượng *anisotropy*, Ethayarajh 2019), nên hai câu chẳng liên quan vẫn có thể có cosine 0.2–0.4 tùy model. **Ngưỡng cosine phải được hiệu chỉnh trên dữ liệu của chính bạn**, không có con số dùng chung.

Bộ nhớ: một vector 768 chiều float32 tốn $768 \times 4 = 3\,072$ B ≈ 3 KB.

#### 1.5 Code chuẩn

```python
import numpy as np

def normalize(X, eps=1e-12):
    """Chuẩn hóa từng hàng về độ dài 1; hàng toàn 0 giữ nguyên 0 thay vì sinh nan."""
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(norms, eps)

def cosine_matrix(A, B):
    return normalize(A) @ normalize(B).T          # (m, d) @ (d, n) -> (m, n)

def angle_deg(u, v):
    c = cosine_matrix(u[None], v[None])[0, 0]
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))   # clip: xem cạm bẫy 2
```

#### 1.6 Cạm bẫy

1. **Vector 0.** Cosine không xác định (chia cho 0). Code ngây thơ trả về `nan`, và `nan` làm hỏng cả phép sắp xếp top-k. Luôn chặn bằng `eps`, hoặc loại vector rỗng từ trước.
2. **Sai số làm tròn của float32.** Đo trên máy bạn: với 1000 vector 768 chiều float32 đã chuẩn hóa, **100 vector** có $u^\top u > 1$ (ví dụ `1.0000001`). Gọi `arccos` trên giá trị đó ra `nan`. Phải `np.clip(c, -1, 1)`.
3. **Dùng tích vô hướng trên vector chưa chuẩn hóa.** Vector có độ dài lớn (thường là văn bản dài) sẽ luôn được xếp lên đầu dù không liên quan hơn.
4. **So sánh cosine giữa hai model embedding khác nhau.** Mỗi model có không gian riêng; cosine 0.8 của model A không tương đương 0.8 của model B.
5. **Nhầm $L_1$ và $L_2$.** $L_2$ phạt nặng sai số lớn (bình phương), $L_1$ phạt đều. Ý này quay lại ở Ngày 3 (L1/L2 regularization).

**Sắc thái hay bị hỏi khi phỏng vấn:** *"Cosine, dot product hay L2 cho embedding?"* Nếu vector đã chuẩn hóa thì ba cách cho **cùng thứ hạng** (Mệnh đề 1.3), nên chọn dot product vì rẻ nhất. Nếu chưa chuẩn hóa, dot product thiên vị vector dài. Chỉ dùng dot product thô khi model được huấn luyện để độ dài vector mang ý nghĩa; tài liệu của model sẽ nói rõ điều này.

#### 1.7 Đọc thêm

- G. Strang, *Introduction to Linear Algebra* (6th ed.), ch. 1.
- Deisenroth, Faisal, Ong, *Mathematics for Machine Learning* (MML, 2020), §3.1–3.4. Bản PDF miễn phí tại mml-book.github.io.
- K. Ethayarajh, *How Contextual are Contextualized Word Representations?*, EMNLP 2019.

---

## 02 · Ma trận, hạng và bình phương tối thiểu

> ⏱ Tầng 1: 15' · Tầng 2: 40'
> **Cần biết trước:** tích vô hướng và hình chiếu (bài 01).

> **📖 Từ mới**
> - **Ma trận:** bảng số $m$ hàng × $n$ cột. Nhân ma trận với vector $Ax$ là "đưa vector vào máy biến đổi $A$".
> - **Ánh xạ tuyến tính:** phép biến đổi giữ nguyên phép cộng và nhân vô hướng (xoay, co giãn, chiếu…).
> - **Chuyển vị ($A^\top$):** đổi hàng thành cột.
> - **Ma trận đơn vị ($I$):** ma trận "không làm gì": $Ix = x$.
> - **Nghịch đảo ($A^{-1}$):** máy "làm ngược lại" $A$: $A^{-1}A = I$.
> - **Định thức (det):** hệ số mà $A$ phóng to diện tích (2D) hoặc thể tích (3D).
> - **Hạng (rank):** số chiều thật sự mà $A$ giữ lại sau khi biến đổi.
> - **Bình phương tối thiểu (least squares):** chọn tham số sao cho tổng bình phương sai số nhỏ nhất.
> - **Số điều kiện (condition number, $\kappa$):** mức độ "nhạy cảm" của bài toán: sai số nhỏ ở đầu vào bị khuếch đại bao nhiêu lần ở đầu ra.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- Ma trận là một **cái máy biến hình**: đưa vào một mũi tên, nhận ra một mũi tên khác. Ghép hai máy liên tiếp ($B$ rồi $A$) chính là phép nhân ma trận $AB$.
- **Định thức** cho biết máy phóng to diện tích bao nhiêu lần. Nếu định thức bằng 0, máy **ép phẳng** mặt phẳng thành một đường thẳng: hai mũi tên khác nhau có thể cho cùng một kết quả, nên không thể "chạy ngược" để biết đầu vào. Máy đó **không có nghịch đảo**.
- **Bình phương tối thiểu:** bạn có 3 điểm đo không thẳng hàng và muốn vẽ một đường thẳng. Không đường nào đi qua cả 3 điểm, nên ta chọn đường làm **tổng bình phương khoảng cách theo chiều dọc** nhỏ nhất.

**Ba ý cốt lõi:**
1. $Ax$ là **tổ hợp tuyến tính các cột** của $A$, với hệ số lấy từ $x$.
2. Ma trận **thiếu hạng** làm mất thông tin, nên không đảo ngược được.
3. Khi có nhiều phương trình hơn ẩn, thường không có nghiệm đúng. Nghiệm bình phương tối thiểu là **hình chiếu** của $y$ lên tập các giá trị mà mô hình tạo ra được.

**Ví dụ nhỏ.** Ba điểm $(0, 1)$, $(1, 3)$, $(2, 4)$. Đường thẳng tốt nhất là $y \approx 1.167 + 1.5x$ (tính chi tiết ở mục 2.4).

---

### TẦNG 2 · Kiến thức chuẩn

#### 2.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Linear map** | $f(\alpha x + \beta y) = \alpha f(x) + \beta f(y)$. Mọi ánh xạ tuyến tính $\mathbb{R}^n \to \mathbb{R}^m$ đều viết được là $x \mapsto Ax$ | Strang ch. 8 |
| **Matrix product** | $(AB)_{ij} = \sum_k A_{ik}B_{kj}$; $(m \times n)(n \times p) \to (m \times p)$. Là **hợp** của hai ánh xạ: $(AB)x = A(Bx)$. Không giao hoán | Strang §2.4 |
| **Hadamard product** | Nhân từng phần tử $A \odot B$ (NumPy: `*`). Khác hoàn toàn với `@` | — |
| **Transpose** | $(A^\top)_{ij} = A_{ji}$; $(AB)^\top = B^\top A^\top$ | Strang §2.7 |
| **Inverse** | $A^{-1}A = AA^{-1} = I$, chỉ với ma trận vuông; tồn tại $\Leftrightarrow \det A \ne 0 \Leftrightarrow$ hạng đầy đủ | Strang §2.5 |
| **Determinant** | Hệ số co giãn thể tích có dấu; $\det(AB) = \det A \cdot \det B$; bằng tích các trị riêng | Strang ch. 5 |
| **Rank** | $\dim$ của không gian cột = $\dim$ của không gian hàng | Strang §3.3 |
| **Rank–nullity** | Với $A \in \mathbb{R}^{m \times n}$: $\mathrm{rank}(A) + \dim\mathcal{N}(A) = n$, với $\mathcal{N}(A) = \{x : Ax = 0\}$ | Strang §3.3 |
| **Trace** | $\mathrm{tr}(A) = \sum_i A_{ii}$ = tổng các trị riêng; $\mathrm{tr}(AB) = \mathrm{tr}(BA)$ | MML §4.1 |
| **Least squares** | $\theta^\star = \arg\min_\theta \|X\theta - y\|_2^2$ | Strang §4.3 |
| **Normal equations** | $X^\top X\, \theta = X^\top y$ | Strang §4.3 |
| **Condition number** | $\kappa(A) = s_{\max}/s_{\min}$ (tỉ số giá trị kỳ dị lớn nhất và nhỏ nhất; xem bài 03) | Golub & Van Loan §2.6 |

#### 2.2 Hệ $Ax = b$: ba trường hợp

- **Có nghiệm** $\Leftrightarrow$ $b$ nằm trong không gian cột của $A$ (là một tổ hợp của các cột).
- **Nghiệm duy nhất** $\Leftrightarrow$ có nghiệm **và** $\mathcal{N}(A) = \{0\}$ (các cột độc lập tuyến tính).
- Trong ML, $X \in \mathbb{R}^{n \times d}$ với $n \gg d$ (nhiều mẫu hơn tham số): hệ **quá xác định** (overdetermined), $y$ gần như chắc chắn không nằm trong không gian cột, nên ta giải bài toán bình phương tối thiểu.

#### 2.3 Mệnh đề và chứng minh

**Mệnh đề 2.1 (phương trình chuẩn).** $\theta^\star$ cực tiểu hóa $\|X\theta - y\|^2$ khi và chỉ khi $X^\top(X\theta^\star - y) = 0$.

*Chứng minh (hình học).* Khi $\theta$ chạy khắp $\mathbb{R}^d$, $X\theta$ chạy khắp không gian cột $\mathcal{C}(X)$. Điểm trong $\mathcal{C}(X)$ gần $y$ nhất là **hình chiếu vuông góc** của $y$ (tổng quát hóa Mệnh đề 1.4). Điều kiện vuông góc: phần dư $r = y - X\theta^\star$ trực giao với **mọi cột** của $X$, tức $X^\top r = 0$. Ngày 2 sẽ chứng minh lại bằng giải tích: gradient $2X^\top(X\theta - y)$ bằng 0. ∎

**Bổ đề 2.2.** $X^\top X$ khả nghịch $\Leftrightarrow$ $X$ có hạng cột đầy đủ ($\mathrm{rank}\,X = d$).

*Chứng minh.* Nếu $X^\top X v = 0$ thì $v^\top X^\top X v = \|Xv\|^2 = 0$, nên $Xv = 0$. Khi các cột độc lập, $Xv = 0$ kéo theo $v = 0$, nên $X^\top X$ không có hạt nhân khác 0 và khả nghịch. Chiều ngược lại: nếu $Xv = 0$ với $v \ne 0$ thì $X^\top X v = 0$. ∎

Khi đó $\theta^\star = (X^\top X)^{-1}X^\top y$. **Nhưng không bao giờ tính đúng như công thức này** (xem 2.5).

**Hệ quả 2.3.** Nếu $X$ có một cột toàn số 1 (cột bias) thì **tổng các phần dư bằng 0**, vì tích vô hướng của cột đó với $r$ chính là $\sum_i r_i$.

**Mệnh đề 2.4 (bình phương số điều kiện).** $\kappa(X^\top X) = \kappa(X)^2$.

*Chứng minh.* Với SVD $X = U\Sigma V^\top$ (bài 03), $X^\top X = V\Sigma^2 V^\top$, nên các giá trị kỳ dị của $X^\top X$ là $s_i^2$. Do đó $\kappa(X^\top X) = s_{\max}^2/s_{\min}^2$. ∎

*Ý nghĩa:* theo quy tắc kinh nghiệm của giải tích số (Higham), giải một hệ có số điều kiện $\kappa$ sẽ **mất khoảng $\log_{10}\kappa$ chữ số chính xác**. float64 có khoảng 16 chữ số. Đi qua $X^\top X$ thì số chữ số mất đi tăng **gấp đôi**.

Đã đo trên máy bạn: hồi quy có 3 cột $[1, t, t + \varepsilon \cdot \text{nhiễu}]$ (hai cột gần trùng nhau), $\theta$ đúng là $(1, 2, 3)$:

| $\varepsilon$ | $\kappa(X)$ | $\kappa(X^\top X)$ | Sai số giải phương trình chuẩn | Sai số `lstsq` (qua SVD) |
|---|---|---|---|---|
| $10^{-2}$ | 174 | $3.0 \times 10^4$ | $1.3 \times 10^{-11}$ | $4.4 \times 10^{-16}$ |
| $10^{-4}$ | $1.9 \times 10^4$ | $3.6 \times 10^8$ | $1.5 \times 10^{-7}$ | $2.9 \times 10^{-12}$ |
| $10^{-6}$ | $2.0 \times 10^6$ | $4.0 \times 10^{12}$ | $\mathbf{1.7 \times 10^{-3}}$ | $2.7 \times 10^{-12}$ |

#### 2.4 Ví dụ tính tay: hồi quy qua 3 điểm

Điểm $(0, 1), (1, 3), (2, 4)$, mô hình $y = \theta_0 + \theta_1 x$:

$$X = \begin{bmatrix}1 & 0\\ 1 & 1\\ 1 & 2\end{bmatrix},\quad y = \begin{bmatrix}1\\3\\4\end{bmatrix},\quad X^\top X = \begin{bmatrix}3 & 3\\ 3 & 5\end{bmatrix},\quad X^\top y = \begin{bmatrix}8\\11\end{bmatrix}.$$

- $\det(X^\top X) = 15 - 9 = 6$, nên $(X^\top X)^{-1} = \dfrac{1}{6}\begin{bmatrix}5 & -3\\ -3 & 3\end{bmatrix}$.
- $\theta^\star = \dfrac{1}{6}\begin{bmatrix}5 \cdot 8 - 3 \cdot 11\\ -3 \cdot 8 + 3 \cdot 11\end{bmatrix} = \begin{bmatrix}7/6\\ 3/2\end{bmatrix} \approx \begin{bmatrix}1.1667\\1.5\end{bmatrix}$.
- Dự đoán: $1.1667,\ 2.6667,\ 4.1667$. Phần dư $r = (-1/6,\ 1/3,\ -1/6)$.
- Kiểm tra: $\sum r_i = 0$ ✓ (Hệ quả 2.3); $\sum x_i r_i = 0 + 1/3 - 2/6 = 0$ ✓ (Mệnh đề 2.1).
- Tổng bình phương sai số: $1/36 + 1/9 + 1/36 = 1/6$.

**Thêm ba phép tính nhỏ:**
- $A = \begin{bmatrix}2 & 1\\1 & 3\end{bmatrix}$, $B = \begin{bmatrix}1 & 0\\2 & 1\end{bmatrix}$: $AB = \begin{bmatrix}4 & 1\\7 & 3\end{bmatrix}$, còn $A \odot B = \begin{bmatrix}2 & 0\\2 & 3\end{bmatrix}$.
- $\det A = 6 - 1 = 5$: hình vuông đơn vị biến thành hình bình hành diện tích 5.
- $\begin{bmatrix}1 & 2\\2 & 4\end{bmatrix}$ có hạng 1 (hàng 2 = 2 × hàng 1), $\det = 0$, mọi vector bị ép lên đường thẳng hướng $(1, 2)$.

#### 2.5 Các con số cần biết

| Phép toán | Chi phí | Ví dụ |
|---|---|---|
| $(m \times n) @ (n \times p)$ | $2mnp$ FLOP (mỗi phần tử: $n$ phép nhân + $n$ phép cộng) | $(32 \times 768) @ (768 \times 3072)$ = **151 triệu FLOP**: một lớp feed-forward của Transformer với 32 token |
| `solve` / `inv` ma trận $d \times d$ | $O(d^3)$ | $d = 10^4$: khoảng $10^{12}$ phép tính |
| `lstsq` qua SVD với $X \in \mathbb{R}^{n \times d}$ | $O(nd^2)$ | Không khả thi khi $d$ lên tới hàng triệu; khi đó phải dùng gradient descent (Ngày 2) |

#### 2.6 Code chuẩn

```python
theta = np.linalg.lstsq(X, y, rcond=None)[0]      # ổn định nhất: qua SVD, chạy được cả khi thiếu hạng
theta = np.linalg.solve(X.T @ X, X.T @ y)         # nhanh hơn, nhưng κ bị bình phương
# KHÔNG: np.linalg.inv(X.T @ X) @ X.T @ y         # chậm hơn solve và kém chính xác hơn
np.linalg.matrix_rank(X)                          # hạng, có ngưỡng dung sai số học
np.linalg.cond(X)                                 # κ(X) = s_max / s_min
```

#### 2.7 Cạm bẫy

1. **`*` thay cho `@`.** `A * B` là nhân từng phần tử; với shape vuông thì không báo lỗi, chỉ sai lặng lẽ.
2. **Mảng 1 chiều.** `x.T` của mảng shape `(d,)` vẫn là `(d,)`; `x @ x` là tích vô hướng, nhưng `np.outer(x, x)` mới là ma trận $d \times d$.
3. **Dùng `inv`.** Tính nghịch đảo tốn hơn và kém chính xác hơn `solve`. Gần như không bao giờ cần ma trận nghịch đảo một cách tường minh.
4. **Dùng định thức để kiểm tra "gần suy biến".** $0.1 \cdot I_{100}$ có $\det = 10^{-100}$ nhưng lại có điều kiện hoàn hảo ($\kappa = 1$). Hãy dùng `cond` hoặc `matrix_rank`.
5. **Quên cột bias.** Khi đó đường thẳng buộc phải đi qua gốc tọa độ, và tổng phần dư không còn bằng 0.
6. **Hai đặc trưng gần trùng nhau** (đa cộng tuyến, *multicollinearity*): $\kappa$ rất lớn, hệ số $\theta$ dao động mạnh khi dữ liệu thay đổi chút ít. Cách chữa: bỏ bớt đặc trưng, hoặc dùng ridge (Ngày 3, bài 10).

**Sắc thái hay bị hỏi khi phỏng vấn:** *"Có công thức nghiệm đóng cho linear regression, sao còn dùng gradient descent?"* Vì (1) chi phí $O(nd^2 + d^3)$ không khả thi khi $d$ lớn, (2) phải chứa cả $X$ trong RAM, trong khi gradient descent chạy theo mini-batch, (3) mất độ chính xác khi $\kappa$ lớn, và (4) hầu hết các model khác (logistic regression, mạng neural) **không có** nghiệm đóng.

#### 2.8 Đọc thêm

- Strang ch. 2–4 (đặc biệt §4.2–4.3: projection và least squares).
- MML §2 và §9.2 (linear regression).
- N. Higham, *Accuracy and Stability of Numerical Algorithms* (2nd ed.), ch. 1 và ch. 20.

---

## 03 · Trị riêng, SVD và PCA

> ⏱ Tầng 1: 15' · Tầng 2: 45'
> **Cần biết trước:** tích ma trận, chuyển vị, trực giao (bài 01–02); phương sai (độ phân tán quanh trung bình).

> **📖 Từ mới**
> - **Vector riêng (eigenvector) và trị riêng (eigenvalue):** một hướng mà ma trận chỉ **co giãn** chứ không xoay: $Av = \lambda v$. $\lambda$ là hệ số co giãn.
> - **Ma trận đối xứng:** $A = A^\top$. Ma trận hiệp phương sai và Hessian (Ngày 2) đều đối xứng.
> - **Ma trận hiệp phương sai (covariance):** bảng $d \times d$ cho biết từng cặp đặc trưng biến thiên cùng nhau tới mức nào.
> - **SVD (phân tích giá trị kỳ dị):** viết **mọi** ma trận thành xoay → co giãn theo trục → xoay.
> - **PCA (phân tích thành phần chính):** tìm các trục mà dữ liệu trải rộng nhất, rồi chỉ giữ vài trục đầu để giảm số chiều.
> - **Phương sai giải thích (explained variance ratio):** phần trăm tổng phương sai mà một trục giữ lại được.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- Đặt một bàn tay lên tấm cao su và kéo giãn. Hầu hết các đường kẻ trên tấm cao su vừa dài ra vừa **bị lệch hướng**. Nhưng có vài hướng đặc biệt chỉ dài ra hoặc co lại mà **không lệch**: đó là vector riêng, và tỉ lệ dài ra là trị riêng.
- Một đám mây điểm dữ liệu có hình **quả trứng nằm nghiêng**. Trục dài của quả trứng là hướng dữ liệu biến thiên nhiều nhất. Nếu chỉ được giữ một con số cho mỗi điểm, hãy giữ **vị trí của điểm trên trục dài**: thông tin mất đi là ít nhất. Đó là PCA.

**Ba ý cốt lõi:**
1. $Av = \lambda v$: hướng $v$ không đổi, chỉ bị nhân với $\lambda$.
2. Với ma trận hiệp phương sai, các vector riêng **vuông góc với nhau** và là các trục chính; trị riêng là **phương sai dọc theo trục đó**.
3. SVD làm được với **mọi** ma trận (kể cả không vuông) và là cách tính PCA chuẩn trong thực tế.

**Ví dụ nhỏ.** 4 điểm $(-3, -1), (-1, -3), (1, 3), (3, 1)$ nằm dọc theo đường chéo $y = x$. Trục chính thứ nhất là $(1, 1)/\sqrt{2}$, giữ được **80%** phương sai.

---

### TẦNG 2 · Kiến thức chuẩn

#### 3.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Eigenpair** | $(\lambda, v)$, $v \ne 0$, $Av = \lambda v$. $\lambda$ là nghiệm của **đa thức đặc trưng** $\det(A - \lambda I) = 0$ | Strang ch. 6 |
| **Spectral theorem** | Ma trận thực đối xứng $A$ có $A = Q\Lambda Q^\top$, với $Q$ trực giao ($Q^\top Q = I$) và $\Lambda$ chéo, thực | Strang §6.4 |
| **Positive semidefinite (PSD)** | $x^\top A x \ge 0\ \forall x$. Với $A$ đối xứng: PSD $\Leftrightarrow$ mọi $\lambda_i \ge 0$ | MML §3.2.3 |
| **Sample covariance** | $C = \dfrac{1}{n-1}\tilde X^\top \tilde X$, với $\tilde X$ là $X$ đã trừ trung bình từng cột | Wasserman, *All of Statistics* |
| **SVD** | $X = U\Sigma V^\top$; $U \in \mathbb{R}^{n \times r}$, $V \in \mathbb{R}^{d \times r}$ có cột trực chuẩn; $\Sigma = \mathrm{diag}(s_1 \ge \dots \ge s_r > 0)$ | Golub & Van Loan §2.4 |
| **PCA** | Các trục chính là vector riêng của $C$, sắp theo trị riêng giảm dần | Pearson (1901); Hotelling (1933) |
| **Explained variance ratio** | $\lambda_k / \sum_j \lambda_j$ | — |
| **Eckart–Young–Mirsky** | Xấp xỉ hạng $k$ tốt nhất của $X$ (theo chuẩn Frobenius) là SVD cắt ngắn $X_k = \sum_{i \le k} s_i u_i v_i^\top$, với sai số $\|X - X_k\|_F^2 = \sum_{i > k} s_i^2$ | Eckart & Young (1936) |
| **Power iteration** | Lặp $x \leftarrow Ax / \|Ax\|$: hội tụ về vector riêng ứng với $\lvert\lambda\rvert$ lớn nhất, tốc độ $\lvert\lambda_2/\lambda_1\rvert^k$ | Golub & Van Loan §7.3 |
| **Rayleigh quotient** | $R(x) = \dfrac{x^\top A x}{x^\top x}$; ước lượng trị riêng từ vector gần đúng | — |

#### 3.2 Mệnh đề và chứng minh

**Mệnh đề 3.1.** Với $A$ đối xứng, hai vector riêng ứng với hai trị riêng khác nhau thì trực giao.

*Chứng minh.* $\lambda_1 v_1^\top v_2 = (Av_1)^\top v_2 = v_1^\top A^\top v_2 = v_1^\top A v_2 = \lambda_2 v_1^\top v_2$. Suy ra $(\lambda_1 - \lambda_2)\,v_1^\top v_2 = 0$, mà $\lambda_1 \ne \lambda_2$ nên $v_1^\top v_2 = 0$. ∎

**Mệnh đề 3.2.** Ma trận hiệp phương sai là PSD, và $w^\top C w$ (với $\|w\| = 1$) là **phương sai của dữ liệu khi chiếu lên hướng $w$**.

*Chứng minh.* Điểm $i$ chiếu lên $w$ có tọa độ $\tilde x_i^\top w$; trung bình các tọa độ này bằng 0 vì dữ liệu đã trừ trung bình. Phương sai mẫu của chúng là $\frac{1}{n-1}\sum_i (\tilde x_i^\top w)^2 = \frac{1}{n-1}\|\tilde X w\|^2 = w^\top C w \ge 0$. ∎

**Mệnh đề 3.3 (PCA là bài toán tối ưu).** $\max_{\|w\|=1} w^\top C w = \lambda_{\max}$, đạt được tại vector riêng $q_1$ tương ứng.

*Chứng minh.* Theo spectral theorem, $C = \sum_i \lambda_i q_i q_i^\top$ với $\{q_i\}$ trực chuẩn. Viết $w = \sum_i c_i q_i$, ta có $\sum_i c_i^2 = 1$. Khi đó $w^\top C w = \sum_i \lambda_i c_i^2 \le \lambda_{\max}\sum_i c_i^2 = \lambda_{\max}$, dấu bằng khi $w = q_1$. Lặp lại lập luận trên không gian trực giao với $q_1$ để được trục thứ hai, v.v. ∎

**Mệnh đề 3.4 (SVD ↔ PCA).** Nếu $\tilde X = U\Sigma V^\top$ thì các cột của $V$ là các trục chính và trị riêng của $C$ là $\lambda_i = s_i^2/(n-1)$.

*Chứng minh.* $\tilde X^\top \tilde X = V\Sigma U^\top U \Sigma V^\top = V\Sigma^2 V^\top$. Đây chính là phân tích phổ của $(n-1)C$. ∎

*Vì sao thực tế tính PCA bằng SVD của $\tilde X$ chứ không bằng `eig` của $C$?* Vì tạo ra $C$ là đi qua $\tilde X^\top\tilde X$, nên số điều kiện bị **bình phương** (Mệnh đề 2.4). scikit-learn cũng dùng SVD.

#### 3.3 Ví dụ tính tay

**(a) Trị riêng của $A = \begin{bmatrix}2 & 1\\1 & 2\end{bmatrix}$.**
- $\det(A - \lambda I) = (2 - \lambda)^2 - 1 = 0$, nên $\lambda \in \{3, 1\}$.
- $\lambda = 3$: $(A - 3I)v = 0 \Rightarrow -v_1 + v_2 = 0$, $v = (1, 1)/\sqrt{2}$.
- $\lambda = 1$: $v = (1, -1)/\sqrt{2}$. Hai vector vuông góc ✓ (Mệnh đề 3.1).
- Kiểm tra: $\mathrm{tr}A = 4 = 3 + 1$; $\det A = 3 = 3 \cdot 1$ ✓.

**(b) PCA cho 4 điểm $(-3,-1), (-1,-3), (1,3), (3,1)$.**
- Trung bình bằng $(0, 0)$, nên $\tilde X = X$.
- $\tilde X^\top \tilde X$: $\sum x_1^2 = 9 + 1 + 1 + 9 = 20$; $\sum x_2^2 = 20$; $\sum x_1x_2 = 3 + 3 + 3 + 3 = 12$.
- $C = \dfrac{1}{3}\begin{bmatrix}20 & 12\\12 & 20\end{bmatrix} = \begin{bmatrix}6.667 & 4\\4 & 6.667\end{bmatrix}$.
- Theo (a) (cùng dạng $\begin{bmatrix}a & b\\b & a\end{bmatrix}$, trị riêng $a \pm b$): $\lambda_1 = 10.667$ với trục $(1, 1)/\sqrt{2}$; $\lambda_2 = 2.667$ với trục $(1, -1)/\sqrt{2}$.
- Tỉ lệ phương sai giải thích: $10.667/13.333 = $ **0.8** và **0.2**.
- Tọa độ trên PC1: $(-3-1)/\sqrt{2} = -2.828$, rồi $-2.828,\ 2.828,\ 2.828$.
- SVD: $s_1 = \sqrt{32} = 5.657$, $s_2 = \sqrt{8} = 2.828$; kiểm tra $s_1^2/3 = 10.667$ ✓.
- Xấp xỉ hạng 1: các điểm trở thành $(-2,-2), (-2,-2), (2,2), (2,2)$; sai số Frobenius $= \sqrt{4 \cdot 2} = 2.828 = s_2$ ✓ (Eckart–Young).

**(c) Power iteration** trên $A$ của câu (a), bắt đầu từ $x = (1, 0)$. Đã chạy:

| Vòng lặp $k$ | $x_k$ | Rayleigh quotient |
|---|---|---|
| 1 | $(0.894,\ 0.447)$ | 2.8 |
| 2 | $(0.781,\ 0.625)$ | 2.976 |
| 3 | $(0.733,\ 0.681)$ | 2.9973 |
| 5 | $(0.710,\ 0.704)$ | 2.99997 |

Sai số của trị riêng giảm khoảng 9 lần mỗi vòng, tức $(\lambda_2/\lambda_1)^2 = (1/3)^2$. Hai trị riêng càng gần nhau thì hội tụ càng chậm.

#### 3.4 Liên hệ với AI

| Ở đâu | Đại số tuyến tính đứng sau |
|---|---|
| **Nén embedding** | PCA 768 → 256 chiều: mỗi vector float32 từ 3 072 B còn 1 024 B; 10 triệu vector từ 30.7 GB còn 10.2 GB. Phải **chuẩn hóa lại** sau khi chiếu nếu dùng cosine |
| **Tốc độ gradient descent** (Ngày 2) | Trị riêng của Hessian quyết định learning rate tối đa ($2/\lambda_{\max}$) và số bước ($\approx \kappa = \lambda_{\max}/\lambda_{\min}$) |
| **LoRA** (Tuần 11) | Cập nhật trọng số bằng ma trận hạng thấp $\Delta W = BA$; ý tưởng gần với Eckart–Young: phần lớn "năng lượng" nằm trong vài hướng |
| **Trực quan hóa embedding** (Tuần 3) | Chiếu xuống 2D bằng PCA; t-SNE/UMAP là các phương pháp phi tuyến cho cùng mục đích |

#### 3.5 Code chuẩn

```python
def pca(X, k):
    mean = X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)   # KHÔNG eig(np.cov(X))
    var = s**2 / (len(X) - 1)
    return Vt[:k], var[:k] / var.sum(), mean                   # trục, tỉ lệ phương sai, trung bình

w, Q = np.linalg.eigh(C)      # ma trận ĐỐI XỨNG: dùng eigh — trị riêng thực, tăng dần, Q trực chuẩn
```

#### 3.6 Cạm bẫy

1. **Quên trừ trung bình.** Khi đó trục "chính" đầu tiên chỉ trỏ về phía điểm trung bình của dữ liệu.
2. **Không chuẩn hóa đặc trưng có đơn vị khác nhau.** Một cột tính bằng đồng (hàng triệu) sẽ chiếm toàn bộ PC1.
3. **Dùng `np.linalg.eig` cho ma trận đối xứng.** Có thể trả về số phức với phần ảo rất nhỏ, và không sắp xếp. Dùng `eigh` (trả về tăng dần, nên đảo ngược nếu cần giảm dần).
4. **Dấu của vector riêng không xác định.** $v$ và $-v$ đều đúng. So sánh hai lần chạy phải theo một quy ước dấu (project dùng quy ước: phần tử có trị tuyệt đối lớn nhất là dương).
5. **Phương sai lớn ≠ quan trọng cho bài toán.** PCA không nhìn nhãn; hướng phân biệt hai lớp có thể có phương sai nhỏ.
6. **$n$ hay $n - 1$?** Không ảnh hưởng tới trục và tỉ lệ phương sai; chỉ ảnh hưởng giá trị phương sai tuyệt đối.

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"PCA và SVD liên hệ thế nào?"* Mệnh đề 3.4.
- *"Chọn số thành phần $k$ thế nào?"* Lấy $k$ nhỏ nhất có tỉ lệ phương sai tích lũy ≥ 90–95%, rồi **kiểm chứng trên tác vụ thật** (ví dụ recall@10 của tìm kiếm sau khi nén).
- *"Vì sao trị riêng của ma trận hiệp phương sai không âm?"* Mệnh đề 3.2.

#### 3.7 Đọc thêm

- Strang ch. 6–7.
- MML ch. 4 (matrix decompositions) và ch. 10 (PCA).
- J. Shlens, *A Tutorial on Principal Component Analysis*, arXiv:1404.1100.

---

## Bài tập Ngày 1 · `optlab/linalg.py` (~2h)

Đây là mảnh đầu tiên của project tuần ([project.md](project.md)). Viết file `project/optlab/linalg.py` gồm các hàm sau. Test: `python -m pytest -q tests/test_linalg.py` (8 test).

| Hàm | Yêu cầu |
|---|---|
| `add_bias(X)` | Thêm cột toàn số 1 vào **đầu**: `(n, d) → (n, d+1)` |
| `standardize(X) -> (Z, mean, std)` | Chuẩn hóa từng cột về trung bình 0, độ lệch chuẩn 1 (dùng `X.std(axis=0)`, tức chia $n$). Cột hằng (std = 0) thì chia cho 1, kết quả là cột 0, **không được ra `nan`** |
| `lstsq(X, y)` | Nghiệm bình phương tối thiểu qua `np.linalg.lstsq(..., rcond=None)` |
| `normal_equation(X, y, l2=0.0)` | Giải $(X^\top X/n + \ell_2 I)\,\theta = X^\top y/n$ bằng `np.linalg.solve`. Dạng có $\ell_2$ là ridge, sẽ giải thích ở Ngày 3 |
| `condition_number(X)` | $s_{\max}/s_{\min}$ từ `np.linalg.svd(X, compute_uv=False)`. Trả về `inf` nếu $s_{\min} \le s_{\max} \cdot \max(n, d) \cdot \varepsilon_{\text{float64}}$ (cùng ngưỡng với `matrix_rank`) |
| `hessian_eigs_mse(X)` | Trị riêng **tăng dần** của $X^\top X/n$ (dùng `eigvalsh`). Ngày 2 sẽ giải thích vì sao đây là Hessian của loss MSE |
| `PCAResult` | `@dataclass(frozen=True)` với 4 trường: `mean`, `components` shape `(k, d)`, `explained_variance` (chia $n-1$), `explained_variance_ratio` |
| `pca_fit(X, k)` | PCA qua SVD. Quy ước dấu: phần tử có $\lvert\cdot\rvert$ lớn nhất của mỗi trục phải **dương**. `k` ngoài $[1, \min(n, d)]$ thì raise `ValueError` |
| `pca_transform(res, X)` | $(X - \text{mean})\,\cdot$ `components.T` |

**Gợi ý:** test `test_pca_hand_example` dùng đúng 4 điểm ở mục 3.3(b). Nếu test đó sai, hãy in `res` ra và so với số tính tay.
