# Quiz · Ngày 1 · Đại số tuyến tính (bài 01–03)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> Làm trên giấy, được dùng máy tính cầm tay. **Không chạy code.** Đáp án ở cuối file.

## Phần 1 — Quick test (14 câu, câu 14 được 2đ, các câu khác 1đ)

1. Với $x = (1, -2, 2)$, tính $\|x\|_1$, $\|x\|_2$, $\|x\|_\infty$.
2. Tính $\cos\big((1, 0), (1, 1)\big)$.
3. $u, v$ là hai vector đơn vị có $\cos(u, v) = 0.5$. Tính $\|u - v\|$.
4. Đúng/Sai: với các vector đã chuẩn hóa, sắp xếp tài liệu theo tích vô hướng **giảm dần** cho cùng thứ tự với sắp xếp theo khoảng cách L2 **tăng dần**.
5. $A$ có shape `(3, 4)`, $B$ có shape `(4, 2)`. `A @ B` có shape gì, và tốn bao nhiêu FLOP?
6. $(AB)^\top$ bằng: A. $A^\top B^\top$ · B. $B^\top A^\top$ · C. $BA$ · D. $A^\top B$
7. Ma trận $\begin{bmatrix}3 & 1\\6 & 2\end{bmatrix}$ có định thức và hạng bằng bao nhiêu?
8. $X$ có 5 cột và hạng 3. Không gian $\{v : Xv = 0\}$ có bao nhiêu chiều?
9. $\kappa(X) = 10^3$. $\kappa(X^\top X)$ bằng bao nhiêu? Giải bằng phương trình chuẩn trên float64 thì mất khoảng bao nhiêu chữ số chính xác?
10. Tìm trị riêng của $\begin{bmatrix}5 & 2\\2 & 2\end{bmatrix}$.
11. Ma trận hiệp phương sai có trị riêng $6, 3, 1$. Hai thành phần chính đầu tiên giữ bao nhiêu phần trăm phương sai?
12. Đúng/Sai: với ma trận hiệp phương sai, nên dùng `np.linalg.eig` thay vì `np.linalg.eigh`.
13. Lấy ngẫu nhiên hai vector đơn vị trong không gian 400 chiều. Độ lệch chuẩn của cosine giữa chúng xấp xỉ bao nhiêu?
14. *(2đ)* $X$ đã trừ trung bình, có các giá trị kỳ dị $10,\ 5,\ 0.1$. Tính: (a) $\kappa(X)$; (b) sai số Frobenius của xấp xỉ hạng 2 tốt nhất; (c) tỉ lệ phương sai giải thích của PC1.

## Phần 2 — Tự luận (4 câu, 30 phút)

**T1 · Chứng minh (4đ).**
- (a) Từ bất đẳng thức Cauchy–Schwarz, chứng minh bất đẳng thức tam giác $\|x + y\|_2 \le \|x\|_2 + \|y\|_2$.
- (b) Chứng minh: nếu các cột của $X$ độc lập tuyến tính thì $X^\top X$ khả nghịch.

**T2 · Tính tay (4đ).** Khớp đường thẳng $y = \theta_0 + \theta_1 x$ qua ba điểm $(1, 2), (2, 3), (3, 5)$ bằng phương trình chuẩn. Ghi rõ $X^\top X$, $X^\top y$, $\theta^\star$, vector phần dư, và kiểm tra hai tính chất của phần dư.

**T3 · PCA tính tay (3đ).**
- (a) Với 4 điểm $(2, 0), (-2, 0), (0, 1), (0, -1)$: tính ma trận hiệp phương sai (chia $n - 1$), các trục chính và tỉ lệ phương sai giải thích.
- (b) Với 4 điểm $(1, 0), (-1, 0), (0, 1), (0, -1)$ thì PCA cho ra kết quả gì? Điều này nói lên gì về tính duy nhất của các trục chính?

**T4 · Code (4đ).** Viết `power_iteration(A, iters=1000, tol=1e-8, seed=0) -> (lam, v, converged)` cho ma trận đối xứng $A$:
- Khởi tạo $v$ ngẫu nhiên, chuẩn hóa.
- Dùng Rayleigh quotient để ước lượng $\lambda$.
- Điều kiện dừng phải dựa trên **phần dư** $\|Av - \lambda v\|$, không dựa trên việc $\lambda$ thôi thay đổi.

Sau đó trả lời: với $A = \begin{bmatrix}0 & 1\\1 & 0\end{bmatrix}$ chuyện gì xảy ra, và vì sao dừng theo "$\lambda$ thôi thay đổi" là sai ở trường hợp này?

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Quick test**

1. $\|x\|_1 = 5$, $\|x\|_2 = 3$, $\|x\|_\infty = 2$.
2. $1/\sqrt{2} \approx 0.707$.
3. $\|u - v\|^2 = 2 - 2 \cdot 0.5 = 1$, nên $\|u - v\| = 1$.
4. **Đúng** (Mệnh đề 1.3).
5. `(3, 2)`; $2 \cdot 3 \cdot 4 \cdot 2 = 48$ FLOP.
6. **B**.
7. $\det = 6 - 6 = 0$; hạng **1** (hàng 2 = 2 × hàng 1).
8. Theo rank–nullity: $5 - 3 =$ **2**.
9. $\kappa(X^\top X) = 10^6$; mất khoảng **6** chữ số (còn khoảng 10 trên 16). Dùng `lstsq` chỉ mất khoảng 3.
10. $\lambda^2 - 7\lambda + 6 = 0$, nên $\lambda \in \{6, 1\}$. Kiểm tra: tổng $= 7 = \mathrm{tr}$, tích $= 6 = \det$.
11. $(6 + 3)/10 =$ **90%**.
12. **Sai**. `eigh` dành cho ma trận đối xứng: trị riêng thực, sắp tăng dần, vector riêng trực chuẩn.
13. $1/\sqrt{400} =$ **0.05**.
14. (a) $10/0.1 = 100$. (b) $s_3 = 0.1$ (Eckart–Young). (c) $100/(100 + 25 + 0.01) \approx$ **0.80**. Trị riêng tỉ lệ với $s_i^2$, không phải $s_i$.

**T1**

(a) $\|x + y\|^2 = \|x\|^2 + 2x^\top y + \|y\|^2 \le \|x\|^2 + 2\|x\|\|y\| + \|y\|^2 = (\|x\| + \|y\|)^2$. Bước "$\le$" dùng Cauchy–Schwarz $x^\top y \le \lvert x^\top y\rvert \le \|x\|\|y\|$. Lấy căn hai vế (đều không âm). ∎ *(2đ)*

(b) Giả sử $X^\top X v = 0$. Nhân trái với $v^\top$: $v^\top X^\top X v = \|Xv\|^2 = 0$, nên $Xv = 0$. Các cột độc lập tuyến tính nên $v = 0$. Vậy $X^\top X$ (vuông $d \times d$) có hạt nhân $\{0\}$ nên khả nghịch. ∎ *(2đ. Thiếu bước $v^\top X^\top X v = \|Xv\|^2$ thì mất 1đ.)*

**T2**

- $X = \begin{bmatrix}1 & 1\\1 & 2\\1 & 3\end{bmatrix}$, $X^\top X = \begin{bmatrix}3 & 6\\6 & 14\end{bmatrix}$, $X^\top y = \begin{bmatrix}10\\23\end{bmatrix}$.
- $\det = 42 - 36 = 6$. $\theta^\star = \dfrac{1}{6}\begin{bmatrix}14 & -6\\-6 & 3\end{bmatrix}\begin{bmatrix}10\\23\end{bmatrix} = \dfrac{1}{6}\begin{bmatrix}2\\9\end{bmatrix} = \begin{bmatrix}1/3\\3/2\end{bmatrix}$.
- Dự đoán $1.833,\ 3.333,\ 4.833$; phần dư $r = (1/6,\ -1/3,\ 1/6)$; tổng bình phương $= 1/6$.
- Kiểm tra: $\sum r_i = 0$ (có cột bias); $\sum x_i r_i = 1/6 - 2/3 + 1/2 = 0$ (phần dư vuông góc với cột $x$). *(Đã chạy kiểm chứng.)*

*(Chấm: $X^\top X$, $X^\top y$ được 1đ · $\theta^\star$ được 2đ · phần dư và 2 phép kiểm tra được 1đ.)*

**T3**

(a) Trung bình $(0, 0)$. $\sum x_1^2 = 8$, $\sum x_2^2 = 2$, $\sum x_1 x_2 = 0$, nên $C = \mathrm{diag}(8/3,\ 2/3)$. Ma trận đã chéo: trục chính là $(1, 0)$ và $(0, 1)$, tỉ lệ phương sai **0.8** và **0.2**. *(2đ)*

(b) $C = \tfrac{2}{3}I$: hai trị riêng **bằng nhau**, nên **mọi** hướng đều là vector riêng và đều có phương sai $2/3$. Trục chính **không duy nhất**; thư viện sẽ trả về một cặp trục trực giao tùy ý. Dữ liệu "tròn" thì PCA không có gì để nén. *(1đ)*

**T4** *(đã chạy kiểm chứng: với $\begin{bmatrix}2 & 1\\1 & 2\end{bmatrix}$ trả về $(3.0, \pm(0.707, 0.707), \text{True})$; với ma trận $6 \times 6$ ngẫu nhiên khớp `eigvalsh` tới 13 chữ số)*

```python
def power_iteration(A, iters=1000, tol=1e-8, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[0])
    v /= np.linalg.norm(v)
    lam = float(v @ A @ v)
    for _ in range(iters):
        w = A @ v
        nw = np.linalg.norm(w)
        if nw == 0:                                   # v nằm trong hạt nhân: trị riêng 0
            return 0.0, v, True
        v = w / nw
        lam = float(v @ A @ v)                        # Rayleigh quotient (‖v‖ = 1)
        if np.linalg.norm(A @ v - lam * v) <= tol * max(1.0, abs(lam)):
            return lam, v, True
    return lam, v, False
```

Với $A = \begin{bmatrix}0 & 1\\1 & 0\end{bmatrix}$ (trị riêng $+1$ và $-1$, **cùng trị tuyệt đối**): $A$ chỉ hoán đổi hai tọa độ, nên $v$ nhảy qua lại giữa $(a, b)$ và $(b, a)$ mãi mãi, **không hội tụ**. Nhưng Rayleigh quotient $2ab$ thì **không đổi** sau mỗi vòng. Nếu dừng khi "$\lambda$ thôi thay đổi", hàm sẽ dừng ngay ở vòng đầu và trả về $\lambda \approx -0.999$, một con số **không phải trị riêng**. Kiểm tra phần dư thì phát hiện được: hàm chạy hết `iters` và trả về `converged=False`.

*(Chấm: code đúng được 2đ · dừng theo phần dư được 1đ · giải thích trường hợp $\lvert\lambda_1\rvert = \lvert\lambda_2\rvert$ được 1đ.)*

</details>
