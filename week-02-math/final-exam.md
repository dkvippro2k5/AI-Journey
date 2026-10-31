# Final · Tuần 2 · Toán cho AI

> ⏱ **180 phút** · **100 điểm** · trung bình – khó · Đạt khi ≥ 70 · Đáp án: [final-solutions.md](final-solutions.md)
> **Được dùng:** giấy nháp và máy tính cầm tay (phần A, B); Python + NumPy + pytest (phần C).
> **Không được dùng:** AI, Internet, tài liệu.
> **Phần A, B và D làm trên giấy, không chạy code.**

| Phần | Nội dung | Điểm | Thời gian gợi ý |
|---|---|---|---|
| A | Quick test: tính nhanh, trắc nghiệm | 24 | 30' |
| B | Tính tay và chứng minh ngắn | 20 | 40' |
| C | Code (có test sẵn) | 32 | 70' |
| D | Thiết kế và phân tích | 24 | 40' |

**Bài nào được kiểm tra ở câu nào:**
- 01 Vector, cosine: A1, D1
- 02 Ma trận, bình phương tối thiểu: A2, B1
- 03 Trị riêng, SVD, PCA: A3, B1, C1, D1
- 04 Gradient, Hessian: A4, B2
- 05 Backprop, softmax + CE: A5, B2, C2
- 06 Gradient descent, $\kappa$: A6, B2, D3
- 07 Momentum, Adam, lịch lr: A7, A8, D3
- 08 Bayes: A9
- 09 Phân phối, CLT, khoảng tin cậy: A10, B3, C3, D2
- 10 MLE, MAP: A11, C2
- 11 Entropy, KL, perplexity: A12, B3, D3

---

## A. Quick test (12 câu × 2đ)

**A1.** Tính $\cos(x, y)$ với $x = (2, -1, 2)$ và $y = (1, 2, 2)$.

**A2.** $A = \begin{bmatrix}1 & 2\\2 & 4\end{bmatrix}$, $b = (1, 3)$. Hệ $Ax = b$ có nghiệm không? Bài toán bình phương tối thiểu $\min\|Ax - b\|$ có nghiệm **duy nhất** không? Giải thích bằng hạng.

**A3.** Ma trận hiệp phương sai có trị riêng $4, 3, 2, 1$. Cần tối thiểu bao nhiêu thành phần chính để giữ ≥ 70% phương sai?

**A4.** Tại một điểm có $\sigma(z) = 0.9$. Tính $\sigma'(z)$.

**A5.** $f = (xy)^2$ với $x = 1$, $y = 3$. Tính $\partial f/\partial x$ bằng chain rule.

**A6.** GD trên $L(\theta) = \tfrac{1}{2}\theta^\top H\theta$, $H$ có trị riêng $2$ và $8$, với $\alpha = 0.3$. GD hội tụ hay phân kỳ? Nếu phân kỳ, thành phần nào phân kỳ và mỗi bước bị nhân với hệ số bao nhiêu?

**A7.** Adam với $\beta_2 = 0.999$. Ở bước $t = 10$, nếu **không** có bias correction thì $v_{10}$ chỉ bằng khoảng bao nhiêu phần của $\mathbb{E}[g^2]$ (giả sử $\mathbb{E}[g^2]$ không đổi)?

**A8.** Momentum với $\beta = 0.9$, $\alpha = 0.01$, gradient luôn bằng 2. Về lâu dài, mỗi bước dịch $\theta$ bao nhiêu?

**A9.** $P(H) = 0.05$, $P(E \mid H) = 0.8$, $P(E \mid \neg H) = 0.1$. Tính $P(H \mid E)$.

**A10.** Accuracy $\hat p = 0.7$ trên $n = 900$ câu. Tính nửa độ rộng của khoảng tin cậy Wald 95%.

**A11.** 0 lần thành công trong 8 lần thử. Tính MLE và MAP (prior $\mathrm{Beta}(2, 2)$) của xác suất thành công.

**A12.** $p = (1, 0)$, $q = (0.5, 0.5)$. Tính $D_{\mathrm{KL}}(p\,\|\,q)$ và $D_{\mathrm{KL}}(q\,\|\,p)$ theo bit.

---

## B. Tính tay và chứng minh ngắn (20đ)

**B1 — PCA và hồi quy trên cùng dữ liệu (7đ).** Bốn điểm đã có trung bình bằng 0: $(-2, -1), (-1, -1), (1, 1), (2, 1)$.
- (a) Tính ma trận hiệp phương sai mẫu (chia $n - 1$), hai trị riêng của nó, và tỉ lệ phương sai giải thích của PC1.
- (b) Tìm vector riêng của PC1 và **hệ số góc** của đường thẳng đi qua gốc theo hướng PC1.
- (c) Tính hệ số góc của hồi quy bình phương tối thiểu $y = \theta x$ (không bias, vì dữ liệu đã trừ trung bình).
- (d) Hai hệ số góc ở (b) và (c) khác nhau. Giải thích vì sao, dựa trên việc mỗi phương pháp cực tiểu hóa **khoảng cách nào**.

**B2 — Softmax và tốc độ hội tụ (7đ).**
- (a) Logit $z = (2, 1, 0)$, nhãn đúng là lớp **thứ 3** (chỉ số 2). Tính $p = \mathrm{softmax}(z)$, loss cross-entropy và $\nabla_z\ell$.
- (b) Một loss bậc hai có Hessian với trị riêng $0.5$ và $4.5$. Tính learning rate lớn nhất để GD hội tụ, $\alpha^\star$, $\rho^\star$, và số bước để sai số giảm $10^6$ lần với $\alpha^\star$.
- (c) Với heavy-ball momentum tối ưu, hệ số co là $(\sqrt\kappa - 1)/(\sqrt\kappa + 1)$. Cần bao nhiêu bước cho cùng yêu cầu ở (b)?

**B3 — Đánh giá và thông tin (6đ).**
- (a) Một LLM trả lời đúng 850/1000 câu. Tính khoảng tin cậy Wald 95%.
- (b) *Distillation:* model thầy dự đoán $p = (0.6, 0.3, 0.1)$, model trò dự đoán $q = (0.5, 0.25, 0.25)$. Tính $H(p, q)$ và $D_{\mathrm{KL}}(p\,\|\,q)$ theo nat, biết $H(p) = 0.898$ nat.
- (c) Một câu 4 token được model gán xác suất $0.5,\ 0.25,\ 0.125,\ 0.5$. Tính perplexity. Chứng minh rằng perplexity bằng nghịch đảo của **trung bình nhân** các xác suất.

---

## C. Code (32đ)

Viết file `code/final_lib.py` (cùng thư mục với file test). Đứng ở thư mục của tuần, chạy `python -m pytest -q code/test_final.py` phải qua hết **9 test**. **Không được sửa test.** Chỉ dùng NumPy (không dùng `np.linalg.eig*`, `svd` hay scikit-learn bên trong `pca_power`).

**C1 — `pca_power` (12đ).** Viết `pca_power(X, k, iters=1000, tol=1e-10, seed=0) -> (components, eigenvalues)`:
- Tính ma trận hiệp phương sai mẫu $C$ (chia $n - 1$) từ $X$ đã trừ trung bình.
- Tìm lần lượt $k$ trục chính bằng **power iteration**:
  - khởi tạo bằng `np.random.default_rng(seed)`;
  - ước lượng trị riêng bằng Rayleigh quotient;
  - dừng khi phần dư $\|Cv - \lambda v\| \le \text{tol}\cdot\max(1, \lvert\lambda\rvert)$.
- Sau mỗi trục, **deflation**: $C \leftarrow C - \lambda vv^\top$.
- Quy ước dấu: phần tử có $\lvert\cdot\rvert$ lớn nhất của mỗi trục là dương.
- Trả về `components` shape `(k, d)` và `eigenvalues` shape `(k,)`.
- `k` ngoài $[1, d]$ thì raise `ValueError`. **Không được sửa `X`.**

**C2 — Softmax regression (12đ).**
- `softmax_ce_loss(W, X, y, l2=0.0) -> (loss, grad)`: `W` shape `(d, K)`, `y` là chỉ số lớp. Loss là cross-entropy trung bình cộng $\tfrac{\ell_2}{2}\|W\|_F^2$; `grad` cùng shape với `W`. **Không được overflow** với logit cỡ 1000.
- `fit_softmax(X, y, K, *, lr=0.5, epochs=300, l2=1e-3) -> W`: full-batch GD bắt đầu từ `W = 0`.
- `predict_softmax(W, X) -> nhãn`.

**C3 — Bootstrap (8đ).**
- `bootstrap_ci(x, stat=np.mean, n_boot=2000, level=0.95, seed=0) -> (lo, hi)`:
  - percentile bootstrap: lấy mẫu lại **có hoàn lại** $n$ phần tử, `n_boot` lần, rồi lấy phân vị $(1 - \text{level})/2$ và $(1 + \text{level})/2$ của thống kê;
  - chạy lại với cùng `seed` phải cho cùng kết quả;
  - $n < 2$ thì raise `ValueError`.
- `paired_bootstrap_diff(a, b, n_boot=2000, level=0.95, seed=0) -> (diff, lo, hi)`:
  - `diff = mean(b - a)`;
  - mỗi lần lấy mẫu lại phải dùng **cùng chỉ số** cho `a` và `b`;
  - `a`, `b` khác shape thì raise `ValueError`.

Viết thêm 2 câu: (1) vì sao lấy mẫu lại theo cặp thì khoảng tin cậy hẹp hơn so với lấy mẫu lại `a` và `b` độc lập? (2) Khi nào nên dùng bootstrap thay cho công thức Wald?

---

## D. Thiết kế và phân tích (24đ)

**Bối cảnh.** Nhóm bạn xây hệ thống tìm kiếm ngữ nghĩa cho 5 triệu đoạn văn bản tiếng Việt. Model embedding cho vector 1024 chiều, float32, đã chuẩn hóa về độ dài 1. Tìm kiếm theo cosine.

**D1 — Nén embedding bằng PCA (8đ).**
- (a) Ma trận embedding chiếm bao nhiêu GB? Nếu nén còn $k = 256$ chiều thì còn bao nhiêu?
- (b) Mô tả **đầy đủ các bước** để fit PCA và áp dụng nó cho cả tài liệu lẫn câu truy vấn. Cần lưu lại những gì? Fit trên toàn bộ 5 triệu vector hay trên một mẫu? Ước lượng chi phí tính toán nếu fit trên 100 000 vector.
- (c) Sau khi chiếu, vì sao **phải chuẩn hóa lại** về độ dài 1 trước khi tính cosine? Điều gì xảy ra nếu lúc truy vấn quên trừ trung bình?
- (d) Chọn $k$ thế nào? Tỉ lệ phương sai giải thích có đủ để quyết định không?

**D2 — So sánh model embedding cũ và mới (8đ).** Có một bộ đánh giá gồm các cặp (câu hỏi, đoạn văn đúng). Metric là recall@10: tỉ lệ câu hỏi mà đoạn văn đúng nằm trong top 10.
- (a) Model cũ đạt khoảng 0.7. Cần bao nhiêu câu hỏi để khoảng tin cậy 95% của **một** model có nửa độ rộng 2 điểm %?
- (b) Thiết kế thí nghiệm so sánh hai model sao cho ít câu hỏi nhất mà vẫn kết luận được. Dùng kiểm định gì? Báo cáo những con số nào?
- (c) Nhiều câu hỏi được sinh tự động từ **cùng một** đoạn văn. Điều này ảnh hưởng thế nào tới khoảng tin cậy, và xử lý ra sao?
- (d) Nhóm thử 20 cấu hình (kích thước chunk, $k$ của PCA…) rồi chọn cấu hình tốt nhất trên bộ đánh giá. Con số tốt nhất đó có đáng tin không? Phải làm gì?

**D3 — Chẩn đoán huấn luyện bị hỏng (8đ).** Một đồng nghiệp huấn luyện một classifier head (softmax regression) trên đặc trưng **chưa chuẩn hóa** (có cột lên tới $10^4$). Họ dùng Adam với `lr = 1e-2`, không warmup, không clipping. Softmax được viết tay rồi mới lấy `log`. Loss thành `nan` ở bước 300; trước đó loss dao động mạnh.
- (a) Liệt kê ít nhất **4 giả thuyết**. Với mỗi giả thuyết, chỉ ra nó vi phạm điều gì đã học trong tuần (ghi rõ bài/mệnh đề).
- (b) Với mỗi giả thuyết, đề xuất **một thí nghiệm nhỏ** để xác nhận hoặc bác bỏ.
- (c) Đề xuất một cấu hình huấn luyện đã sửa, và giải thích từng thay đổi.
- (d) Nên theo dõi những đại lượng nào trong lúc huấn luyện để phát hiện sớm vấn đề này?
