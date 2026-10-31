# Final · Tuần 3 · Machine Learning nền tảng

> ⏱ **180 phút** · **100 điểm** · trung bình – khó · Đạt khi ≥ 70 · Đáp án: [final-solutions.md](final-solutions.md)
> **Được dùng:** giấy nháp và máy tính cầm tay (phần A, B, D); Python + NumPy + pandas + scikit-learn + pytest (phần C).
> **Không được dùng:** AI, Internet, tài liệu.

| Phần | Nội dung | Điểm | Thời gian gợi ý |
|---|---|---|---|
| A | Quick test | 24 | 30' |
| B | Tính tay | 20 | 40' |
| C | Code (có test sẵn) | 32 | 70' |
| D | Thiết kế | 24 | 40' |

**Bài nào được kiểm tra ở câu nào:**
- 01 Chia dữ liệu, rò rỉ: A1, C1, D1
- 02 Bias–variance: A2, B2
- 03 Cross-validation: A3, C2, D3
- 04 Metric, ngưỡng: A4, A5, B1, D2
- 05 Model tuyến tính: A6
- 06 Cây: A7, B2
- 07 Ensemble: A8, A9, B2
- 08 Phân cụm: A10, B3
- 09 t-SNE/UMAP: A10
- 10 Pipeline, tiền xử lý: A11, C1
- 11 Tuning: A12, C2
- Hiệu chỉnh xác suất (bài 04, 05): C3, D2

---

## A. Quick test (12 câu × 2đ)

**A1.** Dự đoán một câu trả lời của LLM có bị bộ phận kiểm duyệt gắn cờ hay không. Cột nào gây rò rỉ? A. độ dài câu trả lời · B. `moderation_action` (gỡ bài / cảnh cáo / không) · C. intent của câu hỏi · D. tên model

**A2.** Với kNN, $k = 1$ hay $k = 100$ có variance cao hơn? Bias cao hơn?

**A3.** Bộ đánh giá gồm 3 000 câu hỏi sinh từ 300 tài liệu (mỗi tài liệu 10 câu). Nên dùng kiểu CV nào? Vì sao?

**A4.** TP = 30, FP = 70, FN = 10, TN = 890. Tính precision, recall, F1 và accuracy.

**A5.** Dữ liệu có 5% mẫu dương. Một model chấm điểm ngẫu nhiên có ROC-AUC và AP xấp xỉ bao nhiêu?

**A6.** Xác suất nền $p = 0.1$, hệ số logistic $\beta = \ln 2$. Bật đặc trưng này lên thì $p$ bằng bao nhiêu?

**A7.** Tính Gini impurity của một nút có 3 lớp với tỉ lệ $(0.5,\ 0.3,\ 0.2)$.

**A8.** Random forest có $\rho = 0.5$ và $\sigma^2 = 0.2$. Khi $B \to \infty$, variance của rừng tiến về bao nhiêu? Muốn giảm thêm thì làm gì?

**A9.** Gradient boosting với log-loss, mẫu có $y = 0$ và $\sigma(F) = 0.8$. Pseudo-residual bằng bao nhiêu? Vòng tiếp theo sẽ đẩy $F$ của mẫu này lên hay xuống?

**A10.** Một điểm có $a = 3$, $b = 2$. Tính silhouette của nó. Trên hình t-SNE, cụm X trông to gấp đôi cụm Y. Kết luận được gì?

**A11.** Áp `RobustScaler` lên $(5, 7, 9, 11, 200)$, biết median là 9 và $Q_1 = 7$, $Q_3 = 11$. Các giá trị 5 và 200 trở thành bao nhiêu?

**A12.** "Vùng tốt" chỉ chiếm 2% không gian tìm kiếm. Random search cần tối thiểu bao nhiêu lần thử để có ≥ 90% khả năng rơi vào vùng đó ít nhất một lần?

---

## B. Tính tay (20đ)

**B1 — Metric và ngưỡng (7đ).** 12 mẫu:

| Mẫu dương | 0.92 | 0.81 | 0.64 | 0.45 | 0.30 | | |
|---|---|---|---|---|---|---|---|
| **Mẫu âm** | 0.75 | 0.58 | 0.40 | 0.33 | 0.22 | 0.15 | 0.05 |

- (a) Tính ROC-AUC bằng cách đếm cặp.
- (b) Tính Average Precision.
- (c) Với $c_{FP} = 1$ và $c_{FN} = 4$, lập bảng chi phí theo các ngưỡng, chọn ngưỡng tốt nhất, và tính precision, recall, F1 tại ngưỡng đó.
- (d) Ngưỡng lý thuyết $t^\star$ (nếu xác suất đã hiệu chỉnh) bằng bao nhiêu? Vì sao kết quả (c) có thể khác?

**B2 — Cây và ensemble (7đ).**
- (a) Một nút có 100 mẫu, trong đó 40 mẫu dương. Có hai cách chia:
  - **A**: trái 50 mẫu (35 dương), phải 50 mẫu (5 dương);
  - **B**: trái 20 mẫu (20 dương), phải 80 mẫu (20 dương).

  Tính gain của mỗi cách theo **Gini** và theo **entropy**. Mỗi tiêu chí chọn cách nào? Giải thích sự khác biệt.
- (b) Gradient boosting phân loại bắt đầu từ $F_0 = \log\frac{\bar y}{1 - \bar y}$ với $\bar y = 0.4$. Tính $F_0$ và pseudo-residual của một mẫu có $y = 1$ và một mẫu có $y = 0$.
- (c) Mỗi cây có $\sigma^2 = 0.09$, $\rho = 0.3$. Tính variance của rừng với $B = 1$ và $B = 20$, và giá trị sàn.

**B3 — K-Means (6đ).** Dữ liệu một chiều $1, 3, 4, 10, 12, 32$, $k = 2$, tâm ban đầu $(1, 3)$.
- (a) Chạy Lloyd tới khi hội tụ, ghi phép gán, tâm và inertia ở từng vòng.
- (b) So sánh inertia cuối cùng với cách chia "tự nhiên" {1, 3, 4} · {10, 12, 32}. Nhận xét về ảnh hưởng của điểm 32.

---

## C. Code (32đ)

Viết `code/final_lib.py`. Đứng ở thư mục tuần, chạy `python -m pytest -q code/test_final.py` phải qua hết **8 test**. **Không được sửa test.**

**C1 — Target encoding out-of-fold (12đ).** Thay mỗi giá trị phân loại bằng tỉ lệ dương (đã làm mượt) của nó:

$$\text{enc}(c) = \frac{\sum_{i: x_i = c} y_i + m \cdot \text{prior}}{n_c + m}$$

(Đây chính là MAP với prior Beta, Tuần 2, Mệnh đề 10.7.)

- `target_encode_cv(cats, y, k=5, m=10.0, seed=0) -> (enc_train, mapping, prior)`:
  - `prior = mean(y)`; `mapping` là dict tính trên **toàn bộ** dữ liệu (để dùng cho test set).
  - `enc_train` phải tính **out-of-fold**: dùng `KFold(k, shuffle=True, random_state=seed)`; mỗi dòng được mã hóa bằng thống kê của các fold **khác**, với prior của fold đó là trung bình $y$ trên các dòng train của fold; giá trị chưa xuất hiện trong phần train thì nhận prior của fold.
- `apply_encoding(cats, mapping, prior)`: giá trị lạ thì nhận `prior`.

**C2 — Nested CV (12đ).** `nested_cv(estimator, param_grid, X, y, outer_k=3, inner_k=3, seed=0) -> dict`:
- Vòng ngoài là `StratifiedKFold(outer_k, shuffle=True, random_state=seed)`.
- Ở mỗi fold ngoài, chạy `GridSearchCV` với `clone(estimator)`, `scoring="roc_auc"`, `cv=StratifiedKFold(inner_k, shuffle=True, random_state=seed)` trên phần train; rồi chấm ROC-AUC trên phần test của fold ngoài.
- Trả về `{"outer_scores": [...], "inner_best": [...], "best_params": [...]}`.
- **Không được fit `estimator` gốc.**

**C3 — Calibration (8đ).**
- `brier(y, p)`: trung bình của $(p - y)^2$.
- `calibration_table(y, p, n_bins=10) -> DataFrame`:
  - các cột là `bin, count, mean_pred, frac_pos`, chỉ gồm các bin **không rỗng**;
  - bin của mỗi mẫu là $\min(\lfloor p \cdot n_\text{bins}\rfloor,\ n_\text{bins} - 1)$.
- `ece(y, p, n_bins=10)`: $\sum_b \frac{n_b}{n}\,\lvert\text{frac\_pos}_b - \text{mean\_pred}_b\rvert$ (*Expected Calibration Error*).

Viết thêm 2 câu: vì sao test ở C1 cho thấy mã hóa "naive" có AUC trên 0.85 với một cột hoàn toàn vô nghĩa? Và vì sao model dùng `class_weight="balanced"` sẽ có ECE lớn?

---

## D. Thiết kế (24đ) — Dự đoán thuê bao di động rời mạng

**Bối cảnh** (mọi con số là **giả định**):
- Nhà mạng có 10 triệu thuê bao trả trước; mỗi tháng khoảng **2%** rời mạng.
- Mỗi tháng, bộ phận chăm sóc khách hàng gọi được tối đa **50 000** thuê bao, và tặng mỗi người được gọi một ưu đãi.
- Chi phí cho mỗi thuê bao được gọi (cuộc gọi + ưu đãi) là **60 000 đ**.
- Một thuê bao **sắp rời mạng** mà được gọi thì có **25%** khả năng ở lại; giữ được một thuê bao mang lại **2 400 000 đ** (doanh thu 24 tháng).
- Có 24 tháng lịch sử: dữ liệu nạp tiền, gọi, data, khiếu nại, và trạng thái thuê bao theo ngày.

**D1 — Dữ liệu và nhãn (8đ).**
- (a) Định nghĩa chính xác **thời điểm dự đoán**, **cửa sổ đặc trưng** và **cửa sổ nhãn**. Vì sao nên có một khoảng trống giữa hai cửa sổ?
- (b) Nêu 3 đặc trưng có nguy cơ gây **rò rỉ**, và cách kiểm tra.
- (c) Chia train/validation/test thế nào? Một thuê bao xuất hiện ở nhiều tháng thì có vấn đề gì không?

**D2 — Metric và quyết định (8đ).**
- (a) Gọi ngẫu nhiên 50 000 thuê bao thì lãi hoặc lỗ bao nhiêu mỗi tháng?
- (b) Precision trong top 50 000 thuê bao (precision@50k) phải đạt tối thiểu bao nhiêu để hòa vốn? Tương ứng với lift bằng bao nhiêu?
- (c) Nếu model đạt precision@50k = 30% thì lãi bao nhiêu?
- (d) Nếu model cho ra **xác suất đã hiệu chỉnh** $p_i$, thì nên gọi thuê bao $i$ khi nào? (Viết điều kiện theo $p_i$.) Tại sao điều kiện này cần xác suất **đã hiệu chỉnh**, chứ một score chỉ xếp hạng tốt thì không đủ?
- (e) Vì sao ROC-AUC **không** phải metric chính cho bài toán này? Nên báo cáo metric nào?

**D3 — Model, đánh giá và vận hành (8đ).**
- (a) Đề xuất model baseline và model chính, và lý do chọn.
- (b) Thiết kế quy trình đánh giá (CV, khoảng tin cậy, so sánh với baseline).
- (c) Sau khi triển khai, cần theo dõi những gì? Khi nào cần huấn luyện lại?
- (d) Sau khi chiến dịch chạy, những thuê bao **được gọi** sẽ ít rời mạng hơn. Điều này làm hỏng dữ liệu huấn luyện của các tháng sau như thế nào, và xử lý ra sao?
