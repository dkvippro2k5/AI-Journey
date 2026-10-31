# Quiz · Ngày 2 · Model tuyến tính, cây, ensemble (bài 05–07)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> Làm trên giấy, được dùng máy tính cầm tay. **Không chạy code.** Đáp án ở cuối file.

## Phần 1 — Quick test (14 câu, câu 14 được 2đ, các câu khác 1đ)

1. Xác suất nền $p = 0.2$. Một đặc trưng nhị phân có hệ số logistic $\beta = \ln 3$. Khi bật đặc trưng này lên thì xác suất là bao nhiêu?
2. `LogisticRegression(C=0.01)` và `LogisticRegression(C=100)`: cái nào regularize mạnh hơn?
3. `class_weight="balanced"` với 10% mẫu dương: tính $w_1$ và $w_0$.
4. Đúng/Sai: `class_weight="balanced"` thường làm AUC tăng đáng kể.
5. Một nút có 30% dương, 70% âm. Tính Gini impurity.
6. Tính entropy (bit) của nút có tỉ lệ 50/50, và của nút có 3 lớp phân bố đều.
7. Model nào **cần** chuẩn hóa đặc trưng? A. logistic regression có L2 · B. random forest
8. Một cây hồi quy được huấn luyện với $x \in [0, 10]$. Nó dự đoán gì tại $x = 50$?
9. Random forest có $\rho = 0.4$, $\sigma^2 = 1$, $B = 50$. Tính variance của trung bình.
10. Với $n$ lớn, mỗi cây của random forest để lại khoảng bao nhiêu phần trăm dữ liệu làm OOB?
11. Gradient boosting với log-loss, mẫu có $y = 1$ và hiện tại $\sigma(F) = 0.3$. Pseudo-residual (gradient âm theo $F$) bằng bao nhiêu?
12. Boosting với $\eta = 0.1$ và với $\eta = 1$: cấu hình nào cần nhiều cây hơn? Cấu hình nào thường tổng quát hóa tốt hơn?
13. Một cột ID ngẫu nhiên (không liên quan tới nhãn, có nhiều giá trị khác nhau) lại có impurity importance cao. Vì sao? Nên dùng phương pháp nào thay thế?
14. *(2đ)* 8 mẫu sắp theo $x = 1, \dots, 8$ có nhãn $1, 1, 1, 0, 0, 0, 0, 0$. Tìm ngưỡng chia tốt nhất theo Gini và tính gain.

## Phần 2 — Tự luận (4 câu, 30 phút)

**T1 · Chia nút (4đ).** Đặc trưng `n_turns` và nhãn:

| `n_turns` | 1 | 2 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| bad | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 |

- (a) Liệt kê mọi ngưỡng ứng viên.
- (b) Tính gain theo **Gini** và theo **entropy** cho các ngưỡng 2.5, 3.5 và 7.
- (c) Ngưỡng nào được chọn? Ngưỡng 7 tạo ra một lá thuần chỉ có 1 mẫu. Bình luận về rủi ro của các phép chia kiểu này và hyperparameter nào chặn được chúng.

**T2 · Gradient boosting tính tay (4đ).** $x = (1, 2, 3, 4)$, $y = (2, 1, 5, 8)$, loss bình phương, cây 1 tầng (stump), $\eta = 0.5$. Thực hiện **2 vòng**: ở mỗi vòng ghi phần dư, ngưỡng chia tốt nhất (theo tổng bình phương sai số của phần dư), giá trị hai lá, $F$ mới và MSE train.

**T3 · Random forest (3đ).**
- (a) Mỗi cây có $\sigma^2 = 0.04$, các cây có tương quan $\rho = 0.25$. Tính variance của rừng với $B = 10, 100, 1000$.
- (b) Giảm `max_features` ảnh hưởng thế nào tới $\rho$ và tới $\sigma^2$ của từng cây?
- (c) Vì sao tăng $B$ không gây overfit, trong khi tăng số cây của boosting thì có thể?

**T4 · Code (4đ).** Viết `perm_importance(model, X, y, n_repeats=5, seed=0) -> dict` cho model đã fit có `predict_proba`, với `X` là DataFrame. Hàm trả về mức giảm ROC-AUC trung bình khi xáo trộn **từng cột**, sắp giảm dần. Chỉ ra **một điều kiện** để kết quả đáng tin (dùng dữ liệu nào?), và **một tình huống** mà kết quả gây hiểu lầm.

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Quick test**

1. Odds $0.25 \times 3 = 0.75$, nên $p = 0.75/1.75 \approx$ **0.429**.
2. **`C=0.01`**, vì $C$ là nghịch đảo của độ mạnh phạt.
3. $w_1 = 1/(2 \cdot 0.1) = 5$; $w_0 = 1/(2 \cdot 0.9) \approx 0.556$.
4. **Sai**. AUC gần như không đổi (0.769 so với 0.771 trên dữ liệu project); thứ thay đổi là thang xác suất.
5. $1 - 0.09 - 0.49 =$ **0.42**.
6. **1 bit**; $\log_2 3 \approx$ **1.585 bit**.
7. **A**.
8. Một **hằng số**: giá trị của lá chứa vùng $x$ lớn nhất trong dữ liệu train. Cây không ngoại suy được.
9. $0.4 + 0.6/50 =$ **0.412**.
10. Khoảng **36.8%** ($e^{-1}$).
11. $y - \sigma(F) =$ **0.7**.
12. $\eta = 0.1$ cần nhiều cây hơn, và thường tổng quát hóa tốt hơn.
13. Có nhiều ngưỡng để thử, nên dễ tìm được phép chia "trông có ích" **trên chính dữ liệu train** (thiên lệch về đặc trưng có nhiều giá trị). Thay bằng **permutation importance trên validation/test**.
14. Chia tại **3.5**: hai nút con đều thuần, nên gain $= G_{\text{gốc}} = 1 - (3/8)^2 - (5/8)^2 =$ **0.469**.

**T1** *(đã chạy kiểm chứng)*

(a) Các giá trị phân biệt là 1, 2, 3, 4, 5, 6, 8, nên có 6 ngưỡng: 1.5, 2.5, 3.5, 4.5, 5.5, 7. *(0.5đ)*

(b) Nút gốc: 4/8 dương, nên $G = 0.5$ và $H = 1$. *(2.5đ)*

| $t$ | Trái | Phải | Gini có trọng số | Gain Gini | Entropy có trọng số | Gain entropy |
|---|---|---|---|---|---|---|
| 2.5 | 3 mẫu, 1 dương ($G = 0.444$, $H = 0.918$) | 5 mẫu, 3 dương ($G = 0.48$, $H = 0.971$) | 0.467 | 0.033 | 0.951 | 0.049 |
| **3.5** | 4 mẫu, 1 dương ($G = 0.375$, $H = 0.811$) | 4 mẫu, 3 dương ($G = 0.375$, $H = 0.811$) | 0.375 | **0.125** | 0.811 | **0.189** |
| 7 | 7 mẫu, 3 dương ($G = 0.490$, $H = 0.985$) | 1 mẫu, 1 dương (thuần) | 0.429 | 0.071 | 0.862 | 0.138 |

(c) Cả hai tiêu chí đều chọn **3.5**. Các phép chia như ở ngưỡng 7 (hoặc 1.5) tách riêng **một mẫu** để có lá thuần: gain dương nhưng dựa trên 1 điểm dữ liệu, rất có thể chỉ là nhiễu. Đây là nguồn gốc variance của cây sâu. Chặn bằng `min_samples_leaf` (ví dụ ≥ 20–50) hoặc `max_depth`. *(1đ)*

**T2** *(đã chạy kiểm chứng bằng `L2Boost` và quét mọi ngưỡng)*

$F_0 = \bar y = 4$, MSE ban đầu $= (4 + 9 + 1 + 16)/4 = 7.5$.

**Vòng 1:**
- Phần dư $r = (-2, -3, 1, 4)$.
- SSE theo từng ngưỡng: 1.5 → 24.67; **2.5 → 5.0**; 3.5 → 8.67. Chọn **2.5**.
- Lá: trái $= (-2 - 3)/2 = -2.5$; phải $= (1 + 4)/2 = 2.5$.
- $F_1 = 4 \pm 0.5 \cdot 2.5 = (2.75,\ 2.75,\ 5.25,\ 5.25)$.
- MSE $= (0.5625 + 3.0625 + 0.0625 + 7.5625)/4 =$ **2.8125**.

**Vòng 2:**
- Phần dư $r = (-0.75, -1.75, -0.25, 2.75)$.
- SSE: 1.5 → 10.5; 2.5 → 5.0; **3.5 → 1.167**. Chọn **3.5**.
- Lá: trái $= (-0.75 - 1.75 - 0.25)/3 = -0.917$; phải $= 2.75$.
- $F_2 = (2.292,\ 2.292,\ 4.792,\ 6.625)$.
- MSE $=$ **0.922**.

Nhận xét: hai vòng chọn **hai ngưỡng khác nhau**. Vòng sau sửa đúng chỗ mà vòng trước làm chưa tốt (điểm $x = 4$).

*(Chấm: mỗi vòng 2đ: phần dư 0.5 · ngưỡng 0.5 · lá 0.5 · $F$ và MSE 0.5.)*

**T3**

(a) $0.25 \cdot 0.04 + 0.75 \cdot 0.04/B$, cho ra **0.0130**, **0.0103**, **0.01003**. Sàn là $\rho\sigma^2 = 0.01$. *(1đ)*

(b) `max_features` nhỏ thì các cây khác nhau hơn, nên **$\rho$ giảm**. Nhưng mỗi cây bị hạn chế lựa chọn nên kém hơn một chút, tức **$\sigma^2$ (và bias) của từng cây tăng nhẹ**. Giá trị tối ưu nằm ở giữa; mặc định $\sqrt d$ là một điểm khởi đầu tốt. *(1đ)*

(c) Thêm cây vào rừng chỉ làm **trung bình** thêm nhiều mẫu, nên variance đơn điệu giảm về $\rho\sigma^2$, không thêm "độ phức tạp" nào. Boosting thì mỗi cây **khớp thêm** phần dư, và cuối cùng phần dư chỉ còn nhiễu. Thêm cây thì model học thuộc nhiễu, giống như chạy gradient descent quá lâu. *(1đ)*

**T4** *(đã chạy kiểm chứng: thứ hạng trên dữ liệu project gần giống `sklearn.inspection.permutation_importance`; hai đặc trưng không có tác dụng là `query_tokens` và `n_docs` đều ≈ 0)*

```python
def perm_importance(model, X, y, n_repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    out = {}
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())     # phá quan hệ giữa cột này và nhãn
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        out[col] = float(np.mean(drops))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
```

- *Điều kiện:* tính trên dữ liệu **không dùng để train** (validation/test). Trên tập train, model đã học thuộc nên kết quả bị thổi phồng.
- *Gây hiểu lầm:* hai đặc trưng **tương quan mạnh** với nhau (ví dụ `latency_s` và một cột `latency_ms` trùng lặp). Xáo một cột thì model vẫn dựa được vào cột kia, nên cả hai trông như không quan trọng.

*(Chấm: code đúng được 2đ · điều kiện được 1đ · tình huống gây hiểu lầm được 1đ.)*

</details>
