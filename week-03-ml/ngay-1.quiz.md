# Quiz · Ngày 1 · Nền tảng ML (bài 01–04)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> Làm trên giấy, được dùng máy tính cầm tay. **Không chạy code.** Đáp án ở cuối file.

## Phần 1 — Quick test (14 câu, câu 14 được 2đ, các câu khác 1đ)

1. Dự đoán khách hàng có hủy gói cước trong tháng tới không. Cột nào gây **rò rỉ**? A. số lần đăng nhập tháng trước · B. ngày hủy hợp đồng · C. loại gói cước · D. tuổi tài khoản
2. MSE train là 0.02, MSE validation là 0.45. Model đang bị bias cao hay variance cao?
3. MSE train là 0.40, MSE validation là 0.42; sai số không thể tránh khoảng 0.05. Chẩn đoán là gì? Thêm dữ liệu có giúp được không?
4. `GridSearchCV` với lưới 4 × 5 giá trị và 5-fold CV phải fit bao nhiêu lần (chưa tính lần fit lại cuối cùng)?
5. Dữ liệu theo thời gian thì dùng kiểu chia fold nào?
6. Dữ liệu có 2% mẫu dương. Một model luôn đoán âm có accuracy bao nhiêu? Average Precision của một model đoán ngẫu nhiên xấp xỉ bao nhiêu?
7. TP = 40, FP = 10, FN = 20, TN = 930. Tính precision và recall.
8. Tính F1 từ câu 7.
9. Một model có AUC = 0.3. Có cách nào biến nó thành model có AUC 0.7 không?
10. Xác suất của model đã được hiệu chỉnh tốt, $c_{FP} = 2$, $c_{FN} = 8$. Ngưỡng tối ưu là bao nhiêu?
11. Đúng/Sai: fit `StandardScaler` trên toàn bộ dữ liệu rồi mới chạy CV là một dạng rò rỉ.
12. Tăng $\alpha$ của ridge thì bias và variance thay đổi theo hướng nào?
13. Nhân mọi score của model với 10 rồi cộng 3. AUC có đổi không? Precision ở ngưỡng 0.5 có đổi không?
14. *(2đ)* Mẫu dương có score $0.8, 0.4, 0.3$; mẫu âm có score $0.6, 0.2, 0.1$. Tính ROC-AUC và Average Precision.

## Phần 2 — Tự luận (4 câu, 30 phút)

**T1 · Kiểm tra rò rỉ (4đ).** Bài toán: tại thời điểm một ticket hỗ trợ **vừa được tạo**, dự đoán ticket đó có được giải quyết trong 24 giờ hay không. Các cột có sẵn:

`created_hour`, `customer_tier`, `text_length`, `category` (gán tự động lúc tạo), `n_agent_replies`, `resolved_by_agent_id`, `satisfaction_score`, `first_response_minutes`, `customer_id`.

- (a) Cột nào **không được** dùng? Giải thích từng cột.
- (b) Nên chia train/test thế nào? Cần lưu ý gì về cột `customer_id`?

**T2 · Tính tay metric (4đ).** Mẫu dương có score $0.95, 0.7, 0.55, 0.2$; mẫu âm có score $0.8, 0.6, 0.3, 0.1$.
- (a) Tại ngưỡng 0.5, tính ma trận nhầm lẫn, precision và recall.
- (b) Tính ROC-AUC bằng cách đếm cặp.
- (c) Với $c_{FP} = 1$, $c_{FN} = 3$: lập bảng chi phí theo mọi ngưỡng ứng viên và chọn ngưỡng tốt nhất.

**T3 · Đọc learning curve (3đ).** Hai model được đo AUC theo số mẫu train:

| $n$ | 500 | 1 000 | 2 000 | 4 000 |
|---|---|---|---|---|
| Model A, train | 0.99 | 0.97 | 0.95 | 0.93 |
| Model A, val | 0.70 | 0.74 | 0.77 | 0.79 |
| Model B, train | 0.76 | 0.75 | 0.75 | 0.75 |
| Model B, val | 0.73 | 0.74 | 0.74 | 0.74 |

Chẩn đoán từng model. Với mỗi model, nêu 2 việc nên làm và 1 việc **không** nên làm.

**T4 · Code (4đ).** Viết `cost_curve(y, s, cost_fp, cost_fn) -> (thresholds, costs)` trả về chi phí của **mọi** ngưỡng ứng viên ($+\infty$ và mọi giá trị score phân biệt, sắp từ lớn tới nhỏ), **không dùng vòng lặp Python theo ngưỡng**: chỉ sắp xếp một lần rồi dùng `cumsum`. Độ phức tạp của cách làm này là bao nhiêu, so với cách lặp qua từng ngưỡng?

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Quick test**

1. **B**. Ngày hủy chỉ có **sau khi** khách đã hủy, tức là hệ quả của nhãn.
2. **Variance cao** (overfit).
3. **Bias cao** (underfit): train và val sát nhau nhưng đều xa mức 0.05. Thêm dữ liệu **không** giúp được; cần model phức tạp hơn hoặc thêm đặc trưng.
4. $20 \times 5 =$ **100** (cộng 1 lần fit lại trên toàn bộ dữ liệu nếu `refit=True`).
5. `TimeSeriesSplit`: luôn train trên quá khứ, validate trên tương lai.
6. Accuracy **98%**; AP ngẫu nhiên khoảng **0.02** (bằng tỉ lệ dương).
7. Precision $40/50 = 0.8$; recall $40/60 = 0.667$.
8. $2 \cdot 0.8 \cdot 0.667/1.467 =$ **0.727**.
9. **Có**: đảo dấu score ($s \to -s$). Mọi cặp xếp sai thành xếp đúng, nên AUC $= 1 - 0.3 = 0.7$. (Thực tế thì nên đi tìm lỗi trước: thường là nhãn bị đảo ở đâu đó.)
10. $2/(2 + 8) =$ **0.2**.
11. **Đúng.** Mean và std của fold validation đã ảnh hưởng tới phép biến đổi. Mức rò rỉ nhỏ với scaler, nhưng rất nặng với các bước như chọn đặc trưng (thí nghiệm 0.795 so với 0.545).
12. Bias **tăng**, variance **giảm**.
13. AUC **không đổi**, vì phép biến đổi đơn điệu tăng giữ nguyên thứ tự. Precision ở ngưỡng 0.5 **thay đổi**, vì tập mẫu vượt ngưỡng đã khác.
14. AUC: 0.8 thắng 3, 0.4 thắng 2, 0.3 thắng 2, tổng $7/9 =$ **0.778**. AP: thứ tự là 0.8 (+), 0.6 (−), 0.4 (+), 0.3 (+); precision tại các mẫu dương là $1,\ 2/3,\ 3/4$, nên AP $= (1 + 0.667 + 0.75)/3 =$ **0.806**. *(1đ mỗi ý)*

**T1**

(a) Không được dùng: *(3đ, 0.75đ mỗi cột)*
- `n_agent_replies`: tăng dần trong suốt vòng đời ticket; lúc vừa tạo luôn bằng 0. Ticket càng sống lâu càng nhiều phản hồi, nên cột này gắn chặt với nhãn.
- `resolved_by_agent_id`: chỉ có **sau khi** ticket đã được giải quyết.
- `satisfaction_score`: khách chấm điểm **sau khi** xong.
- `first_response_minutes`: chưa có tại thời điểm tạo ticket.

Còn dùng được: `created_hour`, `customer_tier`, `text_length`, `category`. `customer_id` là ID, không dùng làm đặc trưng trực tiếp.

(b) Chia **theo thời gian**: train trên các ticket cũ, test trên các ticket mới hơn, đúng với cách model được dùng trong thực tế. Đồng thời, các ticket của **cùng một khách hàng** không nên nằm ở cả hai phía, vì model có thể "nhận ra" khách hàng. Dùng `GroupKFold` hoặc `StratifiedGroupKFold` theo `customer_id`. *(1đ)*

**T2** *(đã chạy kiểm chứng)*

(a) Đoán dương các mẫu $\ge 0.5$: 0.95, 0.8, 0.7, 0.6, 0.55. Ta có TP = 3, FP = 2, FN = 1 (0.2), TN = 2. Precision $= 0.6$, recall $= 0.75$. *(1đ)*

(b) 0.95 thắng 4; 0.7 thắng 3; 0.55 thắng 2; 0.2 thắng 1. AUC $= 10/16 =$ **0.625**. *(1đ)*

(c) Bảng chi phí: *(2đ)*

| $t$ | $+\infty$ | 0.95 | 0.8 | 0.7 | 0.6 | 0.55 | 0.3 | **0.2** | 0.1 |
|---|---|---|---|---|---|---|---|---|---|
| FP | 0 | 0 | 1 | 1 | 2 | 2 | 3 | 3 | 4 |
| FN | 4 | 3 | 3 | 2 | 2 | 1 | 1 | 0 | 0 |
| Chi phí | 12 | 9 | 10 | 7 | 8 | 5 | 6 | **3** | 4 |

Chọn $t = 0.2$, chi phí 3.

**T3** *(1.5đ mỗi model)*

- **Model A: variance cao.** Khoảng cách train–val lớn nhưng đang **thu hẹp** khi $n$ tăng, và val còn đang đi lên.
  - Nên làm: thêm dữ liệu; tăng regularization (giảm độ sâu, tăng `min_samples_leaf`, tăng $\alpha$).
  - Không nên: làm model phức tạp hơn.
- **Model B: bias cao.** Train và val sát nhau và **đã đi ngang**.
  - Nên làm: dùng model mạnh hơn (ví dụ cây hoặc boosting thay cho model tuyến tính); thêm đặc trưng hoặc tương tác.
  - Không nên: thu thêm dữ liệu, vì đường cong đã phẳng.

**T4** *(đã chạy kiểm chứng: khớp với cách lặp trên 20 bộ dữ liệu ngẫu nhiên có giá trị hòa)*

```python
def cost_curve(y, s, cost_fp=1.0, cost_fn=1.0):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]   # cuối mỗi nhóm score bằng nhau
    tp = np.cumsum(y == 1)[last]                              # số TP khi ngưỡng = score đó
    fp = np.cumsum(y == 0)[last]
    fn = tp[-1] - tp                                          # tp[-1] = tổng số dương
    thresholds = np.r_[np.inf, s[last]]
    costs = np.r_[cost_fn * tp[-1], cost_fp * fp + cost_fn * fn]
    return thresholds, costs
```

Độ phức tạp là $O(n\log n)$, do sắp xếp. Cách lặp qua từng ngưỡng là $O(n \cdot m)$ với $m$ là số giá trị score phân biệt, tức tới $O(n^2)$. Với $n = 10^6$ thì chênh lệch là giây so với hàng giờ.

*(Chấm: code đúng được 2đ · xử lý các giá trị hòa được 1đ · nêu độ phức tạp được 1đ.)*

</details>
