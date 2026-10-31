# Đáp án · Final Tuần 3

> Mọi con số của phần A, B, D đã được chạy kiểm chứng. Code phần C đã chạy: 8/8 test pass.

## A. Quick test (24đ, 2đ/câu)

| Câu | Đáp án | Vì sao |
|---|---|---|
| A1 | **B** | Hành động kiểm duyệt là **hệ quả** của việc bị gắn cờ |
| A2 | $k = 1$: variance cao; $k = 100$: bias cao | $k = 1$ thì dự đoán thay đổi theo từng điểm dữ liệu; $k = 100$ thì trung bình trên vùng rộng, ranh giới quá trơn *(1đ mỗi ý)* |
| A3 | `GroupKFold` (hoặc `StratifiedGroupKFold`) theo tài liệu | Các câu hỏi cùng tài liệu tương quan với nhau; nếu nằm ở cả train và val thì điểm bị thổi phồng |
| A4 | P = 0.30 · R = 0.75 · F1 = 0.429 · Acc = 0.92 | Accuracy 0.92 trông đẹp, nhưng 70% số lần báo là báo nhầm *(0.5đ mỗi giá trị)* |
| A5 | AUC ≈ 0.5 · AP ≈ 0.05 | Mức ngẫu nhiên của AP bằng tỉ lệ dương |
| A6 | $\approx 0.182$ | Odds $0.111 \times 2 = 0.222$, nên $p = 0.222/1.222$ |
| A7 | $0.62$ | $1 - 0.25 - 0.09 - 0.04$ |
| A8 | $\rho\sigma^2 = 0.1$ | Muốn giảm thêm thì phải giảm $\rho$ (giảm `max_features`, tăng tính ngẫu nhiên) hoặc giảm $\sigma^2$ của từng cây. Tăng $B$ không giúp được *(1đ mỗi ý)* |
| A9 | $-0.8$; đẩy $F$ **xuống** | Cây tiếp theo khớp giá trị âm nên $\sigma(F)$ giảm về phía nhãn 0 |
| A10 | $s = (2 - 3)/3 = -0.33$; **không kết luận được gì** | Trên hình t-SNE, kích thước cụm không phản ánh độ phân tán thật *(1đ mỗi ý)* |
| A11 | 5 thành $-1$; 200 thành $47.75$ | $(5 - 9)/4$ và $(200 - 9)/4$ |
| A12 | **114** | $n \ge \ln 0.1/\ln 0.98 = 113.97$ |

## B. Tính tay (20đ)

**B1 (7đ)**

(a) Đếm cặp: 0.92 thắng 7; 0.81 thắng 7; 0.64 thắng 6 (thua 0.75); 0.45 thắng 5; 0.30 thắng 3 (0.22, 0.15, 0.05). Tổng 28/35, nên AUC = **0.80**. *(1.5đ)*

(b) Thứ tự giảm dần: 0.92 (+), 0.81 (+), 0.75 (−), 0.64 (+), 0.58 (−), 0.45 (+), 0.40 (−), 0.33 (−), 0.30 (+), … Precision tại các mẫu dương: $1,\ 1,\ 3/4,\ 4/6,\ 5/9$. AP $= (1 + 1 + 0.75 + 0.667 + 0.556)/5 =$ **0.794**. *(1.5đ)*

(c) Bảng chi phí (chi phí $= FP + 4 \cdot FN$): *(3đ)*

| $t$ | $\infty$ | 0.92 | 0.81 | 0.75 | 0.64 | 0.58 | 0.45 | 0.40 | 0.33 | **0.30** | 0.22 | 0.15 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP | 0 | 0 | 0 | 1 | 1 | 2 | 2 | 3 | 4 | 4 | 5 | 6 | 7 |
| FN | 5 | 4 | 3 | 3 | 2 | 2 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| Chi phí | 20 | 16 | 12 | 13 | 9 | 10 | 6 | 7 | 8 | **4** | 5 | 6 | 7 |

Chọn $t = 0.30$. Tại đó: precision $= 5/9 = 0.556$, recall $= 1$, F1 $= 0.714$.

(d) $t^\star = 1/(1 + 4) = 0.2$. Kết quả thực nghiệm khác vì: score chưa chắc đã được hiệu chỉnh; mẫu rất nhỏ (12 điểm); và ngưỡng chỉ có thể nằm ở các giá trị score rời rạc. Ở đây $t \in (0.22, 0.30]$ đều cho cùng chi phí 4. *(1đ)*

**B2 (7đ)**

(a) Gốc: $G = 2 \cdot 0.4 \cdot 0.6 = 0.48$; $H(0.4) = 0.971$. *(4đ)*

| Cách chia | Gini có trọng số | Gain Gini | Entropy có trọng số | Gain entropy |
|---|---|---|---|---|
| A: $p_L = 0.7$, $p_R = 0.1$ | $0.5 \cdot 0.42 + 0.5 \cdot 0.18 = 0.30$ | **0.18** | $0.5 \cdot 0.881 + 0.5 \cdot 0.469 = 0.675$ | 0.296 |
| B: $p_L = 1$, $p_R = 0.25$ | $0.2 \cdot 0 + 0.8 \cdot 0.375 = 0.30$ | **0.18** | $0.8 \cdot 0.811 = 0.649$ | **0.322** |

Gini **hòa**, còn entropy chọn **B**. Entropy thưởng nhiều hơn cho nút **thuần** (nút trái của B), đúng như bài 06. Với Gini, hòa thì tùy cách cài đặt (ví dụ thứ tự xét đặc trưng).

(b) $F_0 = \ln(0.4/0.6) = -0.405$, nên $\sigma(F_0) = 0.4$. Pseudo-residual: $y = 1$ cho $0.6$; $y = 0$ cho $-0.4$. *(1.5đ)*

(c) $B = 1$: $0.09$. $B = 20$: $0.3 \cdot 0.09 + 0.7 \cdot 0.09/20 = 0.0302$. Sàn: $0.027$. *(1.5đ)*

**B3 (6đ)**

(a) *(4đ)*

| Vòng | Gán | Tâm | Inertia |
|---|---|---|---|
| 1 | {1} · {3, 4, 10, 12, 32} | 1 · 12.2 | 548.8 |
| 2 | {1, 3, 4} · {10, 12, 32} | 2.667 · 18 | 300.67 |
| 3 | {1, 3, 4, 10} · {12, 32} | 4.5 · 22 | 245 |
| 4 | {1, 3, 4, 10, 12} · {32} | **6 · 32** | **90** |
| 5 | không đổi | — | dừng |

(b) Cách chia "tự nhiên" {1, 3, 4} · {10, 12, 32} có inertia **300.67**, lớn hơn nhiều so với 90. Theo **hàm mục tiêu của K-Means** (tổng bình phương), tách riêng điểm 32 là tối ưu, vì bình phương khoảng cách làm outlier chiếm phần lớn tổng. Nếu "cụm tự nhiên" mới là điều ta muốn thì phải: loại hoặc cắt outlier, dùng $k = 3$, dùng K-Medians, hoặc dùng DBSCAN (đánh dấu 32 là nhiễu). *(2đ)*

## C. Code (32đ)

Chấm theo test: **C1** 3 test (12đ) · **C2** 2 test (12đ) · **C3** 3 test (6đ) + 2 câu trả lời (2đ).

```python
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold


# ---------- C1 ----------
def _encode(cats, y, m, prior):
    g = pd.DataFrame({"c": cats, "y": y}).groupby("c")["y"].agg(["sum", "count"])
    return ((g["sum"] + m * prior) / (g["count"] + m)).to_dict()


def target_encode_cv(cats, y, k=5, m=10.0, seed=0):
    cats, y = np.asarray(cats, dtype=object), np.asarray(y, dtype=float)
    prior = float(y.mean())
    mapping = _encode(cats, y, m, prior)                    # dùng cho test set
    enc = np.empty(len(y))
    for tr, te in KFold(n_splits=k, shuffle=True, random_state=seed).split(cats):
        p = float(y[tr].mean())
        mp = _encode(cats[tr], y[tr], m, p)                 # CHỈ thống kê của các fold khác
        enc[te] = [mp.get(c, p) for c in cats[te]]
    return enc, mapping, prior


def apply_encoding(cats, mapping, prior):
    return np.array([mapping.get(c, prior) for c in cats], dtype=float)


# ---------- C2 ----------
def nested_cv(estimator, param_grid, X, y, outer_k=3, inner_k=3, seed=0):
    X, y = np.asarray(X), np.asarray(y)
    res = {"outer_scores": [], "inner_best": [], "best_params": []}
    for tr, te in StratifiedKFold(outer_k, shuffle=True, random_state=seed).split(X, y):
        search = GridSearchCV(clone(estimator), param_grid, scoring="roc_auc",
                              cv=StratifiedKFold(inner_k, shuffle=True, random_state=seed))
        search.fit(X[tr], y[tr])                            # tune CHỈ trên phần train của fold ngoài
        res["outer_scores"].append(float(roc_auc_score(y[te], search.predict_proba(X[te])[:, 1])))
        res["inner_best"].append(float(search.best_score_))
        res["best_params"].append(search.best_params_)
    return res


# ---------- C3 ----------
def brier(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def calibration_table(y, p, n_bins=10):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)     # p = 1.0 vào bin cuối
    t = (pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin")
         .agg(count=("y", "size"), mean_pred=("p", "mean"), frac_pos=("y", "mean")).reset_index())
    return t[["bin", "count", "mean_pred", "frac_pos"]]


def ece(y, p, n_bins=10):
    t = calibration_table(y, p, n_bins)
    return float(np.sum(t["count"] / t["count"].sum() * np.abs(t["frac_pos"] - t["mean_pred"])))
```

**Hai câu trả lời của C3:**
1. Với cột gần như mỗi dòng một giá trị, mã hóa naive dùng **chính nhãn của dòng đó**. Với $m = 1$, một giá trị chỉ xuất hiện một lần có $y = 1$ được mã hóa thành $(1 + \text{prior})/2 \approx 0.75$, còn có $y = 0$ thì thành $\approx 0.25$. Nhãn bị chép thẳng vào đặc trưng: đây là target leakage. Bản out-of-fold không nhìn thấy nhãn của chính dòng đó, nên AUC ≈ 0.5, đúng với sự thật.
2. `class_weight="balanced"` làm loss coi mỗi mẫu dương nặng gấp $w_1/w_0$ lần, nên xác suất dự đoán bị **đẩy lên** (bài 05: trung bình từ 0.154 lên 0.422). Trong mỗi bin, `mean_pred` lớn hơn hẳn `frac_pos`, nên ECE lớn.

**Lỗi hay gặp:**
- C1: tính `mapping` trên toàn bộ dữ liệu rồi dùng luôn cho tập train (chính là bản naive).
- C2: gọi `estimator.fit` trực tiếp thay vì `clone`; hoặc báo `best_score_` thay cho điểm của vòng ngoài.
- C3: `p = 1.0` rơi vào bin `n_bins` (ngoài phạm vi), làm ra một bin thứ 11.

## D. Thiết kế (24đ)

**D1 (8đ)**

(a) *(3đ)*
- **Thời điểm dự đoán:** cuối ngày cuối cùng của tháng $M$.
- **Cửa sổ đặc trưng:** 3–6 tháng trước thời điểm đó: nạp tiền, số phút gọi, data, khiếu nại, số ngày từ lần nạp cuối…
- **Cửa sổ nhãn:** "rời mạng" xảy ra trong tháng $M + 2$ (cần định nghĩa chính xác, ví dụ không phát sinh cước 30 ngày liên tiếp, hoặc chuyển mạng giữ số).
- **Khoảng trống một tháng ($M + 1$)** giữa hai cửa sổ dành cho việc chạy model và gọi điện. Nếu dự đoán churn ngay trong tháng $M + 1$, nhiều thuê bao đã rời đi trước khi kịp gọi. Khoảng trống này cũng tránh việc các tín hiệu "đang rời đi" lọt vào đặc trưng.

(b) Ba đặc trưng có nguy cơ rò rỉ: *(3đ)*
- Trạng thái thuê bao (khóa một chiều / hai chiều), vì có thể là **hệ quả** của việc đã rời mạng.
- Yêu cầu chuyển mạng giữ số.
- Các đặc trưng tổng hợp tính trên **cả** tháng của nhãn (ví dụ "tổng nạp 30 ngày gần nhất" bị tính tới ngày chạy batch thay vì tới thời điểm dự đoán).

Cách kiểm tra:
- Với mỗi cột, ghi rõ **mốc thời gian** mà giá trị của nó được biết.
- Huấn luyện một model chỉ dùng từng đặc trưng riêng lẻ; một cột cho AUC rất cao (ví dụ > 0.95) là đáng nghi.
- Kiểm tra permutation importance xem có cột nào đóng góp bất thường.

(c) *(2đ)*
- Chia **theo thời gian**: train các tháng 1–18, validation các tháng 19–21, test các tháng 22–24 (các snapshot cách nhau đúng khoảng trống ở (a)).
- Một thuê bao xuất hiện ở nhiều tháng thì các dòng của họ tương quan với nhau. Chia theo thời gian đã đảm bảo không dùng tương lai. Có thể thêm group theo thuê bao **trong** tập train khi chạy CV, để điểm CV không bị thổi phồng.

**D2 (8đ)**

(a) Ngẫu nhiên: $50\,000 \times 2\% = 1\,000$ thuê bao sắp rời đi được gọi, giữ lại $250$, thu về $250 \times 2.4$ triệu $= 0.6$ tỉ. Chi phí $50\,000 \times 60\,000 = 3.0$ tỉ. Kết quả: **lỗ 2.4 tỉ đ/tháng**. *(1.5đ)*

(b) Hòa vốn khi $50\,000 \cdot p \cdot 0.25 \cdot 2.4\text{ tr} = 3.0\text{ tỉ}$, tức $p = 60\,000/600\,000 =$ **10%**; **lift = 10%/2% = 5**. *(2đ)*

(c) $50\,000 \times 0.3 = 15\,000$ thuê bao sắp rời đi được gọi, giữ lại $3\,750$, thu về $9.0$ tỉ. Trừ chi phí 3.0 tỉ: **lãi 6.0 tỉ đ/tháng**. *(1.5đ)*

(d) Gọi thuê bao $i$ khi lợi nhuận kỳ vọng dương: $p_i \cdot 0.25 \cdot 2.4\text{ tr} > 60\,000 \iff p_i >$ **0.1**. Lấy những thuê bao có $p_i$ cao nhất, tối đa 50 000, và dừng sớm hơn nếu $p_i$ đã xuống dưới 0.1. Điều kiện này so sánh $p_i$ với một **con số tuyệt đối** (0.1), nên cần $p_i$ đúng là xác suất. Một score chỉ xếp hạng tốt (AUC cao) cho biết **ai trước ai sau**, nhưng không cho biết **dừng ở đâu**. *(1.5đ)*

(e) Chỉ 0.5% số thuê bao được gọi, nên điều quan trọng là **đầu bảng xếp hạng**. ROC-AUC tính trên **toàn bộ** thứ hạng, kể cả phần không bao giờ được gọi. Nên báo cáo: **precision@50k / lift@50k**, **lợi nhuận kỳ vọng** theo số thuê bao được gọi, và **ECE / reliability diagram** nếu dùng quy tắc ở (d). *(1.5đ)*

**D3 (8đ)**

(a) *(2đ)*
- **Baseline:** logistic regression với các đặc trưng đã chuẩn hóa; nhanh và dễ giải thích cho bộ phận kinh doanh. Và một baseline luật: "không nạp tiền 20 ngày".
- **Model chính:** HistGradientBoosting hoặc LightGBM, vì dữ liệu bảng có ngưỡng và tương tác (bài 07). Dữ liệu 10 triệu dòng thì HGB/LightGBM xử lý tốt nhờ histogram.

(b) *(2.5đ)*
- **Backtest theo thời gian** (rolling): train đến tháng $t$, đánh giá tháng $t + 2$, lặp lại cho nhiều tháng.
- Báo trung bình và độ dao động của precision@50k qua các tháng, kèm khoảng tin cậy bootstrap.
- So sánh với baseline trên **cùng** các tháng (ghép cặp).
- Tune hyperparameter chỉ trên phần train/validation.
- Kiểm tra calibration trên validation, và hiệu chỉnh nếu cần.

(c) Cần theo dõi: *(2đ)*
- **Drift** của các đặc trưng (ví dụ PSI) và phân phối score.
- Tỉ lệ churn thực tế.
- Precision@50k khi nhãn của tháng đó về (trễ 2 tháng).
- ECE.

Huấn luyện lại định kỳ (ví dụ hàng tháng hoặc hàng quý), hoặc khi metric giảm vượt ngưỡng, hoặc khi có thay đổi lớn (gói cước mới, đối thủ khuyến mãi).

(d) Những thuê bao được gọi và ở lại sẽ có nhãn "không rời mạng" **do có can thiệp**, không phải vì họ vốn sẽ ở lại. Model học trên dữ liệu đó sẽ hiểu sai rằng nhóm rủi ro cao thực ra an toàn (*feedback loop*). Hai cách xử lý: *(1.5đ)*
- Giữ một **nhóm đối chứng ngẫu nhiên** (holdout, ví dụ 5–10% số thuê bao có rủi ro cao **không** được gọi). Nhóm này cho nhãn sạch để huấn luyện và để đo **hiệu quả thật** của chiến dịch (uplift).
- Hoặc loại các thuê bao đã được can thiệp khỏi tập huấn luyện, hoặc đánh dấu họ bằng một đặc trưng riêng. Mở rộng: uplift modeling, tức dự đoán **ai sẽ thay đổi quyết định nhờ được gọi**, thay vì chỉ dự đoán ai sẽ rời đi.
