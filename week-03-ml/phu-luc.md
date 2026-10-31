# Phụ lục Tuần 3 · Feature engineering, dữ liệu mất cân bằng, metric hồi quy

> **Loại tài liệu: Lý thuyết (bản nhẹ).** Có công thức và ví dụ tính tay đầy đủ; chứng minh chỉ phác thảo vài dòng.
> **Vì sao có phụ lục này:** Tuần 3 dạy feature engineering rải rác (splines, tương tác ở bài 05; target encoding ở final), xử lý mất cân bằng mới dừng ở `class_weight`, và metric hồi quy chỉ có một dòng định nghĩa ở bài 04. Ba bài dưới đây lấp các chỗ đó.
> **Gồm 3 bài:**
> - **P1** Feature engineering: biến đổi số, mã hóa phân loại, thời gian, tương tác
> - **P2** Dữ liệu mất cân bằng: resampling, SMOTE, hiệu chỉnh xác suất
> - **P3** Metric hồi quy: MAE, RMSE, R², MAPE, pinball, và chuyện train trên $\log y$
>
> **Thời gian:** đọc ~1h45 · quiz 30' (cuối file).
> **Dữ liệu:** P1 dùng lại bộ log chatbot `quality` (6 000 dòng, 15.5% 👎). P2 dùng thêm một bộ tổng hợp mất cân bằng mạnh hơn (20 000 dòng, **3.54%** dương, `make_classification`). P3 dùng cột `latency_s` của `quality` làm biến mục tiêu hồi quy. Mọi con số đo bằng scikit-learn 1.9 trên máy bạn; CV là `RepeatedStratifiedKFold` (5 fold, lặp 3 lần với P1, lặp 2 lần với P2).

---

## P1 · Feature engineering: biến đổi số, mã hóa phân loại, thời gian, tương tác

> ⏱ Tầng 1: 10' · Tầng 2: 30'
> **Cần biết trước:** model tuyến tính chỉ cộng điểm độc lập (Tuần 3, bài 05); cây tự học ngưỡng và tương tác (bài 06–07); Pipeline/ColumnTransformer (bài 10); rò rỉ dữ liệu (bài 01).

> **📖 Từ mới**
> - **Feature engineering:** tạo đặc trưng mới từ dữ liệu thô sao cho model **dễ học** quan hệ cần học.
> - **Biến đổi đơn điệu:** đổi thang đo mà giữ thứ tự, ví dụ $\log$.
> - **Binning (rời rạc hóa):** chia một cột số thành các khoảng, mỗi khoảng thành một cột 0/1.
> - **Đặc trưng chéo (cross feature):** ghép hai cột phân loại thành một, ví dụ `intent|model` = `tech_support|small`.
> - **Mã hóa chu kỳ (cyclic encoding):** biểu diễn giờ trong ngày hay thứ trong tuần bằng $(\sin, \cos)$ để 23h nằm cạnh 0h.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Model là **đầu bếp**, đặc trưng là **nguyên liệu**. Đầu bếp giỏi (gradient boosting) tự thái, tự lọc được; đầu bếp đơn giản (logistic regression) chỉ biết **trộn theo tỉ lệ**, nên cần nguyên liệu đã sơ chế sẵn. Feature engineering là bước sơ chế: dữ liệu vẫn như cũ, nhưng được trình bày theo cách model "hiểu" được.

**Ví dụ có số: giờ trong ngày.** Nếu đưa `hour` vào model dưới dạng số thô, 23h và 0h cách nhau $|23 - 0| = 23$, **xa hơn** 0h với 12h ($|0 - 12| = 12$). Điều này vô lý: 23h và 0h chỉ cách nhau một giờ. Hãy đặt mỗi giờ lên một **mặt đồng hồ** (hình tròn bán kính 1): giờ $h$ thành điểm $\big(\sin(2\pi h/24),\ \cos(2\pi h/24)\big)$.

| Giờ | $\sin$ | $\cos$ |
|---|---|---|
| 0 | 0 | 1 |
| 6 | 1 | 0 |
| 12 | 0 | −1 |
| 18 | −1 | 0 |
| 23 | −0.259 | 0.966 |

Khoảng cách Euclid giữa hai điểm: $d(23\text{h}, 0\text{h}) = 0.261$, bằng $d(0\text{h}, 1\text{h})$; còn $d(0\text{h}, 12\text{h}) = 2$ (hai đầu đường kính). Giờ đã "khép vòng".

**Ba ý cốt lõi:**
1. Model tuyến tính chỉ học được quan hệ **cộng và tuyến tính** trong các đặc trưng bạn đưa vào. Ngưỡng, tương tác, chu kỳ phải được **tạo sẵn** thành đặc trưng.
2. Model cây tự học ngưỡng và tương tác, nên feature engineering giúp cây **ít hơn** nhiều. Cây vẫn cần các đặc trưng **mang thông tin mới**: tổng hợp theo người dùng, khoảng thời gian từ lần trước.
3. Mọi đặc trưng phải tính được **tại thời điểm dự đoán**, và mọi thống kê "học từ dữ liệu" (trung vị, bin, target encoding) phải fit **trong fold train**.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### P1.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Feature engineering** | Quá trình tạo, biến đổi, chọn đặc trưng để cải thiện model | Kuhn & Johnson, *Feature Engineering and Selection* (2019); Zheng & Casari, *Feature Engineering for Machine Learning* (2018) |
| **Log / Box-Cox / Yeo-Johnson** | Biến đổi để giảm độ lệch (skew) của phân phối; Yeo-Johnson chấp nhận cả giá trị âm | Box & Cox, JRSS-B 1964; Yeo & Johnson, Biometrika 2000; `PowerTransformer` |
| **Binning (`KBinsDiscretizer`)** | Chia khoảng theo độ rộng đều hoặc theo **quantile**, rồi one-hot | scikit-learn docs |
| **One-hot / ordinal encoding** | Mỗi giá trị một cột 0/1 / mỗi giá trị một số nguyên (chỉ hợp lý khi có thứ tự) | scikit-learn docs |
| **Target encoding** | Thay giá trị phân loại bằng trung bình (đã làm mượt) của $y$ trong nhóm đó; phải tính **out-of-fold** | Micci-Barreca, SIGKDD Explorations 2001 |
| **Count / frequency encoding** | Thay giá trị bằng số lần xuất hiện của nó | Zheng & Casari 2018 |
| **Feature hashing** | Băm giá trị phân loại vào $2^k$ cột cố định; chấp nhận va chạm | Weinberger et al., ICML 2009 |
| **Cross feature** | Tích Descartes của hai đặc trưng phân loại | Wide & Deep, Cheng et al. 2016 |
| **Cyclic encoding** | $(\sin 2\pi t/P,\ \cos 2\pi t/P)$ cho biến có chu kỳ $P$ | scikit-learn docs, *Time-related feature engineering* |

#### P1.2 Danh mục kỹ thuật theo loại dữ liệu

| Loại dữ liệu | Kỹ thuật | Model nào cần | Lưu ý |
|---|---|---|---|
| **Số, lệch phải** (token, latency, giá tiền) | $\log(1 + x)$, Yeo-Johnson | Tuyến tính, mạng neural, KNN | Cây không cần (biến đổi đơn điệu không đổi cách chia) |
| **Số, có ngưỡng** | Cờ nghiệp vụ (`latency > SLA`), binning, splines | Tuyến tính | Ngưỡng nên đến từ nghiệp vụ hoặc từ dữ liệu train |
| **Số, vượt khoảng hợp lý** | Cắt (clip) ở phân vị 1%/99% | Tuyến tính | Tính phân vị trên train |
| **Phân loại, ít giá trị** | One-hot | Mọi model | `handle_unknown="ignore"` |
| **Phân loại, rất nhiều giá trị** (user_id, sản phẩm) | Target encoding (out-of-fold), count encoding, hashing, embedding (Tuần 4) | Mọi model | One-hot thì quá nhiều cột thưa |
| **Thời gian** | Giờ, thứ, ngày lễ; mã hóa chu kỳ; "bao lâu kể từ lần trước" | Mọi model | Đổi múi giờ trước (Tuần 1, phụ lục P5) |
| **Hai cột phân loại tương tác** | Cross feature | Tuyến tính | Số cột bằng tích số giá trị; chỉ chéo các cặp có lý do |
| **Lịch sử theo thực thể** (user, phiên) | Tổng hợp: số lần 👎 trước đó, latency trung bình 7 ngày | Mọi model | **Chỉ dùng quá khứ** tính tới thời điểm dự đoán |
| **Giá trị thiếu** | Điền + cột chỉ báo thiếu (`add_indicator=True`) | Mọi model | Việc "thiếu" thường mang thông tin |
| **Văn bản** | Độ dài, TF-IDF (Tuần 5), embedding (Tuần 8) | | |

#### P1.3 Ví dụ tính tay

**(a) Target encoding có làm mượt.** Công thức (giống final Tuần 3):

$$\text{enc}(c) = \frac{\sum_{i: x_i = c} y_i + m \cdot \text{prior}}{n_c + m}.$$

Một giá trị `intent` hiếm xuất hiện $n_c = 4$ lần, có 3 lần 👎; prior $= 0.155$; $m = 10$.
- Không làm mượt: $3/4 = 0.75$. Chỉ dựa vào 4 dòng nên rất nhiễu.
- Làm mượt: $(3 + 10 \times 0.155)/(4 + 10) = 4.55/14 =$ **0.325**. Kéo mạnh về prior vì $n_c$ nhỏ.
- Nếu cùng tỉ lệ 75% nhưng $n_c = 400$ (300 lần 👎): $(300 + 1.55)/410 = 0.735$, gần như giữ nguyên.

$m$ là "số mẫu ảo" mang giá trị prior: giá trị hiếm thì tin prior, giá trị phổ biến thì tin dữ liệu.

**(b) Log nén đuôi dài.** `answer_tokens` = 50, 150, 1 000 thành $\log(1 + x)$ = 3.93, 5.02, 6.91. Khoảng cách 50→150 (×3) và 150→1 000 (×6.7) trở nên tương đương nhau trên thang log. Model tuyến tính sẽ coi "gấp đôi" là một mức thay đổi cố định, bất kể đang ở mức nào.

**(c) Khoảng cách giờ** (đã làm ở Tầng 1): $d = 2\sin(\pi \Delta h / 24)$, với $\Delta h$ là số giờ chênh lệch (lấy theo chiều ngắn trên vòng tròn). $\Delta h = 1$: 0.261; $\Delta h = 4$: $2\sin(\pi/6) = 1$; $\Delta h = 12$: 2.

#### P1.4 Đo trên dữ liệu project: feature engineering giúp logistic regression đuổi kịp HGB

AUC trung bình qua 15 fold (5 fold × 3 lần lặp). Mỗi dòng **cộng dồn** thêm vào dòng trên, trừ khi ghi khác. Các đặc trưng được tạo bằng một hàm thuần `add_features(df)` (không fit gì nên không rò rỉ); binning nằm trong `ColumnTransformer` nên được fit trong từng fold.

| Cấu hình | Đặc trưng thêm | AUC |
|---|---|---|
| Logistic regression gốc (như bài 05) | số + one-hot | 0.768 |
| + log | `log1p` của latency, answer_tokens, query_tokens (thay cột gốc) | **0.760** (giảm!) |
| + cờ nghiệp vụ | `slow` = latency > 5 s (SLA), `night` = 0–5h, `n_turns` cắt ở 8 | 0.785 |
| + binning | 10 bin quantile cho `retrieval_score`, `latency_s` | 0.791 |
| + cross feature | `intent|model`, `intent|long_answer` (answer > 250 token) | **0.813** |
| *(thay `night` bằng $\sin/\cos$ của giờ)* | | 0.808 |
| *(không dùng cờ nghiệp vụ; chỉ binning + cross)* | | 0.789 |
| HGB gốc | | 0.809 |
| HGB + cùng đặc trưng | | 0.811 |
| Trần: AUC của xác suất thật `p_true` | | 0.828 |

**Đọc bảng:**
- **LR đầy đủ ngang HGB:** chênh +0.0035 ± 0.0084 theo cặp fold, LR thắng 10/15 fold, tức là **ngang nhau**. So với LR gốc: +0.044, thắng **15/15** fold.
- **Biến đổi không tự động là tốt.** Log làm AUC giảm ở **cả 15/15 fold** (−0.0086). Lý do: tác động của latency là một **ngưỡng** ở 5 s nằm ở **đuôi** phân phối, và log lại **nén** chính phần đuôi đó. Hãy nhìn quan hệ với $y$ (partial dependence của HGB, hoặc tỉ lệ 👎 theo bin) trước khi chọn phép biến đổi.
- **Cross feature đóng góp nhiều nhất** (+0.022): đúng hai tương tác có trong dữ liệu. Bài 05 đã đo: thêm **mọi** tương tác bậc 2 làm AUC **giảm** còn 0.743. Chéo **có chọn lọc** thì giúp, chéo hàng loạt thì không.
- **Chu kỳ không phải lúc nào cũng thắng.** Hiệu ứng giờ ở đây là bậc thang (0–5h), nên cờ `night` khớp hơn một sóng $\sin/\cos$ trơn.
- **Hiểu nghiệp vụ đáng giá.** Không có cờ nghiệp vụ, binning chỉ tìm được một phần ngưỡng (0.789). Ngưỡng 5 s nằm ở đuôi nên 10 bin quantile không có bin nào bắt đầu đúng ở đó.
- **HGB gần như không hưởng lợi** (+0.002): cây đã tự học ngưỡng và tương tác.

*Ghi chú trung thực:* dữ liệu là tổng hợp, nên ta **biết** các ngưỡng thật. Ngoài đời, các ngưỡng đến từ hiểu biết nghiệp vụ (SLA, ca trực đêm) hoặc từ việc đọc HGB/partial dependence trên **dữ liệu train**.

#### P1.5 Code chuẩn

```python
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

def add_features(d):
    """Hàm thuần, theo từng dòng: không học gì từ dữ liệu, nên gọi trước khi chia vẫn an toàn."""
    d = d.copy()
    d["log_query"] = np.log1p(d["query_tokens"])
    d["slow"] = (d["latency_s"] > 5).astype(float)           # SLA 5 s
    d["night"] = d["hour"].between(0, 5).astype(float)
    d["turns_c"] = d["n_turns"].clip(upper=8)
    d["intent_x_model"] = d["intent"] + "|" + d["model"]
    long = np.where(d["answer_tokens"] > 250, "long", "short")
    d["intent_x_long"] = d["intent"] + "|" + long
    return d

num = ["log_query", "n_docs", "retrieval_score", "slow", "night", "turns_c"]
cat = ["intent", "model", "intent_x_model", "intent_x_long"]
binned = ["retrieval_score", "latency_s"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                      ("sc", StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("bin", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                      ("kb", KBinsDiscretizer(n_bins=10, encode="onehot", strategy="quantile",
                                              quantile_method="averaged_inverted_cdf"))]), binned),
])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000))])
# cross_val_score(model, add_features(df), y, cv=..., scoring="roc_auc")
```

Chạy đúng đoạn code trên với CV như P1.4 (đã chạy): AUC **0.813**, bằng cấu hình tốt nhất trong bảng, dù đã bỏ ba cột log và cột `hour` thô.

**Lý do thiết kế:**
- **Hai loại biến đổi.** Biến đổi **không học** (log, cờ, cross) đặt trong một hàm thuần. Biến đổi **có học** (điền trung vị, bin quantile, scaler, target encoding) đặt trong Pipeline, để CV tự fit lại trong từng fold.
- **Dùng chung một hàm** `add_features` cho cả lúc train lẫn lúc phục vụ. Hai bản code khác nhau thì sớm muộn sẽ lệch nhau (*training–serving skew*, Tuần 13).

#### P1.6 Cạm bẫy

1. **Đặc trưng tổng hợp dùng tương lai:** "tỉ lệ 👎 của user này" tính trên **toàn bộ** log rồi đưa vào train là rò rỉ (giá trị chứa chính nhãn của dòng đó và các dòng sau). Phải tính bằng dữ liệu **trước** thời điểm của mỗi dòng.
2. **Target encoding không out-of-fold:** với giá trị hiếm, encoding gần như **bằng chính nhãn**, nên model học thuộc. Phải dùng out-of-fold như final Tuần 3, hoặc `TargetEncoder` của scikit-learn (mặc định đã có cross-fitting bên trong `fit_transform`).
3. **Fit bin/scaler/imputer trên toàn bộ dữ liệu** trước khi chia: rò rỉ nhẹ, nhưng là lỗi quy trình.
4. **Chéo mọi cặp đặc trưng:** số cột nổ ra, model overfit (0.743 ở bài 05).
5. **Biến đổi theo thói quen** ("cột lệch thì cứ log"): log ở trên làm giảm AUC trên 15/15 fold.
6. **Ordinal encoding cho biến không có thứ tự** (`intent` → 0, 1, 2, 3, 4) với model tuyến tính: áp đặt một thứ tự vô nghĩa.

**Sắc thái phỏng vấn:**
- *"Đã có deep learning, còn cần feature engineering không?"* Với dữ liệu bảng: có, và nó thường là chỗ cải thiện lớn nhất; gradient boosting kèm đặc trưng tốt vẫn là baseline mạnh (Grinsztajn et al., NeurIPS 2022). Với ảnh và văn bản, mạng tự học biểu diễn, còn feature engineering chuyển thành thiết kế đầu vào: prompt, chunking, metadata (Tuần 8).
- *"Làm sao biết một đặc trưng mới có ích?"* So bằng CV **theo cặp fold** (cùng fold cho cả hai cấu hình); báo cáo chênh lệch trung bình và số fold thắng, như ở P1.4. Kiểm tra permutation importance trên tập validation (Tuần 3, bài 07).

#### P1.7 Đọc thêm

- M. Kuhn & K. Johnson, *Feature Engineering and Selection* (2019), bản online miễn phí tại feat.engineering.
- A. Zheng & A. Casari, *Feature Engineering for Machine Learning* (O'Reilly 2018).
- scikit-learn User Guide: *Preprocessing data*, *Target Encoder*, ví dụ *Time-related feature engineering*.
- D. Micci-Barreca, *A Preprocessing Scheme for High-Cardinality Categorical Attributes*, SIGKDD Explorations 2001.
- L. Grinsztajn, E. Oyallon, G. Varoquaux, *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS 2022 (Datasets & Benchmarks).

---

## P2 · Dữ liệu mất cân bằng: resampling, SMOTE, hiệu chỉnh xác suất

> ⏱ Tầng 1: 10' · Tầng 2: 25'
> **Cần biết trước:** precision/recall, AP, ngưỡng theo chi phí $t^\star = c_{FP}/(c_{FP} + c_{FN})$ (Tuần 3, bài 04); `class_weight` đẩy xác suất lên (bài 05); Brier score và ECE (final Tuần 3); Bayes dạng odds (Tuần 2, bài 08).

> **📖 Từ mới**
> - **Mất cân bằng lớp (class imbalance):** một lớp hiếm hơn hẳn lớp kia, ví dụ 3.5% giao dịch gian lận.
> - **Undersampling:** bỏ bớt mẫu của lớp đông. **Oversampling:** nhân bản mẫu của lớp hiếm.
> - **SMOTE:** tạo mẫu hiếm **nhân tạo** bằng cách nội suy giữa một mẫu hiếm và một hàng xóm hiếm của nó.
> - **Prior (tỉ lệ nền):** tỉ lệ lớp dương trong dữ liệu, ký hiệu $\pi$.
> - **Hiệu chỉnh prior:** đổi xác suất của một model được huấn luyện ở prior này sang xác suất đúng ở prior khác.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Một bác sĩ thực tập học chẩn đoán bệnh hiếm (3.5% bệnh nhân). Nếu chỉ cho họ xem hồ sơ thật, họ gặp quá ít ca bệnh để học. Nếu cho xem **một nửa ca bệnh, một nửa ca khỏe** (resampling), họ học nhận dạng tốt hơn. Nhưng họ sẽ **quen nghĩ** rằng cứ 2 người thì có 1 người bệnh, và khi ra phòng khám thật, họ báo động quá nhiều. Muốn dùng được thì phải **nhắc lại tỉ lệ thật**: đó là hiệu chỉnh prior.

**Ví dụ có số, đi từng bước: resampling làm lệch xác suất bao nhiêu?** Dữ liệu thật có $\pi = 3.54\%$ dương. Train trên dữ liệu đã cân bằng 50/50.

Theo Bayes dạng odds (Tuần 2, bài 08): odds hậu nghiệm = odds tiên nghiệm × tỉ số likelihood. Model học được **tỉ số likelihood** từ đặc trưng (giống nhau dù prior là gì), nhưng odds tiên nghiệm thì khác:
- Odds tiên nghiệm lúc train: $0.5/0.5 = 1$.
- Odds tiên nghiệm thật: $0.0354/0.9646 = 0.0367$.
- Vậy mọi odds của model bị **phóng to** $1/0.0367 = 27.2$ lần.

Model nói 0.5 (odds 1) thì odds thật là $1 \times 0.0367$, tức xác suất thật **0.0354**. Model "phân vân 50/50" thật ra đang nói "bằng đúng tỉ lệ nền".

Đo trên máy bạn (logistic regression, bộ dữ liệu 3.54%): train trên dữ liệu gốc thì xác suất dự đoán trung bình là 0.0354 (khớp tỉ lệ thật); train với `class_weight="balanced"` thì trung bình **0.255**, cao gấp 7 lần thực tế.

**Ba ý cốt lõi:**
1. Resampling và class weight **gần như không đổi khả năng xếp hạng** (AUC), nhưng **làm sai thang xác suất**. Thứ thay đổi là điểm làm việc ở ngưỡng 0.5.
2. Đổi ngưỡng trên model gốc đạt hiệu ứng **gần như tương đương**, mà vẫn giữ xác suất đúng.
3. Nếu đã resample, hãy **hiệu chỉnh prior** trước khi dùng xác suất; và **chỉ** resample trong fold train.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### P2.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Random under/oversampling** | Bỏ ngẫu nhiên mẫu lớp đông / lấy lặp có hoàn lại mẫu lớp hiếm cho tới khi cân bằng | He & Garcia, *Learning from Imbalanced Data*, IEEE TKDE 2009 |
| **SMOTE** | Với mỗi mẫu hiếm mới: chọn mẫu hiếm $x_i$, chọn một trong $k$ hàng xóm hiếm gần nhất $x_j$, lấy $x_i + u(x_j - x_i)$, $u \sim U(0, 1)$ | Chawla et al., JAIR 16 (2002) |
| **Cost-sensitive learning** | Gán chi phí khác nhau cho từng loại lỗi; tương đương đổi ngưỡng hoặc đổi trọng số mẫu | Elkan, IJCAI 2001 |
| **Prior shift correction** | $\text{odds}_\text{đúng} = \text{odds}_\text{model} \cdot \dfrac{\pi/(1 - \pi)}{\pi_s/(1 - \pi_s)}$, với $\pi_s$ là prior lúc train | Saerens, Latinne, Decaestecker, *Neural Computation* 2002; Elkan 2001 |
| **Threshold moving** | Giữ model, chỉ đổi ngưỡng quyết định | Tuần 3, bài 04 |
| **Average Precision (PR-AUC)** | Metric nhạy với lớp hiếm; baseline ngẫu nhiên bằng $\pi$ | Davis & Goadrich, ICML 2006 |

#### P2.2 SMOTE từng bước

Mẫu hiếm $x_i = (1, 2)$. Trong $k = 5$ hàng xóm hiếm gần nhất, chọn ngẫu nhiên được $x_j = (3, 6)$. Rút $u = 0.25$:

$$x_\text{mới} = x_i + u\,(x_j - x_i) = (1, 2) + 0.25 \times (2, 4) = (1.5,\ 3).$$

Điểm mới nằm **trên đoạn thẳng** nối hai mẫu hiếm, cách $x_i$ một phần tư quãng đường. SMOTE "tô đậm" vùng của lớp hiếm thay vì chồng các bản sao lên đúng một điểm như random oversampling.

Hạn chế:
- Đoạn nối hai mẫu hiếm có thể **cắt qua vùng của lớp đông**, tạo ra điểm sai nhãn.
- Nội suy **không có nghĩa** với cột one-hot (`intent` = 0.25 × faq + 0.75 × billing?). Dùng biến thể SMOTE-NC cho dữ liệu có cột phân loại.
- Cần khoảng cách có ý nghĩa, nên phải scale đặc trưng trước.

#### P2.3 Hiệu chỉnh prior: ví dụ tính tay

Model train trên dữ liệu 50/50 ($\pi_s = 0.5$), dữ liệu thật có $\pi = 0.04$. Hệ số đổi odds: $\dfrac{0.04/0.96}{0.5/0.5} = 0.04167$.

| Model nói $q$ | odds model $q/(1 - q)$ | × 0.04167 | Xác suất đúng $p = \text{odds}/(1 + \text{odds})$ |
|---|---|---|---|
| 0.5 | 1 | 0.04167 | **0.040** |
| 0.8 | 4 | 0.1667 | **0.143** (= 1/7) |
| 0.96 | 24 | 1.0 | **0.5** |

Model phải nói 0.96 thì mới tương đương "50/50" ngoài đời thật.

**Vì sao (phác thảo).** Logistic regression ước lượng log-odds $= \log\frac{p(x \mid 1)}{p(x \mid 0)} + \log\frac{\pi}{1 - \pi}$. Resampling chỉ đổi số hạng thứ hai (prior), không đổi số hạng thứ nhất (likelihood). Vì vậy chỉ cần trừ đi rồi cộng lại đúng số hạng prior. Với class weight, phép hiệu chỉnh là **xấp xỉ**, vì trọng số còn làm thay đổi cả cách fit.

#### P2.4 Đo trên máy bạn

Bộ dữ liệu 20 000 dòng, 3.54% dương; 10 fold; resampling **chỉ** trên phần train của mỗi fold. "Sau hiệu chỉnh" là áp công thức P2.3 với $\pi_s = 0.5$.

| Model | Cách xử lý | AUC | AP | Brier | ECE | ECE sau hiệu chỉnh | Precision @0.5 | Recall @0.5 | $\bar p$ |
|---|---|---|---|---|---|---|---|---|---|
| LR | không | 0.891 | **0.704** | **0.016** | **0.007** | — | 0.96 | 0.53 | 0.035 |
| LR | `class_weight` | 0.901 | 0.686 | 0.098 | 0.220 | 0.013 | 0.19 | 0.80 | 0.255 |
| LR | undersample | 0.902 | 0.683 | 0.100 | 0.223 | 0.013 | 0.19 | 0.81 | 0.258 |
| LR | oversample | 0.901 | 0.684 | 0.098 | 0.221 | 0.013 | 0.19 | 0.80 | 0.257 |
| LR | SMOTE | 0.902 | 0.681 | 0.093 | 0.204 | 0.010 | 0.20 | 0.80 | 0.239 |
| HGB | không | 0.910 | **0.789** | **0.012** | **0.005** | — | 0.95 | 0.64 | 0.032 |
| HGB | `class_weight` | 0.909 | 0.766 | 0.037 | 0.109 | 0.016 | 0.49 | 0.78 | 0.144 |
| HGB | undersample | 0.905 | 0.725 | 0.072 | 0.129 | 0.011 | 0.25 | 0.82 | 0.164 |
| HGB | oversample | 0.913 | 0.789 | 0.018 | 0.035 | 0.015 | 0.72 | 0.77 | 0.071 |
| HGB | SMOTE | 0.911 | 0.778 | 0.033 | 0.082 | 0.011 | 0.50 | 0.79 | 0.118 |

**Đọc bảng:**
- **AUC gần như không đổi** (dao động trong ±0.01). **AP còn giảm nhẹ** với mọi cách xử lý: khả năng xếp hạng không tốt lên.
- **Calibration bị phá:** ECE của LR tăng từ 0.007 lên ~0.22; $\bar p$ từ 0.035 lên ~0.25. Hiệu chỉnh prior kéo ECE về 0.010–0.016, gần mức ban đầu nhưng không bằng hẳn.
- **Undersampling với HGB tệ nhất** (AP 0.725): vứt đi ~96% số mẫu âm làm model mất thông tin.
- **Điều duy nhất "tốt lên" là recall ở ngưỡng 0.5**, và cái đó có được bằng cách hạ ngưỡng:

| LR | Ngưỡng | Recall | Precision |
|---|---|---|---|
| `class_weight="balanced"` | 0.5 | 0.804 | 0.192 |
| Không xử lý, ngưỡng lý thuyết $t = \pi = 0.0354$ (ảnh của 0.5 qua phép hiệu chỉnh) | 0.0354 | 0.793 | 0.187 |

Hai điểm làm việc gần như trùng nhau. Khác biệt nhỏ đến từ chỗ class weight cũng đổi nhẹ cách fit của model.

**Trên dữ liệu project `quality`** (15.5% 👎, LR): `balanced` có AUC 0.771 so với 0.769; ECE từ 0.025 lên **0.267**, và sau hiệu chỉnh prior về lại 0.026.

**Rò rỉ khi resample trước khi chia** (Random Forest, 5 fold):

| | Oversample | SMOTE |
|---|---|---|
| **Sai:** resample cả bộ dữ liệu rồi mới chia fold | AUC **1.000** | AUC **0.999** |
| **Đúng:** chỉ resample phần train của mỗi fold | AUC 0.910 | AUC 0.911 |

Oversample trước khi chia đặt **bản sao giống hệt** của một mẫu vào cả train lẫn test, nên Random Forest chỉ cần "nhớ". SMOTE trước khi chia tạo điểm nội suy từ các mẫu test, và cũng gần như hoàn hảo một cách giả tạo. Đây là một dạng rò rỉ của bài 01.

#### P2.5 Quyết định: khi nào dùng gì

| Mục tiêu | Khuyến nghị |
|---|---|
| Cần **xác suất đúng** (tính chi phí kỳ vọng, xếp hạng rủi ro, ghép với hệ thống khác) | **Không** resample. Train trên dữ liệu gốc, chọn ngưỡng theo chi phí $t^\star = c_{FP}/(c_{FP} + c_{FN})$ |
| Chỉ cần **quyết định có/không** với một recall mục tiêu | Đổi ngưỡng trên model gốc; chọn ngưỡng trên validation |
| Lớp hiếm có **quá ít mẫu tuyệt đối** (vài chục) | Thu thêm dữ liệu dương là cách tốt nhất; `class_weight` để model không bỏ qua lớp hiếm; vẫn hiệu chỉnh xác suất khi dùng |
| Dữ liệu quá lớn, cần train nhanh | Undersample lớp đông **có ghi lại tỉ lệ**, rồi hiệu chỉnh prior (hoặc dùng trọng số $1/\text{tỉ lệ giữ lại}$) |
| Mạng neural, minibatch hiếm khi có mẫu dương | Weighted loss hoặc sampler cân bằng, rồi hiệu chỉnh; hoặc focal loss (Lin et al., ICCV 2017) |
| Báo cáo | AP/PR-AUC, recall ở precision cố định (hoặc ngược lại), Brier/ECE nếu dùng xác suất; **không** dùng accuracy |

Nhận định này khớp với một phân tích trên logistic regression cho model rủi ro y khoa (van den Goorbergh et al., JAMIA 2022): các phép "sửa mất cân bằng" làm xác suất sai lệch nặng, trong khi AUC không cải thiện.

#### P2.6 Code chuẩn

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def correct_prior(p, pi_train, pi_true):
    """Đổi xác suất của model train ở prior pi_train sang prior thật pi_true (Saerens 2002)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    odds = p / (1 - p) * (pi_true / (1 - pi_true)) / (pi_train / (1 - pi_train))
    return odds / (1 + odds)

def smote(X, y, k=5, seed=0):
    """SMOTE tối giản cho đặc trưng SỐ đã scale; cân bằng hai lớp. Chỉ gọi trên phần TRAIN."""
    rng = np.random.default_rng(seed)
    Xp = X[y == 1]
    n_new = int((y == 0).sum() - len(Xp))
    _, nbr = NearestNeighbors(n_neighbors=k + 1).fit(Xp).kneighbors(Xp)   # cột 0 là chính nó
    i = rng.integers(0, len(Xp), n_new)
    j = nbr[i, rng.integers(1, k + 1, n_new)]
    u = rng.random((n_new, 1))
    X_new = Xp[i] + u * (Xp[j] - Xp[i])
    return np.vstack([X, X_new]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])
```

Trong production, thư viện `imbalanced-learn` (chưa cài trên máy bạn) có `SMOTE`, `SMOTENC` và `imblearn.pipeline.Pipeline`. Pipeline đó chỉ resample **lúc `fit`**, không resample lúc `predict`, nên dùng an toàn bên trong `cross_val_score`. `Pipeline` của scikit-learn **không** hỗ trợ bước resample.

#### P2.7 Cạm bẫy

1. **Resample trước khi chia train/test hoặc trước CV:** AUC giả 1.000 (P2.4).
2. **Dùng xác suất của model đã resample như xác suất thật:** ước lượng số ca, chi phí, hay rủi ro bị thổi phồng ~7 lần (bảng P2.4).
3. **Báo cáo accuracy:** đoán "âm" cho tất cả đã được 96.5%.
4. **Resample tập validation/test:** metric không còn phản ánh tỉ lệ thật; precision bị thổi phồng.
5. **SMOTE trên cột one-hot hoặc đặc trưng chưa scale:** tạo ra điểm vô nghĩa.
6. **Chọn ngưỡng 0.5 theo thói quen:** với 3.5% dương, model gốc ở ngưỡng 0.5 chỉ bắt được 53% ca dương. Ngưỡng phải đến từ chi phí hoặc từ recall mục tiêu (bài 04).

**Sắc thái phỏng vấn:** *"Dữ liệu gian lận chỉ có 0.5%. Bạn xử lý mất cân bằng thế nào?"* Câu trả lời yếu là "dùng SMOTE". Câu trả lời mạnh: (1) chia dữ liệu theo **thời gian**, stratify; (2) metric là AP và recall ở mức precision mà đội vận hành chịu được; (3) baseline là model gốc với ngưỡng theo chi phí; (4) chỉ thử class weight/resampling **trong fold train**, so bằng AP theo cặp fold, và hiệu chỉnh prior nếu dùng xác suất; (5) thu thêm nhãn dương (active learning, luật nghiệp vụ) thường đáng giá hơn mọi kỹ thuật resampling.

#### P2.8 Đọc thêm

- N. Chawla et al., *SMOTE: Synthetic Minority Over-sampling Technique*, JAIR 16 (2002).
- H. He & E. Garcia, *Learning from Imbalanced Data*, IEEE TKDE 21 (2009).
- C. Elkan, *The Foundations of Cost-Sensitive Learning*, IJCAI 2001.
- M. Saerens, P. Latinne, C. Decaestecker, *Adjusting the Outputs of a Classifier to New a Priori Probabilities*, Neural Computation 14 (2002).
- R. van den Goorbergh et al., *The harm of class imbalance corrections for risk prediction models*, JAMIA 29 (2022).
- imbalanced-learn docs (imbalanced-learn.org): *Common pitfalls and recommended practices*.

---

## P3 · Metric hồi quy: MAE, RMSE, R², MAPE, pinball, và chuyện train trên $\log y$

> ⏱ Tầng 1: 10' · Tầng 2: 25'
> **Cần biết trước:** MSE, hồi quy tuyến tính (Tuần 2, bài 02); MLE với nhiễu Gauss cho ra MSE (Tuần 2, bài 10); phân phối lognormal (Tuần 2, bài 09).

> **📖 Từ mới**
> - **Sai số (residual):** $e_i = y_i - \hat y_i$, giá trị thật trừ giá trị dự đoán.
> - **Outlier:** điểm có sai số lớn bất thường.
> - **Quantile (phân vị) mức $\tau$:** giá trị mà $\tau$ phần dữ liệu nằm dưới nó; $\tau = 0.5$ là trung vị.
> - **Lệch hệ thống (bias):** trung bình dự đoán khác trung bình thật, tức là luôn đoán cao hoặc luôn đoán thấp.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Dự báo thời gian giao hàng.
- **MAE** trả lời: "trung bình lệch bao nhiêu phút?" Mọi phút trễ có giá như nhau.
- **RMSE** phạt **bình phương**: một lần trễ 60 phút tệ bằng 36 lần trễ 10 phút. Hợp với tình huống một lần trễ lớn gây hậu quả lớn.
- **MAPE** đo theo **phần trăm**: lệch 5 phút cho chuyến 10 phút (50%) tệ hơn lệch 5 phút cho chuyến 50 phút (10%).

**Ví dụ có số: hai model, hai metric, hai kết luận.** $y = (10, 20, 30, 40, 50)$.

| | Sai số $e$ | MAE | MSE | RMSE | R² |
|---|---|---|---|---|---|
| Model A | $(1, -1, 1, -1, 16)$ | $20/5 =$ **4** | $(1 + 1 + 1 + 1 + 256)/5 = 52$ | **7.21** | 0.74 |
| Model B | $(5, -5, 5, -5, 5)$ | $25/5 =$ **5** | $125/5 = 25$ | **5.00** | 0.875 |

MAE chọn **A** (4 < 5), RMSE chọn **B** (5 < 7.21). A gần như hoàn hảo nhưng có một lần trượt 16; B lần nào cũng lệch 5. Chọn metric chính là chọn **loại sai số nào đắt hơn** với nghiệp vụ của bạn.

**Ba ý cốt lõi:**
1. Mỗi metric có một **"hằng số tốt nhất"** riêng: MSE ↔ **trung bình**, MAE ↔ **trung vị**, pinball loss mức $\tau$ ↔ **phân vị $\tau$**. Train bằng loss nào thì model ước lượng đại lượng đó.
2. R² là "giảm được bao nhiêu phần sai số so với việc đoán trung bình", và **có thể âm**.
3. Train trên $\log y$ rồi lấy $\exp$ là ước lượng **trung vị**, không phải trung bình. Tổng dự đoán sẽ bị **thấp hơn** thực tế.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### P3.1 Định nghĩa

| Metric | Công thức | Đơn vị | Hằng số tối ưu | Nguồn |
|---|---|---|---|---|
| **MAE** | $\frac1n\sum\lvert e_i\rvert$ | như $y$ | trung vị | Hyndman & Koehler, IJF 2006 |
| **MSE / RMSE** | $\frac1n\sum e_i^2$ / căn của nó | $y^2$ / như $y$ | trung bình | như trên |
| **R²** | $1 - \dfrac{\sum e_i^2}{\sum (y_i - \bar y)^2}$ | không đơn vị | — (đoán $\bar y$ cho R² = 0) | scikit-learn `r2_score` |
| **Adjusted R²** | $1 - (1 - R^2)\dfrac{n - 1}{n - p - 1}$ | | | phạt số đặc trưng $p$ |
| **MAPE** | $\frac1n\sum \dfrac{\lvert e_i\rvert}{\lvert y_i\rvert}$ | % | thiên về đoán **thấp** | Hyndman & Koehler 2006 |
| **RMSLE** | RMSE trên $\log(1 + y)$ | | | sai số tương đối |
| **Huber loss** | $\tfrac12 e^2$ khi $\lvert e\rvert \le \delta$; $\delta(\lvert e\rvert - \tfrac12\delta)$ khi lớn hơn | | giữa trung bình và trung vị | Huber, Ann. Math. Stat. 1964 |
| **Pinball (quantile) loss** | $\tau e$ nếu $e \ge 0$; $(\tau - 1)e$ nếu $e < 0$ | như $y$ | phân vị $\tau$ | Koenker & Bassett, Econometrica 1978 |

**Vì sao trung vị tối ưu cho MAE (phác thảo).** Dịch hằng số $c$ lên một chút: mọi điểm nằm **trên** $c$ giảm sai số, mọi điểm nằm **dưới** $c$ tăng sai số, mỗi điểm cùng một lượng. Có lợi chừng nào số điểm phía trên còn nhiều hơn phía dưới. Điểm cân bằng là trung vị. Pinball loss gán "giá" $\tau$ cho phía trên và $1 - \tau$ cho phía dưới, nên điểm cân bằng là phân vị $\tau$.

#### P3.2 Ví dụ tính tay

**(a) R² âm.** Với $y = (10, 20, 30, 40, 50)$: $\bar y = 30$, tổng bình phương quanh trung bình $\text{SST} = 400 + 100 + 0 + 100 + 400 = 1000$.
- Model A: $\text{SSE} = 260$, $R^2 = 1 - 260/1000 =$ **0.74**.
- Đoán hằng số 30: $R^2 =$ **0**.
- Đoán toàn 0: $\text{SSE} = 100 + 400 + 900 + 1600 + 2500 = 5500$, $R^2 = 1 - 5.5 =$ **−4.5**.

**(b) Hằng số tối ưu và outlier.** Dữ liệu $(1, 2, 3, 4, 100)$: trung bình 22, trung vị 3.

| Hằng số $c$ | MAE | MSE |
|---|---|---|
| 3 (trung vị) | $(2 + 1 + 0 + 1 + 97)/5 =$ **20.2** | $9415/5 = 1883$ |
| 22 (trung bình) | $(21 + 20 + 19 + 18 + 78)/5 = 31.2$ | $7610/5 =$ **1522** |

Một outlier (100) kéo trung bình từ 2.5 lên 22, nhưng không làm trung vị nhúc nhích. Đây là lý do MAE "bền với outlier".

**(c) MAPE bất đối xứng.** $y = 100$: đoán 50 → 50%; đoán 150 → 50%; đoán 0 → **100%** (trần); đoán 300 → **200%** (không có trần). Đoán thấp bị phạt tối đa 100%, đoán cao bị phạt vô hạn, nên model tối ưu MAPE có xu hướng **đoán thấp**. MAPE còn **không xác định** khi $y = 0$.

**(d) Pinball loss, $\tau = 0.75$.** Latency (s) của 10 request: $0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0$. Pinball loss trung bình khi đoán hằng số $q$:

| $q$ | 2.0 | 3.0 | **4.0** | 5.0 | 6.0 |
|---|---|---|---|---|---|
| Loss | 1.25 | 1.05 | **1.00** | 1.05 | 1.10 |

Nhỏ nhất tại $q = 4.0$, đúng là phân vị 75% (giá trị thứ 8 trong 10; $0.75 \times 10 = 7.5$ nên lấy giá trị thứ 8). Đây là cách huấn luyện model dự đoán **p75/p90 latency** cho SLA thay vì latency trung bình.

**(e) Adjusted R².** $n = 50$, $p = 10$, $R^2 = 0.80$: $1 - 0.2 \times 49/39 =$ **0.749**.

#### P3.3 Train trên $\log y$: chậm lại từng bước

Latency, doanh thu, số token thường **lệch phải** (lognormal: $\log y$ có phân phối chuẩn với trung bình $\mu$, độ lệch chuẩn $\sigma$). Người ta hay train trên $\log y$ cho dễ học. Nhưng phải hiểu mình nhận lại đại lượng nào.

- Model trên thang log dự đoán **trung bình của $\log y$**, tức $\mu$.
- Lấy $\exp(\mu)$: vì $\exp$ đơn điệu, trung vị của $\log y$ biến thành trung vị của $y$. Với phân phối chuẩn thì trung vị = trung bình = $\mu$. Vậy $e^\mu$ là **trung vị** của $y$.
- Nhưng **trung bình** của $y$ lognormal là $e^{\mu + \sigma^2/2}$, lớn hơn trung vị $e^{\sigma^2/2}$ lần.

Với $\sigma = 0.45$ (độ nhiễu của `latency_s` trong `data.py`): $e^{0.45^2/2} = e^{0.101} = 1.107$. Dự đoán bằng $\exp$ thấp hơn trung bình thật **~10%**.

**Sửa bằng smearing** (Duan, JASA 1983): nhân dự đoán với $\frac1n\sum \exp(\hat e_i)$, trung bình $\exp$ của residual trên thang log ở tập train. Cách này không cần giả định phân phối.

**Đo trên máy bạn** (dự đoán `latency_s` từ `model`, `intent`, `answer_tokens`; 70/30; tập test có trung bình 2.05 s, trung vị 1.70 s):

| Model | MAE | RMSE | R² | MAPE | Dự đoán TB | Lệch **tổng** |
|---|---|---|---|---|---|---|
| Linear trên $y$ | 0.736 | **1.029** | **0.352** | 0.437 | 2.069 | +0.7% |
| Linear trên $\log y$ → $\exp$ | **0.714** | 1.045 | 0.332 | **0.386** | 1.871 | **−9.0%** |
| … + smearing (hệ số đo được **1.106**) | 0.734 | 1.029 | 0.352 | 0.436 | 2.069 | +0.7% |
| Huber | 0.716 | 1.039 | 0.339 | 0.397 | 1.912 | −6.9% |
| HGB, loss bình phương | 0.762 | 1.059 | 0.314 | 0.449 | 2.079 | +1.2% |
| HGB, loss tuyệt đối | 0.733 | 1.065 | 0.306 | 0.400 | 1.888 | −8.1% |

**Đọc bảng:**
- **Hệ số smearing đo được (1.106) khớp với lý thuyết** $e^{\sigma^2/2} = 1.107$.
- Các model "kiểu trung vị" (log + exp, Huber, HGB absolute) có **MAE và MAPE tốt hơn** nhưng **tổng thấp hơn 7–9%**. Các model "kiểu trung bình" có RMSE/R² tốt hơn và tổng gần đúng.
- **Hệ quả nghiệp vụ.** Nếu dùng dự đoán để **cộng dồn** (ước lượng tổng GPU-giờ, tổng chi phí token tháng tới, dung lượng cần chuẩn bị), model kiểu trung vị sẽ **thiếu ~9%**. Nếu dùng để báo cho **từng người dùng** "khoảng bao lâu", trung vị lại là con số trung thực hơn.
- R² chỉ ~0.35 vì nhiễu lognormal là **không thể giảm**: model đúng hoàn toàn cũng chỉ đạt quanh mức này.

**Quantile model cho SLA.** HGB với `loss="quantile", quantile=0.9`: **86.8%** latency test nằm dưới dự đoán (mục tiêu 90%; thiếu một chút, cần hiệu chỉnh trên validation). Pinball loss ($\tau = 0.9$) là 0.219, so với 0.371 nếu dùng dự đoán trung bình làm "p90". Muốn nói "90% request xong trong X giây" thì phải train cho phân vị 0.9, không lấy trung bình.

#### P3.4 Chọn metric

| Câu hỏi nghiệp vụ | Metric / loss |
|---|---|
| Sai số lớn đặc biệt đắt (quá tải hệ thống, phạt hợp đồng) | RMSE / MSE |
| Mọi đơn vị sai số như nhau; dữ liệu có outlier | MAE (loss tuyệt đối hoặc Huber) |
| Sai số tương đối quan trọng; $y$ trải nhiều bậc độ lớn | RMSLE, hoặc train trên $\log y$; MAPE nếu $y$ luôn xa 0 |
| Cần **tổng** chính xác (ngân sách, capacity) | Loss bình phương; nếu train trên log thì smearing; kiểm tra lệch tổng |
| Cần cận trên/cận dưới (SLA p90, tồn kho) | Pinball loss ở $\tau$ tương ứng; báo **coverage** |
| So sánh nhiều model trên **cùng** tập test | R² hoặc RMSE; **không** so R² giữa hai tập dữ liệu khác nhau |

#### P3.5 Code chuẩn

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error, r2_score,
                             mean_absolute_percentage_error, mean_pinball_loss)

def regression_report(y, p, tau=None):
    out = {"MAE": mean_absolute_error(y, p), "RMSE": root_mean_squared_error(y, p),
           "R2": r2_score(y, p), "MAPE": mean_absolute_percentage_error(y, p),
           "sum_bias": (p.sum() - y.sum()) / y.sum()}                  # lệch tổng, tương đối
    if tau is not None:
        out[f"pinball@{tau}"] = mean_pinball_loss(y, p, alpha=tau)
        out["coverage"] = np.mean(y <= p)                               # phải gần tau
    return out

# train trên log y, kèm smearing (Duan 1983)
lin = LinearRegression().fit(X_tr, np.log(y_tr))
smear = np.mean(np.exp(np.log(y_tr) - lin.predict(X_tr)))
pred_mean = np.exp(lin.predict(X_te)) * smear          # ước lượng trung bình
pred_median = np.exp(lin.predict(X_te))                # ước lượng trung vị

# model phân vị cho SLA
q90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.9).fit(X_tr, y_tr)
```

#### P3.6 Cạm bẫy

1. **Báo cáo một metric duy nhất:** A và B ở Tầng 1 đổi thứ hạng tùy metric. Báo MAE **và** RMSE, kèm lệch tổng nếu tổng có ý nghĩa.
2. **So R² giữa hai tập test khác nhau:** R² phụ thuộc phương sai của $y$ trong tập đó. Tập test "dễ" có phương sai lớn cho R² cao hơn với cùng sai số.
3. **MAPE khi $y$ có thể bằng 0 hoặc rất nhỏ:** bùng nổ hoặc không xác định. Dùng MAE/RMSLE, hoặc WAPE $= \sum\lvert e\rvert / \sum\lvert y\rvert$.
4. **Train trên $\log y$ rồi báo cáo tổng mà không smearing:** thiếu ~9% ở ví dụ trên.
5. **Dùng `r2_score` trên tập train để chọn model:** luôn tăng khi thêm đặc trưng. Dùng CV, hoặc ít nhất adjusted R².
6. **Lấy dự đoán trung bình làm "p90":** coverage thấp hơn nhiều so với 90%.

**Sắc thái phỏng vấn:**
- *"MAE hay RMSE?"* Không có đáp án chung. Nói rõ: RMSE ↔ ước lượng trung bình, phạt nặng sai số lớn, hợp khi nhiễu gần Gauss; MAE ↔ ước lượng trung vị, bền với outlier. Rồi hỏi lại: sai số lớn có đắt hơn tỉ lệ không? Có cần tổng chính xác không?
- *"R² = 0.35 thì model có tệ không?"* Chưa chắc. So với trần đạt được (nhiễu không thể giảm), so với baseline, và xem sai số tính bằng **đơn vị nghiệp vụ** (MAE 0.7 s có chấp nhận được không?).

#### P3.7 Đọc thêm

- R. Hyndman & A. Koehler, *Another look at measures of forecast accuracy*, International Journal of Forecasting 22 (2006).
- R. Koenker & G. Bassett, *Regression Quantiles*, Econometrica 46 (1978).
- N. Duan, *Smearing Estimate: A Nonparametric Retransformation Method*, JASA 78 (1983).
- P. Huber, *Robust Estimation of a Location Parameter*, Annals of Mathematical Statistics 35 (1964).
- scikit-learn User Guide: *Regression metrics*; ví dụ *Prediction Intervals for Gradient Boosting Regression*.

---

## Quiz phụ lục Tuần 3

> ⏱ **30 phút** · 10 câu nhanh (1đ) + 2 câu tự luận (5đ) · **20 điểm** · Đạt khi ≥ 14
> Làm trên giấy, được dùng máy tính cầm tay. **Không có câu chứng minh.**

**Phần 1 — Câu nhanh**

1. Mã hóa chu kỳ giờ: tính khoảng cách Euclid giữa 22h và 2h.
2. Target encoding với prior 0.1, $m = 8$: một giá trị xuất hiện 2 lần, cả 2 đều dương. Giá trị mã hóa là bao nhiêu?
3. Vì sao log của `latency_s` làm logistic regression **kém đi** trên dữ liệu project?
4. Kể 2 phép biến đổi giúp logistic regression nhưng gần như vô ích với HGB. Vì sao?
5. Dữ liệu thật có 2% dương. Model được train sau khi undersample về 1:1, và cho ra 0.6 với một mẫu. Xác suất đã hiệu chỉnh là bao nhiêu?
6. SMOTE: $x_i = (2, 4)$, hàng xóm $x_j = (6, 0)$, $u = 0.5$. Điểm mới là gì?
7. Một bạn oversample toàn bộ dữ liệu rồi mới chạy 5-fold CV với Random Forest, được AUC 0.999. Giải thích con số đó.
8. Sai số $e = (2, -2, 2, -2, 10)$. Tính MAE và RMSE.
9. Với dữ liệu $(1, 3, 4, 10, 50)$, hằng số nào tối ưu MAE, hằng số nào tối ưu MSE?
10. Model luôn dự đoán **trung bình của tập train** cho mọi điểm test. R² trên tập test bằng 0, dương hay âm? Vì sao?

**Phần 2 — Tự luận**

**T1 · Mất cân bằng trong phát hiện gian lận (5đ).** Ví điện tử có 0.5% giao dịch gian lận. Một đồng nghiệp train model trên dữ liệu đã undersample về 1:1 và đề xuất "chặn nếu score ≥ 0.5". Chi phí: chặn nhầm một giao dịch sạch $c_{FP} = 1$; bỏ lọt một giao dịch gian lận $c_{FN} = 50$.
- (a) Đổi các score 0.3, 0.6, 0.9 của model sang xác suất đúng.
- (b) Tính ngưỡng tối ưu theo chi phí trên **xác suất đúng**. Trong ba giao dịch trên, giao dịch nào bị chặn? So với đề xuất "≥ 0.5".
- (c) Đề xuất metric báo cáo và quy trình đánh giá (chia dữ liệu, resample ở đâu).

**T2 · Chọn model hồi quy (5đ).** Dự đoán thời gian xử lý (giây) cho 5 job: $y = (1.0,\ 1.5,\ 2.0,\ 3.0,\ 8.0)$.
- Model A dự đoán $(1.2,\ 1.4,\ 2.1,\ 2.8,\ 4.0)$.
- Model B dự đoán $(1.8,\ 2.2,\ 2.6,\ 3.6,\ 6.5)$.
- (a) Tính MAE, RMSE, R² và MAPE của mỗi model.
- (b) Đội hạ tầng cần ước lượng **tổng** thời gian để đặt trước tài nguyên. Chọn model nào? Tính lệch tổng của mỗi model.
- (c) Một model khác train trên $\log y$, residual trên thang log có độ lệch chuẩn 0.5. Hệ số smearing (theo giả định lognormal) là bao nhiêu, và nó sửa được vấn đề gì?

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Phần 1**

1. Lệch 4 giờ: $d = 2\sin(\pi \cdot 4/24) = 2\sin(30°) =$ **1.0**.
2. $(2 + 8 \times 0.1)/(2 + 8) = 2.8/10 =$ **0.28**.
3. Tác động của latency là một **ngưỡng ở 5 s nằm ở đuôi** phân phối; log **nén** đuôi, nên khoảng 4–10 s gần nhau hơn và LR khó tách. (Đo được: giảm 0.0086 AUC trên 15/15 fold.)
4. Ví dụ: log/Yeo-Johnson, scaling, binning, splines (bất kỳ 2 cái). Cây chỉ dùng **thứ tự** giá trị để chia, nên biến đổi đơn điệu không đổi các phép chia có thể có; ngưỡng thì cây tự học. (Cross feature giúp cây ít hơn LR nhiều, nhưng không phải vô ích hoàn toàn.)
5. odds $= 1.5 \times (0.02/0.98) = 0.0306$; $p = 0.0306/1.0306 =$ **0.0297**.
6. $(2, 4) + 0.5 \times (4, -4) =$ **$(4, 2)$**.
7. Bản sao giống hệt của cùng một mẫu dương nằm ở **cả** fold train lẫn fold test; Random Forest "nhớ" các điểm đó, nên AUC giả. Đây là rò rỉ; cách đúng là chỉ resample phần train của mỗi fold (đo được: 1.000 so với 0.910).
8. MAE $= 18/5 =$ **3.6**; MSE $= (4 \times 4 + 100)/5 = 23.2$; RMSE $=$ **4.82**.
9. MAE: **trung vị = 4**; MSE: **trung bình = 13.6**.
10. **Thường âm** (hoặc bằng 0 nếu trung bình train trùng trung bình test). R² của scikit-learn so với trung bình của **tập test**; hằng số đó là hằng số tốt nhất cho MSE trên test, nên mọi hằng số khác đều cho SSE lớn hơn, tức $R^2 \le 0$.

**T1 (5đ)**

(a) 2đ. Hệ số đổi odds $= (0.005/0.995)/(0.5/0.5) = 0.005025$.
- 0.3: odds $0.4286 \times 0.005025 = 0.00215$, nên $p \approx$ **0.0022**.
- 0.6: odds $1.5 \times 0.005025 = 0.00754$, nên $p \approx$ **0.0075**.
- 0.9: odds $9 \times 0.005025 = 0.0452$, nên $p \approx$ **0.0433**.

(b) 1.5đ. $t^\star = 1/(1 + 50) =$ **0.0196**. Chỉ giao dịch có score 0.9 ($p = 0.043$) bị chặn. Đề xuất "≥ 0.5" chặn cả 0.6 và 0.9. Kết quả trùng nhau ở **ba điểm này**, nhưng đó là tình cờ: ngưỡng 0.5 trên model đã resample tương ứng với $p = 0.005$ trên thang thật, thấp hơn ngưỡng chi phí ~4 lần, nên nhìn chung sẽ chặn **quá nhiều**. (Muốn ra quyết định đúng, đổi $t^\star = 0.0196$ ngược về thang của model: odds $0.02/0.005025 \approx 3.98$, tức score $\approx 0.80$.)

(c) 1.5đ. Chia theo **thời gian** (train quá khứ, test tương lai), stratify khi chia fold; resample (nếu dùng) **chỉ** trên phần train của mỗi fold; validation/test giữ tỉ lệ thật. Metric: AP/PR-AUC, recall ở mức precision vận hành chấp nhận được, chi phí kỳ vọng ở ngưỡng chọn; **không** dùng accuracy.

**T2 (5đ)**

(a) 2.5đ. $\bar y = 3.1$, SST $= 32.2$.
- A: $e = (-0.2, 0.1, -0.1, 0.2, 4.0)$. MAE $= 4.6/5 =$ **0.92**; SSE $= 16.1$, RMSE $= \sqrt{3.22} =$ **1.79**; $R^2 = 1 - 16.1/32.2 =$ **0.50**; MAPE $=$ **17.7%**.
- B: $e = (-0.8, -0.7, -0.6, -0.6, 1.5)$. MAE $= 4.2/5 =$ **0.84**; SSE $= 4.1$, RMSE $=$ **0.906**; $R^2 =$ **0.873**; MAPE $=$ **39.1%**.
- MAE, RMSE, R² đều chọn B; MAPE chọn A, vì A rất chính xác ở các job **nhỏ** (MAPE chia cho $y$ nhỏ).

(b) 1.5đ. Tổng thật 15.5 s. A dự đoán 11.5 s (**−26%**), B dự đoán 16.7 s (**+7.7%**). Chọn **B**: thiếu tài nguyên (A) nguy hiểm hơn dư một chút; A bỏ sót gần hết job dài.

(c) 1đ. $e^{0.5^2/2} = e^{0.125} =$ **1.133**. Nhân dự đoán $\exp(\cdot)$ với 1.133 để chuyển từ ước lượng **trung vị** sang ước lượng **trung bình**, nên **tổng** không còn bị thấp có hệ thống (~12% ở đây).

</details>
