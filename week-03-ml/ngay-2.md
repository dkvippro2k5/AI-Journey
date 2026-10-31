# Ngày 2 · Model tuyến tính, cây quyết định và ensemble

> **Loại tài liệu: Lý thuyết (bản nhẹ).** Công thức đầy đủ, ví dụ tính tay; chứng minh chỉ phác thảo.
> **Hôm nay gồm 3 bài:**
> - **05** Linear & logistic regression trong thực tế
> - **06** Cây quyết định
> - **07** Ensemble: Random Forest và Gradient Boosting
>
> **Thời gian:** đọc ~2h30 · quiz 45' · bài tập ~2h
> **Liên hệ Tuần 2:** phần toán của linear/logistic regression (gradient, MLE, L1/L2) đã học ở Tuần 2. Hôm nay tập trung vào **dùng chúng đúng cách** và vào họ model mạnh nhất cho dữ liệu dạng bảng: **cây và ensemble của cây**.

---

## 05 · Linear & logistic regression trong thực tế

> ⏱ Tầng 1: 10' · Tầng 2: 30'
> **Cần biết trước:** Tuần 2, bài 05 (gradient $p - y$) và bài 10 (ridge, lasso, MAP).

> **📖 Từ mới**
> - **Hệ số (coefficient) $\beta_j$:** mức thay đổi của đầu ra tuyến tính khi đặc trưng $j$ tăng 1 đơn vị, các đặc trưng khác giữ nguyên.
> - **Odds:** $p/(1 - p)$. Ví dụ $p = 0.2$ thì odds bằng $0.25$, tức "1 ăn 4".
> - **Odds ratio:** odds được nhân lên bao nhiêu lần khi đặc trưng tăng 1 đơn vị.
> - **One-hot encoding:** biến cột phân loại (ví dụ `intent`) thành nhiều cột 0/1, mỗi giá trị một cột.
> - **`C`:** tham số regularization của scikit-learn, **bằng nghịch đảo** của độ mạnh phạt: $C$ nhỏ là phạt mạnh.
> - **Class weight:** cho mẫu thuộc lớp hiếm trọng số lớn hơn trong loss.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Logistic regression giống một **bảng cộng điểm rủi ro**: mỗi yếu tố cộng hoặc trừ một số điểm cố định, rồi tổng điểm được đổi sang xác suất bằng sigmoid. Ví dụ: "intent là tech_support: +0.8 điểm", "model small: +0.7 điểm", "retrieval score cao hơn 1 độ lệch chuẩn: −0.5 điểm".

Ưu điểm: dễ giải thích, nhanh, ổn định. Nhược điểm: **mỗi yếu tố cộng điểm độc lập** với các yếu tố khác. Model không tự hiểu được "small **và** tech_support thì đặc biệt tệ" (tương tác), cũng không hiểu "latency vượt quá 5 giây thì tệ hẳn" (ngưỡng), trừ khi ta tự tạo đặc trưng cho các điều đó.

**Ba ý cốt lõi:**
1. Phải chuẩn hóa đặc trưng trước khi regularize; khi đó hệ số mới so sánh được với nhau.
2. Hệ số của logistic regression là **log odds ratio**.
3. Model tuyến tính là **baseline bắt buộc**: nhanh, khó overfit, dễ giải thích. Nếu model phức tạp không thắng được nó rõ ràng thì giữ nó.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 5.1 Thuật ngữ và các estimator

| Estimator (scikit-learn) | Bài toán | Loss + phạt | Ghi chú |
|---|---|---|---|
| `LinearRegression` | Hồi quy | MSE | Không regularize |
| `Ridge(alpha)` | Hồi quy | MSE $+\ \alpha\|\beta\|_2^2$ | Ổn định khi đặc trưng tương quan |
| `Lasso(alpha)` | Hồi quy | $\frac{1}{2n}$MSE $+\ \alpha\|\beta\|_1$ | Cho nghiệm thưa (Tuần 2, Mệnh đề 10.6) |
| `ElasticNet(alpha, l1_ratio)` | Hồi quy | Trộn L1 và L2 | Thưa nhưng ổn định hơn lasso khi các đặc trưng tương quan (Zou & Hastie 2005) |
| `LogisticRegression(C)` | Phân loại | BCE/CE $+\ \frac{1}{2C}\|\beta\|_2^2$ (mặc định) | **Mặc định đã có L2**, với $C = 1$ |

**Đa lớp:**
- **Multinomial (softmax)**: một model, loss là cross-entropy. Đây là mặc định của `LogisticRegression` cho bài đa lớp ở các bản scikit-learn mới.
- **One-vs-Rest (OvR)**: $K$ model nhị phân, mỗi model "lớp $k$ hay không phải lớp $k$"; lấy lớp có điểm cao nhất.

#### 5.2 Đọc hệ số của logistic regression

$\log\dfrac{p}{1 - p} = \beta_0 + \sum_j \beta_j x_j$, nên tăng $x_j$ thêm 1 đơn vị thì **odds nhân với $e^{\beta_j}$**.

**Ví dụ tính tay.** Xác suất nền $p = 0.2$, tức odds $= 0.25$. Một đặc trưng có $\beta = \ln 2 \approx 0.693$ (odds ratio 2). Bật đặc trưng này lên: odds $= 0.5$, nên $p = 0.5/1.5 =$ **0.333**. Odds nhân đôi **không** có nghĩa là xác suất nhân đôi.

**Trên dữ liệu project** (đo trên máy bạn; đặc trưng số đã chuẩn hóa, nên hệ số của chúng ứng với "tăng 1 độ lệch chuẩn"):

| Đặc trưng | $\beta$ | Odds ratio |
|---|---|---|
| `intent = tech_support` | +0.805 | 2.24 |
| `model = small` | +0.681 | 1.98 |
| `latency_s` (+1 độ lệch chuẩn) | +0.541 | 1.72 |
| `retrieval_score` (+1 độ lệch chuẩn) | −0.518 | 0.60 |
| `n_turns` (+1 độ lệch chuẩn) | +0.499 | 1.65 |

Model tuyến tính tìm đúng **hướng** của từng yếu tố. Nhưng nó không biết hai điều: "small **chỉ** tệ khi đi với tech_support", và "latency chỉ tệ khi **vượt quá 5 giây**".

#### 5.3 Mất cân bằng lớp: `class_weight`

`class_weight="balanced"` gán trọng số $w_c = \dfrac{n}{K \cdot n_c}$ cho mỗi lớp $c$. Với 15.5% mẫu dương: $w_1 = 1/(2 \times 0.155) = 3.23$ và $w_0 = 1/(2 \times 0.845) = 0.59$.

Đo trên máy bạn (CV 5-fold):

| | AUC | Recall tại ngưỡng 0.5 | Precision tại ngưỡng 0.5 | Xác suất dự đoán trung bình |
|---|---|---|---|---|
| Không dùng class weight | 0.769 | 0.165 | 0.646 | 0.154 (khớp tỉ lệ thật) |
| `balanced` | 0.771 | **0.683** | 0.313 | **0.422** |

Bài học: class weight **gần như không đổi AUC** (khả năng xếp hạng vẫn vậy). Thứ nó thay đổi là **thang xác suất**: mọi xác suất bị đẩy lên, nên với cùng ngưỡng 0.5 thì recall tăng vọt. Đổi ngưỡng (bài 04) cũng đạt được hiệu ứng gần tương tự mà **giữ được** xác suất đã hiệu chỉnh. Nếu bạn cần xác suất có ý nghĩa (ví dụ để tính chi phí kỳ vọng), đừng dùng class weight.

#### 5.4 Giúp model tuyến tính bắt quan hệ phi tuyến

| Kỹ thuật | Ý tưởng | Đo trên dữ liệu project (AUC CV) |
|---|---|---|
| Logistic regression thường | — | 0.769 |
| + `SplineTransformer` cho đặc trưng số | Mỗi đặc trưng số thành một đường cong trơn ghép từ nhiều đoạn, nên bắt được hiệu ứng ngưỡng | **0.792** |
| + tương tác bậc 2 cho mọi cặp (chưa tune `C`) | Thêm $x_ix_j$ | 0.743: **tệ hơn**, vì số đặc trưng tăng vọt và overfit |
| Tham khảo: Random Forest / HistGradientBoosting | Tự học ngưỡng và tương tác | 0.807 / 0.812 |

Thêm đặc trưng có chọn lọc thì giúp được; thêm hàng loạt mà không tune regularization thì phản tác dụng.

#### 5.5 Code chuẩn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logreg = Pipeline([("pre", build_preprocessor("linear")),       # imputer + StandardScaler + one-hot (bài 10)
                   ("clf", LogisticRegression(C=1.0, max_iter=2000))])
logreg.fit(X_tr, y_tr)
coef = dict(zip(logreg["pre"].get_feature_names_out(), logreg["clf"].coef_[0]))
```

#### 5.6 Cạm bẫy

1. **Không chuẩn hóa** trước khi regularize; khi đó so sánh độ lớn các hệ số là vô nghĩa.
2. **Đọc hệ số như quan hệ nhân quả.** Hệ số chỉ cho biết **tương quan khi giữ nguyên các đặc trưng khác**. Nếu hai đặc trưng tương quan mạnh với nhau, hệ số của chúng có thể đổi dấu khi bỏ bớt một cột.
3. **Bẫy biến giả (dummy trap)** khi **không** regularize: one-hot đủ $K$ cột cộng với intercept thì thừa một bậc tự do, làm $X^\top X$ suy biến (Tuần 2, bài 02). Cách xử lý: `OneHotEncoder(drop="first")`, hoặc giữ L2 (mặc định).
4. **`ConvergenceWarning`**: tăng `max_iter`, và **kiểm tra đã chuẩn hóa chưa** ($\kappa$ lớn, Tuần 2, bài 06).
5. **Dùng class weight rồi diễn giải `predict_proba` như xác suất thật.**

**Sắc thái hay bị hỏi khi phỏng vấn:** *"Hệ số logistic regression bằng 0.7 nghĩa là gì?"* Tăng 1 đơn vị (hoặc 1 độ lệch chuẩn, nếu đã chuẩn hóa) thì odds nhân với $e^{0.7} \approx 2$; xác suất tăng bao nhiêu còn tùy vào điểm xuất phát.

#### 5.7 Đọc thêm

- ISLR ch. 3–4 và 6.
- scikit-learn User Guide, *1.1 Linear Models*.

---

## 06 · Cây quyết định

> ⏱ Tầng 1: 10' · Tầng 2: 35'
> **Cần biết trước:** entropy (Tuần 2, bài 11).

> **📖 Từ mới**
> - **Cây quyết định:** chuỗi câu hỏi có/không, ví dụ "latency > 5 s?", dẫn tới một **lá** chứa dự đoán.
> - **Nút gốc / nút trong / lá:** câu hỏi đầu tiên / các câu hỏi tiếp theo / điểm kết thúc.
> - **Độ không thuần (impurity):** mức độ "lẫn lộn" các lớp trong một nút. Nút chỉ chứa một lớp là **thuần**.
> - **Gini impurity, entropy:** hai cách đo độ không thuần.
> - **Information gain:** độ không thuần **giảm** được bao nhiêu nhờ một phép chia.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Trò chơi "20 câu hỏi": người chơi giỏi luôn hỏi câu **chia đôi khả năng** tốt nhất. Cây quyết định làm đúng như vậy: ở mỗi bước, thử mọi câu hỏi có thể ("đặc trưng $j$ ≤ ngưỡng $t$?") và chọn câu hỏi làm hai nhóm con **thuần nhất**.

**Ba ý cốt lõi:**
1. Cây học **ngưỡng** và **tương tác** một cách tự nhiên: "nếu tech_support thì hỏi tiếp về model".
2. Không cần chuẩn hóa đặc trưng: chỉ có **thứ tự** giá trị là quan trọng.
3. Cây sâu không giới hạn sẽ học thuộc dữ liệu (AUC train bằng 1.0, AUC test 0.60; bài 02). Một cây đơn lẻ **variance rất cao**.

---

### TẦNG 2 · Kiến thức chuẩn

#### 6.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **CART** | Cây nhị phân, chia **tham lam** (greedy) từng bước theo tiêu chí impurity; lá dự đoán theo lớp chiếm đa số (phân loại) hoặc giá trị trung bình (hồi quy). scikit-learn cài đặt CART | Breiman et al. (1984) |
| **Gini impurity** | $G = 1 - \sum_k p_k^2$: xác suất phân loại sai nếu gán nhãn ngẫu nhiên theo đúng tỉ lệ các lớp trong nút | — |
| **Entropy** | $H = -\sum_k p_k\log_2 p_k$ | ID3, Quinlan (1986) |
| **Impurity decrease (gain)** | $\Delta = I(\text{cha}) - \dfrac{n_L}{n}I(L) - \dfrac{n_R}{n}I(R)$ | — |
| **Tiêu chí của cây hồi quy** | Giảm MSE: lá dự đoán giá trị **trung bình** của các mẫu trong lá | — |
| **Hyperparameter chính** | `max_depth`, `min_samples_leaf`, `max_leaf_nodes`, `ccp_alpha` (cost-complexity pruning) | scikit-learn |

**Thuật toán tìm phép chia:** với mỗi đặc trưng $j$, sắp xếp các giá trị; mỗi ứng viên ngưỡng là **trung điểm** của hai giá trị phân biệt liên tiếp; chọn cặp $(j, t)$ có gain lớn nhất. Chi phí khoảng $O(d \cdot n\log n)$ cho mỗi nút. Cách tìm tham lam này **không** đảm bảo ra cây tối ưu toàn cục; tìm cây tối ưu là bài toán NP-khó (Hyafil & Rivest 1976).

#### 6.2 Ví dụ tính tay

10 lượt trả lời, chỉ dùng một đặc trưng `latency_s`:

| latency | 1.2 | 2.0 | 2.5 | 3.1 | 4.0 | 4.8 | 5.5 | 6.2 | 7.0 | 8.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| bad | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 1 |

**Nút gốc:** 4 dương / 10, nên $G = 1 - 0.4^2 - 0.6^2 = 0.48$ và $H = 0.971$ bit.

**Hai ngưỡng ứng viên:**

| Ngưỡng | Trái | Phải | Gini trái / phải | Gini có trọng số | **Gain (Gini)** | Entropy có trọng số | **Gain (entropy)** |
|---|---|---|---|---|---|---|---|
| 2.80 | 3 mẫu, 0 dương | 7 mẫu, 4 dương | $0$ / $24/49 = 0.490$ | $0.7 \times 0.490 = 0.343$ | 0.137 | $0.7 \times 0.985 = 0.690$ | **0.281** |
| 5.15 | 6 mẫu, 1 dương | 4 mẫu, 3 dương | $10/36 = 0.278$ / $0.375$ | $0.6 \times 0.278 + 0.4 \times 0.375 = 0.317$ | **0.163** | $0.6 \times 0.650 + 0.4 \times 0.811 = 0.715$ | 0.256 |

Đã quét **cả 9 ngưỡng** (chạy kiểm chứng): **Gini chọn 5.15**, còn **entropy chọn 2.80**.
- Entropy (dùng log) "thưởng" nhiều hơn cho việc tạo ra một nút **hoàn toàn thuần**, như nhóm trái ở ngưỡng 2.80.
- Trong thực tế, hai tiêu chí cho cây có chất lượng gần như nhau. Gini rẻ hơn vì không cần tính log, nên là mặc định của scikit-learn.

#### 6.3 Các con số cần biết

Đo trên dữ liệu project (bảng đầy đủ ở bài 02): `max_depth` bằng 6 cho AUC test 0.767; cây không giới hạn cho AUC train 1.000 nhưng AUC test chỉ 0.604. **Một cây đơn lẻ luôn thua** logistic regression (0.769) hoặc random forest (0.807). Cây chủ yếu đáng giá khi được dùng làm **viên gạch** cho ensemble (bài 07).

| Tính chất | Cây quyết định |
|---|---|
| Cần chuẩn hóa? | Không |
| Bắt được tương tác và ngưỡng? | Có, một cách tự nhiên |
| Ranh giới quyết định | Gồm các đoạn **song song với trục**: một đường chéo phải xấp xỉ bằng nhiều bậc thang |
| Ngoại suy | **Không**: cây hồi quy dự đoán là **hằng số** ở ngoài vùng dữ liệu đã thấy |
| Độ ổn định | Thấp: dữ liệu đổi một chút có thể làm thay đổi cả cây |

#### 6.4 Code chuẩn

```python
from sklearn.tree import DecisionTreeClassifier, export_text

tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=0).fit(X_num, y)
print(export_text(tree, feature_names=list(X_num.columns)))     # in các luật dạng văn bản
```

#### 6.5 Cạm bẫy

1. **Để cây sâu không giới hạn** (đây là mặc định của scikit-learn): overfit gần như chắc chắn.
2. **Dùng cây hồi quy để ngoại suy** (dự báo xu hướng tăng): cây chỉ trả lại giá trị của lá cuối cùng.
3. **Mã hóa số thứ tự cho biến phân loại** (faq = 0, billing = 1, …) với model tuyến tính là sai, vì nó áp đặt một thứ tự giả. Với **cây** thì cách này đôi khi chấp nhận được (`OrdinalEncoder`), vì cây có thể cắt nhiều lần.
4. **Diễn giải một cây sâu** như một bộ luật dễ hiểu: cây 20 tầng không còn dễ hiểu nữa.

**Sắc thái hay bị hỏi khi phỏng vấn:** *"Gini hay entropy?"* Kết quả gần như nhau, Gini nhanh hơn một chút. Câu hỏi đáng giá hơn là: *"Làm sao chống overfit cho cây?"* Giới hạn độ sâu và số mẫu mỗi lá, pruning, hoặc **dùng ensemble**.

#### 6.6 Đọc thêm

- ISLR ch. 8; ESL §9.2.
- L. Breiman, J. Friedman, R. Olshen, C. Stone, *Classification and Regression Trees*, 1984.

---

## 07 · Ensemble: Random Forest và Gradient Boosting

> ⏱ Tầng 1: 15' · Tầng 2: 45'
> **Cần biết trước:** cây (bài 06); phương sai của trung bình (Tuần 2, bài 09); gradient descent (Tuần 2, bài 06).

> **📖 Từ mới**
> - **Ensemble:** kết hợp nhiều model để ra một dự đoán tốt hơn từng model riêng lẻ.
> - **Bagging (bootstrap aggregating):** huấn luyện nhiều model trên các mẫu bootstrap (lấy mẫu có hoàn lại) rồi lấy trung bình.
> - **Random Forest:** bagging các cây, và ở mỗi lần chia chỉ xét **một tập con ngẫu nhiên** các đặc trưng.
> - **Boosting:** huấn luyện các model **nối tiếp nhau**; model sau sửa lỗi của tổng các model trước.
> - **Out-of-bag (OOB):** những mẫu không được bốc vào mẫu bootstrap của một cây; dùng được làm validation "miễn phí".
> - **Feature importance:** mức độ model dựa vào mỗi đặc trưng.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- **Bagging:** hỏi 300 người đoán cân nặng của một con bò rồi lấy trung bình. Mỗi người đoán sai khá nhiều, nhưng các sai số **ngẫu nhiên và khác nhau** thì triệt tiêu một phần lẫn nhau. Điều kiện quan trọng: các ý kiến phải **độc lập**; nếu cả 300 người đều nghe một "chuyên gia" thì trung bình chẳng tốt hơn bao nhiêu.
- **Boosting:** một học sinh làm đề, giáo viên chỉ ra những câu sai, học sinh **chỉ ôn lại các câu sai**, rồi làm lại; cứ thế nhiều vòng. Mỗi vòng sửa một ít, và cộng dồn lại.

**Ba ý cốt lõi:**
1. Random Forest **giảm variance** bằng cách lấy trung bình nhiều cây sâu, ít tương quan với nhau.
2. Gradient Boosting **giảm bias** bằng cách cộng dần nhiều cây nông, mỗi cây khớp **phần dư**. Đây chính là gradient descent, nhưng trong không gian các hàm.
3. Với dữ liệu dạng bảng, gradient boosting (XGBoost, LightGBM, HistGradientBoosting) thường là model mạnh nhất (Grinsztajn et al., NeurIPS 2022).

---

### TẦNG 2 · Kiến thức chuẩn

#### 7.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Bagging** | Trung bình của $B$ model huấn luyện trên $B$ mẫu bootstrap | Breiman (1996) |
| **Random Forest** | Bagging các cây, mỗi lần chia chỉ xét `max_features` đặc trưng ngẫu nhiên (mặc định $\sqrt d$ cho phân loại) để **giảm tương quan** giữa các cây | Breiman, *Machine Learning* 2001 |
| **Gradient Boosting Machine** | $F_m(x) = F_{m-1}(x) + \eta\,h_m(x)$, với $h_m$ là cây khớp **gradient âm** của loss tại $F_{m-1}$ | Friedman, *Annals of Statistics* 2001 |
| **Learning rate (shrinkage) $\eta$** | Co mỗi cây lại; $\eta$ nhỏ cần nhiều cây hơn nhưng thường tổng quát hóa tốt hơn | Friedman (2001) |
| **XGBoost / LightGBM / CatBoost** | Các bản cài đặt GBM tối ưu: dùng đạo hàm bậc 2, regularize lá, histogram, xử lý biến phân loại | Chen & Guestrin (KDD 2016); Ke et al. (NeurIPS 2017); Prokhorenkova et al. (NeurIPS 2018) |
| **HistGradientBoosting** | Bản của scikit-learn theo kiểu LightGBM: chia giá trị thành tối đa **255 bin**, **tự xử lý `NaN`**, rất nhanh khi $n$ lớn | scikit-learn User Guide §1.11 |
| **Impurity-based importance** | Tổng mức giảm impurity mà mỗi đặc trưng đóng góp qua mọi cây | Breiman (2001) |
| **Permutation importance** | Metric giảm bao nhiêu khi **xáo trộn** cột đó trên tập validation | Breiman (2001); Fisher et al. (2019) |
| **SHAP** | Phân bổ dự đoán của **từng mẫu** cho từng đặc trưng theo giá trị Shapley | Lundberg & Lee, NeurIPS 2017 |

#### 7.2 Vì sao Random Forest hoạt động

$B$ cây, mỗi cây có variance $\sigma^2$, tương quan đôi một là $\rho$. Variance của trung bình bằng:

$$\mathrm{Var}\Big(\frac{1}{B}\sum_b T_b\Big) = \rho\,\sigma^2 + \frac{1 - \rho}{B}\,\sigma^2 \quad\text{(ESL §15.2)}.$$

| $\rho$ | $B = 1$ | $B = 10$ | $B = 100$ |
|---|---|---|---|
| 0 | $1.00\,\sigma^2$ | $0.10\,\sigma^2$ | $0.01\,\sigma^2$ |
| 0.3 | $1.00\,\sigma^2$ | $0.37\,\sigma^2$ | $0.31\,\sigma^2$ |
| 0.6 | $1.00\,\sigma^2$ | $0.64\,\sigma^2$ | $0.60\,\sigma^2$ |

Đọc bảng:
- Tăng $B$ chỉ đẩy variance về **sàn $\rho\sigma^2$**, không xuống dưới được. Muốn hạ sàn thì phải **giảm tương quan** giữa các cây. Đó là việc của `max_features`.
- Tăng $B$ **không gây overfit**; nó chỉ tốn thời gian. Đo trên dữ liệu project, AUC CV theo số cây là: 1 cây 0.685, 10 cây 0.788, 100 cây 0.807, 300 cây 0.807 (đã bão hòa).

**OOB:** xác suất một mẫu **không** được bốc vào một mẫu bootstrap cỡ $n$ là $(1 - 1/n)^n \to e^{-1} = 0.368$. Mỗi cây chừa ra khoảng 36.8% dữ liệu, dùng được để đánh giá (`oob_score=True`).

#### 7.3 Vì sao Gradient Boosting hoạt động

Với loss bình phương $\tfrac{1}{2}(y - F)^2$, gradient âm theo $F$ chính là **phần dư** $y - F$. Mỗi vòng:
1. Tính $r_i = y_i - F_{m-1}(x_i)$.
2. Khớp một cây $h_m$ vào $(x_i, r_i)$.
3. Cập nhật $F_m = F_{m-1} + \eta\,h_m$.

Đây là gradient descent (Tuần 2, bài 06) với "tham số" là **chính hàm $F$**. Với loss khác (log-loss cho phân loại), chỉ cần thay phần dư bằng gradient âm tương ứng: $y - \sigma(F)$, tức lại là "nhãn trừ dự đoán" (Tuần 2, Mệnh đề 5.3).

**Ví dụ tính tay** (đây cũng là test của bài tập). Dữ liệu $x = 1, \dots, 6$ và $y = (1.0,\ 1.2,\ 1.1,\ 3.0,\ 3.2,\ 3.1)$; dùng cây 1 tầng (stump), $\eta = 0.5$:

| Vòng | Phần dư $r$ | Stump học được | $F$ sau vòng (nhóm $x \le 3.5$ / nhóm $x > 3.5$) | MSE train |
|---|---|---|---|---|
| 0 | — | $F_0 = \bar y = 2.1$ | 2.1 / 2.1 | 1.007 |
| 1 | $(-1.1, -0.9, -1.0, 0.9, 1.1, 1.0)$ | chia tại 3.5; lá $-1$ / $+1$ | $2.1 \mp 0.5 = 1.6$ / $2.6$ | 0.2567 |
| 2 | $(-0.6, -0.4, -0.5, 0.4, 0.6, 0.5)$ | lá $\mp 0.5$ | 1.35 / 2.85 | 0.0692 |
| 3 | $(-0.35, -0.15, -0.25, 0.15, 0.35, 0.25)$ | lá $\mp 0.25$ | 1.225 / 2.975 | 0.0223 |

Đọc bảng:
- Mỗi vòng, khoảng cách tới trung bình thật của mỗi nhóm (1.1 và 3.1) **giảm một nửa**, đúng bằng hệ số $(1 - \eta)$.
- MSE tiến về **sàn 0.0067**, là phương sai bên trong mỗi nhóm. Một stump chia tại 3.5 không làm tốt hơn được mức này.

**$\eta$ và số cây đánh đổi cho nhau** (đo trên máy bạn, hồi quy 3 chiều, sàn nhiễu 0.09):

| $\eta$ | Số cây | MSE CV |
|---|---|---|
| 1.0 | 50 | 0.196 |
| 0.1 | 50 | **0.122** |
| 0.1 | 500 | 0.125 |
| 0.01 | 500 | 0.123 |

$\eta = 1$ thì mỗi cây "sửa quá tay" và overfit. Khác với Random Forest, boosting **có thể overfit khi quá nhiều cây**; dùng **early stopping** (bài 11) để chọn số cây.

#### 7.4 Feature importance: đừng tin impurity importance

Đo trên máy bạn, với Random Forest trên dữ liệu project:

| Đặc trưng | Impurity importance (thứ hạng) | Permutation importance (giảm AUC trên test) |
|---|---|---|
| `retrieval_score` | 0.224 (hạng 1) | 0.063 |
| `latency_s` | 0.167 (hạng 2) | 0.053 |
| `answer_tokens` | 0.100 (hạng 4) | **0.005** |
| `hour` | 0.093 (hạng 5) | 0.021 |
| `query_tokens` | 0.079 (hạng 6) | **0.002** |
| `intent` | bị chia ra nhiều cột one-hot, mỗi cột nhỏ | **0.077** (hạng 1) |

**Nhận xét:**
- Impurity importance **thiên vị các đặc trưng số có nhiều giá trị phân biệt** (Strobl et al., BMC Bioinformatics 2007): có nhiều ngưỡng để thử thì dễ tìm được một phép chia "trông có ích" trên chính dữ liệu train. `query_tokens` thực ra **không** ảnh hưởng tới nhãn (xem `data.py`), nhưng vẫn được xếp hạng 6.
- Impurity importance được tính **trên dữ liệu train**.
- Permutation importance đo trên **validation/test** và đo trên **cột gốc**, nên đáng tin hơn.
- Nhược điểm của permutation importance: khi hai đặc trưng tương quan mạnh, xáo một cột thì model vẫn dựa được vào cột kia, nên **cả hai** đều trông kém quan trọng.

#### 7.5 Chọn model nào?

| | Random Forest | Gradient Boosting (HGB/LightGBM/XGBoost) |
|---|---|---|
| Giảm chủ yếu | Variance | Bias (và variance, nhờ shrinkage) |
| Cây | Sâu, độc lập, **huấn luyện song song được** | Nông, **nối tiếp nhau** |
| Mặc định đã tốt? | Rất tốt, khó làm hỏng | Cần tune $\eta$, số cây, kích thước cây |
| Overfit khi tăng số cây? | Không | Có, nên cần early stopping |
| Trên dữ liệu project (AUC CV) | 0.807 | 0.812; sau khi tune: 0.817 |

#### 7.6 Code chuẩn

```python
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0)
hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=7,
                                     min_samples_leaf=50, early_stopping="auto", random_state=0)
model = Pipeline([("pre", build_preprocessor("tree")), ("clf", hgb)]).fit(X_tr, y_tr)
imp = permutation_importance(model, X_val, y_val, scoring="roc_auc", n_repeats=10, random_state=0)
```

#### 7.7 Cạm bẫy

1. **Tin impurity importance** (mục 7.4).
2. **Tune Random Forest quá kỹ** trong khi nó đã gần tối ưu với mặc định; ngược lại, **để mặc định cho GBM** trong khi nó rất nhạy với $\eta$ và số cây.
3. **One-hot biến phân loại có rất nhiều giá trị** (hàng nghìn) cho model cây: tạo ra quá nhiều cột thưa. Dùng hỗ trợ biến phân loại có sẵn của HGB/LightGBM/CatBoost, hoặc target encoding (kèm CV để tránh rò rỉ).
4. **Xem SHAP hoặc importance là quan hệ nhân quả.** Chúng giải thích **model**, không giải thích **thế giới**.

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"Bagging và boosting khác nhau thế nào?"* Song song, giảm variance; so với nối tiếp, giảm bias.
- *"Vì sao Random Forest chọn ngẫu nhiên đặc trưng?"* Để giảm $\rho$ (mục 7.2).
- *"Gradient trong gradient boosting là gradient theo cái gì?"* Theo **dự đoán** $F(x_i)$, không theo tham số.

#### 7.8 Đọc thêm

- ESL ch. 10 (Boosting), ch. 15 (Random Forests); ISLR ch. 8.
- J. Friedman, *Greedy Function Approximation: A Gradient Boosting Machine*, Annals of Statistics 2001.
- T. Chen & C. Guestrin, *XGBoost*, KDD 2016.
- L. Grinsztajn, E. Oyallon, G. Varoquaux, *Why do tree-based models still outperform deep learning on tabular data?*, NeurIPS 2022.
- C. Strobl et al., *Bias in random forest variable importance measures*, BMC Bioinformatics 2007.

---

## Bài tập Ngày 2 · `quality/tree.py` (~2h)

Test: `python -m pytest -q tests/test_tree.py` (5 test).

| Thành phần | Yêu cầu |
|---|---|
| `gini(y)`, `entropy(y)` | Mảng rỗng thì trả về 0. `entropy` tính theo **bit**. Dùng được cho nhiều lớp |
| `best_split(x, y, criterion="gini")` | Trên **một** đặc trưng số: ứng viên là trung điểm của các cặp giá trị **phân biệt** liên tiếp; nhánh trái là `x <= t`; trả về `(t, gain)`. Hòa thì lấy $t$ nhỏ nhất. Không có ứng viên thì trả về `(nan, 0.0)` |
| `L2Boost(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)` | Gradient boosting cho loss bình phương (mục 7.3), mỗi vòng dùng `DecisionTreeRegressor(max_depth, random_state)` khớp phần dư. Thuộc tính sau khi `fit`: `init_` (trung bình $y$), `trees_`, `train_mse_` (MSE train sau mỗi vòng). Phương thức: `fit` (trả về `self`), `predict`, `staged_predict` (generator, yield dự đoán sau mỗi cây) |

Test sẽ kiểm tra `L2Boost` cho ra **đúng** dự đoán của `sklearn.ensemble.GradientBoostingRegressor` với cùng tham số (sai số $\le 10^{-8}$), và kiểm tra lại bảng tính tay ở mục 6.2 và 7.3.
