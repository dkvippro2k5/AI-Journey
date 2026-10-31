# Ngày 1 · Nền tảng ML: chia dữ liệu, overfitting, cross-validation, metric

> **Loại tài liệu: Lý thuyết (bản nhẹ).** Công thức đầy đủ và ví dụ tính tay. Chứng minh chỉ còn phác thảo vài dòng ("vì sao") ở những chỗ thật sự cần.
> **Hôm nay gồm 4 bài:**
> - **01** Học có giám sát, train/val/test và rò rỉ dữ liệu
> - **02** Overfitting, bias–variance và regularization
> - **03** Cross-validation
> - **04** Metric phân loại và chọn ngưỡng
>
> **Thời gian:** đọc ~2h45 · quiz 45' · bài tập ~2h
> **Dữ liệu xuyên suốt tuần:** log của chatbot ([project.md](project.md)): mỗi dòng là một lượt trả lời; nhãn `bad = 1` nếu người dùng bấm 👎 (khoảng 15.5%). Mọi con số "đo trên máy bạn" đều chạy bằng scikit-learn 1.9.

---

## 01 · Học có giám sát, train/val/test và rò rỉ dữ liệu

> ⏱ Tầng 1: 10' · Tầng 2: 30'
> **Cần biết trước:** MLE và loss (Tuần 2, bài 10).

> **📖 Từ mới**
> - **Đặc trưng (feature) / nhãn (label):** thông tin đầu vào $x$ / đáp án cần dự đoán $y$.
> - **Học có giám sát:** học từ các cặp $(x, y)$ đã có nhãn. **Không giám sát:** chỉ có $x$, tự tìm cấu trúc (bài 08).
> - **Tổng quát hóa (generalization):** dự đoán tốt trên dữ liệu **chưa từng thấy**.
> - **Tập train / validation / test:** dùng để học / để chọn model và tham số / để đo kết quả cuối cùng, **chỉ một lần**.
> - **Rò rỉ dữ liệu (data leakage):** thông tin mà lúc dự đoán thật không thể có, nhưng lọt vào lúc huấn luyện hoặc đánh giá. Kết quả là điểm đánh giá đẹp giả tạo.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Học sinh ôn thi đại học.
- **Train:** làm bài tập trong sách.
- **Validation:** thi thử, để quyết định nên ôn thêm phần nào.
- **Test:** thi thật, chỉ được làm một lần.

Nếu học sinh **nhìn trộm đề thi thật** trong lúc ôn, điểm thi thử sẽ rất cao nhưng chẳng nói lên năng lực thật. Đó là rò rỉ dữ liệu.

**Ví dụ có thật trong dữ liệu tuần này.** Cột `escalated` cho biết phiên chat có bị chuyển cho nhân viên hay không. Nhưng việc chuyển cho nhân viên xảy ra **sau khi** người dùng bấm 👎. Đo trên máy bạn:
- Không dùng `escalated`: AUC **0.769**.
- Có dùng `escalated`: AUC **0.888**.

Model có vẻ giỏi hơn hẳn, nhưng khi đưa vào chạy thật (lúc câu trả lời vừa sinh ra, chưa ai chuyển cho nhân viên) thì cột này không tồn tại.

**Ba ý cốt lõi:**
1. Mục tiêu là sai số trên dữ liệu **mới**, không phải trên dữ liệu đã học.
2. Test set chỉ được chạm vào **một lần**, ở cuối cùng.
3. Với mỗi đặc trưng, hỏi: *"Tại thời điểm dự đoán, mình có biết giá trị này chưa?"*

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 1.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Supervised / unsupervised / reinforcement learning** | Học từ $(x, y)$ / chỉ từ $x$ / từ phần thưởng khi tương tác với môi trường | ISLR §2.1; Sutton & Barto |
| **Self-supervised learning** | Nhãn được tạo ra từ chính dữ liệu, ví dụ đoán token tiếp theo. **Pretraining của LLM thuộc loại này** | LeCun & Misra (2021) |
| **Empirical risk** | $\hat R(f) = \frac{1}{n}\sum_i \ell(f(x_i), y_i)$ trên tập train | ESL §2.4 |
| **Generalization error** | $R(f) = \mathbb{E}_{(x, y) \sim \mathcal{D}}[\ell(f(x), y)]$ trên phân phối thật | ESL §7.2 |
| **i.i.d. assumption** | Train và test được lấy độc lập từ **cùng** một phân phối | — |
| **Distribution shift** | Phân phối lúc chạy thật khác lúc huấn luyện (*covariate shift*, *label shift*, *concept drift*) | Quiñonero-Candela et al. (2009) |
| **Data leakage** | Thông tin không có sẵn lúc dự đoán lọt vào việc huấn luyện hoặc đánh giá | Kaufman et al., *ACM TKDD* 2012 |

#### 1.2 Ba kiểu rò rỉ hay gặp

| Kiểu | Ví dụ | Cách chặn |
|---|---|---|
| **Target leakage**: đặc trưng là **hệ quả** của nhãn | `escalated`; "số lần hoàn tiền" khi dự đoán khách rời bỏ | Kiểm tra mốc thời gian của từng cột |
| **Train–test contamination**: thông tin của tập test lọt vào lúc huấn luyện | Chuẩn hóa hoặc chọn đặc trưng trên **toàn bộ** dữ liệu rồi mới chia; dòng trùng lặp nằm ở cả train và test | Mọi bước "fit" đặt trong `Pipeline` (bài 10); khử trùng lặp **trước** khi chia |
| **Temporal leakage**: dùng tương lai để dự đoán quá khứ | Chia ngẫu nhiên dữ liệu chuỗi thời gian | Chia theo thời gian (bài 03) |

**Chọn đặc trưng trên toàn bộ dữ liệu: một thí nghiệm đáng nhớ** (ESL §7.10.2). Lấy 200 mẫu, **5 000 đặc trưng hoàn toàn ngẫu nhiên**, và nhãn cũng ngẫu nhiên, nên accuracy thật chỉ là 50%. Đo trên máy bạn:

| Cách làm | Accuracy CV 5-fold |
|---|---|
| Chọn 20 đặc trưng "tốt nhất" trên **toàn bộ** dữ liệu, rồi mới chạy CV | **0.795** (ảo) |
| Đặt bước chọn đặc trưng **bên trong** Pipeline, để mỗi fold tự chọn | 0.545 (đúng) |

Vì sao: với 5 000 cột ngẫu nhiên, luôn có vài cột **tình cờ** tương quan với nhãn trên chính 200 mẫu đó. Chọn chúng trên toàn bộ dữ liệu nghĩa là đã nhìn vào nhãn của fold test.

#### 1.3 Chia dữ liệu

| Cỡ dữ liệu | Cách làm phổ biến |
|---|---|
| Nhỏ (< ~10 000 mẫu) | Giữ một tập test (khoảng 20%); dùng **cross-validation** trên phần còn lại thay cho một tập validation cố định |
| Lớn | Train / val / test, ví dụ 80/10/10 hoặc 98/1/1. Chỉ cần tập val và test đủ lớn để khoảng tin cậy đủ hẹp (Tuần 2, bài 09) |

**Stratify:** khi lớp dương hiếm (15.5%), chia ngẫu nhiên có thể làm tỉ lệ lớp dương trong test lệch hẳn đi. `train_test_split(..., stratify=y)` giữ đúng tỉ lệ ở mọi phần.

**Tính nhanh:** test có 1 200 mẫu với AUC khoảng 0.8 thì khoảng tin cậy 95% rộng cỡ ±0.04 (project đo được $[0.758,\ 0.833]$). Hai model chênh nhau 0.01 AUC trên tập test này là **không phân biệt được**.

#### 1.4 Code chuẩn

```python
from sklearn.model_selection import train_test_split

X = df[FEATURES]                               # chỉ những cột BIẾT ĐƯỢC tại thời điểm dự đoán
y = df["bad"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
# Từ đây tới cuối project, X_te / y_te chỉ được dùng ĐÚNG MỘT LẦN.
```

#### 1.5 Cạm bẫy

1. **Cột ID** (`session_id`, `user_id`) lọt vào làm đặc trưng. Model có thể học thuộc ID, hoặc học được thứ tự thời gian ẩn trong ID.
2. **Tune trên test set.** Thử 20 cấu hình rồi báo con số tốt nhất trên test, thì test đã trở thành validation (bài 11).
3. **Dữ liệu trùng lặp** (cùng một câu hỏi xuất hiện nhiều lần) nằm ở cả train lẫn test, làm điểm bị thổi phồng. Rất hay gặp khi đánh giá LLM.
4. **Kết quả quá đẹp.** AUC 0.99 trên một bài toán khó thì **trước tiên hãy nghi rò rỉ**, chưa vội ăn mừng.

**Sắc thái hay bị hỏi khi phỏng vấn:** *"Model offline rất tốt nhưng khi chạy thật thì tệ, vì sao?"* Ba nghi phạm theo thứ tự: rò rỉ dữ liệu, distribution shift, và khác biệt giữa pipeline huấn luyện với pipeline phục vụ (*training–serving skew*: cùng một đặc trưng nhưng tính bằng hai đoạn code khác nhau).

#### 1.6 Đọc thêm

- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning* (ESL, 2nd ed.), §7.10.2 (*The Wrong and Right Way to Do Cross-validation*).
- James et al., *An Introduction to Statistical Learning* (ISLR / ISLP), ch. 2 và 5.
- S. Kaufman et al., *Leakage in Data Mining: Formulation, Detection, and Avoidance*, ACM TKDD 2012.

---

## 02 · Overfitting, bias–variance và regularization

> ⏱ Tầng 1: 15' · Tầng 2: 30'
> **Cần biết trước:** ridge/L2 (Tuần 2, bài 10); phương sai (Tuần 2, bài 09).

> **📖 Từ mới**
> - **Underfitting:** model quá đơn giản, sai nhiều trên cả train lẫn test.
> - **Overfitting:** model học thuộc cả **nhiễu** của train, nên train rất tốt nhưng test tệ.
> - **Độ phức tạp (capacity):** khả năng của model trong việc khớp các hàm phức tạp, ví dụ bậc của đa thức hay độ sâu của cây.
> - **Bias:** sai số do model **quá cứng nhắc**, đoán lệch một cách có hệ thống. **Variance:** sai số do model **quá nhạy** với từng tập train cụ thể.
> - **Learning curve:** đồ thị sai số train/val theo số mẫu huấn luyện.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Ba học sinh ôn thi:
- **A** chỉ học thuộc một công thức chung cho mọi dạng bài: sai nhiều, cả bài cũ lẫn bài mới. Đó là **bias cao**.
- **B** hiểu bản chất: làm tốt cả bài cũ lẫn bài mới.
- **C** học thuộc lòng đáp án của từng bài trong sách, kể cả lỗi in sai: làm bài cũ thì hoàn hảo, đề mới thì chịu. Đó là **variance cao**.

**Ví dụ có số** (đo trên máy bạn). Lấy 20 điểm từ $y = \sin(2\pi x) + \text{nhiễu}$ (độ lệch chuẩn của nhiễu là 0.3, nên sai số không thể tránh là $0.3^2 = 0.09$). Khớp các đa thức bậc khác nhau:

| Bậc | MSE train | MSE test | Nhận xét |
|---|---|---|---|
| 1 | 0.164 | 0.309 | underfit |
| 3 | 0.086 | **0.131** | vừa |
| 9 | 0.062 | 1.869 | overfit |
| 15 | 0.056 | 9.657 | overfit nặng |

MSE train **luôn giảm** khi model phức tạp hơn; MSE test đi theo hình **chữ U**.

**Ba ý cốt lõi:**
1. Sai số train **không** nói lên chất lượng model.
2. Có một mức phức tạp vừa phải, nằm giữa underfit và overfit.
3. Regularization, thêm dữ liệu, hoặc giảm độ phức tạp đều là cách giảm variance.

---

### TẦNG 2 · Kiến thức chuẩn

#### 2.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Bias–variance decomposition** | Với loss bình phương và $y = f(x) + \varepsilon$, $\mathrm{Var}(\varepsilon) = \sigma^2$: $\mathbb{E}[(y - \hat f(x))^2] = \underbrace{(\mathbb{E}\hat f(x) - f(x))^2}_{\text{bias}^2} + \underbrace{\mathrm{Var}(\hat f(x))}_{\text{variance}} + \sigma^2$. Kỳ vọng lấy trên các tập train ngẫu nhiên | ESL §7.3; Geman et al. (1992) |
| **Irreducible error** | $\sigma^2$: không model nào xuống dưới được | — |
| **Regularization** | Mọi kỹ thuật làm giảm variance bằng cách hạn chế model: phạt L1/L2 (Tuần 2, bài 10), giới hạn độ sâu cây, dropout (Tuần 4), early stopping (bài 11) | Goodfellow ch. 7 |
| **Learning curve** | Sai số train và val theo số mẫu $n$ | — |
| **Double descent** | Với model rất lớn (mạng neural), test error có thể giảm **lần hai** khi vượt qua ngưỡng nội suy | Belkin et al., PNAS 2019 |

*Vì sao có phân tích bias–variance (phác thảo):* cộng và trừ $\mathbb{E}\hat f(x)$ bên trong bình phương, rồi khai triển. Các số hạng chéo có kỳ vọng bằng 0, vì $\varepsilon$ độc lập với tập train.

#### 2.2 Đo bias và variance bằng mô phỏng

Lặp lại 300 lần: lấy một tập train 20 điểm mới, khớp đa thức, ghi lại dự đoán trên một lưới điểm cố định. Đo trên máy bạn:

| Bậc | bias² | variance | bias² + variance + $\sigma^2$ |
|---|---|---|---|
| 1 | **0.155** | 0.031 | 0.276 |
| 3 | 0.003 | 0.023 | **0.116** |
| 5 | 0.001 | 0.173 | 0.264 |
| 7 | ≈ 0 * | **34.5** | rất lớn |

\* Ở bậc 7, bias² ước lượng ra 0.040 chỉ vì nhiễu của chính phép mô phỏng: variance 34.5 chia cho 300 lần lặp vẫn còn cỡ 0.1.

Đọc bảng:
- Bậc thấp thì bias chiếm ưu thế.
- Bậc cao thì variance bùng nổ, đặc biệt ở **hai đầu mút**, nơi có ít dữ liệu.
- Tổng nhỏ nhất ở bậc 3. Con số 0.116 này khớp với MSE test 0.131 ở tầng 1.

#### 2.3 Regularization: giữ model phức tạp nhưng phạt nó

Đa thức bậc 15 kèm ridge (đặc trưng đã chuẩn hóa). Đo trên máy bạn:

| $\alpha$ của ridge | MSE train | MSE test |
|---|---|---|
| $10^{-6}$ | 0.066 | 0.542 |
| $10^{-3}$ | 0.067 | 0.387 |
| $10^{-1}$ | 0.109 | **0.177** |
| 10 | 0.149 | 0.280 |

Lại là hình chữ U, lần này theo **$\alpha$**: $\alpha$ nhỏ thì overfit, $\alpha$ lớn thì underfit. Chọn $\alpha$ bằng cross-validation (bài 03).

#### 2.4 Chẩn đoán bằng learning curve

| Triệu chứng | Chẩn đoán | Nên làm |
|---|---|---|
| Train và val **cùng cao**, sát nhau | Bias cao (underfit) | Model phức tạp hơn, thêm đặc trưng, giảm regularization. **Thêm dữ liệu không giúp được** |
| Train thấp, val cao, **khoảng cách lớn** | Variance cao (overfit) | Thêm dữ liệu, tăng regularization, đơn giản hóa model |
| Cả hai đã gần sai số không thể tránh | Gần tối ưu | Dừng lại, hoặc sửa chính nhãn và dữ liệu |

Trên dữ liệu của project, cây quyết định cho thấy rõ cả hai đầu (đo trên máy bạn, AUC):

| `max_depth` | AUC train | AUC test |
|---|---|---|
| 2 | 0.613 | 0.606 |
| 4 | 0.770 | 0.738 |
| 6 | 0.815 | **0.767** |
| 10 | 0.910 | 0.691 |
| không giới hạn | **1.000** | 0.604 |

#### 2.5 Cạm bẫy

1. **Chọn model theo sai số train.** Model phức tạp nhất luôn "thắng" trên train.
2. **Regularize mà không chuẩn hóa đặc trưng.** Phạt $\|\theta\|^2$ ép các hệ số về 0 như nhau, nên đặc trưng đo bằng đơn vị lớn (hệ số nhỏ) gần như không bị phạt, còn đặc trưng đo bằng đơn vị nhỏ thì bị phạt nặng.
3. **Thêm dữ liệu khi vấn đề là bias.** Learning curve sẽ cho thấy điều đó vô ích.

**Sắc thái hay bị hỏi khi phỏng vấn:** *"Model lớn như LLM có overfit không, khi số tham số còn nhiều hơn số mẫu?"* Bức tranh hình chữ U cổ điển không còn đúng hoàn toàn (double descent). Nhưng overfit vẫn xảy ra khi **fine-tune trên tập nhỏ**, và lúc đó vẫn phải theo dõi validation loss (Tuần 11).

#### 2.6 Đọc thêm

- ESL §7.1–7.3; ISLR §2.2.
- M. Belkin et al., *Reconciling modern machine-learning practice and the classical bias–variance trade-off*, PNAS 2019.

---

## 03 · Cross-validation

> ⏱ Tầng 1: 10' · Tầng 2: 25'
> **Cần biết trước:** bài 01–02.

> **📖 Từ mới**
> - **k-fold cross-validation (CV):** chia dữ liệu thành $k$ phần (fold); lần lượt dùng từng phần làm validation, $k - 1$ phần còn lại để train; lấy trung bình $k$ điểm số.
> - **Stratified:** mỗi fold giữ đúng tỉ lệ các lớp.
> - **Group k-fold:** các mẫu cùng nhóm (cùng người dùng, cùng tài liệu) luôn nằm chung một fold.
> - **Nested CV:** CV lồng trong CV; vòng trong dùng để tune, vòng ngoài dùng để đánh giá.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Muốn biết một học sinh giỏi tới đâu mà chỉ có 5 đề. Thay vì dùng 4 đề để ôn và **1 đề cố định** để thi (có thể đề đó tình cờ dễ), ta làm 5 lượt: mỗi lượt giữ lại một đề khác nhau để thi. Điểm trung bình của 5 lượt đáng tin hơn, và độ chênh giữa 5 lượt cho biết con số đó **dao động** bao nhiêu.

**Ví dụ nhỏ.** 10 mẫu, $k = 5$: mỗi fold có 2 mẫu; mỗi mẫu được dùng làm validation **đúng 1 lần** và làm train 4 lần. Phải fit tổng cộng 5 model.

**Ba ý cốt lõi:**
1. CV tận dụng mọi mẫu cho cả việc học lẫn việc đánh giá, nên ít lãng phí khi dữ liệu nhỏ.
2. Phải chia fold **đúng cấu trúc** của dữ liệu: theo lớp, theo nhóm, theo thời gian.
3. Điểm CV của **cấu hình tốt nhất sau khi tune** vẫn lạc quan; cần một tập test riêng, hoặc nested CV.

---

### TẦNG 2 · Kiến thức chuẩn

#### 3.1 Thuật ngữ và các kiểu chia

| Kiểu | Khi nào dùng | scikit-learn |
|---|---|---|
| **KFold** (`shuffle=True`) | Dữ liệu i.i.d., hồi quy | `KFold` |
| **StratifiedKFold** | Phân loại, nhất là khi lớp mất cân bằng (**mặc định nên dùng**) | `StratifiedKFold` |
| **GroupKFold** | Nhiều mẫu từ cùng người dùng, cùng tài liệu, cùng bệnh nhân. Nếu không, model "nhận ra" nhóm thay vì học quy luật | `GroupKFold`, `StratifiedGroupKFold` |
| **TimeSeriesSplit** | Dữ liệu theo thời gian: luôn train trên quá khứ, validate trên tương lai | `TimeSeriesSplit` |
| **Leave-one-out** | $k = n$; rất tốn kém, variance của ước lượng cao | `LeaveOneOut` |

**Chọn $k$.** $k = 5$ hoặc $10$ là mặc định thực tế (ESL §7.10.1). $k$ lớn hơn thì mỗi model được train trên nhiều dữ liệu hơn (ước lượng ít bi quan hơn) nhưng tốn gấp $k$ lần.

**Chi phí:** $k$ fold × $m$ cấu hình = $k \cdot m$ lần fit. Ví dụ: 5 fold × 20 cấu hình = 100 lần fit; trên dữ liệu project mất khoảng 20 giây (đo trên máy bạn).

**Độ dao động giữa các fold.** Trên dữ liệu project (6 000 mẫu, 5 fold), AUC của HistGradientBoosting có trung bình 0.812 và độ lệch chuẩn khoảng 0.01–0.02 giữa các fold. Hai model chênh nhau 0.004 (RF và HGB) thì **không phân biệt được**. Lưu ý: các fold dùng chung dữ liệu train nên không độc lập; lấy độ lệch chuẩn chia $\sqrt{k}$ sẽ **đánh giá thấp** độ không chắc chắn (Bengio & Grandvalet, JMLR 2004).

#### 3.2 Nested CV: vì sao điểm CV sau khi tune bị lạc quan

Thử $m$ cấu hình rồi chọn cấu hình có điểm CV cao nhất. Điểm cao nhất đó gồm **chất lượng thật cộng với may mắn**: cấu hình được chọn thường là cấu hình được nhiễu "ưu ái". Giống như tung 20 đồng xu, mỗi đồng 10 lần, rồi khen đồng xu ra mặt ngửa nhiều nhất là "đồng xu thiên về mặt ngửa".

Cách đúng:
- **Đủ dữ liệu:** tune bằng CV trên tập train, rồi đo **một lần** trên test.
- **Ít dữ liệu:** nested CV. Vòng ngoài $k_1$ fold dùng để đánh giá; trong mỗi fold ngoài, chạy một vòng CV bên trong để tune.

Project của bạn sẽ thấy đúng hiện tượng này: AUC CV của cấu hình tốt nhất là 0.817, AUC trên test là 0.796.

#### 3.3 Code chuẩn

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=["roc_auc", "average_precision"])
print(res["test_roc_auc"].mean(), res["test_roc_auc"].std())
# `pipe` gồm CẢ tiền xử lý → mỗi fold tự fit imputer/scaler trên phần train của fold đó (bài 10)
```

#### 3.4 Cạm bẫy

1. **Tiền xử lý nằm ngoài CV** (fit scaler hoặc chọn đặc trưng trên toàn bộ dữ liệu): đúng thí nghiệm 0.795 so với 0.545 ở bài 01.
2. **Dùng KFold có xáo trộn cho chuỗi thời gian:** model được "nhìn thấy tương lai".
3. **Không dùng group khi cần**, ví dụ nhiều câu hỏi sinh từ cùng một tài liệu (đúng tình huống ở final Tuần 2, D2c).
4. **Báo điểm CV của cấu hình tốt nhất như kết quả cuối cùng.**

**Sắc thái hay bị hỏi khi phỏng vấn:** *"Có 500 mẫu, đánh giá thế nào?"* Stratified 5- hoặc 10-fold, lặp lại vài lần với seed khác nhau (*repeated CV*), và báo trung bình kèm độ dao động. Nếu còn phải tune thì dùng nested CV.

#### 3.5 Đọc thêm

- ESL §7.10; scikit-learn User Guide, *3.1 Cross-validation*.
- G. Cawley & N. Talbot, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, JMLR 2010.

---

## 04 · Metric phân loại và chọn ngưỡng

> ⏱ Tầng 1: 15' · Tầng 2: 40'
> **Cần biết trước:** Bayes và tỉ lệ nền (Tuần 2, bài 08); khoảng tin cậy (Tuần 2, bài 09).

> **📖 Từ mới**
> - **Ma trận nhầm lẫn (confusion matrix):** bảng đếm đúng/sai theo từng loại: **TP** (dương, đoán dương), **FP** (âm, đoán dương: báo nhầm), **FN** (dương, đoán âm: bỏ sót), **TN**.
> - **Precision:** trong những ca **bị báo**, bao nhiêu phần trăm là đúng. **Recall:** trong những ca **dương thật**, bắt được bao nhiêu phần trăm.
> - **F1:** trung bình điều hòa của precision và recall.
> - **Ngưỡng (threshold):** model cho ra điểm $s \in [0, 1]$; dự đoán dương khi $s \ge t$.
> - **ROC-AUC:** đo khả năng **xếp hạng**, không phụ thuộc ngưỡng.
> - **PR-AUC / Average Precision (AP):** tương tự nhưng tập trung vào lớp dương; hợp với dữ liệu mất cân bằng.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Lưới đánh cá:
- Mắt lưới **nhỏ**: bắt được hầu hết cá (recall cao), nhưng vớt lên cả rác (precision thấp).
- Mắt lưới **to**: thứ vớt lên phần lớn là cá (precision cao), nhưng lọt mất nhiều (recall thấp).

Ngưỡng $t$ chính là kích thước mắt lưới. Không có ngưỡng "đúng" chung cho mọi bài toán: ngưỡng phụ thuộc vào việc **bỏ sót đắt hơn hay báo nhầm đắt hơn**.

**Nghịch lý accuracy.** Dữ liệu tuần này có 15.5% câu trả lời bị 👎. Một model "luôn đoán là tốt" đạt accuracy **84.5%** mà không bắt được câu tệ nào. Với dữ liệu mất cân bằng, accuracy gần như vô dụng.

**Ba ý cốt lõi:**
1. Luôn bắt đầu từ ma trận nhầm lẫn; mọi metric khác suy ra từ nó.
2. AUC đánh giá **thứ hạng**; precision, recall, F1 đánh giá **một ngưỡng cụ thể**.
3. Ngưỡng nên chọn theo **chi phí** của từng loại sai, trên tập validation.

---

### TẦNG 2 · Kiến thức chuẩn

#### 4.1 Định nghĩa

| Metric | Công thức | Ghi chú |
|---|---|---|
| Accuracy | $\dfrac{TP + TN}{n}$ | Vô nghĩa khi mất cân bằng |
| Precision | $\dfrac{TP}{TP + FP}$ | Phụ thuộc **tỉ lệ nền** (Tuần 2, bài 08) |
| Recall (TPR, sensitivity) | $\dfrac{TP}{TP + FN}$ | |
| FPR | $\dfrac{FP}{FP + TN}$ | |
| F1 | $\dfrac{2PR}{P + R}$ | Trung bình điều hòa: kéo về phía giá trị **nhỏ hơn** |
| ROC curve | Đường $(\text{FPR}(t), \text{TPR}(t))$ khi $t$ chạy từ $+\infty$ xuống $-\infty$ | Fawcett (2006) |
| **ROC-AUC** | $P(s^+ > s^-)$: xác suất một mẫu dương ngẫu nhiên được chấm điểm cao hơn một mẫu âm ngẫu nhiên; hòa tính $\tfrac{1}{2}$ | Bằng thống kê Mann–Whitney $U/(n_+ n_-)$ |
| **Average Precision** | $\sum_i (R_i - R_{i-1})\,P_i$ trên các ngưỡng (định nghĩa của scikit-learn) | Mức ngẫu nhiên = **tỉ lệ dương**, không phải 0.5 |
| Macro / micro / weighted F1 (đa lớp) | Trung bình F1 các lớp như nhau / gộp TP, FP, FN của mọi lớp rồi tính / trung bình theo số mẫu mỗi lớp | Macro coi lớp hiếm quan trọng ngang lớp phổ biến |

**Metric hồi quy:** MAE $= \frac{1}{n}\sum\lvert y - \hat y\rvert$ (bền với outlier); RMSE $= \sqrt{\frac{1}{n}\sum(y - \hat y)^2}$ (phạt nặng sai số lớn, cùng đơn vị với $y$); $R^2 = 1 - \dfrac{\sum(y - \hat y)^2}{\sum(y - \bar y)^2}$ (có thể âm nếu model tệ hơn việc đoán trung bình).

#### 4.2 Ngưỡng tối ưu theo chi phí

Gọi $c_{FP}$, $c_{FN}$ là chi phí của một lần báo nhầm và một lần bỏ sót. Nếu điểm $s$ là **xác suất đã hiệu chỉnh** ($s \approx P(y = 1 \mid x)$), thì đoán dương khi chi phí kỳ vọng của việc đoán dương nhỏ hơn:

$$c_{FP}\,(1 - s) < c_{FN}\,s \iff s > t^\star = \frac{c_{FP}}{c_{FP} + c_{FN}}.$$

Ví dụ $c_{FP} = 1$ (một nhân viên đọc lại câu trả lời thừa), $c_{FN} = 5$ (người dùng bực mình và rời đi): $t^\star = 1/6 \approx$ **0.167**. Project chọn ngưỡng bằng cách quét thực nghiệm trên dự đoán out-of-fold, và tìm ra **0.174**, rất sát lý thuyết. Lệch nhỏ vì xác suất của model chỉ gần đúng đã hiệu chỉnh. (Elkan, *The Foundations of Cost-Sensitive Learning*, IJCAI 2001.)

#### 4.3 Ví dụ tính tay

10 mẫu: 4 dương, 6 âm.

| Mẫu dương ($y = 1$) | 0.9 | 0.8 | 0.6 | 0.35 | | |
|---|---|---|---|---|---|---|
| **Mẫu âm ($y = 0$)** | 0.7 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 |

**(a) Ngưỡng 0.5.** Đoán dương các mẫu có $s \ge 0.5$: 0.9, 0.8, 0.7, 0.6, 0.5.
- TP = 3 (0.9, 0.8, 0.6); FP = 2 (0.7, 0.5); FN = 1 (0.35); TN = 4.
- Precision $= 3/5 = 0.6$; recall $= 3/4 = 0.75$; F1 $= 2 \cdot 0.6 \cdot 0.75/1.35 = 0.667$; accuracy $= 0.7$.

**(b) ROC-AUC bằng cách đếm cặp.** Có $4 \times 6 = 24$ cặp (dương, âm). Đếm số cặp mà mẫu dương có điểm cao hơn:
- 0.9 thắng cả 6 mẫu âm.
- 0.8 thắng cả 6.
- 0.6 thắng 5 (thua 0.7).
- 0.35 thắng 3 (0.3, 0.2, 0.1).

Tổng là 20, nên AUC $= 20/24 =$ **0.833**.

**(c) Average Precision.** Sắp giảm dần: 0.9 (+), 0.8 (+), 0.7 (−), 0.6 (+), 0.5 (−), 0.4 (−), 0.35 (+), … Tại mỗi lần gặp một mẫu dương, precision lần lượt là $1/1,\ 2/2,\ 3/4,\ 4/7$; mỗi lần recall tăng $1/4$. AP $= \tfrac{1}{4}(1 + 1 + 0.75 + 0.571) =$ **0.830**.

**(d) Chọn ngưỡng với $c_{FP} = 1$, $c_{FN} = 5$:**

| $t$ | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 | 0.4 | **0.35** | 0.3 |
|---|---|---|---|---|---|---|---|---|
| FP | 0 | 0 | 1 | 1 | 2 | 3 | 3 | 4 |
| FN | 3 | 2 | 2 | 1 | 1 | 1 | 0 | 0 |
| Chi phí $= FP + 5\,FN$ | 15 | 10 | 11 | 6 | 7 | 8 | **3** | 4 |

Ngưỡng tốt nhất là **0.35**, với chi phí 3. Nếu hai loại lỗi cùng giá ($c_{FN} = 1$), chi phí nhỏ nhất bằng 2, đạt ở cả 0.8 và 0.6.

Mọi con số trên đã chạy kiểm chứng bằng `sklearn.metrics` và là test của bài tập hôm nay.

#### 4.4 ROC hay PR?

| | ROC-AUC | PR-AUC / AP |
|---|---|---|
| Mức ngẫu nhiên | 0.5, **luôn luôn** | **tỉ lệ dương** (ở đây 0.155) |
| Nhạy với tỉ lệ nền | Không: TPR và FPR tính riêng trên từng lớp | Có: precision phụ thuộc tỉ lệ nền |
| Khi nào dùng | So sánh khả năng xếp hạng nói chung | Lớp dương hiếm và ta quan tâm tới những ca **bị báo** (Saito & Rehmsmeier, PLOS ONE 2015) |

Trên dữ liệu project: AUC 0.81 nhưng AP chỉ khoảng 0.52. Cả hai đều tốt hơn ngẫu nhiên nhiều (0.5 và 0.155), nhưng AP cho thấy rõ hơn: trong số những câu bị báo là tệ, vẫn còn nhiều câu báo nhầm.

#### 4.5 Code chuẩn

```python
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report

proba = model.predict_proba(X_val)[:, 1]          # dùng XÁC SUẤT cho AUC/AP, không dùng nhãn 0/1
print(roc_auc_score(y_val, proba), average_precision_score(y_val, proba))
pred = (proba >= t).astype(int)                    # t chọn theo chi phí, KHÔNG mặc định 0.5
print(confusion_matrix(y_val, pred))               # [[TN, FP], [FN, TP]]
print(classification_report(y_val, pred, digits=3))
```

#### 4.6 Cạm bẫy

1. **Tính AUC trên nhãn 0/1** thay vì trên xác suất: mất toàn bộ thông tin thứ hạng.
2. **Dùng ngưỡng mặc định 0.5** với dữ liệu mất cân bằng: thường bắt được rất ít ca dương.
3. **Chọn ngưỡng trên test set**: rò rỉ. Chọn trên validation hoặc trên dự đoán out-of-fold.
4. **Báo precision mà không nêu tỉ lệ nền**: cùng một model, precision thay đổi khi tỉ lệ dương thay đổi.
5. **So sánh AP giữa hai tập dữ liệu có tỉ lệ dương khác nhau.**

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"AUC có ý nghĩa xác suất gì?"* $P(s^+ > s^-)$.
- *"Dữ liệu 1% dương, dùng metric gì?"* PR-AUC hoặc recall ở một mức precision cố định, không dùng accuracy.
- *"Chọn ngưỡng thế nào?"* Theo chi phí, trên validation; nếu xác suất đã hiệu chỉnh thì $t^\star = c_{FP}/(c_{FP} + c_{FN})$.

#### 4.7 Đọc thêm

- T. Fawcett, *An Introduction to ROC Analysis*, Pattern Recognition Letters 2006.
- T. Saito & M. Rehmsmeier, *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, PLOS ONE 2015.
- C. Elkan, *The Foundations of Cost-Sensitive Learning*, IJCAI 2001.
- scikit-learn User Guide, *3.4 Metrics and scoring*.

---

## Bài tập Ngày 1 · `quality/metrics.py` (~2h)

Mảnh đầu tiên của project ([project.md](project.md)). Viết **từ đầu bằng NumPy** (được dùng `scipy.stats.rankdata`); test sẽ đối chiếu với `sklearn.metrics`. Lệnh test: `python -m pytest -q tests/test_metrics.py` (6 test).

| Hàm | Yêu cầu |
|---|---|
| `confusion(y_true, y_pred)` | Mảng `[[TN, FP], [FN, TP]]` |
| `precision_recall_f1(y_true, y_pred)` | Trả về `(p, r, f1)`; mẫu số bằng 0 thì metric tương ứng bằng 0 |
| `roc_auc(y_true, score)` | Qua tổng hạng: $\dfrac{\sum \text{rank}(s^+) - n_+(n_+ + 1)/2}{n_+ n_-}$, dùng `rankdata` (hạng trung bình khi hòa). Thiếu một trong hai lớp thì raise `ValueError` |
| `average_precision(y_true, score)` | Sắp giảm dần; **gộp các score bằng nhau thành một ngưỡng**; $\sum(R_i - R_{i-1})P_i$. Phải khớp `sklearn` cả khi có nhiều giá trị hòa |
| `best_threshold(y_true, score, cost_fp=1, cost_fn=1)` | Trả về `(t, tổng chi phí)` nhỏ nhất. Ứng viên: mọi giá trị score phân biệt và `inf` (không đoán dương nào). Dự đoán dương khi `score >= t`. Hòa thì chọn $t$ **lớn hơn** |
| `stratified_kfold(y, k=5, seed=0)` | Danh sách `k` cặp `(train_idx, test_idx)` (đã sắp xếp). Với mỗi lớp: xáo bằng `rng.permutation`, chia bằng `np.array_split` thành `k` phần; fold `i` lấy phần `i` của mọi lớp. `k < 2` thì raise `ValueError` |

**Gợi ý:** viết `average_precision` bằng cách xác định vị trí **cuối** của mỗi nhóm score bằng nhau (sau khi sắp xếp), rồi lấy `cumsum` tại các vị trí đó.
