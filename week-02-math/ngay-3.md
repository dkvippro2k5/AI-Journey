# Ngày 3 · Xác suất, Thống kê & Lý thuyết thông tin

> **Loại tài liệu: Lý thuyết.** Đầy đủ ký hiệu, định nghĩa, mệnh đề kèm chứng minh, và bài tính tay.
> **Hôm nay gồm 4 bài:**
> - **08** Xác suất có điều kiện và định lý Bayes
> - **09** Biến ngẫu nhiên, phân phối, luật số lớn và định lý giới hạn trung tâm
> - **10** MLE và MAP: loss function đến từ đâu
> - **11** Entropy, cross-entropy, KL divergence và perplexity
>
> **Thời gian:** đọc ~3h · quiz 45' · bài tập ~2h
> **Vì sao bài này quan trọng với AI Engineer:** loss của mọi LLM là cross-entropy (bài 11), loss đó đến từ MLE (bài 10), và mọi con số đánh giá model đều cần khoảng tin cậy (bài 09). Bạn sẽ dùng lại bài 09 khi làm evaluation ở Tuần 13.

---

## 08 · Xác suất có điều kiện và định lý Bayes

> ⏱ Tầng 1: 15' · Tầng 2: 35'
> **Cần biết trước:** phân số, phần trăm.

> **📖 Từ mới**
> - **Biến cố:** một điều có thể xảy ra, ví dụ "bình luận này độc hại".
> - **Xác suất đồng thời $P(A, B)$:** xác suất **cả** $A$ **và** $B$ cùng xảy ra.
> - **Xác suất biên $P(A)$:** xác suất của $A$, không quan tâm tới $B$.
> - **Xác suất có điều kiện $P(A \mid B)$:** xác suất của $A$ **khi đã biết** $B$ xảy ra.
> - **Độc lập:** biết $B$ không làm thay đổi xác suất của $A$.
> - **Prior / likelihood / posterior:** niềm tin **trước** khi thấy bằng chứng / mức độ bằng chứng phù hợp với giả thuyết / niềm tin **sau** khi thấy bằng chứng.
> - **Tỉ lệ nền (base rate, prevalence):** biến cố phổ biến tới mức nào trong tổng thể.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Một bộ lọc kiểm duyệt bình luận:
- Bắt được 95% bình luận độc hại.
- Báo nhầm 5% bình luận bình thường.
- Chỉ **1%** bình luận là độc hại thật.

Bộ lọc vừa báo "độc hại". Khả năng bình luận đó độc hại thật là bao nhiêu? Nhiều người đoán 95%. Hãy đếm trên 10 000 bình luận:

| | Bị báo | Không bị báo | Tổng |
|---|---|---|---|
| Độc hại (1%) | 95 | 5 | 100 |
| Bình thường (99%) | **495** | 9 405 | 9 900 |
| Tổng | **590** | 9 410 | 10 000 |

Trong 590 bình luận bị báo, chỉ 95 là độc hại thật: **16.1%**. Số báo nhầm (495) nhiều gấp 5 lần số báo đúng, vì bình luận bình thường nhiều gấp 99 lần.

**Ba ý cốt lõi:**
1. $P(A \mid B)$ khác $P(B \mid A)$: "bị báo khi độc hại" (95%) khác "độc hại khi bị báo" (16%).
2. Kết luận phụ thuộc mạnh vào **tỉ lệ nền**. Biến cố càng hiếm thì báo động càng dễ là báo nhầm.
3. Bayes là công thức để **cập nhật niềm tin** khi có bằng chứng mới.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 8.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Kolmogorov axioms** | $P(A) \ge 0$; $P(\Omega) = 1$; $P(\bigcup A_i) = \sum P(A_i)$ với các $A_i$ đôi một rời nhau | Blitzstein & Hwang, *Intro to Probability* §1.6 |
| **Conditional probability** | $P(A \mid B) = \dfrac{P(A, B)}{P(B)}$, với $P(B) > 0$ | như trên, §2.2 |
| **Product rule** | $P(A, B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A)$ | — |
| **Sum rule (marginalization)** | $P(A) = \sum_b P(A, B = b)$ | MML §6.3 |
| **Law of total probability** | $P(A) = \sum_i P(A \mid B_i)\,P(B_i)$, với $\{B_i\}$ là một phân hoạch của $\Omega$ | Blitzstein & Hwang §2.3 |
| **Independence** | $P(A, B) = P(A)\,P(B)$, tương đương $P(A \mid B) = P(A)$ | — |
| **Conditional independence** | $P(A, B \mid C) = P(A \mid C)\,P(B \mid C)$. **Không** suy ra được độc lập, và ngược lại | §2.5 |
| **Bayes' theorem** | $P(H \mid E) = \dfrac{P(E \mid H)\,P(H)}{P(E)}$: posterior = likelihood × prior / evidence | Bayes (1763); Laplace (1812) |
| **Likelihood ratio** | $\mathrm{LR} = P(E \mid H)/P(E \mid \neg H)$ | — |
| **Naive Bayes** | Phân loại theo $\arg\max_c P(c)\prod_j P(x_j \mid c)$, giả định các đặc trưng **độc lập có điều kiện** khi biết lớp | Jurafsky & Martin, *SLP3* ch. 4 |

#### 8.2 Mệnh đề và chứng minh

**Mệnh đề 8.1 (Bayes).** *Chứng minh:* viết product rule theo hai cách, $P(H \mid E)P(E) = P(E \mid H)P(H)$, rồi chia cho $P(E)$. Mẫu số tính bằng định luật xác suất toàn phần: $P(E) = P(E \mid H)P(H) + P(E \mid \neg H)P(\neg H)$. ∎

**Mệnh đề 8.2 (dạng odds).** Với odds $O(H) = P(H)/P(\neg H)$:

$$O(H \mid E) = \mathrm{LR} \times O(H).$$

*Chứng minh.* Viết Bayes cho $H$ và cho $\neg H$ rồi chia hai vế cho nhau; $P(E)$ triệt tiêu. ∎

Dạng odds giúp **cập nhật nhiều lần** rất nhanh. Với bộ lọc ở tầng 1, $\mathrm{LR} = 0.95/0.05 = 19$:
- Prior odds $= 1 : 99$. Sau một lần báo: $19 : 99$, nên $P = 19/118 = 16.1\%$ ✓.
- Nếu một bộ lọc **thứ hai, độc lập có điều kiện** với bộ lọc đầu, cũng báo: odds $= 361 : 99$, nên $P = 78.5\%$.

#### 8.3 Ví dụ tính tay

**(a) Bảng xác suất đồng thời.** Đánh giá một chatbot trên hai loại câu hỏi:

| | Đúng | Sai | Biên theo loại |
|---|---|---|---|
| Toán | 0.12 | 0.18 | 0.30 |
| Văn bản | 0.56 | 0.14 | 0.70 |
| Biên theo kết quả | 0.68 | 0.32 | 1 |

- $P(\text{đúng}) = 0.68$ (sum rule).
- $P(\text{đúng} \mid \text{toán}) = 0.12/0.30 = 0.4$; $P(\text{đúng} \mid \text{văn bản}) = 0.56/0.70 = 0.8$.
- $P(\text{toán} \mid \text{sai}) = 0.18/0.32 = 0.5625$: hơn một nửa số câu sai là câu toán, dù câu toán chỉ chiếm 30%.
- Độc lập không? $P(\text{đúng})\,P(\text{toán}) = 0.204 \ne 0.12$, nên **không** độc lập.

**(b) Tỉ lệ nền thay đổi kết luận** (cùng bộ lọc TPR 95%, FPR 5%):

| Tỉ lệ độc hại thật | $P(\text{độc hại} \mid \text{bị báo})$ |
|---|---|
| 1% | 16.1% |
| 10% | 67.9% |
| 50% | 95.0% |

**(c) Naive Bayes tính tay.** Tập huấn luyện 6 tin nhắn (tách từ theo dấu cách cho đơn giản; tách từ tiếng Việt thực sự sẽ học ở Tuần 5):

| Lớp | Tin nhắn |
|---|---|
| spam | "khuyến mãi trúng thưởng" · "trúng thưởng miễn phí" · "khuyến mãi miễn phí" |
| ham | "lịch họp ngày mai" · "gửi tài liệu họp" · "nộp tài liệu ngày mai" |

Thống kê: spam có 12 token, ham có 13 token, từ điển $|V| = 14$ từ. Phân loại tin "**miễn phí họp**" với *Laplace smoothing* $P(w \mid c) = \dfrac{\text{count}(w, c) + 1}{\text{tổng token của } c + |V|}$ (bài 10 sẽ giải thích vì sao cộng 1):

| | $P(c)$ | miễn | phí | họp | Tích |
|---|---|---|---|---|---|
| spam | 0.5 | $3/26$ | $3/26$ | $1/26$ | $2.560 \times 10^{-4}$ |
| ham | 0.5 | $1/27$ | $1/27$ | $3/27$ | $0.762 \times 10^{-4}$ |

Chuẩn hóa: $P(\text{spam} \mid \text{tin}) = 2.560/(2.560 + 0.762) =$ **0.771**. Đã chạy kiểm chứng.

**Không có smoothing** thì cả hai tích đều bằng 0 ("họp" chưa từng xuất hiện trong spam, "miễn" chưa từng xuất hiện trong ham), và mô hình không phân loại được gì.

#### 8.4 Code chuẩn: Naive Bayes trong không gian log

```python
import math
from collections import Counter

def train_nb(docs):                                   # docs: list[(label, text)]
    counts, n_docs = {}, Counter()
    for label, text in docs:
        n_docs[label] += 1
        counts.setdefault(label, Counter()).update(text.split())
    vocab = {w for c in counts.values() for w in c}
    return counts, n_docs, vocab

def predict_nb(model, text):
    counts, n_docs, vocab = model
    total = sum(n_docs.values())
    scores = {}
    for c, cnt in counts.items():
        denom = sum(cnt.values()) + len(vocab)
        scores[c] = math.log(n_docs[c] / total) + sum(math.log((cnt[w] + 1) / denom) for w in text.split())
    m = max(scores.values())                          # trừ max trước khi exp: xem bài 11
    z = sum(math.exp(s - m) for s in scores.values())
    return {c: math.exp(s - m) / z for c, s in scores.items()}
```

#### 8.5 Cạm bẫy

1. **Bỏ qua tỉ lệ nền** (*base rate fallacy*). Precision của một bộ phân loại phụ thuộc vào tỉ lệ lớp dương trong dữ liệu thật, không chỉ vào model.
2. **Nhầm $P(A \mid B)$ với $P(B \mid A)$** (*prosecutor's fallacy*).
3. **Nhân nhiều xác suất nhỏ** làm tràn số dưới (underflow) về 0. Luôn cộng log.
4. **Đếm bằng 0** làm cả tích bằng 0. Dùng smoothing.
5. **Giả định độc lập có điều kiện sai** (các từ trong câu rõ ràng phụ thuộc nhau): xác suất của Naive Bayes thường **quá tự tin** (rất gần 0 hoặc 1), dù thứ hạng phân loại vẫn thường đúng.

**Sắc thái hay bị hỏi khi phỏng vấn:** *"Model kiểm duyệt có recall 95% và FPR 5%, sao người dùng than bị chặn nhầm nhiều thế?"* Trả lời bằng bảng tầng 1: khi tỉ lệ nền thấp, phần lớn các lần báo là báo nhầm. Cách sửa: nâng ngưỡng, thêm tầng kiểm tra thứ hai (Mệnh đề 8.2), hoặc cho người duyệt lại.

#### 8.6 Đọc thêm

- J. Blitzstein & J. Hwang, *Introduction to Probability* (2nd ed.), ch. 1–2. Bản miễn phí tại probabilitybook.net.
- D. Jurafsky & J. Martin, *Speech and Language Processing* (3rd ed. draft), ch. 4 (Naive Bayes).
- MML §6.1–6.3.

---

## 09 · Biến ngẫu nhiên, phân phối, luật số lớn và định lý giới hạn trung tâm

> ⏱ Tầng 1: 15' · Tầng 2: 45'
> **Cần biết trước:** xác suất có điều kiện (bài 08); cosine và Cauchy–Schwarz (bài 01).

> **📖 Từ mới**
> - **Biến ngẫu nhiên:** một con số phụ thuộc vào may rủi, ví dụ số câu chatbot trả lời đúng.
> - **Rời rạc / liên tục:** nhận giá trị đếm được (0, 1, 2…) / nhận mọi giá trị trong một khoảng.
> - **PMF / PDF / CDF:** bảng xác suất của từng giá trị / mật độ xác suất (với biến liên tục) / xác suất $\le x$.
> - **Kỳ vọng $\mathbb{E}[X]$:** trung bình dài hạn. **Phương sai:** độ phân tán quanh kỳ vọng. **Độ lệch chuẩn** = căn của phương sai.
> - **Hiệp phương sai / hệ số tương quan:** hai biến có xu hướng tăng giảm cùng nhau không, và mạnh tới đâu.
> - **i.i.d.:** các mẫu **độc lập** và **cùng phân phối**.
> - **Sai số chuẩn (standard error, SE):** độ lệch chuẩn của một **ước lượng** (ví dụ của trung bình mẫu).
> - **Khoảng tin cậy 95%:** khoảng được xây dựng sao cho, nếu lặp lại thí nghiệm nhiều lần, 95% số khoảng chứa giá trị thật.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Bạn chấm chatbot trên 200 câu hỏi và nó đúng 160 câu: **80%**.
- Nếu chấm trên 200 câu hỏi **khác**, có thể ra 78% hoặc 83%. Con số 80% có "độ rung".
- Độ rung đó cỡ $\sqrt{0.8 \times 0.2/200} \approx 2.8$ điểm phần trăm. Nhân 1.96 ra khoảng **±5.5 điểm %**: kết quả thật nằm đâu đó trong khoảng 74.5%–85.5%.
- Model thứ hai được 83% trên 200 câu **chưa chứng minh** được nó tốt hơn: chênh lệch 3 điểm nhỏ hơn độ rung.
- Muốn độ rung giảm một nửa thì cần **gấp 4 lần** số câu hỏi, không phải gấp 2.

**Ba ý cốt lõi:**
1. Kỳ vọng là trung bình dài hạn; phương sai đo độ phân tán.
2. Trung bình của $n$ mẫu dao động ít hơn một mẫu, nhưng chỉ giảm theo $\sqrt{n}$.
3. Theo định lý giới hạn trung tâm, trung bình mẫu gần như có phân phối chuẩn, nên khoảng "trung bình ± 1.96 × SE" chứa giá trị thật khoảng 95% số lần.

---

### TẦNG 2 · Kiến thức chuẩn

#### 9.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Expectation** | $\mathbb{E}[X] = \sum_x x\,p(x)$ (rời rạc) hoặc $\int x f(x)\,dx$ (liên tục) | Wasserman, *All of Statistics* ch. 3 |
| **Variance** | $\mathrm{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - \mu^2$ | như trên |
| **Covariance** | $\mathrm{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$ | như trên |
| **Correlation (Pearson)** | $\rho = \dfrac{\mathrm{Cov}(X, Y)}{\sigma_X\sigma_Y} \in [-1, 1]$; chỉ đo quan hệ **tuyến tính** | như trên |
| **Law of Large Numbers** | $\bar X_n \to \mu$ (theo xác suất) khi $n \to \infty$, với $X_i$ i.i.d. có kỳ vọng hữu hạn | Wasserman ch. 5 |
| **Central Limit Theorem** | $\dfrac{\sqrt{n}(\bar X_n - \mu)}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1)$, với $X_i$ i.i.d. có phương sai hữu hạn | Wasserman ch. 5 |
| **Standard error** | $\mathrm{SE}(\bar X) = \sigma/\sqrt{n}$, ước lượng bằng $s/\sqrt{n}$ | — |
| **Wald interval (tỉ lệ)** | $\hat p \pm z\sqrt{\hat p(1 - \hat p)/n}$ | — |
| **Wilson interval** | $\dfrac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1 - \hat p)}{n} + \frac{z^2}{4n^2}}}{1 + z^2/n}$; vẫn dùng được khi $\hat p$ gần 0 hoặc 1 | Wilson (1927); Brown, Cai, DasGupta, *Stat. Science* 2001 |

**Các phân phối cần thuộc:**

| Phân phối | Kỳ vọng | Phương sai | Dùng ở đâu trong AI |
|---|---|---|---|
| Bernoulli($p$) | $p$ | $p(1 - p)$ | Một câu trả lời đúng/sai; nhãn nhị phân |
| Binomial($n, p$) | $np$ | $np(1 - p)$ | Số câu đúng trong $n$ câu |
| Categorical / Multinomial | $p_k$ (mỗi lớp) | — | **Phân phối token tiếp theo** của LLM |
| Uniform($a, b$) | $(a + b)/2$ | $(b - a)^2/12$ | Full jitter $U(0, t)$ ở Tuần 1 có kỳ vọng $t/2$ |
| Gaussian $\mathcal{N}(\mu, \sigma^2)$ | $\mu$ | $\sigma^2$ | Nhiễu đo đạc; khởi tạo trọng số |
| Poisson($\lambda$) | $\lambda$ | $\lambda$ | Số request tới trong 1 giây |
| Exponential($\lambda$) | $1/\lambda$ | $1/\lambda^2$ | Thời gian giữa hai request liên tiếp |

#### 9.2 Mệnh đề và chứng minh

**Mệnh đề 9.1 (tuyến tính của kỳ vọng).** $\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]$, **không cần** độc lập. *(Chứng minh trực tiếp từ định nghĩa tổng/tích phân.)*

**Mệnh đề 9.2.** $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X, Y)$. Nếu $X, Y$ độc lập thì $\mathrm{Cov} = 0$. Chiều ngược lại **sai**: $X$ đối xứng quanh 0 và $Y = X^2$ có $\mathrm{Cov} = 0$ dù $Y$ phụ thuộc hoàn toàn vào $X$ (đo trên máy bạn: $r = 1.4 \times 10^{-16}$).

**Mệnh đề 9.3 (phương sai của trung bình mẫu).** Với $X_1, \dots, X_n$ i.i.d. có phương sai $\sigma^2$: $\mathrm{Var}(\bar X_n) = \sigma^2/n$.

*Chứng minh.* Các $X_i$ độc lập nên mọi Cov chéo bằng 0: $\mathrm{Var}\left(\frac{1}{n}\sum X_i\right) = \frac{1}{n^2}\sum\mathrm{Var}(X_i) = \frac{n\sigma^2}{n^2}$. ∎ Đây chính là kết quả đã dùng cho mini-batch ở Mệnh đề 6.6.

**Mệnh đề 9.4 (luật số lớn yếu, qua bất đẳng thức Chebyshev).** $P(\lvert\bar X_n - \mu\rvert \ge \varepsilon) \le \dfrac{\sigma^2}{n\varepsilon^2} \to 0$.

*Chứng minh.* Bất đẳng thức Chebyshev $P(\lvert Z - \mathbb{E}Z\rvert \ge \varepsilon) \le \mathrm{Var}(Z)/\varepsilon^2$ (hệ quả của bất đẳng thức Markov áp dụng cho $(Z - \mathbb{E}Z)^2$), với $Z = \bar X_n$ và Mệnh đề 9.3. ∎

**Mệnh đề 9.5 (tương quan là cosine).** Với dữ liệu mẫu đã trừ trung bình $\tilde x, \tilde y$: $r = \dfrac{\tilde x^\top\tilde y}{\|\tilde x\|\,\|\tilde y\|} = \cos(\tilde x, \tilde y)$. Do đó $r \in [-1, 1]$ theo Cauchy–Schwarz (bài 01). ∎

**CLT** (không chứng minh ở đây, xem Wasserman ch. 5). Mô phỏng trên máy bạn: lấy trung bình của 30 mẫu từ phân phối mũ $\mathrm{Exp}(1)$ (rất lệch, độ xiên bằng 2), lặp 100 000 lần:

| Đại lượng | Đo được | Lý thuyết |
|---|---|---|
| Trung bình của $\bar X$ | 1.000 | 1 |
| Độ lệch chuẩn của $\bar X$ | 0.182 | $1/\sqrt{30} = 0.183$ |
| Độ xiên (skewness) | 0.357 | $2/\sqrt{30} = 0.365$ (phân phối gốc: 2) |
| Tỉ lệ nằm trong $\mu \pm 1.96\,\mathrm{SE}$ | **95.2%** | 95% |

Phân phối gốc lệch rất mạnh, nhưng trung bình 30 mẫu đã gần chuẩn.

#### 9.3 Ví dụ tính tay

**(a) Khoảng tin cậy cho accuracy.** 160/200 câu đúng: $\hat p = 0.8$, $\mathrm{SE} = \sqrt{0.8 \cdot 0.2/200} = 0.0283$.
- Wald 95%: $0.8 \pm 1.96 \times 0.0283 = 0.8 \pm 0.055$, tức $[0.745,\ 0.855]$.
- Wilson 95%: $[0.739,\ 0.850]$ (lệch nhẹ về phía 0.5).

**(b) Cần bao nhiêu câu hỏi?** Muốn nửa độ rộng $E = 2$ điểm %: $n = z^2\hat p(1 - \hat p)/E^2$.
- Với $\hat p = 0.8$: $1.96^2 \cdot 0.16/0.02^2 =$ **1 537** câu.
- Trường hợp xấu nhất $\hat p = 0.5$: **2 401** câu.

**(c) Không có lỗi nào trong 50 lần thử.**
- Wald cho $[0, 0]$: vô lý, vì 50 lần thử không thể chứng minh tỉ lệ lỗi bằng đúng 0.
- Wilson cho $[0,\ 0.071]$.
- Quy tắc nhẩm *rule of three*: cận trên 95% ≈ $3/n = 0.06$.

**(d) So sánh hai model** A (80%) và B (83%), mỗi model 500 câu.
- *Nếu hai model được chấm trên hai bộ câu hỏi độc lập:* $\mathrm{SE}_{\text{diff}} = \sqrt{0.8 \cdot 0.2/500 + 0.83 \cdot 0.17/500} = 0.0245$, $z = 0.03/0.0245 = 1.22$, $p \approx 0.22$: **chưa có ý nghĩa thống kê**.
- *Nếu chấm trên **cùng** 500 câu (thiết kế ghép cặp):* chỉ các câu mà hai model **bất đồng** mới mang thông tin. Giả sử A đúng B sai 40 câu, B đúng A sai 55 câu. Kiểm định McNemar: $z = (55 - 40)/\sqrt{95} = 1.54$, $p \approx 0.12$. Vẫn chưa đủ, nhưng thiết kế ghép cặp thường mạnh hơn vì loại bỏ được độ khó riêng của từng câu.

**(e) Request tới theo Poisson**, trung bình $\lambda = 10$ req/s:
- $P(N \ge 15) = 0.083$: nếu server chỉ xử lý được 15 req/s, cứ khoảng 12 giây sẽ có 1 giây bị quá tải.
- $P(N \ge 20) = 0.0035$.
- Thời gian giữa hai request có phân phối $\mathrm{Exp}(10)$: $P(\text{khoảng cách} > 0.3\text{ s}) = e^{-3} = 0.050$.

**(f) Hiệp phương sai.** $x = (1, 2, 3, 4)$, $y = (2, 4, 5, 9)$, trung bình $2.5$ và $5$:
- $\mathrm{Cov} = \frac{1}{3}[(-1.5)(-3) + (-0.5)(-1) + (0.5)(0) + (1.5)(4)] = 11/3 = 3.667$.
- $s_x^2 = 5/3$, $s_y^2 = 26/3$, nên $r = 3.667/\sqrt{1.667 \times 8.667} = 0.965$.

#### 9.4 Code chuẩn

```python
from scipy import stats

def mean_ci(samples, level=0.95):
    x = np.asarray(samples, dtype=float)
    n = x.size
    half = stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)   # ddof=1: chia n−1
    return x.mean(), x.mean() - half, x.mean() + half
```

**Vì sao dùng phân phối $t$ chứ không dùng 1.96?** Khi $n$ nhỏ, ta ước lượng $\sigma$ bằng $s$, và chính $s$ cũng có sai số. Phân phối $t$ có đuôi dày hơn để bù cho điều đó. $t_{0.975}$ bằng **2.776** với 4 bậc tự do (5 seed), 2.262 với 9, 2.045 với 29, và tiến về 1.96 khi $n$ lớn. Project dùng hàm này để báo kết quả thí nghiệm chạy trên 5 seed.

#### 9.5 Cạm bẫy

1. **Báo một con số accuracy không kèm khoảng tin cậy.** "Model mới tốt hơn 1.5 điểm" trên 300 câu hỏi gần như chắc chắn là nhiễu.
2. **Dùng 1.96 với $n = 5$.** Khoảng tin cậy hẹp hơn thực tế khoảng 30%.
3. **Dữ liệu không i.i.d.** Nhiều câu hỏi sinh ra từ cùng một tài liệu thì tương quan với nhau; SE thật lớn hơn công thức. Cần *clustered standard errors* (Miller 2024).
4. **Tương quan bằng 0 không có nghĩa là độc lập** (Mệnh đề 9.2). Và tương quan không phải nhân quả.
5. **Wald interval khi $\hat p$ gần 0 hoặc 1.** Dùng Wilson.
6. **Nhầm độ lệch chuẩn (độ phân tán của dữ liệu) với sai số chuẩn** (độ không chắc chắn của trung bình).

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"Cần bao nhiêu mẫu test?"* Dùng công thức ở 9.3(b); nêu rõ độ rộng khoảng tin cậy mong muốn.
- *"Model B hơn A 2 điểm, có nên deploy không?"* Hỏi lại cỡ mẫu, thiết kế ghép cặp hay không, và khoảng tin cậy của **hiệu**.

#### 9.6 Đọc thêm

- L. Wasserman, *All of Statistics*, ch. 2–6.
- Blitzstein & Hwang ch. 3–5 và 10.
- E. Miller, *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations*, arXiv:2411.00640 (2024).
- L. Brown, T. Cai, A. DasGupta, *Interval Estimation for a Binomial Proportion*, Statistical Science 2001.

---

## 10 · MLE và MAP: loss function đến từ đâu

> ⏱ Tầng 1: 15' · Tầng 2: 40'
> **Cần biết trước:** Bayes (bài 08); phân phối Bernoulli và Gaussian (bài 09); đạo hàm (bài 04); bình phương tối thiểu (bài 02).

> **📖 Từ mới**
> - **Likelihood $p(D \mid \theta)$:** với tham số $\theta$, dữ liệu $D$ mà ta đã thấy "dễ xảy ra" tới mức nào.
> - **MLE (Maximum Likelihood Estimation):** chọn $\theta$ làm likelihood lớn nhất.
> - **Negative log-likelihood (NLL):** $-\log p(D \mid \theta)$. Cực đại likelihood tương đương cực tiểu NLL.
> - **MAP (Maximum A Posteriori):** chọn $\theta$ làm **posterior** lớn nhất, tức kết hợp dữ liệu với niềm tin ban đầu (prior).
> - **Regularization:** thêm vào loss một số hạng phạt tham số lớn, để model không học thuộc dữ liệu.
> - **Ridge / Lasso:** linear regression có phạt L2 / phạt L1.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- Tung một đồng xu lạ 10 lần, được 7 lần ngửa. Đoán $p = 0.7$ là tự nhiên, vì trong mọi giá trị $p$, đó là giá trị làm "7/10 ngửa" **dễ xảy ra nhất**. Đó là MLE.
- Tung 3 lần, được 0 lần ngửa. MLE nói $p = 0$, tức đồng xu **không bao giờ** ra ngửa. Quá tự tin! Bạn vốn tin đồng xu "chắc cũng gần cân bằng"; kết hợp niềm tin đó với dữ liệu sẽ cho một con số hợp lý hơn, ví dụ $0.2$. Đó là MAP.
- Điều bất ngờ: **mọi loss quen thuộc đều là NLL** của một giả định xác suất. MSE ứng với nhiễu Gauss; BCE ứng với nhãn Bernoulli; cross-entropy của LLM ứng với token Categorical; và L2 regularization ứng với prior Gauss trên trọng số.

**Ba ý cốt lõi:**
1. Huấn luyện model = MLE = cực tiểu NLL.
2. Chọn loss là **chọn giả định** về dữ liệu.
3. Regularization = MAP = thêm prior.

---

### TẦNG 2 · Kiến thức chuẩn

#### 10.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Likelihood** | $\mathcal{L}(\theta) = p(D \mid \theta) = \prod_{i=1}^n p(x_i \mid \theta)$ với dữ liệu i.i.d. | Bishop, *PRML* §1.2.4 |
| **MLE** | $\hat\theta_{\text{MLE}} = \arg\max_\theta \sum_i \log p(x_i \mid \theta)$ | Goodfellow §5.5 |
| **MAP** | $\hat\theta_{\text{MAP}} = \arg\max_\theta\,[\log p(D \mid \theta) + \log p(\theta)]$ (bỏ $p(D)$ vì không phụ thuộc $\theta$) | Goodfellow §5.6.1 |
| **Conjugate prior** | Prior mà posterior cùng họ với nó. Beta là conjugate của Bernoulli; Dirichlet là conjugate của Categorical | Murphy, *PML1* §4.6 |
| **Bessel's correction** | Chia $n - 1$ để ước lượng phương sai không chệch | — |
| **Ridge / Lasso** | $\min \|X\theta - y\|^2 + \lambda\|\theta\|_2^2$ / $\min \tfrac{1}{2}\|X\theta - y\|^2 + \lambda\|\theta\|_1$ | Hoerl & Kennard (1970) / Tibshirani (1996) |

#### 10.2 Mệnh đề và chứng minh

**Mệnh đề 10.1 (MLE của Bernoulli).** $k$ lần thành công trong $n$ lần thử thì $\hat p = k/n$.

*Chứng minh.* $\ell(p) = k\ln p + (n - k)\ln(1 - p)$. $\ell'(p) = \dfrac{k}{p} - \dfrac{n - k}{1 - p} = 0 \Rightarrow p = k/n$. $\ell''(p) = -\dfrac{k}{p^2} - \dfrac{n - k}{(1 - p)^2} < 0$, nên đây là cực đại. ∎

**Mệnh đề 10.2 (MLE của Gaussian).** $\hat\mu = \bar x$ và $\hat\sigma^2 = \frac{1}{n}\sum(x_i - \bar x)^2$. Ước lượng phương sai này **bị chệch**: $\mathbb{E}[\hat\sigma^2] = \frac{n - 1}{n}\sigma^2$.

*Chứng minh (ý chính).* Đạo hàm của $\sum\left[-\frac{(x_i - \mu)^2}{2\sigma^2} - \frac{1}{2}\ln\sigma^2\right]$ theo $\mu$ và $\sigma^2$, cho bằng 0. Độ chệch đến từ việc dùng $\bar x$ thay cho $\mu$ thật: $\sum(x_i - \bar x)^2 = \sum(x_i - \mu)^2 - n(\bar x - \mu)^2$; lấy kỳ vọng được $n\sigma^2 - \sigma^2$. ∎

Mô phỏng với $n = 5$, $\sigma^2 = 1$: trung bình của $\hat\sigma^2$ là **0.7995**, lý thuyết $4/5 = 0.8$ ✓. Ví dụ dữ liệu $(2, 4, 4, 6)$: $\bar x = 4$, $\hat\sigma^2_{\text{MLE}} = 8/4 = 2$, còn bản không chệch là $8/3 = 2.667$.

**Mệnh đề 10.3 (MSE = MLE với nhiễu Gauss).** Giả sử $y_i = \theta^\top x_i + \varepsilon_i$ với $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ i.i.d. Khi đó

$$-\log p(y \mid X, \theta) = \frac{1}{2\sigma^2}\sum_i(y_i - \theta^\top x_i)^2 + \frac{n}{2}\ln(2\pi\sigma^2).$$

Số hạng thứ hai không phụ thuộc $\theta$, nên **MLE chính là nghiệm bình phương tối thiểu** của bài 02. ∎

**Mệnh đề 10.4 (BCE và CE = MLE).**
- Với $y_i \mid x_i \sim \mathrm{Bernoulli}(\sigma(\theta^\top x_i))$: NLL $= \sum_i -[y_i\ln p_i + (1 - y_i)\ln(1 - p_i)]$, đúng là BCE.
- Với nhãn Categorical và $p = \mathrm{softmax}(z)$: NLL là cross-entropy.
- **Pretraining LLM là MLE:** mỗi token tiếp theo được mô hình hóa là Categorical trên từ điển, và loss là trung bình $-\ln p_\theta(x_t \mid x_{<t})$.

**Mệnh đề 10.5 (L2 = prior Gauss, L1 = prior Laplace).** Thêm prior $\theta \sim \mathcal{N}(0, \tau^2 I)$ vào Mệnh đề 10.3:

$$-\log p(\theta \mid D) = \frac{1}{2\sigma^2}\|X\theta - y\|^2 + \frac{1}{2\tau^2}\|\theta\|^2 + \text{const},$$

nên MAP chính là **ridge** với $\lambda = \sigma^2/\tau^2$. Prior càng chặt ($\tau$ nhỏ) thì phạt càng nặng. Với prior Laplace $p(\theta_j) \propto e^{-\lvert\theta_j\rvert/b}$, số hạng phạt là $\|\theta\|_1/b$, tức **lasso**.

*Ridge còn sửa được điều kiện của bài toán:* nghiệm là $(X^\top X + \lambda I)^{-1}X^\top y$, và cộng $\lambda I$ làm mọi trị riêng tăng thêm $\lambda$. Ma trận luôn khả nghịch khi $\lambda > 0$, và $\kappa = \dfrac{\lambda_{\max} + \lambda}{\lambda_{\min} + \lambda}$. Ví dụ với $\lambda_{\max} = 100$, $\lambda_{\min} = 0.01$:

| $\lambda$ | 0 | 0.1 | 1 |
|---|---|---|---|
| $\kappa$ | 10 000 | 910 | 100 |

**Mệnh đề 10.6 (L1 cho nghiệm thưa, L2 thì không).** Trong một chiều, xét $\min_\theta \tfrac{1}{2}(\theta - a)^2 + \text{phạt}$:
- Phạt L2 $\tfrac{\lambda}{2}\theta^2$: $\theta^\star = \dfrac{a}{1 + \lambda}$, **không bao giờ bằng 0** khi $a \ne 0$.
- Phạt L1 $\lambda\lvert\theta\rvert$: $\theta^\star = \mathrm{sign}(a)\max(\lvert a\rvert - \lambda,\ 0)$ (*soft thresholding*), **bằng 0** khi $\lvert a\rvert \le \lambda$.

*Chứng minh (L1).* Với $\theta > 0$: đạo hàm $\theta - a + \lambda = 0$ cho $\theta = a - \lambda$, chỉ hợp lệ khi $a > \lambda$. Tương tự với $\theta < 0$: $\theta = a + \lambda$ khi $a < -\lambda$. Khi $\lvert a\rvert \le \lambda$, hàm giảm trên $(-\infty, 0)$ và tăng trên $(0, \infty)$, nên cực tiểu tại 0. ∎

Ví dụ $\lambda = 0.5$: $a = 0.3$ cho L1 → **0**, L2 → 0.2; $a = 2$ cho L1 → 1.5, L2 → 1.333.

**Mệnh đề 10.7 (Laplace smoothing = MAP với prior Beta).** Prior $\mathrm{Beta}(a, b)$ và $k$ thành công trong $n$ lần thử cho posterior $\mathrm{Beta}(a + k,\ b + n - k)$, có mode là $\dfrac{k + a - 1}{n + a + b - 2}$. Với $a = b = 2$, ta được $\dfrac{k + 1}{n + 2}$: **cộng 1 vào mỗi ô đếm**, đúng như smoothing của Naive Bayes ở bài 08 (bản nhiều lớp dùng prior Dirichlet).
- 7/10: MLE 0.7, MAP 0.667.
- 0/3: MLE **0**, MAP **0.2**.

#### 10.3 Bảng tổng kết: loss ↔ giả định

| Loss | Giả định xác suất | Bài |
|---|---|---|
| MSE | $y \mid x \sim \mathcal{N}(f_\theta(x), \sigma^2)$ | 02, 05 |
| BCE | $y \mid x \sim \mathrm{Bernoulli}(\sigma(f_\theta(x)))$ | 05 |
| Cross-entropy (phân loại, LLM) | $y \mid x \sim \mathrm{Categorical}(\mathrm{softmax}(f_\theta(x)))$ | 05, 11 |
| + L2 / weight decay | Prior $\theta \sim \mathcal{N}(0, \tau^2 I)$ | 07, 10 |
| + L1 | Prior Laplace | 10 |

#### 10.4 Ví dụ tính tay: vì sao phân loại dùng cross-entropy chứ không dùng MSE

Một mẫu có $y = 1$ nhưng model **sai trầm trọng**: logit $z = -5$, nên $p = \sigma(-5) = 0.0067$.
- Với BCE: $\partial\ell/\partial z = p - y = $ **−0.993**: tín hiệu sửa sai mạnh.
- Với MSE $\tfrac{1}{2}(p - y)^2$: $\partial\ell/\partial z = (p - y)\,p(1 - p) = $ **−0.0066**: yếu hơn **150 lần**, vì đạo hàm của sigmoid gần 0 ở vùng bão hòa.

Kết luận: MSE trên sigmoid học **rất chậm** đúng ở những mẫu mà model sai nhiều nhất. BCE "triệt tiêu" $\sigma'$ (Mệnh đề 5.3).

**BCE trung bình của 4 mẫu**, $y = (1, 0, 1, 1)$, $p = (0.9, 0.2, 0.6, 0.99)$: $-\tfrac{1}{4}(\ln 0.9 + \ln 0.8 + \ln 0.6 + \ln 0.99) = $ **0.2123**. Mẫu thứ 3 ($p = 0.6$) đóng góp nhiều nhất.

#### 10.5 Cạm bẫy

1. **MLE trên dữ liệu nhỏ** cho xác suất bằng 0 hoặc 1, và sau đó $\log 0 = -\infty$. Dùng MAP hoặc smoothing.
2. **Cường độ phạt phụ thuộc cách chuẩn hóa loss.** $\lambda$ đi với loss dạng **tổng** khác $\lambda$ đi với loss dạng **trung bình** $n$ lần. Project dùng loss trung bình: $(X^\top X/n + \ell_2 I)\theta = X^\top y/n$.
3. **Phạt cả bias.** Thường không nên: bias chỉ dịch dự đoán, không làm model phức tạp hơn. (Project phạt cả bias cho đơn giản.)
4. **MSE với dữ liệu có ngoại lệ (outlier).** Giả định Gauss có đuôi rất mỏng, nên một outlier kéo lệch cả nghiệm. Dùng L1 loss hoặc Huber loss.
5. **Dùng $\hat\sigma^2$ chia $n$** khi $n$ nhỏ để làm khoảng tin cậy.

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"Vì sao phân loại dùng cross-entropy mà không dùng MSE?"* Hai lý do: (1) CE là MLE của mô hình Bernoulli/Categorical; (2) gradient không bị triệt tiêu ở vùng bão hòa (mục 10.4).
- *"Weight decay có ý nghĩa xác suất gì?"* Prior Gauss (Mệnh đề 10.5).
- *"L1 hay L2?"* L1 tạo nghiệm thưa (chọn đặc trưng); L2 co đều và ổn định hơn khi các đặc trưng tương quan.

#### 10.6 Đọc thêm

- C. Bishop, *Pattern Recognition and Machine Learning*, §1.2 và §3.1–3.3.
- K. Murphy, *Probabilistic Machine Learning: An Introduction* (2022), ch. 4.
- Goodfellow §5.5–5.6.

---

## 11 · Entropy, cross-entropy, KL divergence và perplexity

> ⏱ Tầng 1: 15' · Tầng 2: 40'
> **Cần biết trước:** MLE (bài 10); softmax (bài 04); logarit.

> **📖 Từ mới**
> - **Lượng thông tin (surprise):** $-\log p(x)$. Sự kiện càng hiếm, ta càng "bất ngờ".
> - **Entropy:** độ bất ngờ **trung bình**; cũng là số câu hỏi có/không trung bình cần để đoán ra kết quả.
> - **Bit / nat:** đơn vị khi dùng $\log_2$ / $\ln$.
> - **Cross-entropy:** độ bất ngờ trung bình khi thực tế theo $p$ nhưng ta lại tin theo $q$.
> - **KL divergence:** phần bất ngờ **dư ra** do tin sai: cross-entropy trừ entropy.
> - **Perplexity:** "model đang phân vân giữa bao nhiêu lựa chọn", bằng $e^{\text{cross-entropy}}$.
> - **Log-sum-exp:** cách tính $\log\sum e^{z_i}$ mà không bị tràn số.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- Tin "sáng nay mặt trời mọc" chẳng mang thông tin gì: xác suất gần 1, độ bất ngờ gần 0. Tin "bạn trúng xổ số" rất bất ngờ.
- Đoán kết quả tung đồng xu cân bằng cần đúng **1** câu hỏi có/không: entropy bằng 1 bit. Đồng xu lệch 90/10 thì dễ đoán hơn: entropy chỉ còn **0.469** bit.
- Nếu bạn **tin** đồng xu cân bằng nhưng nó thật ra lệch, bạn bất ngờ nhiều hơn mức cần thiết. Phần dư đó là KL divergence.
- LLM đoán token tiếp theo. Loss của nó là cross-entropy giữa token thật và phân phối mà model dự đoán. Loss bằng 2.0 nat nghĩa là perplexity $e^2 = 7.4$: trung bình, model phân vân như thể phải chọn giữa khoảng 7 token.

**Ba ý cốt lõi:**
1. Entropy đo độ bất định của **một** phân phối.
2. Cross-entropy = entropy + KL. Cực tiểu cross-entropy theo model **chính là** cực tiểu KL tới phân phối dữ liệu, và chính là MLE.
3. Perplexity = $e^{\text{loss}}$ là cách đọc loss của LLM theo đơn vị "số lựa chọn".

---

### TẦNG 2 · Kiến thức chuẩn

#### 11.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Entropy** | $H(p) = -\sum_x p(x)\log p(x)$, quy ước $0\log 0 = 0$ | Shannon (1948); Cover & Thomas §2.1 |
| **Bits / nats** | $\log_2$ / $\ln$; 1 nat $= 1/\ln 2 = 1.4427$ bit | Cover & Thomas |
| **Cross-entropy** | $H(p, q) = -\sum_x p(x)\log q(x)$ | Goodfellow §3.13 |
| **KL divergence** | $D_{\mathrm{KL}}(p\,\|\,q) = \sum_x p(x)\log\dfrac{p(x)}{q(x)}$; bằng $+\infty$ nếu tồn tại $x$ với $p(x) > 0$ mà $q(x) = 0$ | Kullback & Leibler (1951) |
| **Mutual information** | $I(X; Y) = D_{\mathrm{KL}}\big(p(x, y)\,\|\,p(x)p(y)\big) = H(X) - H(X \mid Y)$ | Cover & Thomas §2.4 |
| **Perplexity** | $\mathrm{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^T\ln p_\theta(x_t \mid x_{<t})\right)$ | Jurafsky & Martin, *SLP3* ch. 3 |
| **Log-sum-exp** | $\mathrm{LSE}(z) = \ln\sum_i e^{z_i} = m + \ln\sum_i e^{z_i - m}$, với $m = \max_i z_i$ | — |

#### 11.2 Mệnh đề và chứng minh

**Mệnh đề 11.1 (phân tích cross-entropy).** $H(p, q) = H(p) + D_{\mathrm{KL}}(p\,\|\,q)$.

*Chứng minh.* $-\sum p\log q = -\sum p\log p + \sum p\log\dfrac{p}{q}$. ∎

**Mệnh đề 11.2 (bất đẳng thức Gibbs).** $D_{\mathrm{KL}}(p\,\|\,q) \ge 0$, dấu bằng khi và chỉ khi $p = q$.

*Chứng minh.* Dùng $\ln t \le t - 1$ (dấu bằng khi $t = 1$), lấy tổng trên các $x$ có $p(x) > 0$:

$$-D_{\mathrm{KL}}(p\,\|\,q) = \sum p\ln\frac{q}{p} \le \sum p\left(\frac{q}{p} - 1\right) = \sum_{p(x) > 0}q(x) - 1 \le 0.$$

Dấu bằng cần $q/p = 1$ trên mọi $x$ có $p(x) > 0$, tức $p = q$. ∎

**Hệ quả 11.3 (cross-entropy loss = MLE = cực tiểu KL).** Với $\hat p$ là phân phối thực nghiệm của dữ liệu, $H(\hat p, q_\theta) = -\frac{1}{n}\sum_i\ln q_\theta(x_i)$ đúng bằng NLL trung bình (bài 10). Vì $H(\hat p)$ không phụ thuộc $\theta$, cực tiểu cross-entropy $\Leftrightarrow$ cực tiểu $D_{\mathrm{KL}}(\hat p\,\|\,q_\theta)$ $\Leftrightarrow$ MLE. Ba cách nhìn, một bài toán. ∎

**Mệnh đề 11.4 (cận của entropy).** Với $K$ giá trị: $0 \le H(p) \le \log K$, cực đại khi $p$ đều.

*Chứng minh.* Với $u$ là phân phối đều, $D_{\mathrm{KL}}(p\,\|\,u) = \sum p\log(pK) = \log K - H(p) \ge 0$ theo Gibbs. ∎ Hệ quả: một LM đoán **đều** trên từ điển $V$ token có $\mathrm{PPL} = V$. Mọi LM có ích phải có PPL nhỏ hơn hẳn $V$.

**Mệnh đề 11.5 (log-sum-exp ổn định).** $\ln\sum_i e^{z_i} = m + \ln\sum_i e^{z_i - m}$ với $m = \max_i z_i$.

*Chứng minh.* Đưa $e^m$ ra ngoài tổng. ∎ *Vì sao ổn định:* mọi $z_i - m \le 0$ nên không có `exp` nào tràn số; số hạng lớn nhất bằng $e^0 = 1$ nên tổng $\ge 1$, không bao giờ xảy ra $\ln 0$. Từ đó $\log\mathrm{softmax}(z)_i = z_i - \mathrm{LSE}(z)$.

**KL không đối xứng.** Trong ví dụ ở mục 11.3, $D_{\mathrm{KL}}(p\,\|\,q) = 0.123$ bit nhưng $D_{\mathrm{KL}}(q\,\|\,p) = 0.133$ bit. Khi tối ưu $q$ để xấp xỉ $p$:
- **KL thuận** $D_{\mathrm{KL}}(p\,\|\,q)$ phạt nặng khi $q \approx 0$ ở chỗ $p > 0$, nên $q$ cố **phủ mọi vùng** của $p$ (*mode-covering*). Đây là cái mà MLE và distillation tối ưu.
- **KL ngược** $D_{\mathrm{KL}}(q\,\|\,p)$ phạt nặng khi $q > 0$ ở chỗ $p \approx 0$, nên $q$ **chọn một vùng** của $p$ (*mode-seeking*). Bạn sẽ gặp lại trong variational inference và trong số hạng phạt KL của RLHF (Tuần 11).

#### 11.3 Ví dụ tính tay

**(a)** $p = (0.7, 0.2, 0.1)$, $q = (0.5, 0.3, 0.2)$, đơn vị bit:

| Đại lượng | Phép tính | Kết quả |
|---|---|---|
| $H(p)$ | $0.7\log_2\frac{1}{0.7} + 0.2\log_2 5 + 0.1\log_2 10 = 0.360 + 0.464 + 0.332$ | **1.157** |
| $H(p, q)$ | $0.7\log_2 2 + 0.2\log_2\frac{1}{0.3} + 0.1\log_2 5 = 0.700 + 0.347 + 0.232$ | **1.280** |
| $D_{\mathrm{KL}}(p\,\|\,q)$ | $1.280 - 1.157$ | **0.123** |
| $D_{\mathrm{KL}}(q\,\|\,p)$ | tính trực tiếp | **0.133** |

Theo nat (nhân với $\ln 2 = 0.693$): $H(p) = 0.802$, $H(p, q) = 0.887$, $D_{\mathrm{KL}} = 0.085$.

**(b) Loss với nhãn one-hot.** Nếu nhãn đúng có xác suất dự đoán 0.7, loss $= -\ln 0.7 = 0.357$ nat. Với nhãn one-hot $H(y) = 0$, nên cross-entropy **bằng** KL.

**(c) Perplexity.**

| Mean NLL (nat/token) | PPL |
|---|---|
| 2.0 | 7.39 |
| 3.0 | 20.1 |
| $\ln 32\,000 = 10.37$ (đoán đều trên 32 000 token) | 32 000 |

**(d) Mutual information** từ bảng ở bài 08: $I(\text{loại câu};\ \text{đúng/sai}) = \sum p(x, y)\log_2\dfrac{p(x, y)}{p(x)p(y)} = $ **0.108** bit, trong khi $H(\text{đúng/sai}) = 0.904$ bit. Biết loại câu hỏi giảm được khoảng 12% độ bất định về việc chatbot trả lời đúng hay sai.

**(e) Tràn số.** Với $z = (1000, 1001)$, đo trên máy bạn:
- `np.exp(z) / np.exp(z).sum()` cho `[nan nan]` kèm cảnh báo overflow.
- Bản ổn định cho $(0.2689,\ 0.7311)$; $\mathrm{LSE}(z) = 1001.3133$.

#### 11.4 Code chuẩn

```python
def logsumexp(a, axis=-1):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)), axis=axis)

def cross_entropy_from_logits(z, y):          # z: (n, K) logit, y: (n,) chỉ số lớp đúng
    log_p = z - logsumexp(z, axis=1)[:, None]  # log-softmax, không bao giờ tạo ra log(0)
    return -log_p[np.arange(len(y)), y].mean()
```

#### 11.5 Cạm bẫy

1. **`p * np.log(p)` khi $p = 0$** cho `0 * -inf = nan`. Phải lọc riêng các $p > 0$ (quy ước $0\log 0 = 0$).
2. **$q(x) = 0$ ở chỗ $p(x) > 0$** làm KL và cross-entropy bằng $+\infty$. Đây là lý do có *label smoothing*, và lý do không bao giờ được để model dự đoán xác suất 0 tuyệt đối.
3. **Lẫn bit và nat.** Loss của PyTorch dùng $\ln$ (nat).
4. **So sánh perplexity giữa hai model dùng tokenizer khác nhau.** "Mỗi token" có ý nghĩa khác nhau, nên con số không so sánh được. Phải quy về cùng một đơn vị, ví dụ bits-per-byte.
5. **Tự viết softmax rồi `log`.** Luôn dùng log-softmax hoặc hàm loss nhận thẳng logit.

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"Vì sao LLM dùng cross-entropy?"* Hệ quả 11.3: đó là MLE, và tương đương cực tiểu KL tới phân phối dữ liệu.
- *"Perplexity là gì? So sánh được giữa các model không?"* Chỉ khi cùng tokenizer và cùng tập dữ liệu.
- *"Forward KL và reverse KL khác nhau thế nào?"* Mode-covering và mode-seeking.

#### 11.6 Đọc thêm

- C. Shannon, *A Mathematical Theory of Communication*, Bell System Technical Journal, 1948.
- T. Cover & J. Thomas, *Elements of Information Theory* (2nd ed.), ch. 2.
- Goodfellow §3.13.
- Jurafsky & Martin, *SLP3*, ch. 3 (perplexity).

---

## Bài tập Ngày 3 · `optlab/stats.py` (~2h)

Test: `python -m pytest -q tests/test_stats.py` (7 test). File này cần `scipy` (cho phân phối $t$).

| Hàm | Yêu cầu |
|---|---|
| `logsumexp(a, axis=-1)` | Mệnh đề 11.5; không overflow với `[1000, 1001]`; đúng theo `axis` với mảng 2 chiều |
| `log_softmax(z, axis=-1)`, `softmax(z, axis=-1)` | Dựa trên `logsumexp` |
| `entropy(p, base=math.e)` | Quy ước $0\log 0 = 0$. Raise `ValueError` nếu `p` có phần tử âm hoặc tổng khác 1 (dung sai $10^{-9}$) |
| `cross_entropy(p, q, base=math.e)` | Trả về `math.inf` nếu $q(x) = 0$ tại $x$ có $p(x) > 0$ |
| `kl_divergence(p, q, base=math.e)` | Như trên. `kl_divergence(p, p) == 0` |
| `mean_ci(samples, level=0.95) -> (mean, lo, hi)` | Phân phối $t$ với $n - 1$ bậc tự do (mục 9.4). $n < 2$ thì raise `ValueError` |
| `wilson_interval(k, n, z=1.96) -> (lo, hi)` | Công thức Wilson; kẹp kết quả vào $[0, 1]$. Raise `ValueError` nếu không thỏa $0 \le k \le n$ và $n > 0$ |

**Gợi ý:** test dùng đúng các con số ở mục 11.3(a) và 9.3(a), (c).
