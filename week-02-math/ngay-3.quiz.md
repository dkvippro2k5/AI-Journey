# Quiz · Ngày 3 · Xác suất, Thống kê & Thông tin (bài 08–11)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> Làm trên giấy, được dùng máy tính cầm tay. **Không chạy code.** Đáp án ở cuối file.

## Phần 1 — Quick test (14 câu, câu 14 được 2đ, các câu khác 1đ)

1. $P(A) = 0.3$, $P(B) = 0.5$, $P(A, B) = 0.15$. $A$ và $B$ có độc lập không? Tính $P(A \mid B)$.
2. Một xét nghiệm có TPR 90% và FPR 10%; tỉ lệ mắc bệnh là 2%. Người được xét nghiệm có kết quả dương tính thì xác suất mắc bệnh là bao nhiêu?
3. Prior $P(H) = 0.2$, likelihood ratio của bằng chứng bằng 6. Tính posterior $P(H \mid E)$ bằng dạng odds.
4. $X \sim \mathrm{Binomial}(100, 0.3)$. Tính $\mathbb{E}[X]$ và $\mathrm{Var}(X)$.
5. $X \sim \mathrm{Uniform}(0, 8)$. Tính kỳ vọng và phương sai.
6. Đúng/Sai: $\mathrm{Cov}(X, Y) = 0$ thì $X$ và $Y$ độc lập.
7. Model đạt accuracy 0.9 trên 400 câu. Tính SE và nửa độ rộng của khoảng tin cậy Wald 95%.
8. Cần tối thiểu bao nhiêu câu hỏi để khoảng tin cậy 95% của accuracy có nửa độ rộng ≤ 1 điểm %, trong trường hợp xấu nhất?
9. 25 mẫu i.i.d. có độ lệch chuẩn $\sigma = 10$. Tính độ lệch chuẩn của trung bình mẫu.
10. 3 lần thành công trong 12 lần thử. Tính MLE của $p$, và MAP với prior $\mathrm{Beta}(2, 2)$.
11. Ridge ứng với prior nào? Với nhiễu $\sigma^2 = 1$ và prior $\tau^2 = 0.1$ thì $\lambda$ bằng bao nhiêu?
12. Cực tiểu $\tfrac{1}{2}(\theta - a)^2 + \lambda\lvert\theta\rvert$ với $a = -0.8$, $\lambda = 0.5$. Nếu thay bằng phạt $\tfrac{\lambda}{2}\theta^2$ thì nghiệm là bao nhiêu?
13. Tính entropy (bit) của phân phối $(0.5,\ 0.25,\ 0.25)$.
14. *(2đ)* Một LM có mean NLL 2.5 nat/token. Tính perplexity, và loss theo bit/token. Model này có tốt hơn một model đoán đều trên 50 000 token không? Vì sao?

## Phần 2 — Tự luận (4 câu, 30 phút)

**T1 · Chứng minh (4đ).**
- (a) Chứng minh $D_{\mathrm{KL}}(p\,\|\,q) \ge 0$ (bất đẳng thức Gibbs), chỉ rõ khi nào xảy ra dấu bằng.
- (b) Từ (a), chứng minh entropy của một phân phối trên $K$ giá trị không vượt quá $\log K$.

**T2 · Naive Bayes (4đ).** Dùng đúng tập huấn luyện 6 tin nhắn và Laplace smoothing ở bài 08 (spam có 12 token, ham có 13 token, $|V| = 14$).
- (a) Tính $P(\text{spam} \mid \text{"trúng thưởng ngày mai"})$ với prior $0.5 / 0.5$.
- (b) Tin nhắn có 2 từ "kiểu spam" và 2 từ "kiểu ham", vậy vì sao kết quả không đúng bằng 0.5?
- (c) Ngoài thực tế chỉ 10% tin nhắn là spam. Tính lại posterior với prior này.

**T3 · Đánh giá hai hệ thống (3đ).** Hai hệ thống RAG A và B được chấm trên **cùng** 400 câu hỏi. A đúng 300 câu, B đúng 316 câu. Có 20 câu A đúng B sai, và 36 câu B đúng A sai.
- (a) Tính khoảng tin cậy Wald 95% cho accuracy của từng hệ thống. Hai khoảng có chồng lên nhau không?
- (b) Dùng kiểm định McNemar (xấp xỉ chuẩn, không hiệu chỉnh liên tục) để kết luận ở mức ý nghĩa 5%.
- (c) Nếu bỏ qua việc ghép cặp và dùng kiểm định hai tỉ lệ độc lập thì $z$ bằng bao nhiêu? Vì sao hai cách cho kết luận khác nhau?

**T4 · Code (4đ).** Viết `cross_entropy_from_logits(z, y) -> (loss, grad)`, trong đó `z` có shape `(n, K)` là logit, `y` có shape `(n,)` là chỉ số lớp đúng. `loss` là cross-entropy trung bình; `grad` là $\partial\,\text{loss}/\partial z$, shape `(n, K)`. Yêu cầu không overflow. Tính tay kết quả với `z = [[1000, 1001], [0, 0]]`, `y = [0, 1]`.

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Quick test**

1. $P(A)P(B) = 0.15 = P(A, B)$: **độc lập**. $P(A \mid B) = 0.15/0.5 = 0.3 = P(A)$.
2. $P(+) = 0.9 \cdot 0.02 + 0.1 \cdot 0.98 = 0.116$; $P(\text{bệnh} \mid +) = 0.018/0.116 \approx$ **15.5%**.
3. Prior odds $0.2/0.8 = 1:4$; posterior odds $6:4$; $P = 6/10 =$ **0.6**.
4. $\mathbb{E} = 30$, $\mathrm{Var} = 100 \cdot 0.3 \cdot 0.7 = 21$.
5. $\mathbb{E} = 4$, $\mathrm{Var} = 64/12 \approx 5.33$.
6. **Sai**. Ví dụ $Y = X^2$ với $X$ đối xứng quanh 0.
7. $\mathrm{SE} = \sqrt{0.09/400} = 0.015$; nửa độ rộng $= 1.96 \times 0.015 \approx$ **0.029**.
8. $1.96^2 \cdot 0.25/0.01^2 = 9\,604$ câu.
9. $10/\sqrt{25} =$ **2**.
10. MLE $3/12 = 0.25$; MAP $(3 + 1)/(12 + 2) = 4/14 \approx$ **0.286**.
11. Prior Gauss $\mathcal{N}(0, \tau^2 I)$; $\lambda = \sigma^2/\tau^2 =$ **10**.
12. L1: $\mathrm{sign}(-0.8)\max(0.8 - 0.5,\ 0) =$ **−0.3**. L2: $-0.8/1.5 \approx$ **−0.533**.
13. $0.5 \cdot 1 + 2 \cdot 0.25 \cdot 2 =$ **1.5 bit**.
14. $\mathrm{PPL} = e^{2.5} \approx 12.2$; $2.5/\ln 2 \approx 3.61$ bit/token. **Tốt hơn nhiều**: model đoán đều có PPL $= 50\,000$, tức loss $\ln 50\,000 = 10.8$ nat. *(0.5đ mỗi ý: PPL, bit, so sánh, lý do)*

**T1**

(a) Mệnh đề 11.2. Dùng $\ln t \le t - 1$ cho $t = q/p$ trên các $x$ có $p(x) > 0$. Dấu bằng $\Leftrightarrow q = p$ trên mọi $x$ có $p(x) > 0$ **và** $\sum_{p > 0} q = 1$, tức $p = q$. *(2.5đ; thiếu điều kiện dấu bằng thì trừ 0.5đ)*

(b) Với $u$ là phân phối đều trên $K$ giá trị: $D_{\mathrm{KL}}(p\,\|\,u) = \sum p\log p + \sum p\log K = \log K - H(p) \ge 0$. ∎ *(1.5đ)*

**T2** *(đã chạy kiểm chứng)*

(a) $P(w \mid \text{spam}) = (\text{count} + 1)/26$; $P(w \mid \text{ham}) = (\text{count} + 1)/27$.

| | trúng | thưởng | ngày | mai | Tích (× prior 0.5) |
|---|---|---|---|---|---|
| spam | $3/26$ | $3/26$ | $1/26$ | $1/26$ | $9/26^4 \times 0.5 = 9.85 \times 10^{-6}$ |
| ham | $1/27$ | $1/27$ | $3/27$ | $3/27$ | $9/27^4 \times 0.5 = 8.47 \times 10^{-6}$ |

$P(\text{spam} \mid \text{tin}) = 9.85/(9.85 + 8.47) \approx$ **0.538**. *(2đ)*

(b) Lớp spam có **ít token hơn** (12 so với 13), nên mẫu số smoothing nhỏ hơn ($26 < 27$): mỗi từ, kể cả từ chưa gặp, đều có xác suất hơi cao hơn ở lớp spam. Tỉ số $(27/26)^4 = 1.163$ chính là odds $0.538/0.462$. *(1đ)*

(c) Posterior odds $= 1.163 \times (0.1/0.9) = 0.129$, nên $P = 0.129/1.129 \approx$ **0.114**. *(1đ)*

**T3** *(đã chạy kiểm chứng)*

(a)
- A: $0.75 \pm 1.96\sqrt{0.75 \cdot 0.25/400} = 0.75 \pm 0.042$, tức $[0.708,\ 0.792]$.
- B: $0.79 \pm 0.040$, tức $[0.750,\ 0.830]$.
- Hai khoảng **chồng lên nhau**. Nhưng chồng nhau **không** có nghĩa là "không khác biệt": cần kiểm định trên **hiệu**. *(1đ)*

(b) McNemar chỉ dùng các câu bất đồng: $z = (36 - 20)/\sqrt{36 + 20} = 16/7.48 = 2.14$, $p \approx 0.033 < 0.05$: B **tốt hơn có ý nghĩa thống kê**. (Kiểm định nhị thức chính xác cho $p = 0.044$, cùng kết luận.) *(1đ)*

(c) $\mathrm{SE}_{\text{diff}} = \sqrt{0.75 \cdot 0.25/400 + 0.79 \cdot 0.21/400} = 0.0297$, $z = 0.04/0.0297 = 1.35$, $p \approx 0.18$: không có ý nghĩa. Lý do khác nhau: 280 câu cả hai cùng đúng và 64 câu cả hai cùng sai **không mang thông tin** về sự khác biệt, nhưng kiểm định độc lập vẫn tính phương sai của chúng vào. Ghép cặp loại bỏ được độ khó riêng của từng câu, nên mạnh hơn. **Luôn đánh giá các hệ thống trên cùng bộ câu hỏi và dùng kiểm định ghép cặp.** *(1đ)*

**T4** *(đã chạy kiểm chứng: khớp `torch.nn.functional.cross_entropy` tới 13 chữ số, cả loss lẫn gradient)*

```python
def cross_entropy_from_logits(z, y):
    z = np.asarray(z, dtype=float)
    n = len(y)
    m = z.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True)))[:, 0]   # Mệnh đề 11.5
    loss = np.mean(lse - z[np.arange(n), y])                             # −log softmax của lớp đúng
    p = np.exp(z - lse[:, None])                                         # softmax
    p[np.arange(n), y] -= 1                                              # p − y (Mệnh đề 5.4)
    return loss, p / n                                                   # chia n vì loss là trung bình
```

Tính tay:
- Hàng 1: $\mathrm{LSE} = 1001 + \ln(1 + e^{-1}) = 1001.3133$; loss $= 1001.3133 - 1000 = 1.3133$. Softmax $= (0.2689,\ 0.7311)$, nên gradient $= (0.2689 - 1,\ 0.7311)/2 = (-0.3655,\ 0.3655)$.
- Hàng 2: loss $= \ln 2 = 0.6931$; gradient $= (0.5,\ 0.5 - 1)/2 = (0.25,\ -0.25)$.
- Loss trung bình $= (1.3133 + 0.6931)/2 =$ **1.0032**.

*(Chấm: ổn định số học được 1đ · loss đúng được 1đ · gradient đúng, có chia $n$, được 1đ · tính tay đúng được 1đ.)*

</details>
