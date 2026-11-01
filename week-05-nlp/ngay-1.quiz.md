# Quiz · Ngày 1 · Văn bản cổ điển (bài 11–13)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> Làm trên giấy, được dùng máy tính cầm tay. **Không chạy code.** Đáp án ở cuối file.

## Phần 1 — Quick test (14 câu, câu 14 được 2đ, các câu khác 1đ)

1. Chữ "Việt" có bao nhiêu code point ở dạng NFC, và bao nhiêu ở dạng NFD? Vì sao hai số khác nhau?
2. Vì sao trong `normalize`, bước thay tiền bằng `<money>` phải chạy **trước** bước mở rộng teencode?
3. Kể 2 bước chuẩn hóa **không** làm mất thông tin, và 1 bước **có** làm mất thông tin. Cho một ví dụ cụ thể về thông tin bị mất.
4. Trên dữ liệu của tuần, bỏ dấu làm F1 theo template giảm từ 0.733 xuống 0.717, còn trên chia ngẫu nhiên thì không đổi. Giải thích ngắn gọn.
5. Tập train có $N = 9$ văn bản. Tính idf (công thức scikit-learn, `smooth_idf=True`) của một từ có df = 2, và của một từ có mặt trong cả 9 văn bản.
6. Một văn bản có hai từ, với tf × idf lần lượt là 3 và 2. Vector TF-IDF sau khi chuẩn hóa L2 là gì?
7. Liệt kê các n-gram ký tự `char_wb` với $n$ từ 2 tới 3 của từ "gói". Có bao nhiêu n-gram?
8. Một câu có 5 âm tiết khác nhau, không lặp lại. Với `ngram_range=(1, 2)`, câu đó có bao nhiêu đặc trưng khác 0?
9. Cùng một model đạt macro-F1 0.98 khi chia ngẫu nhiên nhưng chỉ 0.73 khi chia theo template. Con số nào nên báo cáo cho đội vận hành, và vì sao?
10. Cosine giữa vector one-hot của hai từ khác nhau bằng bao nhiêu? Điều đó gây ra vấn đề gì?
11. Câu 4 từ, cửa sổ skip-gram $m = 1$: có bao nhiêu cặp (trung tâm, ngữ cảnh)? Với $m = 2$ thì sao?
12. Ba từ có số lần xuất hiện 81, 16, 1. Tính xác suất mỗi từ được rút làm từ nhiễu theo $\text{count}^{0.75}$.
13. Vì sao FastText có vector cho từ "khôngg" (chưa từng thấy), còn word2vec thì không?
14. *(2đ)* Đoạn code sau có **3 lỗi**. Chỉ ra và sửa.
    ```python
    SW = ["tôi", "của", "là", "không", "được", "thì"]
    vec = CountVectorizer(preprocessor=lambda t: " ".join(w for w in normalize(t).split() if w not in SW))
    X = vec.fit_transform(df.text)                       # df là TOÀN BỘ dữ liệu
    X_tr, X_te, y_tr, y_te = train_test_split(X, df.intent, test_size=0.2)
    clf = LogisticRegression().fit(X_tr, y_tr)
    joblib.dump(clf, "model.joblib")
    ```

## Phần 2 — Tự luận (4 câu, 30 phút)

**T1 · TF-IDF và tìm câu gần nhất (4đ).** Bốn tin nhắn train: (1) "nạp tiền", (2) "nạp thẻ", (3) "trừ tiền sai", (4) "nạp tiền nạp thẻ". Dùng công thức idf của scikit-learn và tách theo âm tiết.
- (a) Lập vocabulary (sắp theo thứ tự chuỗi: `nạp`, `sai`, `thẻ`, `tiền`, `trừ`), df và idf của từng từ.
- (b) Tính vector TF-IDF đã chuẩn hóa L2 của văn bản (4).
- (c) Tin nhắn mới "bị trừ tiền". Tính vector của nó, và cosine với văn bản (1) và (3). Nó gần văn bản nào hơn?
- (d) Vì sao "trừ" có idf cao hơn "tiền"? Điều đó giúp gì cho bài toán khiếu nại?

**T2 · Có nên bỏ dấu không? (4đ).** F1 theo template trên 5 split (cùng split cho hai cấu hình):

| Split | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| `normalize` | 0.871 | 0.640 | 0.835 | 0.690 | 0.628 |
| `normalize` + bỏ dấu | 0.837 | 0.682 | 0.769 | 0.659 | 0.640 |

- (a) Tính chênh lệch theo từng split, chênh lệch trung bình và số split mà "bỏ dấu" thắng. Kết luận?
- (b) Một đồng nghiệp chỉ so trung bình của từng cột (0.733 và 0.717) rồi nói "chênh 0.016, nhỏ hơn độ lệch chuẩn 0.11 nên không có ý nghĩa". Lập luận đó sai ở đâu?
- (c) Nêu một tình huống dữ liệu khiến bỏ dấu **có lợi**, và một cách giữ được cả hai lợi ích (thông tin của dấu và khả năng chịu được tin không dấu).

**T3 · Thiết kế tiền xử lý (3đ).** Một nhà mạng nhận tin nhắn khách hàng từ SMS, Zalo và web chat, rồi phân loại intent bằng TF-IDF + logistic regression. Thiết kế bước tiền xử lý: liệt kê các bước theo thứ tự, nói rõ bước nào **không** làm và vì sao, và làm sao bảo đảm lúc phục vụ chạy đúng như lúc train.

**T4 · Skip-gram (4đ).**
- (a) Viết hàm `skipgram_pairs(tokens: list[str], m: int) -> list[tuple[str, str]]` trả về mọi cặp (trung tâm, ngữ cảnh) theo thứ tự: duyệt từ trung tâm từ trái sang phải; với mỗi từ trung tâm, duyệt ngữ cảnh từ trái sang phải. Cho kết quả với `["nạp", "thẻ", "cào"]`, `m = 1`.
- (b) Với $v_c = (1, 0)$, $u_o = (1, 1)$, hai từ nhiễu $u_{n_1} = (1, 0)$, $u_{n_2} = (0, 2)$, tính loss negative sampling $\ell = -\log\sigma(u_o^\top v_c) - \sum_i \log\sigma(-u_{n_i}^\top v_c)$.
- (c) Số hạng nào đóng góp lớn nhất vào loss, và gradient sẽ đẩy $v_c$ theo hướng nào so với $u_{n_1}$?

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Phần 1**

1. NFC: **4** (V, i, ệ, t). NFD: **6** (V, i, e, dấu nặng U+0323, dấu mũ U+0302, t). NFD tách "ệ" thành chữ gốc và hai dấu kết hợp.
2. "50k" chứa chữ "k" nghĩa là **nghìn**. Nếu mở rộng teencode trước, "50 k" sẽ thành "50 không" và không còn nhận ra là số tiền.
3. Không mất thông tin (với bài toán phân loại intent): NFC, thống nhất kiểu đặt dấu, chữ thường, placeholder cho SĐT (bất kỳ 2). Có mất: **bỏ dấu**, ví dụ "đăng"/"đang", "hông"/"hỏng" thành một. (Bỏ stopword cũng được chấp nhận: mất "không".)
4. Chia ngẫu nhiên: mọi template đã có ở train, kể cả bản không dấu của chúng, nên model nhận ra câu dù có dấu hay không. Chia theo template: model phải dựa vào **từng từ** để khái quát; bỏ dấu gộp các từ khác nghĩa, làm đặc trưng kém phân biệt hơn.
5. df = 2: $\ln(10/3) + 1 =$ **2.204**. df = 9: $\ln(10/10) + 1 =$ **1** (không bằng 0 nhờ "+1" cuối).
6. $(3, 2)/\sqrt{13} =$ **(0.832, 0.555)**.
7. `" g"`, `"gó"`, `"ói"`, `"i "`, `" gó"`, `"gói"`, `"ói "`: **7** n-gram (" gói " dài 5 ký tự, nên $n = 3$ còn lấy được 3 cái).
8. 5 unigram + 4 bigram = **9**.
9. Báo **0.73** (chia theo template), vì người dùng thật sẽ diễn đạt theo những cách chưa có trong dữ liệu train. 0.98 bị thổi phồng do câu gần trùng nhau nằm ở cả train lẫn test: một dạng rò rỉ.
10. **0**. Mọi cặp từ khác nhau đều "xa" như nhau, nên model không chuyển được kiến thức từ "hủy" sang "cắt", "ngừng".
11. $m = 1$: $1 + 2 + 2 + 1 =$ **6**. $m = 2$: $2 + 3 + 3 + 2 =$ **10**.
12. $81^{0.75} = 27$, $16^{0.75} = 8$, $1^{0.75} = 1$; tổng 36, nên xác suất là **0.75, 0.222, 0.028** (so với 0.827, 0.163, 0.010 theo tần suất thuần).
13. Vector của FastText là trung bình (hoặc tổng) vector của các **n-gram ký tự** (`<kh`, `khô`, `hôn`, `ông`, `ngg`, …). Phần lớn các n-gram đó đã thấy trong "không". Word2vec chỉ có một vector cho mỗi **từ** đã thấy.
14. *(mỗi lỗi khoảng 0.67đ)*
    - Stoplist có **"không"** (và "được"): mất phủ định. Bỏ chúng khỏi list, hoặc không bỏ stopword.
    - **Fit vectorizer trên toàn bộ dữ liệu** trước khi chia: vocabulary nhìn thấy test. Chia trước, rồi `fit` trên train (tốt nhất là gộp vectorizer và model vào một `Pipeline`).
    - **Chỉ lưu `clf`**, không lưu vectorizer, nên lúc phục vụ không biến đổi văn bản giống lúc train. Ngoài ra, `lambda` làm Pipeline không pickle được. Lưu cả `Pipeline`, dùng hàm cấp module hoặc `functools.partial`.
    - *(Chấp nhận thêm:)* `CountVectorizer` mặc định bỏ token 1 ký tự như "k", "j", nên cần `token_pattern=r"<\w+>|\w+"`.

**T1 (4đ)**

(a) *(1đ)* $N = 4$.

| Từ | df | idf |
|---|---|---|
| nạp | 3 | $\ln(5/4) + 1 = 1.223$ |
| sai | 1 | $\ln(5/2) + 1 = 1.916$ |
| thẻ | 2 | $\ln(5/3) + 1 = 1.511$ |
| tiền | 3 | 1.223 |
| trừ | 1 | 1.916 |

(b) *(1đ)* tf = (nạp 2, thẻ 1, tiền 1), nên tf × idf $= (2.446,\ 0,\ 1.511,\ 1.223,\ 0)$, chuẩn bằng $3.125$, cho **(0.783, 0, 0.484, 0.392, 0)**.

(c) *(1.5đ)* "bị" không có trong vocabulary nên bị bỏ. tf × idf $= (\text{tiền } 1.223,\ \text{trừ } 1.916)$, chuẩn $2.273$, nên $q = (0, 0, 0, 0.538, 0.843)$.
- Văn bản (1) $= (0.707, 0, 0, 0.707, 0)$: cosine $= 0.707 \times 0.538 =$ **0.380**.
- Văn bản (3) $= (0, 0.645, 0, 0.411, 0.645)$: cosine $= 0.411 \times 0.538 + 0.645 \times 0.843 =$ **0.765**.
- Gần **(3) "trừ tiền sai"** hơn.

(d) *(0.5đ)* "trừ" chỉ có trong 1/4 văn bản, "tiền" có trong 3/4. Từ hiếm phân biệt tốt hơn: "tiền" có ở cả nạp tiền lẫn khiếu nại, còn "trừ" gần như chỉ có ở khiếu nại. Vì vậy câu hỏi được kéo về đúng nhóm khiếu nại.

**T2 (4đ)**

(a) *(1.5đ)* Chênh lệch (bỏ dấu − normalize): −0.034, +0.042, −0.066, −0.031, +0.012. Trung bình **−0.015**, bỏ dấu thắng **2/5** split. Kết luận: bỏ dấu không có lợi, hơi hại; **không** dùng (ít nhất là không dùng thay thế).

(b) *(1.5đ)* Độ lệch chuẩn 0.11 chủ yếu là **khác biệt giữa các split** (split 1 và 4 khó hơn hẳn), và khác biệt đó tác động lên **cả hai** cấu hình như nhau. So sánh đúng là xét **chênh lệch trên cùng split** (paired): các chênh lệch dao động khoảng ±0.04, nhỏ hơn nhiều so với 0.11. Chia cho độ lệch chuẩn của từng cột là dùng sai thang đo. Với 5 split, nên kết luận thận trọng: "không có bằng chứng bỏ dấu giúp ích".

(c) *(1đ)* Có lợi khi phần lớn tin nhắn **không dấu** (ví dụ SMS từ máy cũ), hoặc dữ liệu train toàn có dấu mà lúc phục vụ lại không dấu (lệch phân phối). Giữ cả hai: dùng **cả** đặc trưng có dấu lẫn bỏ dấu (nối hai vectorizer bằng `FeatureUnion`), hoặc dùng n-gram ký tự / subword, hoặc thêm bản bỏ dấu của dữ liệu train như một dạng augmentation.

**T3 (3đ)** *(mỗi ý khoảng 0.5đ)*
- Thứ tự: NFC → chữ thường → placeholder cho URL, email, SĐT, tiền → thống nhất kiểu đặt dấu → bỏ emoji, dấu câu (hoặc map emoji giận dữ thành token `<angry>` nếu đo thấy có ích) → teencode bằng bảng tra → token hóa.
- **Không** bỏ stopword (mất phủ định, câu ngắn); **không** thay thế văn bản bằng bản bỏ dấu (chỉ thêm, nếu đo thấy có lợi).
- Kênh khác nhau thì phân phối khác nhau (SMS ít dấu, Zalo nhiều emoji): đánh giá **theo từng kênh**.
- Bảo đảm train–serve: tiền xử lý là một **hàm thuần** trong package; nằm **trong** Pipeline (`preprocessor=`) được lưu cùng model; có test đơn vị với các câu mẫu; ghi version.
- Đánh giá bằng chia theo template (hoặc theo thời gian), so ghép cặp từng bước tiền xử lý.

**T4 (4đ)**

(a) *(1.5đ)*
```python
def skipgram_pairs(tokens, m):
    return [(c, tokens[j]) for i, c in enumerate(tokens)
            for j in range(max(0, i - m), min(len(tokens), i + m + 1)) if j != i]
```
Với `["nạp", "thẻ", "cào"]`, $m = 1$: `[("nạp", "thẻ"), ("thẻ", "nạp"), ("thẻ", "cào"), ("cào", "thẻ")]`.

(b) *(1.5đ)* Tích vô hướng: $u_o^\top v_c = 1$; $u_{n_1}^\top v_c = 1$; $u_{n_2}^\top v_c = 0$.
$$\ell = -\log\sigma(1) - \log\sigma(-1) - \log\sigma(0) = 0.313 + 1.313 + 0.693 = \mathbf{2.320}.$$

(c) *(1đ)* Số hạng của $n_1$ (1.313) lớn nhất: từ nhiễu này đang "hợp" với $v_c$ bằng đúng từ thật. Gradient của số hạng đó theo $v_c$ là $\sigma(u_{n_1}^\top v_c)\,u_{n_1} = 0.731\,(1, 0)$, nên bước gradient descent đẩy $v_c$ **ngược hướng** $u_{n_1}$ (giảm tích vô hướng), đồng thời kéo $v_c$ về phía $u_o$.

</details>
