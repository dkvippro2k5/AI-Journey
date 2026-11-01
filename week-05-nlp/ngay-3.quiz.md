# Quiz · Ngày 3 · Mô hình ngôn ngữ, pretrained, metric (bài 17–19)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> Làm trên giấy, được dùng máy tính cầm tay. **Không chạy code.** Đáp án ở cuối file.

## Phần 1 — Quick test (14 câu, câu 14 được 2đ, các câu khác 1đ)

1. Viết $P(\text{"ab"})$ của một LM mức ký tự có token bắt đầu `^` và token kết thúc `$`, dưới dạng tích các xác suất có điều kiện.
2. Một LM gán xác suất 0.2 cho **mỗi** token đúng trong một câu 4 token. Perplexity bằng bao nhiêu?
3. Model chưa học gì, dự đoán **đều** trên vocab 50 000 token. Perplexity là bao nhiêu?
4. Câu 40 token, theo quy tắc MLM của BERT: trung bình bao nhiêu token được chọn, bao nhiêu thành `[MASK]`, bao nhiêu thành token ngẫu nhiên, bao nhiêu giữ nguyên? Loss tính trên bao nhiêu vị trí?
5. Vì sao BERT không thay **mọi** token được chọn bằng `[MASK]`?
6. Logit $(1, 0)$, temperature 0.5. Tính xác suất hai token.
7. Logit $(3, 1, 2, 0)$, top-k với $k = 2$. Phân phối dùng để lấy mẫu là gì?
8. Xác suất $(0.5, 0.3, 0.15, 0.05)$, top-p với $p = 0.85$. Những token nào được giữ?
9. LM ký tự có perplexity 1.5 mỗi ký tự; trung bình 4 ký tự mỗi từ (kể cả dấu cách). Perplexity tương đương mỗi từ là bao nhiêu?
10. Nêu một điểm khác nhau về chi phí giữa feature extraction và fine-tuning, và một tình huống nên chọn feature extraction.
11. Mean-pooling **không** dùng `attention_mask` gây ra lỗi gì? Vì sao lỗi đó khó phát hiện?
12. Hai lớp: lớp `a` có 100 mẫu, F1 0.9; lớp `b` có 10 mẫu, F1 0.3. Tính macro-F1 và weighted-F1.
13. Câu sinh "gói gói gói", reference "hủy gói". Tính modified unigram precision và BLEU-1.
14. *(2đ)* Đoạn code trích embedding sau có **3 lỗi**. Chỉ ra và sửa.
    ```python
    tok = AutoTokenizer.from_pretrained("MODEL_A")
    model = AutoModel.from_pretrained("MODEL_B")                 # cùng họ, khác vocab
    b = tok(texts, padding=True, return_tensors="pt")
    h = model(**b).last_hidden_state
    emb = h.mean(dim=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)         # fine-tune cả model
    ```

## Phần 2 — Tự luận (4 câu, 30 phút)

**T1 · Decoding (4đ).** Logit cho 4 token: $z = (2, 1, 0, -1)$. Cho sẵn $e^1 = 2.718$, $e^2 = 7.389$, $e^{-1} = 0.368$, $e^4 = 54.60$, $e^{-2} = 0.135$.
- (a) Tính softmax với $T = 1$ và với $T = 0.5$.
- (b) Với $T = 1$, top-p 0.9 giữ những token nào? Viết phân phối sau khi chuẩn hóa lại.
- (c) Một hệ thống trích xuất số điện thoại từ tin nhắn bằng LLM và một hệ thống viết lời chào khách hàng: chọn cấu hình decoding cho mỗi hệ thống và giải thích.

**T2 · BLEU và ROUGE (4đ).** Reference: "khách hàng muốn hủy gói cước" (6 từ). Câu sinh: "khách hàng cần hủy gói" (5 từ).
- (a) Tính $p_1$, $p_2$, BP và BLEU-2 (trung bình nhân của $p_1$, $p_2$).
- (b) Vì sao BLEU-4 (không làm mượt) bằng 0?
- (c) Tính ROUGE-1 (P, R, F) và ROUGE-L (P, R, F).
- (d) "cần" và "muốn" gần nghĩa. Metric nào ghi nhận được điều đó, và nó làm thế nào?

**T3 · Thiết kế đánh giá (3đ).** Một chatbot tổng đài trả lời câu hỏi về gói cước bằng văn bản tự do. Đội sản phẩm muốn một con số để so sánh phiên bản mới với phiên bản cũ mỗi tuần. Thiết kế quy trình đánh giá: dữ liệu, metric tự động (cái nào dùng được, cái nào không), đánh giá bởi người (bao nhiêu người, đo gì), và tín hiệu từ sản phẩm.

**T4 · Có nên chuyển sang model pretrained không? (4đ).** Macro-F1 theo template trên 5 split (cùng split cho mọi model):

| Split | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| TF-IDF âm tiết | 0.834 | 0.621 | 0.798 | 0.679 | 0.637 |
| TF-IDF ký tự | 0.763 | 0.566 | 0.794 | 0.675 | 0.583 |
| e5 + LR (đã `normalize`) | 0.826 | 0.570 | 0.786 | 0.732 | 0.632 |

- (a) Tính chênh lệch ghép cặp và số split thắng của "e5 + LR" so với **từng** baseline.
- (b) Một đồng nghiệp chỉ so "e5 + LR" với TF-IDF ký tự rồi báo "model pretrained thắng 4/5 split, +0.033". Báo cáo đó thiếu gì?
- (c) Kể ít nhất 3 lý do ngoài F1 (chi phí, vận hành) cần cân nhắc trước khi đưa encoder 118M tham số vào production thay cho TF-IDF.
- (d) Đề xuất một thí nghiệm tiếp theo có khả năng làm model pretrained thắng rõ ràng.

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Phần 1**

1. $P(\text{a} \mid \hat{}\,)\cdot P(\text{b} \mid \hat{}\,\text{a})\cdot P(\$ \mid \hat{}\,\text{ab})$: ba bước dự đoán (a, b, rồi kết thúc).
2. $\exp(-\ln 0.2) = 1/0.2 =$ **5**.
3. **50 000** (perplexity của phân phối đều bằng số lựa chọn).
4. Được chọn $40 \times 0.15 =$ **6**; `[MASK]` $6 \times 0.8 =$ **4.8**; ngẫu nhiên **0.6**; giữ nguyên **0.6**. Loss tính trên **6** vị trí được chọn.
5. Lúc fine-tune và dùng thật không có `[MASK]`. Nếu chỉ học ở chỗ `[MASK]`, model không cần biểu diễn tốt các token bình thường. 10% ngẫu nhiên và 10% giữ nguyên buộc model phải "nghi ngờ" mọi token.
6. $\mathrm{softmax}(2, 0) =$ **(0.881, 0.119)**.
7. Giữ logit 3 và 2 (token 1 và 3): $\mathrm{softmax}(3, 2) =$ **(0.731, 0, 0.269, 0)**.
8. Tổng tích lũy **trước** mỗi token: 0, 0.5, 0.8, 0.95. Bỏ token có tổng trước ≥ 0.85, nên giữ **ba token đầu** (0.5, 0.3, 0.15).
9. $1.5^4 =$ **5.06**.
10. Feature extraction chạy model **một lần** cho mỗi văn bản (có thể cache vector), không cần backward hay bộ nhớ cho optimizer, nên train chỉ mất vài giây. Fine-tune cần backward qua cả model nhiều epoch (thường cần GPU), và mỗi bài toán có một bản model riêng. Chọn feature extraction khi: ít tài nguyên; cần dùng một vector cho **nhiều** bài toán; dữ liệu rất ít; hoặc cần lặp thử nghiệm nhanh.
11. Vector đệm được cộng vào trung bình, nên câu ngắn bị pha loãng, và vector của **một** câu thay đổi theo câu dài nhất trong batch. Khó phát hiện vì code vẫn chạy, kết quả chỉ kém đi một chút, và chạy từng câu riêng lẻ (không đệm) lại cho kết quả đúng.
12. Macro $= (0.9 + 0.3)/2 =$ **0.6**. Weighted $= (0.9 \cdot 100 + 0.3 \cdot 10)/110 =$ **0.845**.
13. "gói" xuất hiện 3 lần, nhưng ref chỉ có 1 lần, nên số đếm bị chặn ở 1: $p_1 =$ **1/3**. $c = 3 > r = 2$ nên BP $= 1$, và BLEU-1 $=$ **1/3**.
14. *(mỗi lỗi khoảng 0.67đ)*
    - **Tokenizer và model khác nhau:** id token không khớp với embedding của model. Phải dùng cùng `MODEL_ID` cho cả hai.
    - **`h.mean(dim=1)` không có mask:** cộng cả vector đệm. Phải nhân với `attention_mask` rồi chia cho tổng mask.
    - **lr 1e-3 cho cả encoder pretrained:** phá tri thức pretrain. Dùng khoảng 1e-5 tới 5e-5 cho encoder, lr lớn hơn cho head.
    - *(Chấp nhận thêm:)* thiếu tiền tố mà model yêu cầu; thiếu `truncation`/`max_length`; không `eval()` + `no_grad()` khi chỉ trích embedding.

**T1 (4đ)**

(a) *(1.5đ)* $T = 1$: $e^z = (7.389, 2.718, 1, 0.368)$, tổng 11.475, nên **(0.644, 0.237, 0.087, 0.032)**. $T = 0.5$: logit thành $(4, 2, 0, -2)$, $e^{z/T} = (54.60, 7.389, 1, 0.135)$, tổng 63.12, nên **(0.865, 0.117, 0.016, 0.002)**.

(b) *(1.5đ)* Tổng tích lũy trước mỗi token: 0, 0.644, 0.881, 0.968. Token có tổng trước ≥ 0.9 bị bỏ, chỉ có token cuối. Giữ **3 token**, chuẩn hóa lại: $(0.644, 0.237, 0.087)/0.968 =$ **(0.665, 0.245, 0.090, 0)**.

(c) *(1đ)* Trích xuất số điện thoại: **greedy** (temperature 0), vì cần kết quả đúng, lặp lại được, không cần sáng tạo; kèm kiểm tra định dạng bằng regex. Lời chào: $T$ khoảng 0.7–1 với top-p khoảng 0.9 để câu đa dạng, không lặp, mà vẫn cắt được phần đuôi gây lỗi (bài 17: $T = 1.5$ cho 44% câu có từ bịa).

**T2 (4đ)**

(a) *(1.5đ)* Unigram trùng: khách, hàng, hủy, gói, nên $p_1 = 4/5$. Bigram của câu sinh: "khách hàng" ✓, "hàng cần" ✗, "cần hủy" ✗, "hủy gói" ✓, nên $p_2 = 2/4$. $c = 5 < r = 6$: BP $= e^{1 - 6/5} = 0.819$. BLEU-2 $= 0.819 \times \sqrt{0.8 \times 0.5} = 0.819 \times 0.632 =$ **0.518**.

(b) *(0.5đ)* Không có 3-gram nào trùng ("cần" phá vỡ mọi 3-gram), nên $p_3 = 0$, $\log 0 = -\infty$ và BLEU bằng 0. Làm mượt (cộng 1) cho khoảng 0.37.

(c) *(1.5đ)* ROUGE-1: trùng 4, nên $P = 4/5 = 0.8$, $R = 4/6 = 0.667$, $F =$ **0.727**. ROUGE-L: LCS = "khách hàng hủy gói" (dài 4), nên $P = 0.8$, $R = 0.667$, $F =$ **0.727**.

(d) *(0.5đ)* **BERTScore**: so khớp từng token bằng cosine của embedding **theo ngữ cảnh**, nên "cần" được ghép với "muốn" với cosine cao, thay vì bị tính là không trùng.

**T3 (3đ)** *(mỗi ý khoảng 0.5đ)*
- **Dữ liệu:** một bộ câu hỏi thật, cố định, có phân tầng theo intent và theo độ khó (gồm câu ngoài phạm vi); giữ riêng, không dùng để chỉnh prompt hay model; cập nhật có version.
- **Metric tự động:** BLEU/ROUGE kém, vì câu trả lời đúng có nhiều cách viết. Dùng được: kiểm tra **sự kiện** (giá gói, cú pháp SMS có đúng không) bằng so khớp chính xác hoặc regex; BERTScore hoặc độ giống embedding với câu trả lời mẫu, chỉ làm tín hiệu phụ; tỉ lệ từ chối đúng với câu ngoài phạm vi.
- **Đánh giá bởi người:** so cặp A/B (phiên bản cũ và mới, ẩn nhãn) theo hướng dẫn rõ (đúng, đủ, an toàn, giọng điệu); ít nhất 2–3 người chấm mỗi câu; đo **kappa** trên một phần chung; tính khoảng tin cậy.
- **Tín hiệu sản phẩm:** tỉ lệ 👎 (Tuần 3), tỉ lệ chuyển sang tổng đài viên, tỉ lệ khách hỏi lại; A/B test trên lưu lượng thật.
- **Con số hàng tuần:** tỉ lệ thắng trong so cặp, kèm khoảng tin cậy, cộng tỉ lệ đúng sự kiện; luôn báo cáo theo từng intent.
- (Có thể nhắc LLM-as-judge, Tuần 6–7, nhưng phải kiểm định với người chấm.)

**T4 (4đ)**

(a) *(1.5đ)*
- e5 − TF-IDF ký tự: +0.063, +0.004, −0.008, +0.057, +0.049. Trung bình **+0.033**, thắng **4/5**.
- e5 − TF-IDF âm tiết: −0.008, −0.051, −0.012, +0.053, −0.005. Trung bình **−0.005**, thắng **1/5**.

(b) *(1đ)* Thiếu so với **baseline mạnh nhất** (TF-IDF âm tiết). So với nó thì model pretrained thua nhẹ ở 4/5 split. Chọn một baseline yếu để so là một lỗi báo cáo hay gặp. Cũng nên báo cả kết quả trên chia ngẫu nhiên (e5 thua) và báo độ dao động.

(c) *(1đ)* Độ trễ và chi phí: chạy 118M tham số trên CPU chậm hơn TF-IDF hàng trăm lần; cần nhiều RAM hơn (khoảng 470 MB), có thể cần GPU. Vận hành: tải và ghim phiên bản model, giấy phép, kích thước container. Khả năng giải thích: TF-IDF đọc được hệ số. Rủi ro dữ liệu: phân phối pretrain khác dữ liệu tổng đài (teencode). Chi phí cập nhật: train lại nhanh hay chậm.

(d) *(0.5đ)* **Fine-tune** cả model (không chỉ trích embedding), với lr nhỏ và chọn epoch trên val theo template. Hoặc kết hợp: nối vector e5 với đặc trưng TF-IDF, hoặc ensemble xác suất của hai model. Luôn so ghép cặp trên cùng 5 split. (Đo trên máy bạn: fine-tune đạt 0.711, vẫn chỉ ngang TF-IDF âm tiết; ensemble TF-IDF âm tiết + e5 đạt **0.760**, thắng 5/5 split, vì e5 giỏi tin có dấu còn TF-IDF giỏi tin không dấu.)

</details>
