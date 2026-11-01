# Final Tuần 5 · NLP và Transformer

> ⏱ **180 phút** · **100 điểm** · Đạt khi ≥ 70
> - **A** Câu nhanh: 12 × 2đ = 24đ (25')
> - **B** Tính tay: 4 × 5đ = 20đ (40')
> - **C** Code: 32đ (75'), chấm bằng `code/test_final.py`
> - **D** Thiết kế: 24đ (40')
>
> Phần A, B, D làm trên giấy, được dùng máy tính cầm tay. Phần C viết vào `code/final_lib.py`, chạy `python -m pytest -q code/test_final.py` ở thư mục `week-05-nlp`. Được xem docs của Python, NumPy; **không** xem tài liệu tuần này. **Không có câu chứng minh.**
> Lời giải: [final-solutions.md](final-solutions.md). Chỉ mở sau khi làm xong.

---

## Phần A · Câu nhanh (12 × 2đ)

**A1.** Hai tin nhắn "hủy gói" hiển thị giống hệt nhau nhưng TF-IDF coi là hai từ khác nhau. Nguyên nhân có thể là gì (2 khả năng)? Sửa thế nào?

**A2.** Model đạt macro-F1 0.98 khi chia ngẫu nhiên, 0.71 khi chia theo template. Vì sao chênh lớn như vậy? Khi dữ liệu là log hội thoại thật (không có template), cách chia nào cho kết quả trung thực tương tự?

**A3.** Viết công thức idf của scikit-learn (`smooth_idf=True`). Một từ có mặt trong **mọi** văn bản có idf bằng bao nhiêu, và vì sao không đặt idf của nó bằng 0?

**A4.** Trong word2vec với negative sampling, từ nhiễu được rút theo $\text{count}^{0.75}$. So với rút theo tần suất thuần, điều này thay đổi gì, và vì sao có ích?

**A5.** BPE và WordPiece khác nhau ở tiêu chí chọn cặp để ghép như thế nào? Cho một tình huống hai tiêu chí chọn hai cặp khác nhau.

**A6.** Trong multi-head attention với $d = 512$, $h = 8$, phải chia điểm số cho bao nhiêu? Nếu lỡ chia cho $\sqrt{512}$ thì trọng số attention thay đổi theo hướng nào?

**A7.** Batch $B = 8$, $T = 32$, $d = 256$, $h = 8$. Cho shape của: $Q$ sau khi tách head; ma trận điểm số; đầu ra trước `out_proj`.

**A8.** RoPE có tính chất gì đối với tích vô hướng $q_m^\top k_n$? Vì sao tính chất đó hữu ích?

**A9.** Pre-LN khác post-LN ở đâu? Vì sao phần lớn model hiện nay dùng pre-LN?

**A10.** Chọn kiến trúc (encoder-only, decoder-only, encoder–decoder) cho từng bài toán và giải thích ngắn: (a) gán nhãn thực thể trong tin nhắn; (b) tóm tắt cuộc gọi; (c) trợ lý chat trả lời tự do.

**A11.** Định nghĩa perplexity. Một LM mức ký tự có perplexity 1.4, một LM mức từ có perplexity 30. LM nào tốt hơn?

**A12.** Top-p khác top-k ở điểm nào khi model **rất chắc chắn** và khi model **phân vân**?

---

## Phần B · Tính tay (4 × 5đ)

**B1 · BPE (5đ).** Corpus: "cuoc" ×3, "cuon" ×2, "cua" ×4. Mỗi từ được tách ký tự và thêm `</w>` (ký tự `<` đứng trước mọi chữ cái thường khi so sánh chuỗi).
- (a) Đếm các cặp kề nhau và học **3** merge đầu tiên (hòa thì chọn cặp nhỏ nhất theo thứ tự chuỗi, so phần tử đầu trước). *(3đ)*
- (b) Tokenize "cuoi" và "cu" bằng 3 merge đó. *(1đ)*
- (c) Khi **encode** "cuoi" thành id, token nào thành `<unk>`, và vì sao? Byte-level BPE tránh được điều đó bằng cách nào? *(1đ)*

**B2 · Kích thước Transformer (5đ).** Một encoder phân loại intent: $V = 8\,000$, learned position embedding cho 128 vị trí, $d = 256$, $h = 4$, $L = 4$ lớp pre-LN, FFN $d_\text{ff} = 1\,024$ (có bias), 2 LayerNorm mỗi lớp cộng 1 LayerNorm cuối, head Linear ra 9 lớp.
- (a) Số tham số của một lớp. *(2đ)*
- (b) Tổng số tham số của model, và tỉ lệ nằm ở embedding (token + vị trí). *(1.5đ)*
- (c) Nếu dùng **cùng** kích thước làm decoder để sinh văn bản: KV-cache fp16 cho batch 32 câu, mỗi câu 512 token, là bao nhiêu MiB? *(1.5đ)*

**B3 · Mô hình ngôn ngữ và decoding (5đ).** Một bigram LM (không làm mượt) có số đếm: $(\langle s\rangle, \text{hủy}) = 6$, $(\langle s\rangle, \text{nạp}) = 4$, $(\text{hủy}, \text{gói}) = 5$, $(\text{hủy}, \langle/s\rangle) = 1$, $(\text{gói}, \langle/s\rangle) = 5$, $(\text{nạp}, \text{tiền}) = 4$, $(\text{tiền}, \langle/s\rangle) = 4$.
- (a) Tính $P(\text{hủy gói})$, bao gồm cả bước bắt đầu và kết thúc, và perplexity của câu đó (chia cho 3 bước dự đoán). *(2.5đ)*
- (b) Câu "nạp gói" có perplexity bao nhiêu? Làm mượt add-k giúp gì ở đây? *(1đ)*
- (c) Một LM khác cho phân phối token tiếp theo $(0.45, 0.25, 0.15, 0.10, 0.05)$. Top-p với $p = 0.5$ và với $p = 0.8$ giữ những token nào? *(1.5đ)*

**B4 · Metric (5đ).** Router intent trên 20 tin nhắn, ba lớp: hủy gói (10 tin), đăng ký gói (8 tin), khác (2 tin).
- 10 tin hủy gói: đoán đúng 7, nhầm thành đăng ký 3.
- 8 tin đăng ký: đúng 6, nhầm thành hủy gói 2.
- 2 tin khác: nhầm thành hủy gói 1, đúng 1.

Câu hỏi:
- (a) Tính precision, recall, F1 của từng lớp. *(2đ)*
- (b) Tính macro-F1, micro-F1 và weighted-F1. Con số nào phù hợp nhất để báo cáo, và vì sao? *(1.5đ)*
- (c) Câu sinh "nạp thẻ cho số này", reference "nạp thẻ cào cho số này". Tính BLEU-2 (trung bình nhân của $p_1$, $p_2$, có BP). BLEU-4 không làm mượt bằng bao nhiêu? *(1.5đ)*

---

## Phần C · Code (32đ)

Viết vào `code/final_lib.py`. Mỗi test pass được chia đều số điểm của bài (11 test).

**C1 · WordPiece tokenize (8đ, 2 test).** `wordpiece_tokenize(word, vocab, unk="[UNK]", max_chars=100) -> list[str]`
- Từ dài hơn `max_chars` ký tự: trả `[unk]`.
- Khớp **dài nhất trước**: bắt đầu từ `start = 0`, tìm chuỗi con `word[start:end]` dài nhất có trong `vocab`. Với `start > 0`, chuỗi cần tìm là `"##" + word[start:end]`. Thêm nó vào kết quả, đặt `start = end`, lặp lại.
- Nếu ở một vị trí nào đó không tìm được chuỗi con nào: trả về `[unk]` cho **cả từ**.

**C2 · Causal multi-head attention và KV-cache bằng NumPy (10đ, 4 test).**
- `causal_mha(X, Wq, Wk, Wv, Wo, n_heads)`: `X` shape `(T, d)`, các `W` shape `(d, d)` (nhân **bên phải**: `X @ Wq`). Tách $d$ thành `n_heads` head liên tiếp (head $i$ gồm các cột `i*dh:(i+1)*dh`), chia $\sqrt{d_h}$, causal mask, softmax, nối head, nhân `Wo`. Trả về `(T, d)`.
- `init_cache(n_heads, d_head)`: dict `{"k": zeros((n_heads, 0, d_head)), "v": ...}`.
- `decode_step(x_t, cache, Wq, Wk, Wv, Wo, n_heads)`: `x_t` shape `(d,)`. Trả về `(out (d,), cache_mới)`, với cache mới có thêm một vị trí. Lặp `decode_step` qua $t = 0..T-1$ phải cho đúng từng hàng của `causal_mha`.

**C3 · Bigram LM add-k (8đ, 3 test).** `BigramLM(k=1.0)` với `BOS = "<s>"`, `EOS = "</s>"`:
- `.fit(sentences)`: `sentences` là list các list token. `vocab_` = các token đã thấy cộng `EOS`, sắp xếp (**không** có `BOS`). Đếm bigram trên `[BOS] + s + [EOS]`. Trả về `self`.
- `.prob(prev, w)` $= \dfrac{\text{count}(prev, w) + k}{\text{count}(prev, \cdot) + k\,|V|}$.
- `.perplexity(sentences)`: $\exp$ của trung bình $-\log$ xác suất trên **mọi** bước dự đoán (gồm cả bước dự đoán `EOS`).

**C4 · MLM masking (6đ, 2 test).** `mlm_mask(ids, mask_id, vocab_size, special_ids=(), p=0.15, rng=None) -> (inputs, labels)`
- Không sửa `ids`.
- Mỗi token **không** thuộc `special_ids` được chọn độc lập với xác suất `p`.
- `labels` bằng `ids` ở vị trí được chọn, bằng −100 ở các vị trí khác.
- Trong các vị trí được chọn: 80% thay bằng `mask_id`, 10% thay bằng một id ngẫu nhiên trong `[0, vocab_size)`, 10% giữ nguyên (rút một số ngẫu nhiên riêng cho mỗi vị trí).

---

## Phần D · Thiết kế (24đ)

**Bối cảnh.** Tổng đài chăm sóc khách hàng của một nhà mạng lớn nhận khoảng **1 triệu tin nhắn mỗi ngày**, qua ba kênh: SMS tới đầu số tổng đài, Zalo OA, và chat trong app. Lúc cao điểm (19–21h) khoảng **80 tin nhắn/giây**. Hiện tại, mọi tin nhắn đều do tổng đài viên đọc và chuyển tiếp thủ công. Ban lãnh đạo muốn một **router intent tự động**:
- Chuyển tin nhắn tới đúng hàng đợi: tra cước, nạp tiền, đăng ký/hủy gói, lỗi mạng, SIM, khiếu nại, gặp người, và **ngoài phạm vi**.
- Chạy trên hạ tầng **tự host**, CPU là chính, có vài GPU dùng chung.
- Độ trễ p95 dưới **100 ms**.
- **Không được** tự động xử lý sai một khiếu nại (khiếu nại bị chuyển nhầm sẽ bị phạt theo quy định chất lượng dịch vụ).
- Hiện có 20 000 tin nhắn cũ đã được tổng đài viên chuyển tiếp (nhãn có nhiễu), và chưa có nhãn cho tin ngoài phạm vi.

**D1 · Dữ liệu và nhãn (6đ).** Bạn xây bộ train/val/test như thế nào từ 20 000 tin nhắn cũ? Làm gì với nhãn nhiễu, với tin ngoài phạm vi, với tin chứa **hai** intent ("hủy gói A rồi đăng ký gói B"), và với ba kênh có văn phong khác nhau? Cách chia nào để đánh giá trung thực?

**D2 · Model và kiến trúc phục vụ (6đ).** Đề xuất model (có thể nhiều tầng), tiền xử lý, và kiến trúc phục vụ đạt p95 < 100 ms ở 80 tin/giây. Ước lượng nhanh tài nguyên cần cho phương án của bạn. Khi nào dùng tới GPU hoặc một LLM?

**D3 · Đánh giá và ngưỡng (6đ).** Chọn metric chính và các metric phụ. Đặt ngưỡng từ chối (chuyển cho người) thế nào để đáp ứng yêu cầu về khiếu nại? Làm sao biết ngưỡng chọn trên val còn đúng khi đưa vào thật?

**D4 · Vận hành và cải tiến (6đ).** Theo dõi gì sau khi triển khai? Khi nhà mạng ra một gói cước mới (tên mới, cú pháp mới) thì hệ thống phản ứng thế nào? Lên kế hoạch học lại (tần suất, dữ liệu, kiểm thử trước khi thay model).

---

## Bài → câu

| Bài | Câu |
|---|---|
| 11 Tiền xử lý tiếng Việt | A1, A2, D1, D2 |
| 12 BoW, TF-IDF | A3, D2 |
| 13 Word embeddings | A4 |
| 14 Subword tokenization | A5, B1, C1 |
| 15 Self-attention, MHA, PE | A6, A7, A8, C2 |
| 16 Block Transformer, KV-cache | A9, A10, B2, C2 |
| 17 Mô hình ngôn ngữ, decoding | A11, A12, B3, C3, C4 |
| 18 Model pretrained, Hugging Face | A10, D2 |
| 19 Metric | B4, D3 |
| Project (router, ngưỡng, chia theo template) | A2, D1–D4 |
