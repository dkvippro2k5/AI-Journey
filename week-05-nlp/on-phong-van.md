# Ôn phỏng vấn · Tuần 5 · NLP và Transformer

> **Dùng khi nào:** sau final, hoặc ôn lại vào các tuần sau (khuyến nghị: lần đầu ngay sau Tuần 5, rồi ôn lại sau 1 tuần và sau 1 tháng).
> **Cách luyện (khoảng 45'):**
> 1. Đọc câu hỏi, **chưa mở đáp án**.
> 2. Hẹn giờ **1 phút**, **nói to** câu trả lời như đang ngồi phỏng vấn (ghi âm càng tốt).
> 3. Mở đáp án, đối chiếu với **"Ý phải có"**. Thiếu ý nào thì đánh dấu, và nói lại lần nữa.
> 4. Câu nào thiếu từ 2 ý trở lên thì đọc lại bài được dẫn ở cuối câu.
>
> **Mẹo chung:** Transformer là chủ đề **gần như chắc chắn** bị hỏi ở vị trí AI Engineer. Hãy viết được công thức attention, nói được shape ở từng bước, và đếm được tham số mà không cần nhìn tài liệu. Câu 5, 6, 10, 11 nên luyện tới mức trả lời trơn tru.

---

## A. Văn bản cổ điển và tokenization

**1. Bạn tiền xử lý văn bản tiếng Việt như thế nào trước khi phân loại?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Đầu tiên là chuẩn hóa Unicode về **NFC**: một số bàn phím gửi chữ và dấu tách rời (NFD), nhìn giống hệt nhưng là chuỗi khác. Sau đó mình đổi về chữ thường, thống nhất **kiểu đặt dấu** ("hủy"/"huỷ"), thay số điện thoại, tiền, link bằng placeholder, và xử lý teencode bằng bảng tra. Có hai bước mình **không** làm theo thói quen: bỏ dấu và bỏ stopword. Trên project của mình, bỏ dấu tạo ra 246 cụm từ bị gộp nhầm ("dung" là đúng, dùng hay dừng), và làm F1 giảm ở 3/5 split. Bỏ stopword với câu ngắn thì có thể mất phủ định. Hàm tiền xử lý nằm **trong** pipeline của model, để lúc phục vụ chạy y hệt lúc train.

**Ý phải có:** NFC · placeholder · teencode · bỏ dấu/stopword xóa thông tin, phải đo · train và serve dùng chung một hàm.
**Nếu bị hỏi tiếp:** *"Thế còn tách từ?"* → tiếng Việt viết theo âm tiết; tách từ ("đăng_ký") cần cho model mức từ như PhoBERT; với subword/char n-gram thì không bắt buộc.
**Xem lại:** bài 11.
</details>

**2. TF-IDF là gì? Bạn còn dùng nó khi đã có Transformer không?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** TF-IDF biểu diễn văn bản bằng số lần mỗi từ xuất hiện, nhân với độ hiếm của từ, $\ln\frac{1 + N}{1 + \text{df}} + 1$, rồi chuẩn hóa L2. Từ có ở mọi nơi như "tôi", "gói" bị hạ trọng số. Mình vẫn dùng nó làm **baseline đầu tiên**: train vài giây, chạy dưới một mili giây, và đọc được hệ số. Trên router intent của mình, TF-IDF âm tiết cộng logistic regression đạt macro-F1 0.71 trên cách diễn đạt chưa từng thấy. Con số đó **ngang** feature extraction từ một encoder pretrained 118M tham số, và hơn hẳn một Transformer nhỏ tự train từ đầu (0.52). Mọi model phức tạp hơn đều phải thắng nó theo cặp split thì mới đáng đưa vào production.

**Ý phải có:** công thức và ý nghĩa idf · chuẩn hóa L2 · baseline rẻ, đọc được · số so sánh thực tế.
**Xem lại:** bài 12; bài 18 (mục 18.4).
</details>

**3. Word2vec học embedding thế nào? Hạn chế của nó?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Word2vec dựa trên giả thuyết phân bố: từ có ngữ cảnh giống nhau thì nghĩa giống nhau. Skip-gram lấy mỗi cặp (từ trung tâm, từ trong cửa sổ) làm một mẫu dương, rút thêm $k$ từ nhiễu theo $\text{count}^{0.75}$, rồi train một bài toán phân loại nhị phân: cặp nào là thật. Mỗi từ có hai vector, một khi là trung tâm và một khi là ngữ cảnh. Mình train thử trên tin nhắn tổng đài: model **tự** tìm ra "đk", "đc" gần "được", "ko", "hông" gần "không", mà không cần bảng teencode. Hạn chế: "hủy" lại nằm sát "đăng ký" vì cả hai đứng trước "gói", nên **trái nghĩa** cũng gần nhau. Mỗi từ chỉ có một vector, nên không phân biệt "gói cước" với "gói quà". Và không có vector cho từ chưa thấy; FastText sửa được điểm này bằng n-gram ký tự.

**Ý phải có:** giả thuyết phân bố · skip-gram + negative sampling, lũy thừa 0.75 · ví dụ đã đo · trái nghĩa gần nhau, tĩnh, OOV.
**Xem lại:** bài 13.
</details>

**4. Vì sao LLM dùng subword tokenization? BPE hoạt động thế nào?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Token mức **từ** thì vocab khổng lồ và luôn gặp từ lạ. Mức **ký tự** thì chuỗi quá dài, mà attention tốn $O(T^2)$. Subword nằm ở giữa: vocab vài chục tới vài trăm nghìn, không có OOV, chuỗi ngắn. BPE bắt đầu từ ký tự (hoặc byte), lặp lại việc **ghép cặp kề nhau phổ biến nhất**, mỗi lần thêm một token vào vocab. Khi tokenize, nó áp các merge theo thứ tự đã học. WordPiece giống vậy nhưng chọn cặp theo likelihood; Unigram thì đi ngược lại, cắt dần một vocab lớn. Một điểm thực tế: tokenizer của model đa ngôn ngữ mình dùng tách câu tiếng Việt không dấu ra nhiều hơn 16% số token, nên gọi API theo token sẽ đắt hơn.

**Ý phải có:** từ / ký tự / subword đánh đổi gì · thuật toán BPE · byte-level không có `<unk>` · tokenizer gắn với model, ảnh hưởng chi phí.
**Nếu bị hỏi tiếp:** *"Vì sao tiếng Việt tốn nhiều token hơn tiếng Anh?"* → tokenizer học chủ yếu trên tiếng Anh, và mỗi ký tự có dấu chiếm 2–3 byte UTF-8.
**Xem lại:** bài 14.
</details>

## B. Transformer

**5. Viết công thức self-attention và giải thích từng phần. Vì sao chia $\sqrt{d_k}$?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** $\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(QK^\top/\sqrt{d_k})\,V$, với $Q = XW_Q$, $K = XW_K$, $V = XW_V$. Mỗi token có một query ("mình đang tìm gì"), một key ("mình là gì") và một value ("mình đóng góp gì"). $QK^\top$ đo mức khớp của mọi cặp, softmax theo hàng biến thành trọng số có tổng bằng 1, rồi nhân với $V$ là trộn value theo trọng số. Chia $\sqrt{d_k}$ vì tích vô hướng của hai vector ngẫu nhiên có phương sai bằng $d_k$. Không chia thì với $d_k = 64$, điểm số chênh nhau cỡ ±8, softmax gần như one-hot, và gradient nhỏ đi vài trăm lần. Mask được áp **trước** softmax bằng cách đặt $-\infty$.

**Ý phải có:** công thức đúng · ý nghĩa Q/K/V · softmax theo hàng · phương sai $d_k$ → bão hòa · mask trước softmax.
**Xem lại:** bài 15; Tuần 4, bài 09.
</details>

**6. Multi-head attention là gì? Nói shape ở từng bước.**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Thay vì một attention với $d$ chiều, ta chạy $h$ attention song song, mỗi head $d/h$ chiều, rồi nối lại và nhân $W_O$. Mỗi head có thể học một kiểu quan hệ riêng. Shape với $X$ là $(B, T, d)$: chiếu một lần thành $(B, T, d)$; `view` thành $(B, T, h, d/h)$; `transpose` thành $(B, h, T, d/h)$; điểm số $(B, h, T, T)$; nhân $V$ được $(B, h, T, d/h)$; `transpose` và `reshape` về $(B, T, d)$; cuối cùng $W_O$. Số tham số $4(d^2 + d)$, **không phụ thuộc** $h$, vì mỗi head chỉ dùng một lát của cùng ma trận. Mình đã tự viết và so với `nn.MultiheadAttention`: lệch dưới $10^{-6}$. Một chỗ hay sai là ngữ nghĩa mask: PyTorch coi `True` trong `key_padding_mask` là **bị che**.

**Ý phải có:** $h$ head, $d/h$ chiều, nối rồi $W_O$ · đủ shape · tham số không đổi theo $h$ · (bonus) mask ngược nghĩa giữa các API.
**Xem lại:** bài 15 (mục 15.2).
</details>

**7. Vì sao cần positional encoding? Sinusoidal, learned và RoPE khác nhau thế nào?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Self-attention không có phép tính nào phụ thuộc vào chỉ số vị trí: đổi chỗ các token thì đầu ra chỉ đổi chỗ theo, nên "không muốn hủy" và "hủy không muốn" giống hệt nhau. Phải đưa vị trí vào. **Sinusoidal** cộng vào embedding các sóng sin/cos với nhiều tần số, như một đồng hồ nhiều kim, không cần học. **Learned** là một bảng embedding cho vị trí, như BERT và GPT-2, nhưng không có vector cho vị trí vượt `max_len`. **RoPE** xoay từng cặp chiều của $q$ và $k$ một góc tỉ lệ với vị trí, ngay trong mỗi lớp. Kết quả là $q_m^\top k_n$ chỉ phụ thuộc vào $m - n$, tức vị trí **tương đối**. Phần lớn LLM mã nguồn mở hiện nay dùng RoPE.

**Ý phải có:** attention không biết thứ tự · ba cách, mỗi cách một câu · RoPE → vị trí tương đối, áp vào q/k.
**Xem lại:** bài 15 (mục 15.3).
</details>

**8. Vì sao Transformer thay thế RNN? Nhược điểm của Transformer là gì?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Có hai lý do. Một là **song song hóa**: RNN phải tính tuần tự theo thời gian, còn Transformer xử lý mọi vị trí trong một phép nhân ma trận. Với causal mask, một lần forward cho ra loss của **mọi** vị trí cùng lúc. Hai là **đường đi ngắn**: thông tin giữa hai token bất kỳ chỉ qua một bước attention, $O(1)$, thay vì truyền qua $T$ bước như RNN, nơi gradient dễ vanishing. Ở Tuần 4 mình đo: LSTM mặc định không nhớ nổi token đầu sau 20 bước. Nhược điểm: chi phí $O(T^2)$ cả phép tính lẫn bộ nhớ cho ma trận attention (1 GiB cho mỗi lớp với 4 096 token và 32 head ở fp16); và khi sinh văn bản vẫn phải tuần tự từng token, nên cần KV-cache.

**Ý phải có:** song song theo $T$ · đường đi $O(1)$ · $O(T^2)$ · sinh vẫn tuần tự.
**Xem lại:** bài 15–16; Tuần 4, bài 08.
</details>

**9. BERT khác GPT thế nào? Khi nào dùng encoder, khi nào dùng decoder?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** BERT là **encoder**: attention hai chiều, pretrain bằng masked LM (che 15% token, trong đó 80% thay bằng `[MASK]`, 10% thay ngẫu nhiên, 10% giữ nguyên). Mỗi token có vector nhìn được cả trái lẫn phải, rất hợp với phân loại, gán nhãn thực thể, embedding cho tìm kiếm. GPT là **decoder**: causal mask, pretrain bằng dự đoán token tiếp theo, nên sinh văn bản tự nhiên, và một model làm được nhiều việc qua prompt. Encoder–decoder như T5 hợp với dịch và tóm tắt. Với router intent của mình (câu ngắn, cần nhanh trên CPU, chỉ phân loại), mình chọn encoder nhỏ, không dùng LLM sinh văn bản.

**Ý phải có:** hai chiều + MLM so với causal + next-token · tác vụ phù hợp · có thêm encoder–decoder · quyết định theo bài toán, độ trễ, chi phí.
**Xem lại:** bài 16 (mục 16.2), bài 17.
</details>

**10. Đếm số tham số của một Transformer như BERT-base.**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Mỗi lớp có attention $4d^2 + 4d$ (bốn ma trận $d \times d$ có bias), FFN với hidden $4d$ là $8d^2 + 5d$, và hai LayerNorm $4d$. Tổng $12d^2 + 13d$. Với $d = 768$ là 7 087 872; mình đếm trên một lớp thật của BERT-base và của GPT-2 và ra đúng con số này. Mười hai lớp là khoảng 85M. Cộng embedding: vocab 30 522 cộng 512 vị trí cộng 2 segment, nhân 768, khoảng 23.8M, và pooler 0.6M, được **109.5M**. GPT-2 small là 124.4M, vì vocab lớn hơn (50 257) và lớp output dùng chung trọng số với embedding. Một điểm hay quên: ở model nhỏ, embedding có thể chiếm phần lớn tham số; encoder đa ngôn ngữ mình dùng có 82% tham số ở embedding vì vocab 250k.

**Ý phải có:** $12d^2$ mỗi lớp, nguồn gốc 4 + 8 · cộng $Vd$ · ra đúng khoảng 110M · weight tying.
**Xem lại:** bài 16 (mục 16.4).
</details>

**11. KV-cache là gì? Tốn bao nhiêu bộ nhớ?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Khi sinh văn bản, ở mỗi bước, key và value của các token **cũ** không đổi, nhờ causal mask: token cũ không bao giờ nhìn token mới. Vì vậy ta lưu $K, V$ của mọi lớp lại; mỗi bước chỉ tính $q, k, v$ cho token mới, nối vào cache, rồi attention với bảng điểm $1 \times t$. Mình đã kiểm tra: kết quả khớp với tính lại toàn bộ chuỗi tới $10^{-7}$, và với 1 024 token thì phần chiếu cộng attention giảm khoảng 600 lần. Bộ nhớ là $2 \times L \times T \times d_{kv} \times$ số byte, **cho mỗi câu**. Với một decoder cỡ 7B ở fp16 và 4 096 token, đó là khoảng 2 GiB mỗi câu, nên batch 16 câu đã tốn hơn cả trọng số. Vì vậy có GQA/MQA (dùng chung head KV) và PagedAttention khi phục vụ.

**Ý phải có:** vì sao được phép cache (causal) · chỉ tính cho token mới · công thức bộ nhớ, ví dụ số · GQA / PagedAttention.
**Xem lại:** bài 16 (mục 16.3); final C2.
</details>

## C. Mô hình ngôn ngữ, pretrained, đánh giá

**12. Perplexity là gì? Có so được perplexity giữa hai model không?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Perplexity là $e$ mũ cross-entropy trung bình trên các token: $\exp(-\frac1N\sum\log P(x_t \mid x_{<t}))$. Có thể hiểu là "trung bình model phân vân giữa bao nhiêu lựa chọn đều nhau". Model đoán đều trên $V$ token có perplexity đúng bằng $V$. Chỉ so được khi **cùng tokenizer** và **cùng tập test**. LM mức ký tự của mình có perplexity 1.41 mỗi ký tự, nghe rất tốt, nhưng quy ra mỗi âm tiết là khoảng 4.2. Perplexity thấp cũng không có nghĩa là model trả lời đúng hay hữu ích, nên với LLM còn phải đánh giá bằng bài toán thật.

**Ý phải có:** công thức · ý nghĩa "số lựa chọn" · phụ thuộc tokenizer · không đủ để đánh giá LLM.
**Xem lại:** bài 17.
</details>

**13. Temperature, top-k, top-p khác nhau thế nào? Bạn chọn cấu hình nào?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Temperature chia logit trước softmax: nhỏ hơn 1 thì nhọn hơn, gần greedy; lớn hơn 1 thì phẳng hơn. Top-k chỉ lấy mẫu trong $k$ token tốt nhất. Top-p lấy tập **nhỏ nhất** có tổng xác suất ≥ p, nên tự co giãn: model chắc chắn thì giữ ít token, phân vân thì giữ nhiều. Trên LM tự train, mình đo được: temperature 0.5 thì 60% câu là bản sao nguyên văn dữ liệu train; temperature 1.5 thì 44% câu có từ bịa ra; còn top-p 0.9 vẫn đa dạng mà tỉ lệ lỗi chỉ 1.3%. Với trích xuất thông tin hay gọi tool, mình dùng temperature 0; với văn bản tự do, khoảng 0.7–1 kèm top-p 0.9.

**Ý phải có:** định nghĩa ba cách · top-p tự co giãn · thứ tự áp dụng (temperature → top-k → top-p) · cấu hình theo bài toán.
**Xem lại:** bài 17 (mục 17.3, 17.5).
</details>

**14. Có 5 000 câu có nhãn: fine-tune BERT hay dùng TF-IDF?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Mình không chọn trước mà **đo**. Đầu tiên là baseline TF-IDF cộng logistic regression, chỉ mất vài giây. Rồi thử feature extraction từ một encoder pretrained (rẻ, chạy model một lần) và fine-tune (đắt hơn). Tất cả được so **ghép cặp** trên cùng nhiều split, với cách chia phản ánh thực tế: theo thời gian, hoặc theo cách diễn đạt. Kết quả của mình khá bất ngờ: trên router intent, feature extraction từ encoder 118M tham số đạt 0.709, chỉ ngang TF-IDF âm tiết (0.714) trên cách diễn đạt mới, và thua trên chia ngẫu nhiên. Fine-tune cả model cũng chỉ đạt 0.711, mà mỗi lần train mất 11 phút thay vì dưới một giây, và trả lời chậm hơn khoảng 30 lần (17 ms so với 0.5 ms). Cái thắng rõ ràng là **ensemble** hai model, 0.760 và thắng 5/5 split, vì model pretrained giỏi tin có dấu còn TF-IDF giỏi tin không dấu. Ngoài F1 còn phải tính độ trễ, chi phí phục vụ, khả năng giải thích. Model pretrained có lợi nhất khi bài toán cần hiểu nghĩa và dữ liệu giống phân phối pretrain.

**Ý phải có:** baseline trước · feature extraction và fine-tune · so ghép cặp, cách chia đúng · số thật · chi phí phục vụ.
**Xem lại:** bài 18; project.
</details>

**15. Macro hay micro F1? BLEU có hạn chế gì?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Macro-F1 lấy trung bình F1 các lớp đều nhau, nên lớp hiếm vẫn được tính; micro-F1 gộp TP, FP, FN của mọi lớp, và với phân loại đơn nhãn thì nó **chính là accuracy**. Với router intent, mình báo macro-F1 kèm bảng từng intent, vì lớp ngoài phạm vi có recall chỉ 0.54 dù accuracy vẫn ổn. BLEU đo precision n-gram có chặn số đếm, lấy trung bình nhân từ 1 tới 4-gram, rồi nhân brevity penalty để phạt câu ngắn. Nó không hiểu đồng nghĩa: "tôi muốn ngừng gói" so với "tôi muốn hủy gói" cho BLEU-4 bằng 0. Ngoài ra corpus BLEU không phải trung bình sentence BLEU, và BLEU phụ thuộc cách token hóa. Với hệ thống sinh văn bản tự do, mình dùng đánh giá bởi người có đo kappa, hoặc so cặp A/B.

**Ý phải có:** macro, micro (= accuracy), khi nào dùng · BLEU: n-gram + chặn + BP · không hiểu đồng nghĩa · đánh giá bởi người, kappa.
**Xem lại:** bài 19.
</details>

## D. Kể về project (câu gần như chắc chắn bị hỏi)

**16. "Kể về một project NLP bạn đã làm."** *(luyện bản 2 phút)*

<details><summary>Khung trả lời và bản mẫu</summary>

**Khung:** bối cảnh → mục tiêu đo được → cách làm → **kết quả có số** → điều học được hoặc điều bất ngờ → sẽ làm gì tiếp.

**Bản mẫu (project `intent`).**
- **Bối cảnh:** router intent cho tổng đài nhà mạng. Tin nhắn tiếng Việt có teencode, không dấu, gõ sai; 8 intent cộng một lớp ngoài phạm vi. Tin nhắn có độ tin cậy thấp được chuyển cho tổng đài viên.
- **Cách làm:** mình viết toàn bộ từ đầu: tiền xử lý tiếng Việt, TF-IDF, BPE, multi-head attention và một Transformer encoder nhỏ. Mình so chúng với feature extraction và fine-tune từ một encoder pretrained.
- **Đánh giá là phần quan trọng nhất:** chia ngẫu nhiên thì model nào cũng đạt khoảng 0.98 macro-F1. Nhưng khi giữ lại những **câu mẫu chưa từng thấy** làm test, con số rơi xuống 0.52–0.71. Mình báo con số thứ hai, so ghép cặp trên 5 split.
- **Kết quả:** TF-IDF âm tiết 0.714, fine-tune model pretrained 0.711, feature extraction 0.709, TF-IDF ký tự 0.676, Transformer tự train 0.521. Ensemble TF-IDF với model pretrained đạt **0.760**, thắng 5/5 split, vì hai model sai ở những nhóm tin khác nhau (có dấu và không dấu).
- **Bất ngờ:** model đơn giản nhất lại khó vượt nhất. Ngưỡng từ chối chọn trên val để đạt precision 95% chỉ cho khoảng 0.85–0.93 trên test, nên phải có shadow mode trước khi bật thật.
- **Nếu làm tiếp:** thu thập dữ liệu thật theo thời gian, ensemble TF-IDF với model pretrained, và đặt ngưỡng riêng cho lớp khiếu nại.

**Câu hỏi đào sâu hay gặp** (chuẩn bị sẵn):
- *"Vì sao chia theo template?"* Câu gần trùng nhau ở cả train lẫn test là một dạng rò rỉ. Với log thật, tương đương là chia theo thời gian và theo khách hàng.
- *"Vì sao Transformer tự train thua?"* 141 câu mẫu là quá ít để học ngôn ngữ từ đầu; nó overfit ngay từ epoch đầu (best epoch = 0). Thiên kiến bag-of-words của TF-IDF hợp với dữ liệu ít hơn.
- *"Ngưỡng từ chối hoạt động thế nào?"* Ngưỡng nhỏ nhất trên max-probability sao cho phần được tự động xử lý đạt precision mục tiêu trên val; dưới ngưỡng thì chuyển cho người.

Thay các chi tiết bằng **số của chính bạn** trong `REPORT.md`.
</details>

---

**Tự chấm:** mỗi câu được 1 điểm nếu nói đủ "Ý phải có" trong khoảng 1 phút (câu 16 trong khoảng 2 phút). Đạt **≥ 12/16** là sẵn sàng cho phần NLP và Transformer của vòng phỏng vấn kỹ thuật.
