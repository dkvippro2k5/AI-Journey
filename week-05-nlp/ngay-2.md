# Ngày 2 · Tokenization và Transformer

> **Loại tài liệu: Lý thuyết (bản nhẹ).** Công thức đầy đủ và ví dụ tính tay. Chứng minh chỉ phác thảo vài dòng.
> **Hôm nay gồm 3 bài:**
> - **14** Subword tokenization: BPE, WordPiece, SentencePiece, byte-level
> - **15** Self-attention, multi-head attention, positional encoding, causal mask
> - **16** Block Transformer, ba kiểu kiến trúc, đếm tham số, KV-cache
>
> **Thời gian:** đọc ~2h55 · quiz 45' · bài tập ~2h
> **Nối với Tuần 4:** bạn đã viết `dot_attention` (mask, chia $\sqrt d$), LayerNorm/RMSNorm, residual, GELU và cách đếm tham số FFN. Hôm nay ghép chúng thành một block Transformer hoàn chỉnh, có **nhiều head**, **vị trí** và **causal mask**.
> Mọi con số "đo trên máy bạn" chạy bằng PyTorch 2.14 (CPU), `tokenizers` 0.23, `transformers` 5.18. Số tham số của BERT-base và GPT-2 được đếm bằng cách dựng model từ **config** (không tải trọng số).

---

## 14 · Subword tokenization: BPE, WordPiece, SentencePiece, byte-level

> ⏱ Tầng 1: 10' · Tầng 2: 45'
> **Cần biết trước:** tokenization, NFC (bài 11); UTF-8 và byte (Tuần 1, phụ lục P1); `nn.Embedding` (Tuần 4, bài 03).

> **📖 Từ mới**
> - **Subword:** đơn vị nằm giữa ký tự và từ, ví dụ "khuyến" = `kh` + `uy` + `ế` + `n`.
> - **Vocabulary (vocab):** tập mọi token mà tokenizer biết; mỗi token có một **id** nguyên.
> - **Merge (phép ghép):** quy tắc "gặp cặp `a` `b` kề nhau thì ghép thành `ab`", học từ dữ liệu.
> - **Ký hiệu cuối từ / đầu từ:** đánh dấu ranh giới từ: `</w>` (BPE gốc), `##` (WordPiece, đánh dấu phần **nối tiếp**), `▁` (SentencePiece, đánh dấu **đầu từ**).
> - **Special token:** token có vai trò riêng, không đến từ văn bản: `<pad>`, `<unk>`, `[CLS]`, `[SEP]`, `[MASK]`, `<s>`, `</s>`.
> - **Byte-level:** coi văn bản là dãy **byte** UTF-8 (256 giá trị), nên không bao giờ gặp ký tự lạ.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Bạn phải nhắn tin bằng một bộ "con dấu" có sẵn chữ in. Có ba cách làm bộ dấu:
- **Một con dấu cho mỗi từ:** in rất nhanh, nhưng cần hàng triệu con, và gặp từ mới ("GX30", "khôngg") là bó tay.
- **Một con dấu cho mỗi chữ cái:** chỉ cần vài chục con, nhưng in một câu mất rất nhiều lần đóng dấu.
- **Subword:** làm sẵn con dấu cho những **mẩu hay gặp** ("ng", "kh", "gói", "cước"), còn mẩu hiếm thì ghép từ chữ cái. Ít con dấu, ít lần đóng, và không bao giờ bó tay.

**Ví dụ có số (đo trên máy bạn).** BPE tự viết, học trên 4 800 tin nhắn đã chuẩn hóa, đo trên 1 200 tin nhắn khác (trung bình 8.3 âm tiết và 35 ký tự mỗi tin):

| Số merge | Vocab | Token mỗi tin nhắn | Tỉ lệ âm tiết thành đúng 1 token |
|---|---|---|---|
| 0 (chỉ ký tự) | 94 | 36.0 | 0% |
| 100 | 194 | 17.4 | 39% |
| 300 | 394 | 11.7 | 72% |
| 1 000 (chỉ học được 934) | 1 028 | 8.35 | 99.6% |

Vocab tăng thì chuỗi ngắn lại. 934 merge là hết: các cặp còn lại xuất hiện ít hơn 2 lần.

**Ba ý cốt lõi:**
1. Subword cân bằng giữa **kích thước vocab** (số tham số embedding) và **độ dài chuỗi** (chi phí attention, tiền API tính theo token).
2. BPE học merge bằng cách **đếm**: lặp lại "ghép cặp kề nhau phổ biến nhất". Với từ mới, nó áp lại các merge đã học, nên không bao giờ hết từ vựng.
3. Tokenizer là **một phần của model**: model được train với tokenizer nào thì phải dùng **đúng** tokenizer đó.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 14.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn | Dùng trong |
|---|---|---|---|
| **BPE** | Bắt đầu từ ký tự; lặp: ghép cặp kề nhau có **tần suất** lớn nhất | Gage 1994 (nén dữ liệu); Sennrich, Haddow, Birch, ACL 2016 | GPT-2/3/4 (byte-level), RoBERTa |
| **Byte-level BPE** | BPE trên byte UTF-8, vocab gốc 256 byte, không có `<unk>` | Radford et al., *GPT-2* 2019 | GPT-2, nhiều LLM hiện nay |
| **WordPiece** | Như BPE, nhưng chọn cặp làm **tăng likelihood** nhiều nhất: điểm $\dfrac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$; phần nối tiếp có tiền tố `##`; khi tokenize thì khớp **dài nhất trước** | Schuster & Nakajima, ICASSP 2012; Wu et al., *GNMT* 2016 | BERT |
| **Unigram LM** | Bắt đầu từ vocab lớn, **bỏ dần** token ít làm giảm likelihood nhất; mỗi từ có nhiều cách tách, có xác suất | Kudo, ACL 2018 | T5, XLM-R, ALBERT |
| **SentencePiece** | Thư viện coi văn bản là dòng ký tự thô, **dấu cách cũng là ký hiệu** (`▁`); chạy BPE hoặc Unigram; detokenize không mất thông tin | Kudo & Richardson, EMNLP 2018 (demo) | T5, LLaMA 1–2, XLM-R |
| **Special tokens** | `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[UNK]` (BERT); `<s>`, `</s>`, `<pad>`, `<unk>` (XLM-R, T5); `<bos>`, `<eos>` | tài liệu của từng model | |

#### 14.2 🐢 Đi chậm từng bước: học BPE trên một corpus đồ chơi

Corpus: "goi" ×5, "gia" ×3, "gian" ×2, "han" ×4. Mỗi từ được tách thành ký tự và thêm `</w>` ở cuối.

*Bước 0 · Trạng thái ban đầu.*

| Từ (tần suất) | Ký hiệu |
|---|---|
| goi (5) | `g o i </w>` |
| gia (3) | `g i a </w>` |
| gian (2) | `g i a n </w>` |
| han (4) | `h a n </w>` |

*Bước 1 · Đếm mọi cặp kề nhau, có nhân tần suất của từ.*
- `a n`: từ "gian" (2) và "han" (4), tổng **6**.
- `n </w>`: cũng từ "gian" và "han", tổng **6**.
- `g i`: 3 + 2 = 5; `g o`: 5; `i </w>`: 5; `i a`: 5; `o i`: 5; `h a`: 4; `a </w>`: 3.

Có **hòa** ở 6. Luật của ta: hòa thì chọn cặp **nhỏ hơn theo thứ tự chuỗi**. Vì `"a" < "n"`, chọn **`a n` → `an`**.

*Bước 2.* Sau khi ghép: `g i an </w>` (2), `h an </w>` (4). Cặp `an </w>` có 6, lớn nhất, nên ghép **`an </w>` → `an</w>`**.

*Bước 3.* Còn bốn cặp cùng 5: `g i`, `g o`, `i </w>`, `o i`. Cặp nhỏ nhất theo thứ tự chuỗi là **`g i` → `gi`**.

*Bước 4–6.* Tiếp tục được `g o` → `go`, rồi `go i` → `goi`, rồi `goi </w>` → `goi</w>`.

Merge đã học, theo thứ tự (gọi là **hạng**, rank): `(a,n)`, `(an,</w>)`, `(g,i)`, `(g,o)`, `(go,i)`, `(goi,</w>)`.

*Bước 7 · Tokenize một từ mới, "giang".* Bắt đầu: `g i a n g </w>`. Mỗi vòng, ghép cặp kề nhau có **hạng nhỏ nhất** (học sớm nhất):
1. Các cặp có hạng: `a n` (0), `g i` (2). Ghép `a n`: `g i an g </w>`.
2. `an g` không phải merge đã học; `g i` (2) ghép: `gi an g </w>`.
3. Không còn cặp nào đã học. Kết quả: **`gi` `an` `g` `</w>`**.

Chú ý: "an" ở **giữa** từ "giang" là token `an`, khác với "an" ở **cuối** từ "gian" là token `an</w>`. Ký hiệu cuối từ giúp model biết ranh giới.

*Bước 8 · Ký tự chưa từng thấy.* "xin goi": `x` không có trong vocab nên thành `<unk>`; "goi" thành `goi</w>`. Byte-level BPE không gặp vấn đề này, vì mọi văn bản đều là dãy của 256 byte.

#### 14.3 Ví dụ tính tay

**(a) WordPiece: khớp dài nhất trước.** Vocab: `goi`, `go`, `dang`, `##ky`, `##n`, `##i`, `[UNK]`.
- "dangky": từ đầu, tiền tố dài nhất có trong vocab là `dang`; phần còn lại phải có tiền tố `##`: `##ky`. Kết quả **`dang` `##ky`**.
- "gon": `goi` không khớp; `go` khớp; còn `n`, tìm `##n`. Kết quả **`go` `##n`**.
- "xin": không tiền tố nào khớp, nên **cả từ** thành `[UNK]` (WordPiece không tách từng ký tự lạ).

**(b) Điểm chọn merge của WordPiece so với BPE.** Đếm được: `count(g, i) = 50`, `count(g) = 100`, `count(i) = 200`; `count(q, u) = 10`, `count(q) = 10`, `count(u) = 40`.
- BPE chọn theo tần suất: **`g i`** (50 > 10).
- WordPiece chọn theo $\frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$: `g i` được $50/(100 \cdot 200) = 0.0025$; `q u` được $10/(10 \cdot 40) = 0.025$. Chọn **`q u`**: "q" **luôn** đi với "u", nên ghép chúng là "chắc chắn", dù cặp này hiếm.

**(c) Byte UTF-8 của tiếng Việt.**

| Chuỗi | Ký tự | Byte UTF-8 |
|---|---|---|
| `a` | 1 | 1 |
| `ủ` | 1 | **3** (225 187 167) |
| `hủy` | 3 | 5 |
| `Việt Nam` | 8 | 10 |
| `🙏` | 1 | 4 |

Trên dữ liệu, trung bình **1.22 byte mỗi ký tự**. Byte-level BPE phải học merge để gộp 3 byte của "ủ" thành một token trước khi học được các từ.

**(d) Vocab và số tham số.** Embedding có $V \times d$ tham số. Encoder pretrained dùng ở bài 18 có $V = 250\,002$, $d = 384$: embedding chiếm $96.0$M trên tổng $117.7$M tham số, tức **82%**. Model đa ngôn ngữ cần vocab lớn để phủ nhiều chữ viết.

#### 14.4 Đo trên máy bạn

**(a) BPE tự viết:** bảng ở Tầng 1. Một số từ sau 300 merge: `cước` → `cước</w>`; `chuyển` → `chuyển</w>`; `khuyến` → `kh` `uy` `ế` `n</w>` (từ ít gặp hơn nên bị tách). Merge đầu tiên học được là `i </w>`, `n g`, `n </w>`, `ng </w>`: các vần cuối phổ biến của tiếng Việt.

**(b) Thư viện `tokenizers` (Hugging Face)**, BPE cùng kích thước vocab 394: **10.9** token mỗi tin nhắn, so với 11.7 của bản tự viết. Thư viện dùng tiền tách theo khoảng trắng **và dấu câu**, cộng một số chi tiết khác về cách đếm, nên không trùng hoàn toàn.

**(c) Tokenizer của một model pretrained đa ngôn ngữ** (SentencePiece, vocab 250 002):

| Văn bản | Token mỗi tin nhắn |
|---|---|
| Đã chuẩn hóa, có dấu | 9.9 |
| Cùng văn bản, bỏ dấu | **11.5** (+16%) |

| Câu | Token |
|---|---|
| tôi muốn hủy gói cước | `▁tôi` `▁muốn` `▁hủy` `▁gói` `▁c` `ước` |
| toi muon huy goi cuoc | `▁toi` `▁mu` `on` `▁huy` `▁go` `i` `▁cu` `oc` |
| ko dc | `▁ko` `▁d` `c` |

Tokenizer này được học trên văn bản tiếng Việt **có dấu**, nên văn bản không dấu bị cắt vụn. Khi trả tiền theo token (Tuần 6), cùng một nội dung có thể đắt hơn 16%.

#### 14.5 Code chuẩn

Bản tự viết là bài tập hôm nay (`intent/bpe.py`, xem cuối file). Với thư viện (offline):

```python
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.train_from_iterator(train_texts, trainers.BpeTrainer(vocab_size=400, special_tokens=["<pad>", "<unk>", "<cls>"]))
enc = tok.encode("hủy gói gx30")
enc.tokens, enc.ids
tok.save("bpe.json")                                       # lưu CÙNG với model

# Tokenizer của model pretrained (cần tải lần đầu; MODEL_ID là tên trên Hub)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID)
batch = tok(["hủy gói", "mạng yếu quá"], padding=True, truncation=True, max_length=64, return_tensors="pt")
batch["input_ids"], batch["attention_mask"]                # attention_mask: 1 = token thật, 0 = đệm
```

#### 14.6 Cạm bẫy

1. **Dùng tokenizer khác với lúc train** (khác phiên bản, khác bước chuẩn hóa): id lệch, model nhận rác mà không báo lỗi. Lưu tokenizer **cùng** checkpoint.
2. **Đếm độ dài bằng ký tự hoặc từ** khi giới hạn là theo token (context window, giá API). Tiếng Việt không dấu tốn hơn có dấu 16% (mục 14.4).
3. **Chuẩn hóa trước tokenizer khác lúc train của model pretrained:** tokenizer được học trên văn bản có dấu, NFC. Bỏ dấu "cho sạch" lại làm hại.
4. **Special token lọt từ dữ liệu người dùng:** người dùng gõ đúng chuỗi `</s>` hoặc `<|endoftext|>`. Nếu tokenizer biến nó thành special token thật, model có thể kết thúc sớm hoặc hiểu sai cấu trúc prompt (một dạng prompt injection, Tuần 7).
5. **Số bị tách vụn:** "200000" có thể thành nhiều token tùy tokenizer, nên model tính toán số kém. Placeholder `<money>` (bài 11) tránh được việc này khi số không quan trọng.
6. **Quên `</w>` hoặc `▁`:** không phân biệt được "an" cuối từ với "an" giữa từ, nên detokenize sai khoảng trắng.

**Sắc thái phỏng vấn:**
- *"Vì sao LLM dùng subword chứ không dùng từ hay ký tự?"* Từ: vocab khổng lồ, có OOV. Ký tự: chuỗi dài, attention $O(T^2)$ đắt, mỗi token mang ít nghĩa. Subword: vocab vừa phải (32k–250k), không có OOV (byte-level), chuỗi ngắn.
- *"BPE khác WordPiece khác Unigram?"* Tiêu chí ghép (tần suất / likelihood) và hướng xây (ghép dần / cắt dần); Unigram cho nhiều cách tách có xác suất, nên dùng được cho *subword regularization* (Kudo 2018).
- *"Vì sao gọi API LLM bằng tiếng Việt đắt hơn tiếng Anh?"* Tokenizer học chủ yếu trên tiếng Anh, nên tiếng Việt bị tách thành nhiều token hơn cho cùng nội dung. Đo thử trên dữ liệu của bạn.

#### 14.7 Đọc thêm

- R. Sennrich, B. Haddow, A. Birch, *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016.
- T. Kudo, *Subword Regularization*, ACL 2018; T. Kudo & J. Richardson, *SentencePiece*, EMNLP 2018 (demo).
- Hugging Face LLM Course, chương 6 *The Tokenizers library* (các mục BPE, WordPiece, Unigram).
- A. Karpathy, *Let's build the GPT Tokenizer* (video, 2024) và repo `minbpe`.

---

## 15 · Self-attention, multi-head attention, positional encoding, causal mask

> ⏱ Tầng 1: 15' · Tầng 2: 50'
> **Cần biết trước:** attention như tra cứu mềm, chia $\sqrt d$, mask (Tuần 4, bài 09); `dot_attention` của bạn (Tuần 4, Ngày 3); LayerNorm, residual (Tuần 4, bài 04–05).

> **📖 Từ mới**
> - **Self-attention:** attention trong đó query, key, value đều sinh ra từ **cùng** một chuỗi: mỗi token "hỏi" mọi token khác trong câu.
> - **Phép chiếu Q/K/V:** ba ma trận học được $W_Q, W_K, W_V$ biến vector của một token thành query, key, value của nó.
> - **Head:** một bộ $(W_Q, W_K, W_V)$ riêng với số chiều nhỏ hơn; **multi-head** chạy nhiều head song song rồi nối kết quả.
> - **Positional encoding:** thông tin vị trí được cộng (hoặc xoay) vào vector token, vì attention tự nó **không** biết thứ tự.
> - **Causal mask:** mask tam giác dưới; token ở vị trí $t$ chỉ được nhìn các vị trí $\le t$.
> - **Padding mask:** che các vị trí là token đệm.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Một cuộc họp, mỗi người là một token. Mỗi người cầm ba tấm thẻ:
- **Query**, "tôi đang tìm gì": người nói chữ "hủy" giơ thẻ "tôi cần biết hủy **cái gì**".
- **Key**, "tôi là gì": "gói" giơ thẻ "tôi là một **đối tượng** có thể bị hủy".
- **Value**, "tôi đóng góp gì nếu được chọn": "gói" đưa ra nội dung "gói cước GX30".

Mỗi người so query của mình với key của **mọi** người, rồi tổng hợp value theo mức khớp. **Multi-head** là cùng lúc chạy nhiều cuộc họp nhỏ, mỗi cuộc hỏi một loại câu hỏi khác nhau (một head tìm "đối tượng", một head tìm "phủ định ở gần"). Nhưng nếu mọi người đứng lẫn lộn thì không ai biết ai nói trước ai sau. Vì vậy mỗi người được đeo thêm **số thứ tự**: đó là positional encoding.

**Ví dụ có số: causal self-attention cho 3 token.** Lấy $Q = K = V = X$ (bỏ qua các phép chiếu), $d = 2$:

$$X = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \qquad \frac{XX^\top}{\sqrt 2} = \begin{pmatrix} 0.707 & 0 & 0.707 \\ 0 & 0.707 & 0.707 \\ 0.707 & 0.707 & 1.414 \end{pmatrix}.$$

Causal mask đặt $-\infty$ ở phía trên đường chéo, rồi softmax theo từng hàng:

| Token | Trọng số (được nhìn) | Đầu ra |
|---|---|---|
| 1 | $(1,\ 0,\ 0)$ | $(1, 0)$ |
| 2 | $(0.330,\ 0.670,\ 0)$ | $(0.330,\ 0.670)$ |
| 3 | $(0.248,\ 0.248,\ 0.503)$ | $(0.752,\ 0.752)$ |

Token 1 chỉ thấy chính nó. Token 3 thấy cả ba, và nhìn mình nhiều nhất (điểm 1.414).

**Ba ý cốt lõi:**
1. Self-attention cho **mọi cặp** token tương tác trực tiếp trong một bước: đường đi giữa hai token bất kỳ dài $O(1)$, so với $O(T)$ ở RNN. Chi phí là ma trận $T \times T$.
2. Multi-head chia $d$ thành $h$ phần, mỗi head học một kiểu quan hệ. **Tổng số tham số không đổi** so với một head lớn.
3. Attention không biết thứ tự (đổi chỗ các token thì đầu ra đổi chỗ y hệt), nên phải thêm vị trí. Causal mask biến encoder thành decoder sinh văn bản.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 15.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Scaled dot-product attention** | $\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\big(QK^\top/\sqrt{d_k}\big)V$ | Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 |
| **Multi-head attention** | $\mathrm{MHA}(X) = \mathrm{Concat}(\text{head}_1..\text{head}_h)W_O$, $\text{head}_i = \mathrm{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)})$, $d_k = d/h$ | như trên |
| **Sinusoidal PE** | $PE_{p,2i} = \sin(p/10000^{2i/d})$, $PE_{p,2i+1} = \cos(p/10000^{2i/d})$, cộng vào embedding | như trên |
| **Learned PE** | Một `nn.Embedding(max_len, d)` cho vị trí | BERT, GPT-2 |
| **RoPE** | Xoay từng cặp chiều của $q$ và $k$ một góc tỉ lệ với vị trí; $q_m^\top k_n$ chỉ phụ thuộc $m - n$ | Su et al., *RoFormer*, 2021 (Neurocomputing 2024) |
| **ALiBi** | Không thêm PE; cộng vào điểm attention một khoản phạt tuyến tính theo khoảng cách | Press, Smith, Lewis, ICLR 2022 |
| **Causal (look-ahead) mask** | $M_{ts} = 0$ nếu $s \le t$, $-\infty$ nếu $s > t$ | Vaswani et al. 2017 |
| **Cross-attention** | Query từ decoder, key/value từ encoder | như trên (Tuần 4, bài 09) |
| **FlashAttention** | Tính attention theo khối, không lưu ma trận $T \times T$ vào bộ nhớ chính của GPU; kết quả giống hệt | Dao et al., NeurIPS 2022 |

#### 15.2 🐢 Đi chậm từng bước: shape trong multi-head attention

Batch $B = 2$ câu, mỗi câu $T = 5$ token, $d = 16$, $h = 4$ head, nên $d_\text{head} = d/h = 4$.

| Bước | Phép tính | Shape | Ghi chú |
|---|---|---|---|
| 1 | $X$ | $(2, 5, 16)$ | đầu vào |
| 2 | $Q = XW_Q$, $K = XW_K$, $V = XW_V$ | mỗi cái $(2, 5, 16)$ | $W$ có shape $16 \times 16$: **một** phép nhân cho cả 4 head |
| 3 | `view(B, T, h, d_head)` | $(2, 5, 4, 4)$ | cắt 16 chiều thành 4 khúc 4 chiều |
| 4 | `transpose(1, 2)` | $(2, 4, 5, 4)$ | đưa head lên trước: giờ là "8 bài toán attention nhỏ" |
| 5 | $S = QK^\top/\sqrt{4}$ | $(2, 4, 5, 5)$ | mỗi head một bảng điểm $T \times T$ |
| 6 | mask + softmax theo chiều cuối | $(2, 4, 5, 5)$ | mỗi hàng có tổng bằng 1 |
| 7 | $SV$ | $(2, 4, 5, 4)$ | |
| 8 | `transpose(1, 2).reshape(B, T, d)` | $(2, 5, 16)$ | **nối** các head lại |
| 9 | $\cdot\,W_O$ | $(2, 5, 16)$ | trộn thông tin giữa các head |

**Vì sao multi-head không tốn thêm tham số.** Bước 2 dùng một ma trận $d \times d$, không phải $h$ ma trận $d \times d$. Mỗi head chỉ dùng một "lát" $d \times d_\text{head}$ của nó. Tổng: $4(d^2 + d)$ tham số với bias, bất kể $h$. Với $d = 12$: $4(144 + 12) = 624$.

**Mask có shape gì?** Padding mask `(B, T)` được mở rộng thành `(B, 1, 1, T)`: cùng một mask cho mọi head và mọi query. Causal mask `(T, T)` broadcast tới `(B, h, T, T)`. Kết hợp cả hai bằng phép `&`.

#### 15.3 🐢 Đi chậm từng bước: vị trí

*Bước 1 · Vì sao cần vị trí.* Đổi chỗ "tôi muốn hủy" thành "hủy muốn tôi": mỗi token vẫn thấy đúng các token cũ, chỉ ở vị trí khác. Self-attention không có phép tính nào phụ thuộc vào **chỉ số** vị trí, nên đầu ra của mỗi token vẫn y hệt, chỉ đổi chỗ. Cả câu thành một "túi từ" (bài 12).

*Bước 2 · Sinusoidal: mỗi vị trí một "đồng hồ nhiều kim".* Với $d = 4$ có hai tần số: $\omega_0 = 1$ và $\omega_1 = 10000^{-2/4} = 0.01$.

| Vị trí $p$ | $\sin p$ | $\cos p$ | $\sin 0.01p$ | $\cos 0.01p$ |
|---|---|---|---|---|
| 0 | 0 | 1 | 0 | 1 |
| 1 | 0.841 | 0.540 | 0.010 | 1.000 |
| 2 | 0.909 | −0.416 | 0.020 | 1.000 |

Kim nhanh (cặp chiều 0–1) phân biệt các vị trí gần nhau; kim chậm (cặp 2–3) phân biệt các vị trí xa. Vector này được **cộng** vào embedding của token.

*Bước 3 · RoPE: xoay thay vì cộng.* Lấy cặp chiều $(x_0, x_1)$ và xoay một góc $p\,\theta$:
$$\begin{pmatrix} x_0' \\ x_1' \end{pmatrix} = \begin{pmatrix} \cos p\theta & -\sin p\theta \\ \sin p\theta & \cos p\theta \end{pmatrix}\begin{pmatrix} x_0 \\ x_1 \end{pmatrix}.$$
Vector $(1, 0)$ với $\theta = 1$: ở $p = 1$ thành $(0.540,\ 0.841)$; ở $p = 2$ thành $(-0.416,\ 0.909)$. Độ dài giữ nguyên.

*Bước 4 · Vì sao RoPE cho vị trí tương đối.* Xoay $q$ một góc $m\theta$ và $k$ một góc $n\theta$. Tích vô hướng của hai vector đã xoay chỉ phụ thuộc **hiệu** của hai góc, $(m - n)\theta$. Đo với $q = (1, 0)$, $k = (0, 1)$:

| $(m, n)$ | (1, 0) | (2, 1) | (3, 2) | (0, 1) |
|---|---|---|---|---|
| $q_m^\top k_n$ | 0.8415 | 0.8415 | 0.8415 | −0.8415 |

Ba cặp cách nhau một vị trí cho đúng cùng một điểm. RoPE được áp vào $q$ và $k$ (không áp vào $v$) **bên trong** mỗi lớp attention.

#### 15.4 Ví dụ tính tay

**(a) Causal attention:** bảng ở Tầng 1. Kiểm tra hàng 2: $\mathrm{softmax}(0, 0.707) = (1, 2.028)/3.028 = (0.330,\ 0.670)$.

**(b) Bộ nhớ của ma trận attention.** $T = 4\,096$ token, $h = 32$ head, fp16 (2 byte): $4096^2 \times 32 \times 2 = 1$ GiB **cho mỗi lớp, mỗi câu**. Đây là lý do FlashAttention không lưu ma trận này.

**(c) Đếm phép tính.** Một lớp self-attention với $T$ token: các phép chiếu tốn $4Td^2$ phép nhân–cộng; $QK^\top$ và nhân với $V$ tốn $2T^2d$. Phần $T^2$ vượt phần $d^2$ khi $T > 2d$. Với $d = 768$, đó là từ $T > 1\,536$.

#### 15.5 Đo trên máy bạn

- `MultiHeadAttention` tự viết (bài tập hôm nay) với trọng số chép từ `nn.MultiheadAttention`: lệch tối đa **$1.2 \times 10^{-7}$** ($d = 64$, $h = 8$, 8 câu × 20 token). Trọng số attention khớp tới $10^{-6}$.
- **Ngữ nghĩa mask ngược nhau:** `nn.MultiheadAttention(key_padding_mask=...)` coi `True` là **bị che**; code của ta (và `F.scaled_dot_product_attention`) coi `True` là **được nhìn**. Test của bài tập truyền `~keep` cho bên này và `keep` cho bên kia.
- **Causal mask chặn tương lai:** đổi ngẫu nhiên token ở vị trí 4–5 thì đầu ra ở vị trí 0–3 **không đổi một bit**.

#### 15.6 Code chuẩn

```python
import math
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_head = n_heads, d_model // n_heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d_model, d_model) for _ in range(4))
        self.drop = nn.Dropout(dropout)

    def _split(self, x):                                         # (B, T, d) → (B, h, T, d_head)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_head).transpose(1, 2)

    def forward(self, x, kv=None, mask=None):                    # mask: True = được nhìn
        kv = x if kv is None else kv
        q, k, v = self._split(self.q_proj(x)), self._split(self.k_proj(kv)), self._split(self.v_proj(kv))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=-1)
        out = (self.drop(w) @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.out_proj(out), w

causal = torch.ones(T, T, dtype=torch.bool).tril()
# Bản tối ưu của PyTorch (FlashAttention khi có GPU hỗ trợ):
out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
```

#### 15.7 Cạm bẫy

1. **`view` sau `transpose` mà không `reshape`/`contiguous`:** lỗi "view size is not compatible". Ở bước 8 phải dùng `reshape`.
2. **Chia cho $\sqrt d$ thay vì $\sqrt{d_\text{head}}$:** softmax quá "mềm" (với $h = 8$ là chia thêm $\sqrt 8$).
3. **Ngược ngữ nghĩa mask giữa các API** (mục 15.5). Viết test so với một bản tham chiếu.
4. **Hàng bị che hết** (câu toàn đệm): softmax của toàn $-\infty$ cho `nan`. Thêm một token `<cls>` luôn có mặt (project làm vậy), hoặc thay `nan` bằng 0.
5. **Quên positional encoding:** model vẫn train được, nhưng "không muốn hủy" và "hủy không muốn" giống hệt nhau.
6. **Học PE theo `max_len` rồi suy luận với chuỗi dài hơn:** learned PE không có vector cho vị trí mới; sinusoidal và RoPE thì ngoại suy được một phần.
7. **Causal mask cho bài toán phân loại:** che mất ngữ cảnh bên phải một cách vô ích. Encoder dùng attention hai chiều, chỉ cần padding mask.

**Sắc thái phỏng vấn:**
- *"Viết công thức attention và giải thích từng phần."* $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$: so khớp, chia $\sqrt{d_k}$ để softmax không bão hòa (Tuần 4, mục 9.2), chuẩn hóa thành trọng số, trộn value. Nêu được shape ở mỗi bước (mục 15.2).
- *"Vì sao multi-head?"* Nhiều kiểu quan hệ trong cùng một lớp; cùng số tham số; các head có thể chuyên môn hóa (Voita et al., ACL 2019, cũng chỉ ra nhiều head **cắt bỏ được**).
- *"RoPE khác sinusoidal thế nào?"* Xoay $q$, $k$ trong mỗi lớp thay vì cộng một lần ở đầu vào; điểm attention chỉ phụ thuộc vị trí tương đối; là mặc định của phần lớn LLM mã nguồn mở hiện nay.
- *"Độ phức tạp của attention?"* $O(T^2 d)$ phép tính, $O(T^2)$ bộ nhớ cho ma trận điểm (FlashAttention giảm bộ nhớ xuống $O(T)$ nhưng phép tính vẫn $O(T^2)$).

#### 15.8 Đọc thêm

- A. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
- J. Alammar, *The Illustrated Transformer* (blog, 2018).
- J. Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv 2021.
- T. Dao et al., *FlashAttention*, NeurIPS 2022.
- E. Voita et al., *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned*, ACL 2019.

---

## 16 · Block Transformer, ba kiểu kiến trúc, đếm tham số, KV-cache

> ⏱ Tầng 1: 10' · Tầng 2: 45'
> **Cần biết trước:** bài 15; pre-LN residual, GELU/SwiGLU, đếm tham số FFN (Tuần 4, bài 04–06); bộ nhớ khi huấn luyện 16 byte/tham số (Tuần 4, bài 06); seq2seq (Tuần 4, bài 09).

> **📖 Từ mới**
> - **Block (lớp) Transformer:** attention + FFN, mỗi phần bọc trong residual và LayerNorm.
> - **FFN (feed-forward network):** MLP hai lớp áp **riêng cho từng token**: $d \to 4d \to d$.
> - **Pre-LN / Post-LN:** LayerNorm đặt **trước** mỗi nhánh (bên trong residual) hay **sau** phép cộng residual.
> - **Encoder-only / decoder-only / encoder–decoder:** chỉ có attention hai chiều / chỉ có attention causal / cả hai, nối nhau bằng cross-attention.
> - **Autoregressive generation:** sinh từng token một, token vừa sinh được nối vào đầu vào của bước sau.
> - **KV-cache:** lưu lại key và value của các token đã xử lý, để bước sau không phải tính lại.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Mỗi block Transformer là một vòng "họp rồi tự suy nghĩ":
- **Attention** là phần họp: mỗi token lấy thông tin từ các token khác.
- **FFN** là phần tự suy nghĩ: mỗi token xử lý riêng những gì vừa nghe được.
- **Residual** là cuốn sổ ghi chép: ý mới được **ghi thêm** vào sổ, không xóa ý cũ.

Chồng 12 vòng như vậy là BERT-base hay GPT-2. Khi **sinh** văn bản, mỗi lần viết thêm một từ thì không cần họp lại từ đầu: những gì mọi người đã nói (key, value) được ghi sẵn trong biên bản, tức KV-cache.

**Ví dụ có số: một lớp có bao nhiêu tham số?** Với $d = 768$ và FFN $4d$:
- Attention: 4 ma trận $d \times d$ cộng bias, được $4d^2 + 4d$.
- FFN: $d \times 4d + 4d$ và $4d \times d + d$, được $8d^2 + 5d$.
- 2 LayerNorm: $4d$.
- Tổng: **$12d^2 + 13d = 7\,087\,872$**. Đếm trên một lớp thật của BERT-base và của GPT-2 cho **đúng con số này**.

**Ba ý cốt lõi:**
1. Block = `x + Attn(LN(x))` rồi `x + FFN(LN(x))` (pre-LN). Gần như toàn bộ tham số nằm ở $12d^2$ mỗi lớp, cộng với embedding $V \times d$.
2. Encoder hai chiều để **hiểu** (phân loại, NER, embedding); decoder causal để **sinh**; encoder–decoder cho bài toán chuỗi → chuỗi (dịch, tóm tắt).
3. Sinh $T$ token mà không có KV-cache là tính lại toàn bộ chuỗi ở **mỗi** bước. Cache làm phép tính nhỏ đi hàng trăm lần, nhưng tốn **bộ nhớ**, và bộ nhớ đó tăng theo độ dài × batch.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 16.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Transformer block (post-LN, bản gốc)** | $x \leftarrow \mathrm{LN}(x + \mathrm{Attn}(x))$; $x \leftarrow \mathrm{LN}(x + \mathrm{FFN}(x))$ | Vaswani et al. 2017 |
| **Pre-LN** | $x \leftarrow x + \mathrm{Attn}(\mathrm{LN}(x))$; $x \leftarrow x + \mathrm{FFN}(\mathrm{LN}(x))$; thêm một LN cuối. Ổn định hơn, không cần warmup dài | Xiong et al., *On Layer Normalization in the Transformer Architecture*, ICML 2020 |
| **Encoder-only** | Attention hai chiều + padding mask; pretrain bằng MLM (bài 17) | BERT (Devlin et al., NAACL 2019), RoBERTa, XLM-R |
| **Decoder-only** | Attention causal; pretrain bằng dự đoán token tiếp theo | GPT (Radford et al. 2018, 2019), hầu hết LLM hiện nay |
| **Encoder–decoder** | Encoder hai chiều; decoder causal + cross-attention vào encoder | Vaswani et al. 2017; T5 (Raffel et al., JMLR 2020); BART (Lewis et al., ACL 2020) |
| **KV-cache** | Lưu $K, V$ của mọi token trước đó ở mỗi lớp; mỗi bước chỉ tính $q, k, v$ cho token mới | Thực hành chuẩn; Pope et al., *Efficiently Scaling Transformer Inference*, MLSys 2023 |
| **MQA / GQA** | Nhiều head query dùng chung một (MQA) hoặc vài (GQA) head key/value, nên cache nhỏ đi | Shazeer 2019 (MQA); Ainslie et al., EMNLP 2023 (GQA) |
| **Weight tying** | Ma trận embedding đầu vào và lớp chiếu ra vocab dùng chung trọng số | Press & Wolf, EACL 2017 |

#### 16.2 Ba kiểu kiến trúc

| | Encoder-only | Decoder-only | Encoder–decoder |
|---|---|---|---|
| Attention | Hai chiều (padding mask) | Causal | Encoder: hai chiều; decoder: causal + cross |
| Pretrain | MLM: đoán token bị che | Đoán token tiếp theo | Khôi phục đoạn bị che / dịch |
| Đầu ra tự nhiên | Vector cho mỗi token / cả câu | Phân phối token tiếp theo | Chuỗi mới |
| Giỏi | Phân loại, NER, embedding cho tìm kiếm (Tuần 8) | Sinh văn bản, chat, few-shot | Dịch, tóm tắt |
| Ví dụ | BERT, RoBERTa, XLM-R, encoder ở bài 18 | GPT-2, LLM hiện nay | T5, BART |
| **Project tuần này** | **Router intent** (`EncoderClassifier`) | char-LM ở bài 17 | |

Decoder-only thắng thế cho LLM vì một mục tiêu duy nhất (token tiếp theo) dùng được cho mọi bài toán qua prompt, và vì dễ mở rộng quy mô. Với bài toán **chỉ** phân loại, có ít dữ liệu và cần độ trễ thấp, encoder nhỏ vẫn là lựa chọn hợp lý.

#### 16.3 🐢 Đi chậm từng bước: sinh văn bản có và không có KV-cache

Decoder đã đọc prompt 3 token $x_1 x_2 x_3$, và đang sinh token thứ 4.

*Không có cache.* Bước sinh $x_4$: đưa cả $x_1 x_2 x_3$ vào, tính $Q, K, V$ cho 3 vị trí, attention $3 \times 3$, rồi lấy đầu ra ở vị trí cuối. Bước sinh $x_5$: đưa $x_1..x_4$ vào, tính **lại** $K, V$ của $x_1..x_3$ dù chúng không đổi (causal mask bảo đảm điều đó: token cũ không bao giờ nhìn token mới).

*Có cache.* Ở mỗi lớp, giữ hai tensor $K_\text{cache}, V_\text{cache}$ có shape $(B, h, t, d_\text{head})$. Bước sinh tiếp theo:
1. Chỉ đưa **token mới** vào, tính $q, k, v$ của nó: shape $(B, h, 1, d_\text{head})$.
2. Nối: $K_\text{cache} \leftarrow [K_\text{cache}; k]$, và tương tự cho $V$.
3. Attention: $\mathrm{softmax}(q K_\text{cache}^\top/\sqrt{d_\text{head}})\,V_\text{cache}$, với bảng điểm chỉ có shape $1 \times t$. Không cần causal mask, vì cache chỉ chứa quá khứ.

*Kiểm chứng trên máy bạn* ($d = 96$, $h = 4$, $T = 64$): đầu ra giải mã từng bước có cache khớp với tính toàn bộ chuỗi bằng causal mask tới **$4.2 \times 10^{-7}$**.

*Phép tính tiết kiệm được* (một lớp, chỉ tính phép chiếu và attention, sinh $T$ token):

| $T$ | Không cache / có cache |
|---|---|
| 64 | 34 lần |
| 256 | 146 lần |
| 1 024 | 637 lần |

Không có cache, tổng chi phí tăng như $T^2 d^2 + T^3 d$; có cache thì là $T d^2 + T^2 d$.

*Cái giá: bộ nhớ.* Cache của **một** câu, ở mọi lớp:
$$\text{bytes} = 2 \times L \times T \times d_{kv} \times \text{bytes/số},$$
với hệ số 2 cho $K$ và $V$, và $d_{kv} = (\text{số head KV}) \times d_\text{head}$.

#### 16.4 Ví dụ tính tay

**(a) BERT-base** ($L = 12$, $d = 768$, $V = 30\,522$, 512 vị trí, 2 loại segment), đếm thật trên máy bạn:

| Phần | Phép tính | Tham số |
|---|---|---|
| Embedding | $(30\,522 + 512 + 2) \times 768 + \text{LN}(2 \times 768)$ | 23 837 184 |
| 12 lớp | $12 \times (12d^2 + 13d)$ | 85 054 464 |
| Pooler | $768 \times 768 + 768$ | 590 592 |
| **Tổng** | | **109 482 240** |

**(b) GPT-2 small** ($V = 50\,257$, 1 024 vị trí): $38\,597\,376$ (wte) $+ 786\,432$ (wpe) $+ 85\,054\,464 + 1\,536$ (LN cuối) $=$ **124 439 808**. Lớp chiếu ra vocab **dùng chung** trọng số với wte (weight tying), nên không tốn thêm 38.6M.

**(c) Encoder của project** ($V$ token BPE, $d = 64$, $h = 4$, 2 lớp, $d_\text{ff} = 128$, 9 lớp đầu ra):
- Mỗi lớp: attention $4(64^2 + 64) = 16\,640$; FFN $64 \cdot 128 + 128 + 128 \cdot 64 + 64 = 16\,576$; 2 LN $256$. Tổng $33\,472$.
- Tổng model: $64V + 2 \times 33\,472 + 128 + (64 \cdot 9 + 9) = 64V + 67\,657$. Với $V = 900$: **125 257** (đúng số đo).

**(d) KV-cache của một decoder cỡ 7B** (giả định: $L = 32$, $d = 4\,096$, 32 head, fp16), ngữ cảnh $T = 4\,096$:
- Đủ head: $2 \times 32 \times 4096 \times 4096 \times 2$ byte $=$ **2 GiB mỗi câu**. Batch 16 câu là 32 GiB, **nhiều hơn** cả trọng số (khoảng 13 GiB ở fp16).
- GQA với 8 head KV ($d_{kv} = 1\,024$): **0.5 GiB** mỗi câu, nhỏ đi 4 lần.

Đây là lý do bộ nhớ KV-cache quyết định số người dùng đồng thời khi phục vụ LLM (Tuần 12, vLLM / PagedAttention).

#### 16.5 Đo trên máy bạn

- Số tham số mỗi lớp $12d^2 + 13d$: khớp **chính xác** với lớp thật của BERT-base và GPT-2 (dựng từ config, `BertModel(BertConfig())`, `GPT2LMHeadModel(GPT2Config())`).
- KV-cache: khớp tới $4.2 \times 10^{-7}$ (mục 16.3).
- **Embedding chiếm phần lớn tham số ở model nhỏ:** encoder của project có $64 \times 900 = 57\,600$ trên 125 257 tham số (46%); encoder pretrained ở bài 18 có 82%.

#### 16.6 Code chuẩn: block pre-LN

```python
class TransformerBlock(nn.Module):
    def __init__(self, d, h, d_ff, dropout=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, h, dropout)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        a, _ = self.attn(self.ln1(x), mask=mask)
        x = x + self.drop(a)                                  # residual quanh attention
        return x + self.drop(self.ff(self.ln2(x)))            # residual quanh FFN
```

`EncoderClassifier` của project (Ngày 4): embedding + sinusoidal PE → các block → LN cuối → **mean-pool có mask** → Linear. Decoder (bài 17) giống hệt, chỉ khác là truyền `causal_mask(T)` và đầu ra là logit trên vocab tại **mọi** vị trí.

#### 16.7 Cạm bẫy

1. **Post-LN, nhiều lớp, learning rate lớn, không warmup:** dễ phân kỳ (Xiong et al. 2020). Pre-LN là mặc định an toàn.
2. **Quên LayerNorm cuối với pre-LN:** đầu ra của residual stream không được chuẩn hóa, nên lớp head nhận vector có thang đo trôi theo độ sâu.
3. **Đếm tham số mà quên embedding:** GPT-2 small có 31% tham số ở wte. Ngược lại, đếm hai lần khi có weight tying.
4. **KV-cache với padding bên trái/phải không đúng:** câu ngắn trong batch có vị trí khác nhau; phải truyền mask và vị trí (position id) đúng cho từng câu.
5. **Ước lượng bộ nhớ phục vụ chỉ bằng trọng số:** cache có thể lớn hơn trọng số (mục 16.4(d)).
6. **Dùng decoder causal cho bài toán phân loại ngắn với ít dữ liệu:** mỗi token chỉ thấy bên trái; encoder hai chiều hợp hơn.

**Sắc thái phỏng vấn:**
- *"Đếm tham số của một Transformer."* $12Ld^2$ cộng $Vd$ (cộng vị trí). Biết vì sao là 12: $4d^2$ cho attention, $8d^2$ cho FFN với $4d$. Với SwiGLU, FFN có 3 ma trận và hidden $\frac{8}{3}d$, nên vẫn khoảng $8d^2$ (Tuần 4, bài 03).
- *"KV-cache là gì, tốn bao nhiêu bộ nhớ?"* Viết công thức $2LTd_{kv} \times$ byte; nói rằng cache biến chi phí mỗi bước từ $O(T)$ token xuống 1 token; nhắc MQA/GQA và PagedAttention.
- *"BERT khác GPT?"* Encoder hai chiều + MLM so với decoder causal + next-token; hiểu so với sinh; cách dùng cho phân loại (vector `[CLS]`/mean-pool + head) so với prompt.

#### 16.8 Đọc thêm

- R. Xiong et al., *On Layer Normalization in the Transformer Architecture*, ICML 2020.
- J. Devlin et al., *BERT*, NAACL 2019; A. Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019.
- C. Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5), JMLR 2020.
- J. Ainslie et al., *GQA*, EMNLP 2023.
- A. Karpathy, *nanoGPT* (GitHub) và video *Let's build GPT: from scratch, in code, spelled out* (2023).
- Harvard NLP, *The Annotated Transformer* (blog, 2018; bản cập nhật 2022).

---

## Bài tập Ngày 2 · `intent/bpe.py` và `intent/attention.py` (~2h)

Hai mảnh tiếp theo của project. Lệnh test: `python -m pytest -q tests/test_bpe.py tests/test_attention.py` (5 + 6 = 11 test).

**`intent/bpe.py`** (~50'):

| Hàm / lớp | Yêu cầu |
|---|---|
| Hằng số | `EOW = "</w>"`, `SPECIALS = ("<pad>", "<unk>", "<cls>")` |
| `train_bpe(texts, num_merges, min_freq=2) -> BPE` | Tần suất từ = đếm các từ (tách theo khoảng trắng) trên mọi văn bản. Mỗi từ thành tuple ký tự + `EOW`. Lặp tối đa `num_merges` lần: đếm cặp kề nhau (nhân tần suất từ); chọn cặp có **tần suất lớn nhất, hòa thì cặp nhỏ nhất theo thứ tự tuple**; nếu tần suất đó < `min_freq` hoặc không còn cặp nào thì dừng; ghép cặp đó ở mọi từ. Alphabet = các ký tự đã thấy |
| `BPE(merges, alphabet)` | `.merges` (list cặp), `.vocab`: dict token → id theo thứ tự `SPECIALS` + `sorted(alphabet ∪ {EOW})` + các token ghép theo thứ tự học (`a + b`); bỏ trùng, giữ lần xuất hiện đầu |
| `.encode_word(word) -> list[str]` | Từ tuple ký tự + `EOW`; mỗi vòng ghép cặp kề nhau có **hạng nhỏ nhất** (hạng = chỉ số trong `merges`); dừng khi không còn cặp đã học. Nên cache theo từ |
| `.tokenize(text)`, `.encode(text)`, `.decode(ids)` | Tokenize từng từ rồi nối; `encode`: token lạ thành id của `<unk>`; `decode`: bỏ special token, nối chuỗi, thay `EOW` bằng dấu cách, `strip()` |

**`intent/attention.py`** (~70'):

| Hàm / lớp | Yêu cầu |
|---|---|
| `causal_mask(T)` | Bool `(T, T)`, tam giác dưới, `True` = được nhìn |
| `padding_mask(lengths, T)` | Bool `(B, T)`, `True` ở vị trí `< lengths[b]` |
| `sinusoidal_pe(T, d)` | Công thức 15.1, shape `(T, d)`, float32 |
| `apply_rope(x, base=10000.0)` | `x` shape `(..., T, d)`, $d$ chẵn; cặp **xen kẽ** `(x[..., 2i], x[..., 2i+1])` ở vị trí $t$ xoay góc $t\,\theta_i$, $\theta_i = \text{base}^{-2i/d}$ |
| `MultiHeadAttention(d_model, n_heads, dropout=0.0, bias=True)` | Thuộc tính `q_proj`, `k_proj`, `v_proj`, `out_proj` (`nn.Linear(d, d)`), `h`, `d_head`; `assert d_model % n_heads == 0`. `forward(x, kv=None, mask=None) -> (out, weights)`: `kv=None` là self-attention; `mask` bool, `True` = được nhìn, broadcast tới `(B, h, T_q, T_k)`; `weights` shape `(B, h, T_q, T_k)` là trọng số **trước** dropout |

**Gợi ý:**
- `test_matches_torch_multihead_attention_with_padding` chép `in_proj_weight` (shape `(3d, d)`) của PyTorch thành ba khối `q`, `k`, `v` theo thứ tự đó. Nếu chỉ lệch ở trọng số attention, kiểm tra lại phép chia $\sqrt{d_\text{head}}$.
- `test_rope_keeps_norm_and_depends_on_relative_position` là phiên bản tự động của mục 15.3, bước 4.
