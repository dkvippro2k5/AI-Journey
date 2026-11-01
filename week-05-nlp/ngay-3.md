# Ngày 3 · Mô hình ngôn ngữ, model pretrained và Hugging Face, metric NLP

> **Loại tài liệu: Lý thuyết (bản nhẹ).** Bài 18 nghiêng về Kỹ thuật.
> **Hôm nay gồm 3 bài:**
> - **17** Mô hình ngôn ngữ: autoregressive, masked LM, perplexity, decoding
> - **18** Model pretrained và Hugging Face: tokenizer, feature extraction, fine-tuning, gán nhãn token
> - **19** Metric NLP: P/R/F1, EM, BLEU, ROUGE, BERTScore, đánh giá bởi người; các thư viện
>
> **Thời gian:** đọc ~2h50 · quiz 45' · bài tập ~2h
> **Cần mạng (tùy chọn):** bài 18 dùng một encoder pretrained đa ngôn ngữ nhỏ (khoảng 118M tham số, 470 MB; trong code gọi là `MODEL_ID`). Model này đã được tải về máy bạn. Không có nó thì vẫn đọc và làm bài tập được, vì mọi test đều chạy offline.

---

## 17 · Mô hình ngôn ngữ: autoregressive, masked LM, perplexity, decoding

> ⏱ Tầng 1: 15' · Tầng 2: 45'
> **Cần biết trước:** cross-entropy, MLE (Tuần 2, bài 05, 10); chain rule xác suất (Tuần 2, bài 08); causal mask, decoder (bài 15–16); greedy và beam search (Tuần 4, bài 09).

> **📖 Từ mới**
> - **Mô hình ngôn ngữ (language model, LM):** model gán xác suất cho chuỗi token, hoặc tương đương, cho **token tiếp theo** khi biết các token trước.
> - **Autoregressive (tự hồi quy):** sinh từng token, mỗi token dựa vào các token đã sinh.
> - **Masked LM (MLM):** che ngẫu nhiên một số token rồi bắt model đoán lại, dùng ngữ cảnh **hai phía**.
> - **Perplexity:** "trung bình model phân vân giữa bao nhiêu lựa chọn"; càng thấp càng tốt.
> - **Temperature:** chia logit cho $T$ trước softmax. $T < 1$ làm phân phối nhọn hơn, $T > 1$ làm nó phẳng hơn.
> - **Top-k / top-p (nucleus):** chỉ lấy mẫu trong $k$ token tốt nhất / trong tập nhỏ nhất có tổng xác suất $\ge p$.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Trò chơi "đoán chữ tiếp theo": bạn đọc "tôi muốn hủy gó…" và gần như chắc chắn chữ tiếp là "i". Một mô hình ngôn ngữ chơi trò này hàng tỉ lần, và **mỗi lần đoán sai là một lần học**. Không cần ai gán nhãn: chính văn bản là đáp án. Muốn **viết** văn bản thì cho nó đoán chữ tiếp theo, viết chữ đó ra, rồi đoán tiếp.

**Ví dụ có số (đo trên máy bạn).** Mô hình ngôn ngữ **mức ký tự**, train trên 7 000 tin nhắn tổng đài, đo trên 1 000 tin khác:

| Model | Perplexity mỗi ký tự |
|---|---|
| 1-gram (chỉ tần suất ký tự) | 27.6 |
| 2-gram | 7.75 |
| 3-gram | 3.73 |
| 5-gram (làm mượt add-0.01) | 1.94 |
| **Transformer causal tự viết** (3 lớp, $d = 96$, 354k tham số, 2 000 bước) | **1.41** |

Perplexity 1.41 nghĩa là, trung bình, model chỉ phân vân giữa khoảng 1.4 ký tự ở mỗi bước. Cho nó tự viết (greedy, bắt đầu bằng "hủy "): **"hủy đăng ký gói plus120"**.

**Ba ý cốt lõi:**
1. LM = phân rã xác suất câu bằng chain rule: $P(x_1..x_T) = \prod_t P(x_t \mid x_{<t})$. Train bằng cross-entropy trên token tiếp theo, tức MLE.
2. Perplexity $= e^{\text{cross-entropy}}$. Chỉ so được giữa các model dùng **cùng** tokenizer và cùng tập test.
3. Model cho ra **phân phối**; cách chọn token từ phân phối đó (greedy, beam, temperature, top-k, top-p) là một quyết định riêng, đánh đổi giữa **an toàn/lặp lại** và **đa dạng/sai sót**.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 17.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **n-gram LM** | $P(x_t \mid x_{t-n+1..t-1})$ ước lượng bằng đếm, có làm mượt (add-k, Kneser–Ney) | SLP3 ch. 3; Kneser & Ney, ICASSP 1995 |
| **Neural LM** | Mạng neural dự đoán token tiếp theo từ embedding của ngữ cảnh | Bengio et al., JMLR 2003 |
| **Causal (autoregressive) LM** | Decoder Transformer + causal mask; loss là CE trên **mọi** vị trí cùng lúc | Radford et al., GPT, 2018 |
| **Masked LM (MLM)** | Chọn 15% token; trong đó 80% thay bằng `[MASK]`, 10% thay bằng token ngẫu nhiên, 10% giữ nguyên; đoán lại token gốc | Devlin et al., BERT, NAACL 2019 |
| **Span corruption** | Che các **đoạn** liên tiếp, decoder sinh lại đoạn bị che | Raffel et al., T5, JMLR 2020 |
| **Perplexity** | $\mathrm{PPL} = \exp\!\big(-\frac1N\sum_t \log P(x_t \mid x_{<t})\big)$ | SLP3 ch. 3 |
| **Temperature sampling** | Lấy mẫu từ $\mathrm{softmax}(z/T)$ | Ackley, Hinton, Sejnowski 1985 (Boltzmann) |
| **Top-k sampling** | Chỉ lấy mẫu trong $k$ token có logit lớn nhất | Fan, Lewis, Dauphin, ACL 2018 |
| **Nucleus (top-p) sampling** | Tập nhỏ nhất có tổng xác suất $\ge p$ | Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020 |
| **Teacher forcing** | Lúc train đưa token **đúng** của bước trước vào (Tuần 4, bài 09) | Williams & Zipser 1989 |

#### 17.2 🐢 Đi chậm từng bước: một câu thành bài toán huấn luyện

Tin nhắn "hủy gói", mức ký tự, thêm `^` (bắt đầu) và `$` (kết thúc).

*Bước 1 · Đầu vào và đích lệch nhau một vị trí.*

| Vị trí | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Đầu vào | `^` | h | ủ | y | ␣ | g | ó | i |
| Đích | h | ủ | y | ␣ | g | ó | i | `$` |

*Bước 2 · Một lần forward cho cả 8 dự đoán.* Nhờ causal mask, vị trí 3 (đầu vào "y") chỉ nhìn được `^ h ủ y`, nên nó phải **đoán** dấu cách mà không được nhìn thấy dấu cách. Cả 8 bài toán "đoán ký tự tiếp theo" được giải song song trong **một** phép nhân ma trận. Đó là lý do Transformer train nhanh hơn RNN.

*Bước 3 · Loss.* Cross-entropy trung bình trên 8 vị trí (bỏ các vị trí đệm khi ghép batch):
$$\mathcal L = -\frac18\sum_{t=1}^{8}\log P(\text{đích}_t \mid \text{đầu vào}_{\le t}).$$

*Bước 4 · Từ loss tới perplexity.* Giả sử model gán cho ba token đúng các xác suất 0.5, 0.25, 0.125:
- $\mathcal L = -\frac13(\ln 0.5 + \ln 0.25 + \ln 0.125) = \frac13 \cdot 6\ln 2 = 1.386$.
- $\mathrm{PPL} = e^{1.386} =$ **4**: trung bình nhân của $1/0.5$, $1/0.25$, $1/0.125$ (tức 2, 4, 8). Như thể mỗi bước model chọn đều giữa 4 khả năng.
- Model đoán **đều** trên vocab $V$ token có $\mathrm{PPL} = V$. Đây là mức "chưa học gì".

*Bước 5 · Sinh văn bản.* Bắt đầu với `^`, lặp: forward, lấy logit ở **vị trí cuối**, chọn một ký tự (mục 17.3), nối vào, cho tới khi gặp `$`.

#### 17.3 🐢 Đi chậm từng bước: chọn token từ logit

Logit của 4 token: $z = (2, 1, 0, -1)$.

| Cách chọn | Xác suất dùng để lấy mẫu | Ghi chú |
|---|---|---|
| Softmax, $T = 1$ | (0.644, 0.237, 0.087, 0.032) | |
| $T = 0.5$ (logit nhân 2) | (0.865, 0.117, 0.016, 0.002) | Nhọn hơn, gần greedy |
| $T = 2$ (logit chia 2) | (0.455, 0.276, 0.167, 0.102) | Phẳng hơn, đa dạng hơn |
| Greedy ($T \to 0$) | (1, 0, 0, 0) | Luôn chọn token 1 |
| Top-k, $k = 2$ | (0.731, 0.269, 0, 0) | Bỏ token 3, 4, chuẩn hóa lại |
| Top-p, $p = 0.9$ | (0.665, 0.245, 0.090, 0) | Tổng tích lũy 0.644 → 0.881 → **0.968**: cần 3 token mới vượt 0.9 |

**Top-p tự co giãn, top-k thì không.** Khi model chắc chắn (một token chiếm 0.95), top-p 0.9 chỉ giữ **1** token; khi model phân vân, nó giữ nhiều token. Top-k luôn giữ đúng $k$ token, kể cả khi $k - 1$ token cuối gần như vô nghĩa.

**Thứ tự áp dụng** (trong code của bạn, giống phần lớn thư viện): chia temperature → top-k → top-p → lấy mẫu.

#### 17.4 Ví dụ tính tay

**(a) MLM.** Câu 20 token: trung bình $20 \times 0.15 = 3$ token được chọn; trong đó $2.4$ thành `[MASK]`, $0.3$ thành token ngẫu nhiên, $0.3$ giữ nguyên. Loss **chỉ** tính trên 3 vị trí được chọn. Vì sao không che hết bằng `[MASK]`: lúc fine-tune và dùng thật **không có** `[MASK]`; giữ nguyên và thay ngẫu nhiên buộc model phải xây biểu diễn tốt cho **mọi** token, không chỉ cho chỗ có `[MASK]`.

**(b) Đổi perplexity ký tự sang perplexity âm tiết.** Mỗi tin nhắn trung bình 35.0 ký tự (kể cả dấu cách) và 8.31 âm tiết, tức 4.21 ký tự mỗi âm tiết. Cùng một tổng log-likelihood, nên $\mathrm{PPL}_\text{âm tiết} = \mathrm{PPL}_\text{ký tự}^{4.21}$:
- Transformer: $1.409^{4.21} \approx$ **4.2**.
- 5-gram: $1.94^{4.21} \approx$ **16.3**.

Vì vậy "perplexity 1.41" của LM ký tự và "perplexity 20" của LM mức từ **không so được** với nhau. Phải quy về cùng đơn vị (ví dụ bits-per-byte, hoặc perplexity trên cùng tokenizer).

#### 17.5 Đo trên máy bạn

**(a) Perplexity:** bảng ở Tầng 1. Transformer: 3.65 sau 100 bước, 1.54 sau 500 bước, 1.41 sau 2 000 bước (batch 64 tin nhắn). n-gram cao hơn: add-k càng lớn càng tệ (5-gram: 1.94 với $k = 0.01$, 3.45 với $k = 1$), vì add-1 chia quá nhiều xác suất cho các ngữ cảnh chưa thấy.

**(b) Decoding: đánh đổi giữa lặp lại và sai sót.** 300 câu sinh ra cho mỗi cách:

| Cách sinh | Chép nguyên văn một câu train | Câu khác nhau | Câu có "từ" không tồn tại | Ví dụ |
|---|---|---|---|---|
| $T = 0.5$ | 60% | 78% | 1.3% | chao tong dai nap tien bang the visa duoc khong |
| $T = 1.0$ | 51% | 92% | 3.7% | tôi bi mat sim |
| $T = 1.5$ | 23% | 98% | **44%** | em duong dang ky goi sv50 theo co ạ |
| $T = 1.0$, top-p 0.9 | 55% | 87% | 1.3% | alo nối máy tổng đài viên |
| $T = 1.0$, top-k 3 | 52% | 73% | 6.0% | tổng đài ơi tôi cần hủy gói cước gấp |

- **Temperature thấp:** an toàn, nhưng lặp lại dữ liệu train. Với LLM, đó là rủi ro **ghi nhớ và làm lộ** dữ liệu huấn luyện.
- **Temperature cao:** đa dạng, nhưng gần một nửa câu có "từ" bịa ra.
- **Top-p** cắt phần đuôi xác suất thấp, nơi các lỗi xuất phát. Kết quả giữ được độ đa dạng gần như $T = 1$, mà ít lỗi như $T = 0.5$.

#### 17.6 Code chuẩn

```python
def top_p_filter(logits, p):
    sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    remove = probs.cumsum(-1) - probs >= p                        # tổng TRƯỚC token này đã đủ p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)

def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    if temperature == 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    return torch.multinomial(torch.softmax(logits, -1), 1, generator=generator).squeeze(-1)

# Loss của causal LM trên batch có đệm: dịch đích một vị trí, bỏ vị trí đệm
logits = model(x[:, :-1])                                         # (B, T-1, V)
loss = F.cross_entropy(logits.transpose(1, 2), x[:, 1:], ignore_index=PAD)
```

#### 17.7 Cạm bẫy

1. **So perplexity giữa hai tokenizer khác nhau** (mục 17.4(b)).
2. **Tính perplexity mà quên vị trí đệm:** đoán `<pad>` sau `<pad>` rất dễ, nên perplexity thấp giả tạo. Dùng `ignore_index`.
3. **Lệch một vị trí** giữa đầu vào và đích: model học "chép lại token hiện tại", loss gần 0 ngay lập tức. Triệu chứng: loss giảm quá nhanh.
4. **Temperature = 0 trong code chia cho $T$:** chia cho 0. Xử lý riêng thành greedy.
5. **Greedy hoặc beam cho văn bản mở:** lặp từ, câu nhạt (Holtzman et al. 2020). Beam vẫn tốt cho dịch, tóm tắt (Tuần 4, bài 09).
6. **Đánh giá LM chỉ bằng perplexity:** perplexity thấp không có nghĩa là trả lời đúng, an toàn hay hữu ích (Tuần 6–7: đánh giá LLM).

**Sắc thái phỏng vấn:**
- *"Perplexity là gì?"* $e^{\text{CE}}$, số lựa chọn đều nhau tương đương; model đều trên $V$ token có PPL $= V$; phụ thuộc tokenizer.
- *"Temperature, top-k, top-p khác nhau thế nào? Khi nào dùng cái nào?"* Định nghĩa, thứ tự áp dụng, top-p tự co giãn; trích xuất thông tin hay gọi tool thì dùng $T$ thấp hoặc 0; viết sáng tạo thì $T$ khoảng 0.7–1 với top-p khoảng 0.9 (Tuần 6).
- *"BERT có sinh văn bản được không?"* Không tự nhiên: MLM đoán token bị che dựa vào hai phía, chứ không phải xác suất token tiếp theo.

#### 17.8 Đọc thêm

- D. Jurafsky & J. Martin, SLP3, chương 3 *N-gram Language Models*.
- Y. Bengio et al., *A Neural Probabilistic Language Model*, JMLR 2003.
- A. Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020.
- N. Carlini et al., *Extracting Training Data from Large Language Models*, USENIX Security 2021.
- A. Karpathy, *makemore* và *nanoGPT* (GitHub).

---

## 18 · Model pretrained và Hugging Face: tokenizer, feature extraction, fine-tuning, gán nhãn token

> ⏱ Tầng 1: 10' · Tầng 2: 45' · *nghiêng về Kỹ thuật*
> **Cần biết trước:** transfer learning, feature extraction và fine-tuning, discriminative LR (Tuần 4, bài 10); bài 14 (tokenizer); bài 16 (encoder); thiết bị và checkpoint (Tuần 4, phụ lục P4).

> **📖 Từ mới**
> - **Hugging Face Hub:** kho model và dữ liệu công khai; mỗi model có một **id** (`MODEL_ID`) và một **model card** mô tả cách train, giấy phép, hạn chế.
> - **`AutoTokenizer` / `AutoModel`:** lớp tự chọn đúng kiến trúc theo file `config.json` của model.
> - **`last_hidden_state`:** vector của **mọi** token ở lớp cuối, shape `(B, T, d)`.
> - **Pooling:** gộp vector các token thành một vector cho cả câu: lấy token đầu (`[CLS]`/`<s>`), hoặc trung bình có mask.
> - **Feature extraction:** đóng băng model, chỉ dùng vector của nó làm đặc trưng cho một model nhỏ.
> - **Fine-tuning:** train tiếp **toàn bộ** (hoặc một phần) model trên dữ liệu của bạn.
> - **Token classification:** gán nhãn cho **từng token** (NER, tách từ), ví dụ bằng lược đồ BIO.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Bạn thuê một nhân viên đã đọc hàng triệu trang tiếng Việt (model pretrained) để làm việc ở tổng đài:
- **Feature extraction:** nhân viên đó đọc mỗi tin nhắn rồi viết ra một bản tóm tắt 384 con số; một người mới (logistic regression) học cách phân loại dựa trên bản tóm tắt. Nhân viên cũ **không học thêm** gì.
- **Fine-tuning:** cho chính nhân viên đó học thêm một tuần về các intent của tổng đài. Đắt hơn, nhưng nhân viên điều chỉnh được cách đọc cho đúng việc.

Nhân viên đó "hiểu" rằng "gói" trong "hủy gói cước" khác "gói" trong "gói quà", vì vector của mỗi token phụ thuộc vào cả câu.

**Ví dụ có số (đo trên máy bạn).** Vector của token "gói" ở lớp cuối, trong bốn câu khác nhau (cosine):

| | hủy gói cước | đăng ký gói data | mua một gói bánh | gói quà sinh nhật |
|---|---|---|---|---|
| hủy gói cước | 1 | **0.875** | 0.868 | 0.837 |
| đăng ký gói data | | 1 | 0.869 | 0.858 |
| mua một gói bánh | | | 1 | **0.899** |

Hai câu viễn thông gần nhau hơn; hai câu "gói" đời thường gần nhau hơn. Mọi cặp đều có cosine cao (0.84–0.90): vector của các model kiểu này thường dồn vào một vùng hẹp (*anisotropy*, Ethayarajh 2019). Embedding **đầu vào** (trước các lớp Transformer) thì giống hệt nhau ở cả bốn câu.

**Ba ý cốt lõi:**
1. Dùng model pretrained = **cùng** tokenizer + **cùng** trọng số + đúng cách pooling và tiền tố mà model được train. Sai một trong ba thì vector vô nghĩa mà không báo lỗi.
2. Feature extraction rẻ (chạy model một lần, train LR vài giây); fine-tuning đắt hơn nhiều nhưng thường tốt hơn. Cả hai đều phải **đo** so với baseline TF-IDF trên cách chia đúng.
3. Model pretrained không tự động thắng: dữ liệu càng giống phân phối pretrain và bài toán càng cần "hiểu nghĩa" thì nó càng có lợi.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 18.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **BERT / RoBERTa / XLM-R** | Encoder pretrain bằng MLM; RoBERTa train lâu hơn, bỏ NSP; XLM-R train trên 100 ngôn ngữ (có tiếng Việt), tokenizer SentencePiece 250k | Devlin et al. 2019; Liu et al. 2019; Conneau et al., ACL 2020 |
| **PhoBERT** | RoBERTa cho tiếng Việt, train trên văn bản **đã tách từ** (`đăng_ký`) | Nguyen & Nguyen, Findings of EMNLP 2020 |
| **Sentence embedding model** | Encoder được fine-tune thêm bằng contrastive learning để cosine giữa các câu có nghĩa | Reimers & Gurevych, *Sentence-BERT*, EMNLP 2019; Wang et al., *E5*, 2022 |
| **Pooling** | `[CLS]` (vector token đầu) hoặc mean-pooling có mask | Reimers & Gurevych 2019 |
| **Feature extraction / fine-tuning** | Đóng băng model / train tiếp cả model | Howard & Ruder, *ULMFiT*, ACL 2018; Tuần 4, bài 10 |
| **Token classification, BIO** | B = bắt đầu thực thể, I = bên trong, O = ngoài. Chỉ gán nhãn cho **sub-token đầu** của mỗi từ, các sub-token còn lại nhận −100 (bỏ qua trong loss) | Ramshaw & Marcus 1995; Hugging Face docs, *Token classification* |
| **Entity-level F1** | Một thực thể đúng khi **cả biên lẫn loại** đều đúng | Tjong Kim Sang & De Meulder, CoNLL 2003; thư viện `seqeval` |
| **`pipeline`** | API một dòng: `pipeline("text-classification", model=MODEL_ID)` | Hugging Face docs |
| **`datasets`** | Đọc dữ liệu theo kiểu Apache Arrow (ánh xạ bộ nhớ), `map` theo batch, cache | Lhoest et al., EMNLP 2021 (demo) |

#### 18.2 🐢 Đi chậm từng bước: từ chuỗi tới vector câu

Hai câu "hủy gói GX30" và "mạng yếu quá", qua tokenizer của model (họ E5 cần tiền tố `query: `):

*Bước 1 · Tokenizer.* `tok([...], padding=True, return_tensors="pt")` trả về:
- `input_ids` shape `(2, 10)`. Câu 1 là `<s> ▁que ry : ▁hủy ▁gói ▁G X 30 </s>` (10 token). Câu 2 có 8 token, được đệm thêm 2 id `1` (`<pad>`).
- `attention_mask` shape `(2, 10)`: câu 1 toàn `1`; câu 2 là `[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]`.

*Bước 2 · Model.* `model(**batch).last_hidden_state` có shape `(2, 10, 384)`, một vector 384 chiều cho mỗi token. (`pooler_output` shape `(2, 384)` là vector `<s>` đi qua một lớp Linear + tanh; model kiểu E5 **không** được train để dùng nó.)

*Bước 3 · Mean-pooling có mask.*
$$e = \frac{\sum_t m_t h_t}{\sum_t m_t},$$
với $m_t$ là `attention_mask`. Nếu quên mask, vector của câu 2 bị pha với 2 vector đệm.

*Bước 4 · Chuẩn hóa L2.* Sau đó tích vô hướng chính là cosine, dùng được cho tìm kiếm (Tuần 8).

*Bước 5 · Dùng vector.* Hoặc đưa vào logistic regression (feature extraction), hoặc nối một lớp Linear lên trên rồi train **cả** model (fine-tuning).

#### 18.3 Ví dụ tính tay: gán nhãn BIO cho sub-token

Câu "hủy gói GX30 cho số 0912345678", hai loại thực thể là GOI và SDT. Tokenizer tách "GX30" thành `▁G` `X` `30`.

| Từ | hủy | gói | GX30 | | | cho | số | 0912345678 |
|---|---|---|---|---|---|---|---|---|
| Sub-token | `▁hủy` | `▁gói` | `▁G` | `X` | `30` | `▁cho` | `▁số` | (vài sub-token) |
| Nhãn theo từ | O | O | B-GOI | | | O | O | B-SDT |
| Nhãn đưa vào loss | O | O | B-GOI | −100 | −100 | O | O | B-SDT, −100, … |

Lúc dự đoán, lấy nhãn của **sub-token đầu** làm nhãn của từ. Thực thể "GX30" đúng khi model dự đoán B-GOI cho `▁G`. Nếu model dự đoán B-SDT (sai loại) thì entity-level F1 tính là **một FP và một FN**, dù biên đúng.

#### 18.4 Đo trên máy bạn: model pretrained so với TF-IDF

Cùng 5 split theo template như project; mọi model được train trên cùng phần `fit` (64% dữ liệu), chọn ngưỡng trên `val`, đo trên `test`. "e5 + LR" là mean-pooling + chuẩn hóa L2 rồi logistic regression ($C = 10$).

| Model | Ngẫu nhiên: F1 | **Theo template: F1** | Chênh với TF-IDF ký tự (split thắng) |
|---|---|---|---|
| TF-IDF âm tiết 1–2 gram | 0.981 | **0.714** | +0.038 (5/5) |
| TF-IDF ký tự 2–5 gram | 0.983 | 0.676 | — |
| e5 + LR, văn bản gốc | 0.946 | 0.676 | −0.000 (4/5) |
| e5 + LR, văn bản đã `normalize` | 0.962 | 0.709 | +0.033 (4/5) |
| Encoder tự train (project) | 0.980 | 0.521 | −0.155 (0/5) |
| e5 **fine-tune** toàn bộ (4 epoch, lr 3e-5 cho encoder, 1e-3 cho head, chọn epoch theo val F1) | (không đo) | 0.711 | +0.035 (3/5) |
| **Ensemble:** trung bình xác suất của TF-IDF âm tiết và e5 + LR | (không đo) | **0.760** | so với TF-IDF âm tiết: **+0.046 (5/5)** |

**Đọc bảng:**
- **Feature extraction không thắng TF-IDF âm tiết** trên chia theo template (0.709 so với 0.714), và **thua** trên chia ngẫu nhiên (0.962 so với 0.981). Vector chung của model "hiểu nghĩa" chung chung, nhưng không được train để phân biệt "hủy gói" với "đăng ký gói": như bài 13, hai câu đó rất giống nhau về mặt ngữ cảnh.
- **Chuẩn hóa vẫn có ích** với model pretrained (0.676 → 0.709): teencode và không dấu là văn bản "lạ" so với dữ liệu pretrain (bài 14: văn bản không dấu bị cắt vụn hơn).
- **Encoder tự train thua xa**: không có tri thức ngôn ngữ nào, nên khái quát kém nhất.
- **Fine-tune cũng chỉ ngang TF-IDF âm tiết:** −0.003, thắng 2/5 split. Nhưng nó dao động **ít hơn** giữa các split (std 0.074 so với 0.097): thắng đậm ở split khó nhất (+0.12), thua ở các split dễ hơn. Với khoảng 110 câu mẫu trong phần train, model 118M tham số có quá ít "cách diễn đạt" để học. Mỗi split mất khoảng 11 phút trên CPU, so với dưới 1 giây của TF-IDF.
- **Ensemble thắng rõ ràng (5/5 split).** Model pretrained mạnh ở tin **có dấu** (accuracy 0.78 so với 0.73 của TF-IDF) nhưng yếu ở tin **không dấu** (0.58 so với 0.73), vì tokenizer cắt vụn văn bản không dấu (bài 14). TF-IDF thì ổn định ở cả hai. Hai model sai ở **những chỗ khác nhau**, nên trung bình của chúng tốt hơn cả hai.
- **Độ trễ** (một tin nhắn, CPU 4 thread, cắm sạc): TF-IDF khoảng 0.5 ms (p95 1 ms); encoder tự viết 1.4 ms; encoder pretrained **17 ms** (p95 23 ms). Với 80 tin/giây thì cả ba đều chạy được trên CPU, nhưng chi phí chênh nhau khoảng 30 lần.

#### 18.5 Code chuẩn

```python
import torch
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)                      # lần đầu tải về cache, sau đó offline
model = AutoModel.from_pretrained(MODEL_ID).eval()

@torch.no_grad()
def embed(texts, prefix="query: ", batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        b = tok([prefix + t for t in texts[i:i + batch_size]], padding=True, truncation=True,
                max_length=64, return_tensors="pt")
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1))
    return torch.cat(out).numpy()

# Fine-tune: một Linear trên mean-pooling, discriminative LR (xem intent/pretrained.py → finetune)
opt = torch.optim.AdamW([{"params": model.parameters(), "lr": 3e-5},
                         {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
```

Cách viết ngắn hơn dùng thư viện: `AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=9)` cùng `Trainer`. Code ở đây viết tay để thấy rõ từng bước.

**Lý do thiết kế:**
- **Ghim phiên bản model:** `from_pretrained(MODEL_ID, revision="<commit hash>")`. Model trên Hub có thể được cập nhật, và kết quả sẽ đổi theo.
- **Tiền tố, pooling, độ dài tối đa** lấy đúng theo model card.
- **`trust_remote_code=False`** (mặc định): không chạy code Python lạ đi kèm model.

#### 18.6 Cạm bẫy

1. **Dùng tokenizer của model khác**, hoặc bỏ tiền tố bắt buộc (`query: `/`passage: ` với E5): vector vẫn ra đều đặn, nhưng chất lượng giảm mà không có lỗi nào.
2. **Mean-pooling không mask:** câu ngắn bị pha với vector đệm, và vector của một câu phụ thuộc vào câu dài nhất trong batch (Tuần 4, phụ lục P2).
3. **Dùng `pooler_output` của model không được train cho nó.**
4. **Fine-tune với learning rate kiểu train từ đầu** (1e-3) cho toàn bộ model: phá tri thức pretrain (*catastrophic forgetting*). Dùng khoảng 1e-5 tới 5e-5 cho encoder, lr lớn hơn cho head.
5. **PhoBERT cần văn bản đã tách từ**; đưa câu chưa tách từ vào thì vẫn chạy, nhưng kém hơn đáng kể. Đọc model card.
6. **Bỏ qua giấy phép và model card:** có model chỉ cho phép nghiên cứu; có model được train trên dữ liệu không phù hợp với bài toán của bạn.
7. **NER: gán nhãn cho mọi sub-token rồi đếm như token:** F1 bị thổi phồng. Dùng entity-level F1.

**Sắc thái phỏng vấn:**
- *"Bạn có 5 000 câu có nhãn, nên fine-tune BERT hay dùng TF-IDF?"* Bắt đầu bằng TF-IDF làm baseline; thử feature extraction (rẻ) rồi fine-tune; so **ghép cặp** trên cách chia đúng (theo template hoặc theo thời gian); cân nhắc độ trễ và chi phí phục vụ. Nêu kết quả tuần này: model pretrained **không tự động** thắng.
- *"[CLS] hay mean-pooling?"* Tùy cách model được train. BERT gốc không train `[CLS]` cho độ giống câu; sentence-embedding model thường dùng mean-pooling. Đọc model card.

#### 18.7 Đọc thêm

- Hugging Face LLM Course, chương 2 *Using Transformers*, chương 3 *Fine-tuning a pretrained model*, chương 7 *Token classification*.
- A. Conneau et al., *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-R), ACL 2020.
- D. Q. Nguyen & A. T. Nguyen, *PhoBERT: Pre-trained language models for Vietnamese*, Findings of EMNLP 2020.
- L. Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5), arXiv 2022.
- K. Ethayarajh, *How Contextual are Contextualized Word Representations?*, EMNLP 2019.

---

## 19 · Metric NLP: P/R/F1, EM, BLEU, ROUGE, BERTScore, đánh giá bởi người; các thư viện

> ⏱ Tầng 1: 10' · Tầng 2: 40'
> **Cần biết trước:** precision, recall, F1, confusion matrix (Tuần 3, bài 04); n-gram (bài 12); cosine, embedding theo ngữ cảnh (bài 18).

> **📖 Từ mới**
> - **Macro / micro / weighted average:** trung bình F1 của các lớp **đều nhau** / gộp TP, FP, FN của **mọi lớp** rồi mới tính / trung bình có trọng số theo số mẫu của mỗi lớp.
> - **Reference (bản tham chiếu):** câu "đáp án" do người viết, để so với câu model sinh ra.
> - **Modified precision:** precision n-gram trong đó mỗi n-gram chỉ được tính tối đa bằng số lần nó xuất hiện trong reference.
> - **Brevity penalty:** khoản phạt khi câu sinh ra **ngắn hơn** reference.
> - **LCS:** dãy con chung dài nhất (không cần liên tiếp).
> - **Inter-annotator agreement:** mức đồng thuận giữa hai người gán nhãn.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Chấm bài dịch:
- **BLEU** giống giáo viên chấm bằng cách **đếm cụm từ trùng** với đáp án mẫu. Nhanh, khách quan, nhưng viết "ngừng" thay vì "hủy" là mất điểm, dù đúng nghĩa.
- **BERTScore** giống giáo viên hiểu **nghĩa** từng từ: "ngừng" gần "hủy" nên vẫn được điểm.
- **Đánh giá bởi người** giống hội đồng chấm: chính xác nhất, đắt nhất, và phải kiểm tra xem các giám khảo có **đồng ý với nhau** không.

**Ví dụ có số.** Reference: "tôi muốn **hủy** gói". Câu sinh ra: "tôi muốn **ngừng** gói".

| Metric | Điểm |
|---|---|
| BLEU-4 (không làm mượt) | **0** (không có 3-gram nào trùng) |
| BLEU-2 | 0.5 |
| ROUGE-L F | 0.75 |
| BERTScore F (với embedding đồ chơi, mục 19.3) | **0.995** |

Cùng một câu đúng nghĩa: điểm chạy từ 0 tới 0.995 tùy metric.

**Ba ý cốt lõi:**
1. Bài toán **phân loại** dùng P/R/F1; chọn cách lấy trung bình theo câu hỏi nghiệp vụ (macro: mọi intent quan trọng như nhau).
2. Bài toán **sinh** dùng metric so với reference (BLEU, ROUGE: so khớp bề mặt; BERTScore: so khớp ngữ nghĩa), nhưng metric tự động chỉ là **xấp xỉ** của đánh giá bởi người.
3. Đánh giá bởi người cũng có nhiễu: đo độ đồng thuận (kappa). Đó là **trần** của mọi metric.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 19.1 Thuật ngữ chuẩn

| Metric | Định nghĩa | Dùng cho | Nguồn |
|---|---|---|---|
| **Macro-F1** | $\frac1C\sum_c F1_c$ | Phân loại, mọi lớp quan trọng như nhau | Tuần 3, bài 04 |
| **Micro-F1** | F1 trên tổng TP, FP, FN. Với phân loại đơn nhãn, micro-F1 = accuracy | | Manning et al., IIR §13.6 |
| **Weighted-F1** | $\sum_c \frac{n_c}{n} F1_c$ | Báo cáo theo tỉ lệ thực tế | scikit-learn docs |
| **Exact Match (EM)** | 1 nếu câu trả lời trùng đáp án sau khi chuẩn hóa (chữ thường, bỏ dấu câu, mạo từ) | QA trích xuất | Rajpurkar et al., *SQuAD*, EMNLP 2016 |
| **Token-F1 (SQuAD)** | F1 trên **túi token** của câu trả lời và đáp án | QA | như trên |
| **BLEU** | $\mathrm{BP}\cdot\exp\!\big(\frac14\sum_{n=1}^4\log p_n\big)$, $p_n$ là modified precision n-gram; $\mathrm{BP} = \min(1, e^{1 - r/c})$ | Dịch máy | Papineni et al., ACL 2002; Post, *sacreBLEU*, WMT 2018 |
| **ROUGE-N / ROUGE-L** | Recall (và F) của n-gram / của LCS so với reference | Tóm tắt | Lin, ACL Workshop 2004 |
| **BERTScore** | Ghép mỗi token của câu sinh với token **gần nhất** (cosine của embedding theo ngữ cảnh) trong reference và ngược lại; lấy trung bình được P, R, F | Sinh văn bản, so khớp ngữ nghĩa | Zhang et al., ICLR 2020 |
| **Perplexity** | Bài 17 | LM | |
| **Cohen's kappa** | $\kappa = \dfrac{p_o - p_e}{1 - p_e}$: độ đồng thuận đã trừ phần trùng do ngẫu nhiên | Đánh giá bởi người, chất lượng nhãn | Cohen, *Educational and Psychological Measurement* 1960 |

#### 19.2 🐢 Đi chậm từng bước: BLEU cho một câu

Reference: "tôi muốn hủy gói cước này" (6 từ). Câu sinh: "tôi muốn hủy gói này" (5 từ).

*Bước 1 · Modified precision từng bậc n-gram.*

| $n$ | n-gram của câu sinh | Trùng với ref | $p_n$ |
|---|---|---|---|
| 1 | tôi, muốn, hủy, gói, này | cả 5 | 5/5 = 1 |
| 2 | tôi muốn, muốn hủy, hủy gói, gói này | 3 (không có "gói này") | 3/4 |
| 3 | tôi muốn hủy, muốn hủy gói, hủy gói này | 2 | 2/3 |
| 4 | tôi muốn hủy gói, muốn hủy gói này | 1 | 1/2 |

*Bước 2 · Trung bình nhân:* $(1 \cdot 0.75 \cdot 0.667 \cdot 0.5)^{1/4} = 0.25^{1/4} = 0.707$.

*Bước 3 · Brevity penalty:* $c = 5 < r = 6$, nên $\mathrm{BP} = e^{1 - 6/5} = 0.819$.

*Bước 4:* $\mathrm{BLEU} = 0.819 \times 0.707 =$ **0.579**.

*Vì sao "modified"?* Câu sinh "hủy hủy hủy hủy" có precision 1-gram thường là 4/4, vì "hủy" có trong ref. Nhưng ref chỉ có **một** "hủy", nên số đếm bị **chặn** ở 1, thành $p_1 = 1/4$. Không có bước chặn này thì lặp một từ đúng cũng được điểm tuyệt đối.

*Vì sao có BP?* Precision không phạt câu **thiếu** chữ: câu sinh chỉ có "tôi" có $p_1 = 1$. BP bù lại phần recall mà BLEU không đo.

*Corpus BLEU không phải trung bình của sentence BLEU.* Hai câu: một câu khớp hoàn toàn (BLEU-2 = 1), một câu "x y" so với ref "x z" (BLEU-2 = 0, vì không có bigram trùng). Trung bình sentence BLEU = 0.5. Corpus BLEU **cộng các số đếm trước**: $p_1 = 5/6$, $p_2 = 3/4$, nên BLEU-2 $= \sqrt{0.625} =$ **0.791**. Các bài báo dịch máy báo corpus BLEU (sacreBLEU).

#### 19.3 Ví dụ tính tay

**(a) Macro, micro, weighted.** 10 mẫu, 3 lớp. Thật: `a` ×6, `b` ×3, `c` ×1. Dự đoán: 6 mẫu `a` → 5 `a`, 1 `b`; 3 mẫu `b` → 2 `b`, 1 `a`; mẫu `c` → `a`.

| Lớp | Precision | Recall | F1 | Số mẫu |
|---|---|---|---|---|
| a | 5/7 = 0.714 | 5/6 = 0.833 | 0.769 | 6 |
| b | 2/3 = 0.667 | 2/3 = 0.667 | 0.667 | 3 |
| c | 0 (không đoán lần nào) | 0 | 0 | 1 |

- Macro-F1 $= (0.769 + 0.667 + 0)/3 =$ **0.479**.
- Micro-F1 $=$ accuracy $= 7/10 =$ **0.70**.
- Weighted-F1 $= 0.6 \cdot 0.769 + 0.3 \cdot 0.667 + 0.1 \cdot 0 =$ **0.662**.

Lớp hiếm `c` bị bỏ qua hoàn toàn, nhưng chỉ macro "nhìn thấy" điều đó.

**(b) ROUGE.** Ref "tôi muốn hủy gói cước này", câu sinh "hủy gói cước giúp tôi":
- ROUGE-1: trùng 4 từ (hủy, gói, cước, tôi), nên $P = 4/5$, $R = 4/6$, $F = 0.727$.
- ROUGE-2: trùng "hủy gói", "gói cước", nên $R = 2/5$.
- ROUGE-L: LCS là "hủy gói cước" (dài 3; "tôi" ở hai câu nằm ở hai đầu ngược nhau), nên $P = 3/5$, $R = 3/6$, $F = 0.545$.

**(c) BERTScore với embedding đồ chơi.** Các vector: tôi $(1, 0, 0)$, muốn $(0, 1, 0)$, hủy $(0, 0, 1)$, ngừng $(0, 0.2, 0.98)$, gói $(0.6, 0, 0.8)$. Câu sinh "tôi muốn ngừng gói", ref "tôi muốn hủy gói".
- Mỗi token câu sinh ghép với token ref có cosine lớn nhất: tôi–tôi 1; muốn–muốn 1; ngừng–hủy **0.98**; gói–gói 1. Precision $= 3.98/4 = 0.995$.
- Recall (chiều ngược lại) cũng bằng 0.995, nên $F =$ **0.995**.
- BERTScore thật dùng embedding theo ngữ cảnh của một model pretrained (và thường điều chỉnh theo một mức nền).

**(d) Cohen's kappa.** Hai người gán 8 tin nhắn, mỗi người gán `x` 5 lần, `y` 3 lần, và trùng nhau 6/8 lần.
- $p_o = 0.75$; $p_e = (5 \cdot 5 + 3 \cdot 3)/64 = 0.531$.
- $\kappa = (0.75 - 0.531)/(1 - 0.531) =$ **0.467**: "trung bình" theo thang Landis & Koch (1977), dù hai người trùng 75%.

#### 19.4 Đo trên máy bạn

**(a) Từng intent** (TF-IDF âm tiết, gộp dự đoán của 5 split theo template):

| Intent | P | R | F1 |
|---|---|---|---|
| đổi SIM | 0.906 | 0.984 | **0.944** |
| nạp tiền | 0.876 | 0.838 | 0.856 |
| gặp tổng đài viên | 0.775 | 0.804 | 0.789 |
| tra cước | 0.758 | 0.758 | 0.758 |
| hủy gói | 0.677 | 0.711 | 0.693 |
| lỗi mạng | 0.595 | 0.758 | 0.667 |
| **khác (ngoài phạm vi)** | 0.858 | **0.538** | 0.662 |
| đăng ký gói | 0.591 | 0.694 | 0.639 |
| khiếu nại | 0.717 | 0.561 | **0.629** |

Macro-F1 0.737; micro-F1 = accuracy = 0.744; weighted-F1 0.743. Các cặp nhầm nhiều nhất:

| Thật → dự đoán | Số lần |
|---|---|
| hủy gói → đăng ký gói | 148 |
| **khác → lỗi mạng** | 122 |
| đăng ký gói → lỗi mạng | 94 |
| khiếu nại → gặp tổng đài viên | 76 |
| khiếu nại → tra cước | 75 |

- **Hủy ↔ đăng ký** nhầm nhau nhiều nhất, đúng như bài 13 dự báo (hai từ cùng đứng trước "gói").
- **Tin nhắn ngoài phạm vi** chỉ được nhận ra 54%; phần lớn còn lại bị gán thành "lỗi mạng". Trong vận hành, đây là lỗi tốn kém: tạo phiếu kỹ thuật cho một câu hỏi về giá vàng. Đó là lý do project có **ngưỡng từ chối**.

**(b) Trần do nhãn nhập nhằng.** Sinh nhãn lần thứ hai, độc lập, cho cùng 6 000 tin nhắn (6/141 template có nhãn ngẫu nhiên): hai lần trùng 98.2%, $\kappa = 0.980$. Accuracy tối đa đạt được là **0.986** (chọn nhãn phổ biến nhất của mỗi template nhập nhằng). Chia ngẫu nhiên đạt 0.98, tức đã chạm trần.

#### 19.5 Các thư viện (roadmap 5.9)

| Thư viện | Dùng để | Trên máy bạn | Tuần này dùng thay bằng |
|---|---|---|---|
| `NLTK` | NLP cổ điển: tokenizer, stemmer (Porter), corpus, BLEU | chưa cài | `textprep.py`, `metrics.py` tự viết |
| `spaCy` | NLP production: pipeline tokenizer, POS, NER, dependency parsing; model nhỏ, nhanh trên CPU | chưa cài | (NER: bài 18, mục 18.3) |
| `transformers` | Model pretrained: `AutoTokenizer`, `AutoModel`, `pipeline`, `Trainer` | **có** (5.18) | bài 18 |
| `datasets` | Tải và xử lý dữ liệu (Arrow, `map`, cache) | **có** (5.0) | DataFrame trong `data.py` |
| `tokenizers` | Tokenizer nhanh (Rust): train BPE/WordPiece/Unigram | **có** (0.23) | bài 14 |
| `sentence-transformers` | Embedding câu, tìm kiếm ngữ nghĩa | chưa cài | Tuần 8 (RAG) |
| `tiktoken` | Tokenizer BPE của một nhà cung cấp LLM; đếm token để ước lượng chi phí | chưa cài | Tuần 6 |
| `evaluate` | Metric chuẩn hóa (BLEU qua sacreBLEU, ROUGE, BERTScore, seqeval) | chưa cài | `metrics.py` tự viết |

**Khi dùng thư viện để báo cáo**, hãy dùng bản chuẩn: BLEU qua sacreBLEU (in kèm "signature" của cách token hóa); ROUGE qua `rouge-score`. Bản tự viết dễ lệch ở các chi tiết (tokenize, chữ hoa, làm mượt), nên chỉ dùng để học hoặc kiểm tra.

#### 19.6 Code chuẩn

```python
def bleu(candidate, references, max_n=4, smooth=False):
    clipped, totals = [], []
    for n in range(1, max_n + 1):
        c = Counter(tuple(candidate[i:i + n]) for i in range(len(candidate) - n + 1))
        max_ref = Counter()
        for ref in references:
            for g, k in Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)).items():
                max_ref[g] = max(max_ref[g], k)                   # chặn theo ref có nhiều nhất
        clipped.append(sum(min(k, max_ref[g]) for g, k in c.items()))
        totals.append(max(len(candidate) - n + 1, 0))
    r = min((abs(len(ref) - len(candidate)), len(ref)) for ref in references)[1]   # ref gần độ dài nhất
    ...                                                           # log trung bình, BP; xem intent/metrics.py
```

#### 19.7 Cạm bẫy

1. **Báo micro-F1 cho bài toán đơn nhãn rồi gọi là "F1":** nó chính là accuracy, và che mất lớp hiếm.
2. **So BLEU giữa hai bài báo dùng cách token hóa khác nhau:** BLEU lệch vài điểm chỉ vì tokenizer. Dùng sacreBLEU và in signature.
3. **Lấy trung bình sentence BLEU và gọi là corpus BLEU** (mục 19.2).
4. **Dùng BLEU/ROUGE để đánh giá chatbot hay LLM trả lời mở:** có nhiều câu trả lời đúng mà không trùng từ nào với reference. Dùng đánh giá bởi người, hoặc LLM-as-judge có kiểm định (Tuần 6–7).
5. **Thu thập đánh giá bởi người mà không đo độ đồng thuận:** nếu $\kappa$ thấp thì hướng dẫn gán nhãn chưa rõ; metric nào cũng không vượt được mức đó.
6. **EM quá khắt khe với tiếng Việt:** "gói GX30" và "GX30" là cùng một đáp án. Dùng token-F1 kèm EM, và chuẩn hóa (bài 11).
7. **Đánh giá NER theo token:** thổi phồng; dùng entity-level F1 (`seqeval`).

**Sắc thái phỏng vấn:**
- *"Macro hay micro F1 cho router intent?"* Macro, nếu intent hiếm vẫn quan trọng (khiếu nại). Báo kèm bảng từng intent và confusion matrix; báo riêng recall của lớp ngoài phạm vi.
- *"BLEU hoạt động thế nào, hạn chế gì?"* Modified precision n-gram, trung bình nhân, BP; không hiểu đồng nghĩa; tương quan với người ở mức hệ thống tốt hơn ở mức câu.
- *"Đánh giá một hệ thống sinh văn bản thế nào?"* Metric tự động để lặp nhanh; đánh giá bởi người (hoặc so cặp A/B) có hướng dẫn, nhiều người chấm, đo $\kappa$; theo dõi tín hiệu sản phẩm (👎, chuyển sang tổng đài viên).

#### 19.8 Đọc thêm

- K. Papineni et al., *BLEU: a Method for Automatic Evaluation of Machine Translation*, ACL 2002.
- M. Post, *A Call for Clarity in Reporting BLEU Scores*, WMT 2018.
- C.-Y. Lin, *ROUGE: A Package for Automatic Evaluation of Summaries*, ACL Workshop 2004.
- T. Zhang et al., *BERTScore: Evaluating Text Generation with BERT*, ICLR 2020.
- P. Rajpurkar et al., *SQuAD: 100,000+ Questions for Machine Comprehension of Text*, EMNLP 2016.
- J. R. Landis & G. Koch, *The Measurement of Observer Agreement for Categorical Data*, Biometrics 1977.

---

## Bài tập Ngày 3 · `intent/metrics.py` và `intent/decoding.py` (~2h)

Lệnh test: `python -m pytest -q tests/test_metrics.py tests/test_decoding.py` (10 + 6 = 16 test).

**`intent/metrics.py`** (~75'):

| Hàm | Yêu cầu |
|---|---|
| `prf(y_true, y_pred, labels=None, average="macro")` | Trả về `(precision, recall, f1)` giống `precision_recall_fscore_support(..., zero_division=0)`. `labels=None` thì dùng mọi nhãn có trong `y_true` hoặc `y_pred`, sắp xếp. `average`: `None` (ba list theo nhãn), `"macro"`, `"micro"`, `"weighted"` (trọng số = số mẫu **thật** của nhãn) |
| `bleu(candidate, references, max_n=4, smooth=False)` | Sentence BLEU, mục 19.2: số đếm n-gram bị chặn bởi **max** số lần trong một ref; $r$ = độ dài ref **gần** $c$ nhất (hòa thì lấy ref ngắn hơn); có $p_n = 0$ thì trả 0. `smooth=True`: cộng 1 vào tử và mẫu với $n \ge 2$ |
| `corpus_bleu(candidates, references_list, max_n=4)` | Cộng số đếm bị chặn, tổng số n-gram, $c$ và $r$ của mọi câu rồi mới tính (không làm mượt) |
| `rouge_n(candidate, reference, n=1)`, `rouge_l(...)`, `lcs_len(a, b)` | Trả về dict `{"p", "r", "f"}`; trùng n-gram tính theo giao của hai `Counter`; ROUGE-L dùng LCS (quy hoạch động) |
| `normalize_answer`, `exact_match(pred, gold)`, `token_f1(pred, gold)` | Kiểu SQuAD: chữ thường, bỏ `string.punctuation`, gộp khoảng trắng; token-F1 trên giao của hai `Counter`, bằng 0 nếu không có token chung |
| `perplexity(log_probs)` | $\exp(-\text{trung bình})$ |
| `cohen_kappa(a, b)` | Công thức 19.1 |

**`intent/decoding.py`** (~45'):

| Hàm | Yêu cầu |
|---|---|
| `softmax_t(logits, temperature=1.0)` | softmax(logits / T) theo chiều cuối |
| `top_k_filter(logits, k)` | Giữ logit **≥** logit lớn thứ $k$ (hòa ở ranh giới thì giữ cả), còn lại `-inf`; chạy được theo batch |
| `top_p_filter(logits, p)` | Giữ tập nhỏ nhất có tổng xác suất ≥ p (token tốt nhất luôn được giữ); giá trị logit được giữ không đổi |
| `sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None)` | `temperature == 0` thì argmax; ngược lại chia T → top-k → top-p → `torch.multinomial`. Đầu vào `(..., V)`, đầu ra `(...)` |
| `generate(step_fn, ids, max_new_tokens, eos_id=None, **sampling)` | Lặp: `step_fn(ids)` trả logit `(V,)`; nối token mới; dừng khi gặp `eos_id` (token eos **được** nối vào) hoặc đủ số token |

**Gợi ý:**
- `test_prf_matches_sklearn` dùng dữ liệu có nhãn `d` không bao giờ được dự đoán: precision của `d` là 0 (không chia cho 0).
- `top_p_filter`: sắp giảm dần, tính tổng tích lũy **trước** mỗi token (`cumsum - probs`), bỏ token nếu tổng đó đã ≥ p, rồi `scatter` về đúng vị trí cũ.
