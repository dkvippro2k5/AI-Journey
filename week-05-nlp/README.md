# Tuần 5 · NLP và Transformer

**Mục tiêu tuần:**
- Xử lý được văn bản tiếng Việt thật (Unicode, không dấu, teencode).
- Hiểu và **tự viết** các mảnh cốt lõi của NLP hiện đại: TF-IDF, BPE, multi-head attention, positional encoding, block Transformer, KV-cache, các cách decoding, các metric.
- Dùng được model pretrained qua Hugging Face.
- Quan trọng nhất: **đánh giá trung thực** một hệ thống NLP.

**Sản phẩm cuối tuần:** `project/intent/`, một router intent cho tổng đài nhà mạng (8 intent + ngoài phạm vi), có ngưỡng chuyển cho tổng đài viên. Router được so sánh **ghép cặp** giữa TF-IDF, Transformer tự viết và (tùy chọn) model pretrained, trên cách diễn đạt **chưa từng thấy**.
**Loại tài liệu: Lý thuyết (bản nhẹ).** Có công thức và ví dụ tính tay đầy đủ; chứng minh chỉ phác thảo vài dòng. Bài 18 nghiêng về Kỹ thuật.
**Ứng với roadmap:** Phase 5 (5.1–5.9), cộng mục *Using HuggingFace pretrained models* (4.7) dời từ Tuần 4 sang.

## Lịch 5 ngày (khoảng 5 giờ mỗi ngày)

| Ngày | Đọc (~2h45–3h) | Quiz (45') | Làm (~2h) |
|---|---|---|---|
| 1 | [ngay-1.md](ngay-1.md): **11** Tiền xử lý tiếng Việt · **12** BoW, n-gram, TF-IDF · **13** Word embeddings (word2vec, GloVe, FastText) | [ngay-1.quiz.md](ngay-1.quiz.md) | `intent/textprep.py` |
| 2 | [ngay-2.md](ngay-2.md): **14** Subword tokenization · **15** Self-attention, multi-head, positional encoding · **16** Block Transformer, kiến trúc, đếm tham số, KV-cache | [ngay-2.quiz.md](ngay-2.quiz.md) | `intent/bpe.py`, `intent/attention.py` |
| 3 | [ngay-3.md](ngay-3.md): **17** Mô hình ngôn ngữ, decoding · **18** Model pretrained, Hugging Face · **19** Metric NLP, thư viện | [ngay-3.quiz.md](ngay-3.quiz.md) | `intent/metrics.py`, `intent/decoding.py` |
| 4 | — | — | **Project:** [project.md](project.md), viết `intent/encoder.py` và `intent/train.py` |
| 5 | — | — | `REPORT.md` (2h) · **Final 180'**: [final-exam.md](final-exam.md) |

Tuần này **nặng** hơn Tuần 4. Nếu thiếu thời gian, đọc lướt các mục "Đọc thêm" và để dành phần mở rộng của project.

## Chuẩn bị một lần (10 phút)

```
cd "week-05-nlp/project"
python -m venv .venv
.venv\Scripts\activate
python -m pip install numpy pandas scipy scikit-learn torch pytest ruff
python -m pip install transformers tokenizers        # chỉ cần cho bài 14 (mục 14.5) và bài 18
python -m pytest -q        # lúc đầu sẽ báo lỗi vì chưa có code của bạn — đó là điều bình thường
```

- **Không cần GPU.** Mọi bài tập và test chạy trên CPU.
- **Cắm sạc khi chạy.** Khi chạy pin, laptop này hạ xung nhịp (Tuần 4, README), và các phép đo thời gian sẽ sai lệch.
- **Không cần tải dữ liệu.** Tin nhắn là dữ liệu tổng hợp (`intent/data.py`).
- **Model pretrained (tùy chọn):** bài 18 và phần mở rộng của project dùng một encoder đa ngôn ngữ nhỏ (khoảng 118M tham số, 470 MB, tên trong `intent/pretrained.py`). Model này **đã được tải** về cache Hugging Face trên máy bạn. Mọi test đều chạy offline, không cần model này.
- Mỗi ngày chỉ chạy test của ngày đó, ví dụ `python -m pytest -q tests/test_textprep.py`.

## Cách học mỗi bài

Vẫn **hai tầng** như các tuần trước. Các cơ chế khó hình dung có khối **"🐢 Đi chậm từng bước"** ngay trước công thức:
- một tin nhắn đi qua `normalize` (11.2);
- skip-gram với negative sampling (13.2);
- học BPE bằng tay (14.2);
- shape trong multi-head attention (15.2);
- vị trí: sinusoidal và RoPE (15.3);
- KV-cache (16.3);
- câu thành bài toán huấn luyện LM (17.2);
- chọn token từ logit (17.3);
- từ chuỗi tới vector câu (18.2);
- BLEU (19.2).

Hãy đọc khối đó trước, rồi mới đọc công thức.

**Tuần này có nhiều "bất ngờ" được đo thật:**
- Cùng một model, chia ngẫu nhiên cho F1 **0.98**, chia theo câu mẫu chỉ còn **0.71**. Cách đánh giá quan trọng hơn cách chọn đặc trưng (bài 12).
- Bỏ dấu, bỏ stopword "cho sạch" lại làm F1 **giảm** (bài 11).
- Word2vec tự học được "đk" = "đc" = "được" mà không cần bảng teencode, nhưng lại đặt "hủy" sát "đăng ký" (bài 13).
- Tokenizer của model pretrained tách câu không dấu ra nhiều hơn 16% token (bài 14).
- Transformer tự viết **thua** TF-IDF (0.52 so với 0.71). Model pretrained 118M tham số, dù chỉ trích embedding (0.709) hay fine-tune cả model (0.711), cũng chỉ **ngang** TF-IDF. Nhưng ensemble của hai loại model thì đạt 0.760, thắng 5/5 split (bài 18, project).
- Temperature 1.5 làm 44% câu sinh ra chứa từ bịa; top-p 0.9 giữ được độ đa dạng mà chỉ 1.3% câu có lỗi (bài 17).
- "tôi muốn ngừng gói" so với "tôi muốn hủy gói" cho BLEU-4 bằng **0** (bài 19).

**Quy ước:** mọi con số "đo trên máy bạn" chạy bằng PyTorch 2.14 (CPU), scikit-learn 1.9, `transformers` 5.18, `tokenizers` 0.23. Kết quả mạng neural có thể lệch nhẹ ở chữ số thứ ba khi đổi số thread hoặc phiên bản.

## Đối chiếu với roadmap

| Roadmap | Bài |
|---|---|
| 5.1 Text preprocessing (tokenization, lowercasing, stopword, stemming/lemmatization, sentence segmentation, URL/email/hashtag, Unicode) | 11 |
| 5.2 BoW, TF-IDF, n-gram, one-hot, sparse/dense | 12 |
| 5.3 Word2vec (CBOW, skip-gram), GloVe, FastText, cosine, analogy, embedding tĩnh và theo ngữ cảnh | 13, 18 |
| 5.4 BPE, WordPiece, SentencePiece, special tokens, token id, kích thước vocab | 14 |
| 5.5 Transformer: vì sao thay RNN, self-attention, Q/K/V, multi-head, positional encoding, FFN, LayerNorm, residual, encoder/decoder/encoder–decoder, causal mask | 15, 16 |
| 5.6 Language modeling: next token, autoregressive, MLM, perplexity, temperature, top-k, top-p, greedy/sampling/beam | 17 (beam: Tuần 4, bài 09) |
| 5.7 Key pretrained models | 16 (BERT, GPT-2, T5), 18 (XLM-R, PhoBERT). Bảng các LLM cụ thể **dời sang Tuần 6** (LLM API) |
| 5.8 Metric: P/R/F1, BLEU, ROUGE, perplexity, BERTScore, human evaluation, EM | 19 |
| 5.9 Thư viện: NLTK, spaCy, transformers, datasets, sentence-transformers, tiktoken, evaluate | 19 (bảng 19.5); `sentence-transformers` dùng thật ở **Tuần 8**, `tiktoken` ở **Tuần 6** |
| 4.7 Using HuggingFace pretrained models (dời từ Tuần 4) | 18 |
| Phase 5 projects: NER pipeline · Semantic search · Fine-tune BERT multi-label + FastAPI | NER: bài 18 (BIO, mục 18.3), không có project riêng · Semantic search: **Tuần 8** · Fine-tune + API + Docker: **Tuần 11, 13** |

## Kiểm tra

| Loại | Khi nào | Thời lượng | Đạt khi |
|---|---|---|---|
| Quiz ngày | Ngay sau khi đọc xong file của ngày đó | 45' (15' quick test + 30' tự luận) | ≥ 21/30 |
| Test bài tập | Sau khi làm bài tập | — | Pass hết |
| Final tuần | Ngày 5 | 180' | ≥ 70/100 |
| [Ôn phỏng vấn](on-phong-van.md) | Sau final; ôn lại sau 1 tuần và sau 1 tháng | ~45' (16 câu, nói to, mỗi câu 1') | ≥ 12/16 |

Quiz và final **không có câu chứng minh**. Phần tự luận tập trung vào: tính tay (TF-IDF, BPE, attention có mask, đếm tham số, KV-cache, perplexity, BLEU/ROUGE, macro/micro F1), phân tích kết quả ghép cặp, và thiết kế. Phần C của final viết WordPiece, causal multi-head attention có KV-cache, bigram LM và MLM masking.

## Nhận từ tuần trước, chuyển sang tuần sau

| Từ | Nội dung | Dùng ở bài |
|---|---|---|
| Tuần 1, phụ lục P1 | NFC, UTF-8 | 11, 14 |
| Tuần 3, bài 01, 03, 04 | Rò rỉ, CV ghép cặp, macro-F1, ngưỡng | 12, 19, project |
| Tuần 4, bài 09 | `dot_attention`, mask, $\sqrt d$, beam search | 15, 17 |
| Tuần 4, bài 04–06, 10 | Residual, LayerNorm, GELU, đếm tham số FFN, 16 byte/tham số, transfer learning | 16, 18 |
| Tuần 4, phụ lục P1–P2 | `ignore_index`, label smoothing, `collate_fn`, padding, mask, BiLSTM | 17, 18 |

**Dời sang tuần sau:**
- Bảng các LLM cụ thể, `tiktoken`, đếm token để tính chi phí: **Tuần 6**.
- `sentence-transformers`, semantic search, vector database: **Tuần 8**.
- Fine-tune multi-label + API: **Tuần 11, 13**.

## Tuần này nối với các tuần sau thế nào

| Kiến thức Tuần 5 | Được dùng lại ở |
|---|---|
| `normalize`, router intent có ngưỡng chuyển người | Tuần 6 (gateway: chọn model theo intent), Tuần 10 (agent: chọn tool), Tuần 13 (service) |
| Tokenizer, đếm token, special token | Tuần 6 (chi phí API, context window), Tuần 7 (prompt injection qua special token) |
| Decoding: temperature, top-p | Tuần 6 (tham số gọi LLM) |
| `MultiHeadAttention`, `TransformerBlock`, KV-cache, GQA | Tuần 11 (LoRA trên attention), Tuần 12 (vLLM, PagedAttention, bộ nhớ KV) |
| Feature extraction, mean-pooling, embedding câu | Tuần 8 (embedding cho RAG) |
| Metric: BLEU/ROUGE và hạn chế, đánh giá bởi người, kappa | Tuần 6–7 (đánh giá LLM, LLM-as-judge), Tuần 8–9 (đánh giá RAG) |
| Chia theo template / theo thời gian, so ghép cặp | Mọi tuần còn lại |
