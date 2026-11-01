# Project Tuần 5 · `intent` — router intent cho tổng đài, TF-IDF so với Transformer

> **Ngày 4:** ghép các mảnh thành pipeline huấn luyện và đánh giá (~5h).
> **Ngày 5:** chạy, phân tích, viết `REPORT.md` (2h), sau đó làm final (3h).
> **Câu hỏi của tuần:** với vài nghìn tin nhắn có nhãn, một Transformer tự viết, hoặc một model pretrained, có thắng được TF-IDF + logistic regression không, khi người dùng diễn đạt theo **cách chưa từng thấy**?
> **Vị trí trong hệ thống chung:** cửa vào của nền tảng RAG + Agent. Mỗi tin nhắn được gán intent trước khi tới gateway LLM (Tuần 6) hoặc agent (Tuần 10); tin có độ tin cậy thấp thì chuyển cho tổng đài viên. `normalize`, BPE, `MultiHeadAttention`, `TransformerBlock` và các metric sẽ được dùng lại ở Tuần 8 (RAG), Tuần 11 (fine-tuning) và Tuần 13 (service).

## 1. Dữ liệu

`intent/data.py` (cho sẵn) sinh tin nhắn khách hàng gửi tổng đài một nhà mạng:

| Cột | Ý nghĩa |
|---|---|
| `text` | Tin nhắn, có nhiễu: 30% bỏ dấu, 34% teencode, 10% Unicode NFD, gõ sai, emoji, lời chào |
| `intent` | Một trong 9 nhãn: `tra_cuoc`, `nap_tien`, `dang_ky_goi`, `huy_goi`, `loi_mang`, `doi_sim`, `khieu_nai`, `gap_tvv`, `khac` (ngoài phạm vi) |
| `template_id` | Câu mẫu sinh ra tin nhắn, ví dụ `huy_goi/07` (141 câu mẫu) |
| `nodiac`, `teen`, `nfd` | Cờ cho biết tin nhắn có bị loại nhiễu đó không, dùng để phân tích lỗi |

- **6 câu mẫu nhập nhằng** có nhãn ngẫu nhiên (ví dụ "nạp 50k rồi mà chưa thấy cộng tiền" là khiếu nại 75%, nạp tiền 25%). Accuracy tối đa đạt được là **0.986**.
- Tên gói ("GX30"), số điện thoại, email đều là giả định.

**Hai cách chia** (`intent/train.py`):
- `random_split`: chia ngẫu nhiên có stratify.
- `template_split`: với **mỗi** intent, giữ lại khoảng 20% số câu mẫu (ít nhất 1) cho test. Test chỉ chứa **cách diễn đạt chưa từng thấy**.

Mỗi lần chạy: test 20%; trong phần còn lại, tách tiếp 20% làm **val** theo **cùng** kiểu chia (để early stopping và ngưỡng từ chối được chọn một cách trung thực); model train trên 64% còn lại (`fit`).

## 2. Thư mục project

```
project/
├── pyproject.toml
├── intent/
│   ├── __init__.py
│   ├── data.py          # cho sẵn
│   ├── textprep.py      # Ngày 1: strip_accents, normalize, syllables, char_ngrams, word_ngrams, Tfidf
│   ├── bpe.py           # Ngày 2: train_bpe, BPE
│   ├── attention.py     # Ngày 2: causal_mask, padding_mask, sinusoidal_pe, apply_rope, MultiHeadAttention
│   ├── metrics.py       # Ngày 3: prf, bleu, corpus_bleu, rouge_n, rouge_l, exact_match, token_f1, perplexity, cohen_kappa
│   ├── decoding.py      # Ngày 3: softmax_t, top_k_filter, top_p_filter, sample_next, generate
│   ├── encoder.py       # Ngày 4: TransformerBlock, EncoderClassifier
│   ├── train.py         # Ngày 4: chia dữ liệu, baseline TF-IDF, train_encoder, ngưỡng từ chối, so sánh ghép cặp
│   └── pretrained.py    # tùy chọn (cần model đã tải): embed, finetune
├── tests/               # 42 test (cho sẵn)
└── _reference/          # lời giải tham khảo — chỉ mở khi kẹt quá 1h
```

## 3. Ngày 4 — `intent/encoder.py` và `intent/train.py`

Kiểm tra bằng `tests/test_train.py` (6 test).

### 3.1 `intent/encoder.py`

1. **`TransformerBlock(d, h, d_ff, dropout=0.1)`**: pre-LN (bài 16, mục 16.6). Thuộc tính `ln1`, `ln2`, `attn` (`MultiHeadAttention` **của bạn**), `ff` (`Linear → GELU → Linear`), `drop`. `forward(x, mask=None)`.
2. **`EncoderClassifier(vocab_size, n_classes, d=64, h=4, n_layers=2, d_ff=128, max_len=64, dropout=0.1, pad_id=0)`**:
   - `emb = nn.Embedding(vocab_size, d, padding_idx=pad_id)`; sinusoidal PE đăng ký bằng `register_buffer("pe", ..., persistent=False)`; `blocks` là `nn.ModuleList`; `ln` cuối; `head = nn.Linear(d, n_classes)`; dropout sau khi cộng PE.
   - `encode(ids)`: `keep = ids != pad_id`; mask attention shape `(B, 1, 1, T)`; qua các block; LN cuối; **mean-pool có mask** (chỉ trên token thật).
   - `forward(ids)` trả về logit `(B, n_classes)`.
   - Số tham số với mặc định: $64V + 67\,657$ (bài 16, mục 16.4(c)). Thêm cột đệm không được làm đổi kết quả.

### 3.2 `intent/train.py`

1. **`random_split(df, test_frac=0.2, seed=0)`** và **`template_split(df, test_frac=0.2, seed=0)`**, trả về `(train_df, test_df)`.
   - `template_split`: intent gốc của câu mẫu là phần trước dấu `/` của `template_id`. Với mỗi intent trong `INTENTS`, lấy các `template_id` của nó, **sắp xếp**, rồi chọn `k = max(1, round(test_frac * số câu mẫu))` câu mẫu bằng `np.random.default_rng(seed).choice(..., replace=False)`. Dùng **một** `rng` cho cả vòng lặp, duyệt intent theo thứ tự của `INTENTS`.
   - `make_split(df, kind, test_frac, seed)` chọn một trong hai hàm theo `kind` (`"template"` hoặc `"random"`).
2. **`build_tfidf(kind="char", strip=False, C=10.0)`**: `Pipeline([("tfidf", ...), ("clf", LogisticRegression(C=C, max_iter=3000))])`; preprocessor là `functools.partial(normalize, strip=strip)`. `kind="word"`: `token_pattern=r"<\w+>|\w+"`, `ngram_range=(1, 2)`. `kind="char"`: `analyzer="char_wb"`, `ngram_range=(2, 5)`.
3. **`encode_texts(bpe, texts, max_len=64)`**: mỗi tin nhắn thành `[<cls>] + bpe.encode(normalize(t))`, cắt ở `max_len`, đệm bằng 0 tới độ dài lớn nhất trong danh sách. Trả về tensor `long` shape `(N, T)`. Token `<cls>` bảo đảm không câu nào rỗng.
4. **`train_encoder(train_texts, train_y, val_texts, val_y, *, seed=0, num_merges=300, epochs=40, patience=5, lr=1e-3, batch_size=64, **net_kw)`**:
   - `torch.manual_seed(seed)`; học BPE trên `normalize` của văn bản **train**; nhãn theo thứ tự `INTENTS`.
   - AdamW (`weight_decay=0.01`), `CrossEntropyLoss`, clip 1.0; xáo bằng `torch.Generator().manual_seed(seed)`.
   - Early stopping theo val loss (cải thiện khi giảm hơn `1e-4`), nạp lại trọng số tốt nhất.
   - Trả về `(model ở chế độ eval, bpe, history)`, với `history` có `train_loss`, `val_loss` (list) và `best_epoch`.
   - `encoder_proba(model, bpe, texts)`: xác suất softmax, numpy, cột theo `INTENTS`.
5. **`choose_threshold(proba, y_true, target_precision=0.95)`**: duyệt các giá trị max-probability khác nhau, **tăng dần**; trả về giá trị **nhỏ nhất** sao cho các mẫu có max-probability ≥ giá trị đó đạt accuracy ≥ mục tiêu. Không giá trị nào đạt thì trả 1.0.
6. **`route(proba, threshold)`**: nhãn có xác suất lớn nhất, hoặc `FALLBACK = "fallback"` nếu max-probability < ngưỡng.
7. **`scores(y_true, proba, threshold=None)`**: dict `macro_f1` (9 nhãn, dùng `prf` của bạn), `acc`; nếu có ngưỡng thì thêm `coverage` (tỉ lệ **không** bị chuyển cho người) và `routed_precision` (accuracy trên phần được nhận).
8. **`run(df, *, seed=0, split="template", models=("word", "char", "encoder"))`** và **`compare(df, seeds=range(5), split="template", models=...)`**: như mô tả ở mục 1. `compare` trả về bảng trung bình theo model, cùng `f1_std`, `diff_vs_char` (trung bình chênh **ghép cặp** so với `"char"`) và `wins_vs_char`.
9. **`main`**: `python -m intent.train [--n 6000] [--seeds 5] [--split template|random]`.

## 4. Chạy thử

```
cd project
python -m pytest -q                                       # 42 test
python -m intent.train --split random --seeds 5
python -m intent.train --split template --seeds 5
```

**Kết quả mong đợi** (đo trên máy bạn bằng lời giải tham khảo; số của mạng neural có thể lệch nhẹ theo số thread và phiên bản). Mọi model train trên cùng phần `fit`, chọn ngưỡng trên `val` với precision mục tiêu 0.95, đo trên `test`; trung bình 5 split.

| Model | Ngẫu nhiên: F1 | **Theo template: F1** | std giữa split | Chênh với `char` (thắng) | Coverage | Precision trên phần được nhận |
|---|---|---|---|---|---|---|
| `word` (TF-IDF âm tiết 1–2) | 0.981 | **0.714** | 0.097 | +0.038 (5/5) | 0.28 | 0.89 |
| `char` (TF-IDF ký tự 2–5) | 0.983 | 0.676 | 0.103 | — | 0.31 | 0.91 |
| `encoder` (Transformer tự viết, 125k tham số) | 0.980 | 0.521 | 0.095 | −0.155 (0/5) | 0.24 | 0.85 |
| *e5 + LR (tùy chọn, model pretrained)* | 0.962 | 0.709 | 0.107 | +0.033 (4/5) | 0.40 | 0.93 |
| *e5 fine-tune (tùy chọn, khoảng 11 phút mỗi split trên CPU)* | — | 0.711 | 0.074 | +0.035 (3/5) | 0.37 | 0.89 |
| *`word` + e5 + LR, trung bình xác suất* | — | **0.760** | | so với `word`: +0.046 (5/5) | | |

Chênh lệch giữa hai cách chia (0.98 so với 0.52–0.71) lớn hơn mọi chênh lệch giữa các model. Encoder tự viết đạt `best_epoch` 0–1 trên chia theo template: nó học thuộc câu mẫu ngay từ epoch đầu.

**Thời gian** (split theo template seed 0, CPU 4 thread, cắm sạc; độ trễ đo trên 300 tin nhắn, mỗi lần một tin):

| Model | Train | Độ trễ p50 / p95 |
|---|---|---|
| `word` | 0.4 s | 0.5 / 1.0 ms |
| `char` | 0.7 s | 0.6 / 1.2 ms |
| `encoder` | 19 s (7 epoch, early stopping) | 1.4 / 2.4 ms |
| encoder pretrained (chỉ forward, chưa tính LR) | — | 17 / 23 ms |

`python -m pytest -q` chạy 42 test trong khoảng 25 giây.

## 5. Ngày 5 — `REPORT.md`

Trả lời bằng **số liệu và phép tính**. Mỗi câu nên có một bảng nhỏ.

1. **Hai cách chia.** Lập bảng F1 của ba model với `--split random` và `--split template`. Giải thích chênh lệch bằng khái niệm rò rỉ (Tuần 3, bài 01). Nếu đây là log thật, không có `template_id`, bạn sẽ chia thế nào để có con số trung thực tương tự?
2. **So ghép cặp.** Từ 5 split theo template, tính độ lệch chuẩn của **hiệu** `word − char` theo từng split, và so với độ lệch chuẩn của từng cột. (Tham khảo: std của hiệu khoảng 0.03, std của từng cột khoảng 0.10.) Vì sao chỉ so trung bình của từng cột là sai?
3. **Phân tích lỗi.** Với `word`, gộp dự đoán của 5 split, in confusion matrix và 20 tin nhắn sai có xác suất cao nhất. Những cặp intent nào nhầm nhiều nhất, và vì sao (liên hệ bài 13)? (Tham khảo: hủy gói → đăng ký gói 148 lần, khác → lỗi mạng 122 lần; recall của lớp `khac` 0.54.)
4. **Ngưỡng từ chối.** Với `word`, chọn ngưỡng cho precision mục tiêu 0.8, 0.9, 0.95, 0.98 trên val, rồi đo coverage và precision trên test. (Tham khảo, trung bình 5 split:)

   | Mục tiêu | Ngưỡng | Val: coverage / precision | Test: coverage / precision |
   |---|---|---|---|
   | 0.80 | 0.43 | 0.76 / 0.80 | 0.78 / 0.79 |
   | 0.90 | 0.73 | 0.38 / 0.93 | 0.43 / 0.89 |
   | 0.95 | 0.84 | 0.27 / 0.97 | 0.28 / 0.89 |
   | 0.98 | 0.87 | 0.24 / 0.99 | 0.25 / 0.89 |

   Vì sao nâng mục tiêu từ 0.9 lên 0.98 không làm precision trên test tăng? Đề xuất cách sửa (hiệu chỉnh xác suất, val theo thời gian, ngưỡng riêng cho khiếu nại).
5. **Độ bền theo loại nhiễu.** Tính accuracy của `word`, `char` và (nếu có) model pretrained trên từng nhóm `nodiac`, `teen`, `nfd`. (Tham khảo, theo template: `word` 0.73 với tin không dấu, model pretrained chỉ **0.58** với tin không dấu nhưng **0.78** với tin có dấu.) Giải thích bằng bài 14 (tokenizer của model pretrained với văn bản không dấu).
6. **Ensemble.** Lấy trung bình xác suất của hai model. (Tham khảo: `word + char` cho 0.700, **thua** `word`; `word` + model pretrained cho **0.760**, thắng `word` ở **5/5** split, +0.046.) Vì sao ensemble hai model "giống nhau" không giúp, còn ensemble hai model **sai ở những chỗ khác nhau** thì giúp? Liên hệ với câu 5.
7. **Encoder tự viết thua vì đâu?** Thử ít nhất hai thay đổi (`num_merges`, `d`, `dropout`, `lr`) trên 2 split và lập bảng. (Tham khảo: các cấu hình nằm trong khoảng 0.31–0.55.) Kết luận về Transformer train từ đầu khi chỉ có khoảng 140 câu mẫu.

**Checklist hoàn thành:**
- [ ] `python -m pytest -q`: cả 42 test pass.
- [ ] `REPORT.md` trả lời đủ câu 1–6, có bảng ghép cặp.
- [ ] Một hàm `predict(text) -> (intent | "fallback", confidence)` dùng model tốt nhất của bạn, kèm 5 ví dụ chạy thật (có cả tin ngoài phạm vi).

**Mở rộng (không bắt buộc):**
- **Model pretrained** (`intent/pretrained.py`, cần model đã tải, khoảng 470 MB): `embed` (feature extraction) và `finetune` (fine-tune toàn bộ). Chạy trên CPU mất khoảng 10 phút mỗi split khi fine-tune; trên Kaggle T4 nhanh hơn nhiều.
- **Đa intent:** viết bộ sinh tin nhắn ghép hai câu mẫu ("hủy gói A rồi đăng ký gói B") và thử phân loại đa nhãn (sigmoid cho từng intent, BCE).
- **Char-LM của bài 17:** dùng `TransformerBlock` + `causal_mask` + `generate` để sinh tin nhắn giả, rồi dùng chúng làm dữ liệu tăng cường cho router. Có giúp trên chia theo template không?
- **(Sentiment, theo roadmap Phase 4 Medium project)** so LSTM tự viết với một encoder pretrained trên một bộ đánh giá sản phẩm tiếng Việt công khai (cần tải dữ liệu).
