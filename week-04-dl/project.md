# Project Tuần 4 · `qnet` — mạng neural cho bài toán 👎, so với gradient boosting

> **Ngày 4:** ghép các mảnh thành pipeline huấn luyện đầy đủ (~5h).
> **Ngày 5:** chạy, phân tích, viết `REPORT.md` (2h), sau đó làm final (3h).
> **Câu hỏi của tuần:** trên **cùng** dữ liệu và **cùng** split với Tuần 3, mạng neural (MLP, và một model attention nhỏ) có thắng HistGradientBoosting không? Kết quả thường gặp trên dữ liệu bảng là **không** (Grinsztajn et al., NeurIPS 2022). Mục tiêu là hiểu **vì sao**, bằng số liệu.
> **Vị trí trong hệ thống chung:** cùng vị trí với model `quality` của Tuần 3 (chấm điểm câu trả lời ngay sau khi chatbot sinh ra). Vòng lặp `fit`, các lớp tự viết và `AttnPool` sẽ được dùng lại ở Tuần 5 (Transformer) và Tuần 11 (fine-tuning).

## 1. Dữ liệu

`qnet/data.py` **giống hệt** `quality/data.py` của Tuần 3 (xem [project Tuần 3](../week-03-ml/project.md) §1): 9 đặc trưng (7 số, 2 phân loại), nhãn `bad` (khoảng 15.5% dương), `NaN` ở `retrieval_score` và `latency_s`. Cột `session_id` và `escalated` (rò rỉ) bị loại bởi `split_xy`.

`qnet/baseline.py` (cho sẵn) có `split_xy(df)` và `build_baseline("logreg" | "hgb")`, với đúng tiền xử lý và tham số mặc định của Tuần 3.

## 2. Thư mục project

```
project/
├── pyproject.toml
├── qnet/
│   ├── __init__.py
│   ├── data.py          # cho sẵn
│   ├── baseline.py      # cho sẵn: logreg và HGB của Tuần 3
│   ├── loop.py          # Ngày 1: make_loader, train_one_epoch, evaluate, predict, fit, checkpoint
│   ├── layers.py        # Ngày 2: Dropout, LayerNorm, RMSNorm, BatchNorm1d, init_, activation_stds, layer_grad_norms
│   ├── attention.py     # Ngày 3: dot_attention, AdditiveAttention, AttnPool
│   ├── tabular.py       # Ngày 4: TabularPrep, TabMLP, TabAttn, build_net
│   └── train.py         # Ngày 4: train_net, so sánh với baseline, lưu/nạp model
├── tests/               # 28 test (cho sẵn)
└── _reference/          # lời giải tham khảo — chỉ mở khi kẹt quá 1h
```

## 3. Hai mạng của tuần

**`TabMLP`** (bài 03): `[x_num ‖ embedding(intent) ‖ embedding(model)]` → (Linear → norm tùy chọn → ReLU → `Dropout`) × 2 → Linear(1). Mặc định `emb_dim=4`, `hidden=(64, 64)`, `dropout=0.1`, `norm="none"`. Có **5 417** tham số (tính tay ở bài 03, mục 3.2).

**`TabAttn`** (bài 09): mỗi đặc trưng thành **một token** $d = 16$ chiều; gộp 11 token bằng `AttnPool`; rồi qua head MLP.
- Token số thứ $i$: $x_i w_i + b_i$, với mỗi cột có $w_i, b_i \in \mathbb R^{d}$ riêng (đây là "numerical embedding" của FT-Transformer, Gorishniy et al., NeurIPS 2021).
- Token phân loại: `nn.Embedding(card, d)`.
- 9 cột số + 2 cột chỉ báo thiếu + 2 cột phân loại = **11 token**.
- Có **2 433** tham số, nhưng trả lời được câu hỏi "với mẫu này, model đang nhìn vào cột nào?".

```
x_num (B, 9) ──► x_i·w_i + b_i ──► (B, 9, 16) ┐
x_cat (B, 2) ──► Embedding      ──► (B, 2, 16) ┴► concat (B, 11, 16) ─► LayerNorm─Linear─ReLU ─► AttnPool ─► (B, 16) ─► head ─► logit
                                                                                                  └► weights (B, 11)
```

## 4. Ngày 4 — `qnet/tabular.py` và `qnet/train.py`

Kiểm tra bằng `tests/test_tabular.py` (4 test) và `tests/test_train.py` (4 test).

### 4.1 `qnet/tabular.py`

1. **`TabularPrep(num_cols=FEATURES_NUM, cat_cols=FEATURES_CAT)`**: phiên bản PyTorch của `ColumnTransformer` ở Tuần 3. Mọi thống kê học **chỉ** trong `fit`.
   - `fit(X)`:
     - `median_`: median từng cột số, bỏ qua NaN.
     - `mean_`, `std_`: mean và std (lệch, `ddof=0`) của dữ liệu **đã điền median**. Cột có `std == 0` thì đặt std = 1.
     - `missing_cols_`: danh sách các cột số có NaN **trong dữ liệu fit**, theo thứ tự của `num_cols`.
     - `vocab_`: dict `{cột: {giá trị: chỉ số}}`, với các giá trị (bỏ NaN) **sắp xếp tăng dần** nhận chỉ số $1..K$.
     - Trả về `self`.
   - Thuộc tính (property):
     - `n_num = len(num_cols) + len(missing_cols_)`.
     - `cardinalities = [K + 1, ...]` (thêm chỉ số 0).
     - `token_names = num_cols + [f"{c}_missing" ...] + cat_cols`.
   - `transform(X) -> (x_num float32 (n, n_num), x_cat int64 (n, len(cat_cols)))`:
     - `x_num` = `[(giá trị đã điền median − mean_) / std_ ‖ chỉ báo thiếu 0/1 của missing_cols_]`.
     - Giá trị phân loại lạ hoặc NaN thì nhận **0**.
2. **`TabMLP(n_num, cardinalities, emb_dim=4, hidden=(64, 64), dropout=0.1, norm="none")`**:
   - Như mục 3, dùng **`Dropout`, `BatchNorm1d` (`norm="batch"`), `LayerNorm` (`norm="layer"`) của Ngày 2**. `norm` khác thì raise `ValueError`.
   - Gọi `init_(self.body, "he")`.
   - `forward(x_num, x_cat)` trả về logit shape `(B,)`.
3. **`TabAttn(n_num, cardinalities, d=16, hidden=64, dropout=0.1)`**:
   - Tham số `num_w`, `num_b` có shape `(n_num, d)` (khởi tạo `randn / sqrt(d)` và `zeros`); `embs`; `token_mlp = Sequential(LayerNorm(d), Linear(d, d), ReLU())`; `pool = AttnPool(d)`; `head = Sequential(Linear(d, hidden), ReLU(), Dropout(dropout), Linear(hidden, 1))`.
   - `forward` trả về logit `(B,)`.
   - `attention(x_num, x_cat)` chạy ở chế độ eval, không gradient, và trả về trọng số `(B, T)`.
4. **`build_net(arch, prep, **kw)`**: `"mlp"` → `TabMLP(prep.n_num, prep.cardinalities, **kw)`; `"attn"` → `TabAttn(...)`; khác thì raise `ValueError`.

### 4.2 `qnet/train.py`

1. **`train_net(arch, X_fit, y_fit, X_val, y_val, *, seed=0, lr=1e-3, weight_decay=1e-4, batch_size=128, max_epochs=200, patience=10, **net_kw) -> (model, prep, hist)`**
   - `set_seed(seed)` → `TabularPrep().fit(X_fit)` → `build_net` → `make_loader` (train: `shuffle=True, seed=seed`; val: batch 1024) → `AdamW` → `fit(..., BCEWithLogitsLoss())` của Ngày 1.
   - `hist` có thêm khóa `time_s`.
2. **`predict_proba(model, prep, X) -> np.ndarray`**: `sigmoid(predict(model, *prep.transform(X)))`.
3. **`attention_summary(model, prep, X) -> dict`**: trọng số attention trung bình trên `X` cho từng `token_names`, sắp **giảm dần**.
4. **`save_bundle(path, arch, net_kw, model, prep)` / `load_bundle(path) -> (model, prep)`**:
   - Lưu `path.with_suffix(".pt")`, là dict `{"arch", "net_kw", "state_dict"}` (đọc bằng `weights_only=True`).
   - Lưu `path.with_suffix(".prep.joblib")` cho `TabularPrep`.
   - `load_bundle` dựng lại model bằng `build_net` rồi trả về ở chế độ eval.
5. **`run(df, *, seed=0, archs=("mlp", "attn"), out=None) -> dict`**
   - Chia 80/20 có `stratify`, `random_state=seed` (**đúng split của Tuần 3**). Chia tiếp phần train 80/20 có `stratify` để có `X_fit`/`X_val` cho early stopping.
   - `logreg` và `hgb` fit trên **toàn bộ** phần train; mỗi mạng dùng `train_net` trên `X_fit`/`X_val`.
   - Kết quả trả về:
     - `test[name] = {"auc", "ap", "time_s"}` cho cả 4 model.
     - `nets[arch] = {"n_params", "best_epoch", "stopped_epoch", "best_val"}`.
     - `attention` (khi có `"attn"`).
     - `n_train`, `n_test`.
   - Nếu có `out`: lưu `out/<arch>.pt`, `out/<arch>.prep.joblib`, `out/report.json` (UTF-8, `ensure_ascii=False`).
6. **`compare_seeds(df, seeds=(0, 1, 2, 3, 4), archs=("mlp", "attn")) -> DataFrame`**
   - Chạy `run` với từng seed (mỗi seed là một split khác).
   - Trả về các cột `model, auc_mean, auc_std (ddof=1), diff_vs_hgb, wins_vs_hgb`, theo thứ tự hàng `logreg, hgb, *archs`.
   - `diff_vs_hgb` là trung bình của **hiệu ghép cặp** AUC(model) − AUC(hgb) trên cùng split; `wins_vs_hgb` là số split mà model thắng.
7. **`main(argv=None)`**: `python -m qnet.train [--n 6000] [--seed 0] [--repeats 0] [--out artifacts]`. Cấu hình `logging`, chạy `run`; nếu `--repeats > 0` thì thêm `report["repeated"]`; in report dạng JSON.

## 5. Chạy thử

```
cd project
python -m pytest -q                                  # 28 test, khoảng 30 giây
python -m qnet.train --out artifacts --repeats 5     # khoảng 1.5 phút
```

**Kết quả mong đợi** (đo trên máy bạn bằng lời giải tham khảo, PyTorch 2.14 CPU; số của mạng neural có thể lệch nhẹ theo số thread và phiên bản):

Split `seed=0` (4 800 train / 1 200 test):

| Model | AUC test | AP test | Thời gian | Ghi chú |
|---|---|---|---|---|
| logreg | 0.7407 | 0.4160 | 0.06 s | |
| **hgb** | **0.7907** | **0.5098** | 0.24 s | |
| mlp | 0.7797 | 0.5014 | 2.8 s | 5 417 tham số; best epoch 18, dừng ở 28 |
| attn | 0.7821 | 0.4992 | 10.8 s | 2 433 tham số; best epoch 43, dừng ở 53 |

```json
"attention": {"model": 0.1831, "intent": 0.1452, "n_turns": 0.1096, "answer_tokens": 0.0947, "n_docs": 0.0884,
              "query_tokens": 0.0853, "hour": 0.0751, "latency_s": 0.0665, "retrieval_score": 0.0653,
              "retrieval_score_missing": 0.0570, "latency_s_missing": 0.0299}
```

So sánh **ghép cặp** qua 5 split (`--repeats 5`):

```
 model  auc_mean  auc_std  diff_vs_hgb  wins_vs_hgb
logreg    0.7560   0.0147      -0.0458            0
   hgb    0.8018   0.0138       0.0000            0
   mlp    0.7854   0.0178      -0.0164            1
  attn    0.7923   0.0164      -0.0095            1
```

## 6. Ngày 5 — `REPORT.md`

Trả lời bằng **số liệu và phép tính**. Mỗi câu nên có một bảng nhỏ.

1. **Ai thắng, và có chắc không?**
   - Từ bảng ghép cặp: MLP thua HGB trung bình 0.016 AUC, nhưng thắng 1/5 split. Tính độ lệch chuẩn của **hiệu ghép cặp** (sửa `compare_seeds` hoặc tự tính từ `run`). Vì sao std của hiệu nhỏ hơn nhiều so với $\sqrt{0.014^2 + 0.018^2}$?
   - Với 5 split, bạn kết luận được gì? (Gợi ý: Tuần 3, bài 03: các split dùng chung dữ liệu nên không độc lập.)
2. **Vì sao cây thắng trên dữ liệu bảng?** Đọc `data.py` và Grinsztajn et al. (2022, ba lý do: hàm mục tiêu không trơn, nhiều đặc trưng vô dụng, không bất biến khi xoay đặc trưng). Chỉ ra ít nhất **hai** quan hệ trong `data.py` mà một cây chỉ cần 1–2 phép chia, còn MLP ReLU phải "ghép" từ nhiều neuron (ví dụ `latency_s > 5.0`).
3. **So sánh có công bằng không?** Hai mạng chỉ thấy 80% dữ liệu train (20% dành cho early stopping), còn HGB thấy 100%.
   - Fit HGB **chỉ trên `X_fit`** và đo lại.
   - Rồi huấn luyện lại MLP trên **toàn bộ** train đúng `best_epoch + 1` epoch (không cần val). Khoảng cách thay đổi thế nào?
   - (Tham khảo, seed 0: HGB chỉ fit trên `X_fit` cho 0.785; MLP huấn luyện lại trên toàn bộ train cho 0.783.)
4. **Ablation.** Với `train_net("mlp", ..., **net_kw)` và 3 seed trên cùng split, lập bảng AUC (mean ± std) cho:
   - `dropout` ∈ {0, 0.1, 0.3};
   - `norm` ∈ {none, batch, layer};
   - `emb_dim` ∈ {1, 4};
   - `hidden` ∈ {(64, 64), (256, 256)}.

   Thay đổi nào **vượt qua** độ dao động giữa các seed? Liên hệ với bài 05, mục 5.4(c) và bài 06, mục 6.4(a). (Tham khảo: chỉ `emb_dim=1` tụt rõ, xuống khoảng 0.764.)
5. **Attention có phải là lời giải thích không?**
   - So sánh bảng `attention` ở trên với **permutation importance** của chính `TabAttn` trên test (viết lại `perm_importance` của Tuần 3, quiz Ngày 2, T4, dùng `predict_proba` của tuần này).
   - `retrieval_score` đứng thứ mấy ở mỗi bảng? Còn `query_tokens` (theo `data.py`, cột này **không** ảnh hưởng tới nhãn)?
   - Giải thích vì sao trọng số attention cao không có nghĩa là quan trọng. Gợi ý: đầu ra là $\sum_j \alpha_j W_v t_j$, nên độ lớn của $W_v t_j$ cũng quan trọng; ngoài ra còn bài 09, cạm bẫy 6.
6. **Calibration và ngưỡng.**
   - Tính ECE (hàm `ece` của final Tuần 3, C3) cho HGB, MLP và Attn trên test.
   - Chọn ngưỡng theo chi phí $c_{FP} = 1$, $c_{FN} = 5$ trên **val** (dùng `best_threshold` của Tuần 3), rồi tính chi phí trên 1 000 lượt ở test. Model nào rẻ nhất?
7. **(Mở rộng) Ensemble.** Lấy trung bình xác suất của HGB và MLP. AUC có tăng không? Vì sao ensemble giữa hai **họ** model khác nhau thường có ích hơn ensemble hai cây? Trên split seed 0, mức tăng là bao nhiêu, và có vượt qua độ dao động không?

**Checklist hoàn thành:**
- [ ] `python -m pytest -q`: cả 28 test pass.
- [ ] `artifacts/mlp.pt` + `mlp.prep.joblib` nạp lại được bằng `load_bundle` và dự đoán được trên dữ liệu thô có `NaN` và có intent lạ.
- [ ] `REPORT.md` trả lời đủ câu 1–6.
- [ ] `artifacts/` nằm trong `.gitignore`.

**Mở rộng (không bắt buộc):**
- **(B) CNN trên `load_digits`:** dựng lại thí nghiệm ở bài 07, mục 7.4 (MLP so với CNN, kiểm tra với ảnh bị dịch) bằng `fit` và `make_loader` của Ngày 1, có thêm early stopping. Thêm augmentation (dịch ngẫu nhiên ±1 pixel) cho cả hai model.
- **Đặc trưng cho MLP:** thử `log1p` cho các cột đuôi dài (`query_tokens`, `answer_tokens`, `latency_s`, `n_turns`); thử *piecewise linear encoding* (Gorishniy et al., NeurIPS 2022) cho `latency_s` và `retrieval_score`. (Tham khảo: `log1p` đơn thuần **không** giúp, AUC 0.774.)
- **Kaggle (cần GPU, tùy chọn):** fine-tune ResNet-18 pretrained (torchvision) trên một bộ ảnh nhỏ 5 lớp, so sánh feature extraction với fine-tune toàn bộ, vẽ loss curve (Phase 4 *Easy project* trong README gốc). Dùng `torch.amp` với fp16 trên T4.
