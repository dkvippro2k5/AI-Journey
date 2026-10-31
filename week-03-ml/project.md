# Project Tuần 3 · `quality` — dự đoán câu trả lời bị người dùng chấm 👎

> **Ngày 4:** ghép các phần thành pipeline huấn luyện đầy đủ (~5h).
> **Ngày 5:** chạy, phân tích, viết `REPORT.md` (2h), sau đó làm final (3h).
> **Vị trí trong hệ thống chung:** model này chạy **ngay sau khi chatbot sinh câu trả lời** (Tuần 6–10). Câu trả lời có xác suất bị 👎 cao sẽ được gửi cho người kiểm duyệt, hoặc được sinh lại bằng model mạnh hơn. Tuần 13 (MLOps) sẽ đóng gói model này thành service và theo dõi drift.

## 1. Bài toán và dữ liệu

`quality/data.py` (cho sẵn) sinh log hội thoại tổng hợp bằng `make_feedback(n, seed)`:

| Cột | Ý nghĩa | Ghi chú |
|---|---|---|
| `session_id` | ID của lượt chat | **Không dùng** (cột ID) |
| `intent` | faq / billing / tech_support / chitchat / complaint | phân loại |
| `model` | small / medium / large | phân loại |
| `n_turns` | số lượt hội thoại trước đó | |
| `query_tokens`, `answer_tokens` | độ dài câu hỏi / câu trả lời | |
| `retrieval_score` | điểm của tài liệu tốt nhất mà RAG tìm được | `NaN` với chitchat (không dùng RAG) và khoảng 5% do lỗi log |
| `n_docs` | số tài liệu tìm được | |
| `latency_s` | thời gian trả lời | khoảng 3% bị thiếu |
| `hour` | giờ trong ngày | |
| `escalated` | phiên có bị chuyển cho nhân viên không | **Rò rỉ**: xảy ra **sau khi** người dùng bấm 👎 |
| `bad` | **nhãn**: 1 nếu người dùng bấm 👎 (khoảng 15.5%) | |

Dữ liệu được sinh với các quan hệ **phi tuyến và có tương tác**, đúng loại quan hệ mà model cây bắt được còn model tuyến tính thì không. `df.attrs["p_true"]` chứa xác suất thật: **không được dùng để train**; chỉ dùng ở `REPORT.md` để tính "trần" AUC.

## 2. Thư mục project

```
project/
├── pyproject.toml
├── quality/
│   ├── __init__.py
│   ├── data.py          # cho sẵn
│   ├── metrics.py       # Ngày 1: metric và stratified k-fold viết từ đầu
│   ├── tree.py          # Ngày 2: gini, entropy, best_split, L2Boost
│   ├── cluster.py       # Ngày 3: K-Means
│   ├── pipeline.py      # Ngày 3: split_xy, build_preprocessor, build_model
│   └── train.py         # Ngày 4: so sánh, tune, chọn ngưỡng, đánh giá, lưu
├── tests/               # 28 test (cho sẵn)
└── _reference/          # lời giải tham khảo — chỉ mở khi kẹt quá 1h
```

## 3. Tham số mặc định của `build_model`

| Tên | Bộ phân loại | Tiền xử lý | Tham số mặc định |
|---|---|---|---|
| `"logreg"` | `LogisticRegression` | `"linear"` | `max_iter=2000` |
| `"rf"` | `RandomForestClassifier` | `"tree"` | `n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0` |
| `"hgb"` | `HistGradientBoostingClassifier` | `"tree"` | `learning_rate=0.05, max_iter=200, max_leaf_nodes=7, min_samples_leaf=50, random_state=0` |

## 4. Ngày 4 — `quality/train.py`

Kiểm tra bằng `tests/test_train.py` (6 test). Mọi CV đều dùng `StratifiedKFold(k, shuffle=True, random_state=seed)`.

1. **`compare_models(X, y, names=("logreg", "rf", "hgb"), k=5, seed=0) -> DataFrame`**
   - Dùng `cross_validate` với `scoring=["roc_auc", "average_precision"]`.
   - Các cột: `model, auc_mean, auc_std, ap_mean, time_s`.
2. **`tune(X, y, name="hgb", n_iter=20, k=5, seed=0) -> (params, best_auc)`**
   - Dùng `RandomizedSearchCV` với `SEARCH_SPACES[name]`; xem bài 11 để thiết kế: `learning_rate` theo **thang log**.
   - `params` trả về **không có** tiền tố `clf__` và chỉ gồm kiểu Python thuần (dùng `.item()` cho số NumPy), để `build_model(name, **params)` dùng được ngay.
3. **`choose_threshold(model, X, y, cost_fp=1, cost_fn=5, k=5, seed=0) -> (t, cost_per_1000)`**
   - Lấy xác suất **out-of-fold** bằng `cross_val_predict(..., method="predict_proba")`, rồi gọi `best_threshold` của Ngày 1.
   - **Không bao giờ chọn ngưỡng trên test.**
4. **`evaluate(model, X, y, threshold, cost_fp=1, cost_fn=5, n_boot=1000, seed=0) -> dict`**
   - Các khóa: `auc, auc_lo, auc_hi, ap, precision, recall, f1, cost_per_1000`.
   - `auc_lo` và `auc_hi` là khoảng tin cậy bootstrap 95%; bỏ qua các mẫu bootstrap chỉ có một lớp.
5. **`run(df, *, seed=0, n_iter=20, out=None) -> dict`**
   - Chia 80/20 có `stratify` → `compare_models` trên train → `tune("hgb")` → `build_model("hgb", **params)` → `choose_threshold` trên train → `fit` trên toàn bộ train → `evaluate` trên test → `permutation_importance` (scoring `"roc_auc"`, `n_repeats=5`) trên test.
   - Nếu có `out` thì lưu `model.joblib` (cả pipeline) và `report.json` (UTF-8, `ensure_ascii=False`).
   - Khóa của dict trả về: `compare, best_params, cv_auc, threshold, cv_cost_per_1000, test, importance`.
6. **`main(argv=None)`**
   - Lệnh: `python -m quality.train [--n 6000] [--seed 0] [--n-iter 20] [--out artifacts]`.
   - Cấu hình `logging`, rồi in JSON gồm `best_params, cv_auc, threshold, test, importance`.

## 5. Chạy thử

```
cd project
python -m pytest -q                          # 28 test, khoảng 30 giây
python -m quality.train --out artifacts      # khoảng 30 giây
```

**Kết quả mong đợi** (đo trên máy bạn bằng lời giải tham khảo; con số có thể lệch nhẹ tùy phiên bản):

```
 model  auc_mean  auc_std  ap_mean   time_s
logreg    0.7729   0.0292   0.4293     0.13
    rf    0.8085   0.0174   0.4984     2.31
   hgb    0.8125   0.0234   0.5189     1.18
```

```json
"cv_auc": 0.8171,
"threshold": 0.1740,
"test": {"auc": 0.7957, "auc_lo": 0.7581, "auc_hi": 0.8333, "ap": 0.5153,
         "precision": 0.3378, "recall": 0.6720, "f1": 0.4496, "cost_per_1000": 458.3},
"importance": {"n_turns": 0.0612, "intent": 0.0607, "latency_s": 0.0532, "retrieval_score": 0.0498,
               "model": 0.0365, "hour": 0.0147, "answer_tokens": 0.0089, "query_tokens": 0.0016, "n_docs": -0.0007}
```

## 6. Ngày 5 — `REPORT.md`

Trả lời bằng **số liệu và phép tính**.

1. **Trần AUC.** Tính `roc_auc_score(y_test, p_true_test)` với `p_true` lấy từ `df.attrs` (chú ý lấy đúng các dòng thuộc tập test). Đây là AUC của "model hoàn hảo, biết đúng xác suất thật". Model của bạn cách trần bao xa? Vì sao **không model nào** đạt AUC 1.0 trên dữ liệu này?
2. **Rò rỉ.** Thêm `escalated` vào đặc trưng, chạy lại `compare_models`. AUC tăng bao nhiêu? Viết một đoạn ngắn giải thích cho quản lý vì sao con số đẹp hơn đó là **sai**.
3. **Model tuyến tính và model cây.** Logistic regression thua khoảng 0.04 AUC. Đọc `data.py` và chỉ ra **ba** quan hệ cụ thể mà model tuyến tính không bắt được. Thử thêm `SplineTransformer` (bài 05) hoặc tự tạo đặc trưng tương tác `is_small_tech = (model == "small") & (intent == "tech_support")`: khoảng cách giảm còn bao nhiêu?
4. **CV và test.** AUC CV tốt nhất (khoảng 0.817) cao hơn AUC test (khoảng 0.796). Giải thích bằng bài 03 và bài 11. Khoảng tin cậy của test có chứa 0.817 không?
5. **Ngưỡng và chi phí.**
   - So sánh ngưỡng đã chọn với ngưỡng lý thuyết $c_{FP}/(c_{FP} + c_{FN})$.
   - Tính chi phí trên 1 000 lượt cho ba chính sách: **không bao giờ báo**, **luôn báo**, và **dùng model**. Model tiết kiệm được bao nhiêu phần trăm?
   - Nếu $c_{FN}$ tăng lên 10 thì ngưỡng dịch theo hướng nào?
6. **Importance.**
   - So sánh permutation importance với impurity importance của RF (bài 07) cho `query_tokens` và `answer_tokens`.
   - Theo `data.py`, `query_tokens` có ảnh hưởng thật không?
   - `n_turns` đứng đầu bảng permutation importance; giải thích dựa trên `data.py`.
7. **(Mở rộng) Phân tích lỗi bằng K-Means.** Lấy các false negative trên test (bị 👎 nhưng model bỏ sót). Chuẩn hóa các đặc trưng số, chạy `kmeans` của Ngày 3 với $k$ từ 2 tới 5, chọn $k$ bằng silhouette, rồi mô tả từng cụm. Có nhóm lỗi nào mà model **không thể** biết (thiếu đặc trưng)?

**Checklist hoàn thành:**
- [ ] `python -m pytest -q`: cả 28 test pass.
- [ ] `artifacts/model.joblib` tải lại được và `predict_proba` chạy được trên dữ liệu thô có `NaN` và có intent lạ.
- [ ] `REPORT.md` trả lời đủ câu 1–6.
- [ ] `artifacts/` nằm trong `.gitignore`.

**Mở rộng (không bắt buộc):**
- Thay `RandomizedSearchCV` bằng `HalvingRandomSearchCV` với cùng ngân sách thời gian, rồi so sánh.
- Hiệu chỉnh xác suất bằng `CalibratedClassifierCV` và vẽ reliability diagram.
- Cài `lightgbm` hoặc `xgboost` và so sánh với HGB.
