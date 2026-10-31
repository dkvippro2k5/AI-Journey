# Tuần 3 · Machine Learning nền tảng

**Mục tiêu tuần:** xây dựng và đánh giá **đúng cách** một model ML trên dữ liệu bảng, bao gồm chia dữ liệu không rò rỉ, chọn metric và ngưỡng theo chi phí, so sánh model tuyến tính / cây / ensemble, tune, và lưu một pipeline triển khai được.
**Sản phẩm cuối tuần:** `project/quality/`, một model dự đoán câu trả lời của chatbot sẽ bị người dùng chấm 👎, kèm báo cáo có số liệu.
**Loại tài liệu: Lý thuyết (bản nhẹ).** Có công thức và ví dụ tính tay đầy đủ; **chứng minh chỉ được phác thảo vài dòng**. Bài 10–11 thuộc loại Kỹ thuật.
**Ứng với roadmap:** Phase 3 (3.1–3.6). Phần TF-IDF và phân loại văn bản (dự án *spam classifier* trong roadmap) được dời sang Tuần 5 (NLP), là nơi có ngữ cảnh phù hợp hơn.

## Lịch 5 ngày (khoảng 5 giờ mỗi ngày)

| Ngày | Đọc (~2h30–2h45) | Quiz (45') | Làm (~2h) |
|---|---|---|---|
| 1 | [ngay-1.md](ngay-1.md): **01** Chia dữ liệu, rò rỉ · **02** Bias–variance · **03** Cross-validation · **04** Metric, ngưỡng | [ngay-1.quiz.md](ngay-1.quiz.md) | `quality/metrics.py` |
| 2 | [ngay-2.md](ngay-2.md): **05** Model tuyến tính · **06** Cây quyết định · **07** Random Forest, Gradient Boosting | [ngay-2.quiz.md](ngay-2.quiz.md) | `quality/tree.py` |
| 3 | [ngay-3.md](ngay-3.md): **08** K-Means, DBSCAN · **09** t-SNE, UMAP · **10** Pipeline · **11** Tuning | [ngay-3.quiz.md](ngay-3.quiz.md) | `quality/cluster.py`, `quality/pipeline.py` |
| 4 | — | — | **Project:** [project.md](project.md), viết `quality/train.py` |
| 5 | — | — | `REPORT.md` (2h) · **Final 180'**: [final-exam.md](final-exam.md) |

## Chuẩn bị một lần (10 phút)

```
cd "week-03-ml/project"
python -m venv .venv
.venv\Scripts\activate
python -m pip install numpy scipy pandas scikit-learn joblib matplotlib pytest ruff
python -m pytest -q        # lúc đầu sẽ báo lỗi vì chưa có code của bạn — đó là điều bình thường
```

Mỗi ngày chỉ chạy test của ngày đó, ví dụ `python -m pytest -q tests/test_metrics.py`.

## Cách học mỗi bài

Vẫn **hai tầng**:
- **Tầng 1 · Trực giác:** ví dụ đời thường, ba ý cốt lõi, ví dụ có số.
- **Tầng 2 · Kiến thức chuẩn** (bắt buộc, đề thi chấm theo tầng này): thuật ngữ kèm nguồn, công thức, **ví dụ tính tay**, số đo thật trên dữ liệu của project, code chuẩn, cạm bẫy, câu hỏi phỏng vấn.

**Khác với Tuần 2:** không còn mục "Mệnh đề và chứng minh". Công thức được nêu kèm ý nghĩa, và tối đa vài dòng "vì sao". Bù lại, có **nhiều ví dụ tính tay và số đo thật** hơn: bài 02 đo bias và variance bằng mô phỏng, bài 04 có bảng chi phí theo ngưỡng, bài 07 so sánh impurity importance với permutation importance trên cùng một model.

**Dữ liệu xuyên suốt:** mọi bài đều quay về cùng một bộ log chatbot (`project/quality/data.py`). Nhờ vậy bạn thấy được mỗi khái niệm ảnh hưởng tới **cùng một bài toán** ra sao.

**Quy ước:** mọi con số "đo trên máy bạn" đều chạy bằng scikit-learn 1.9, NumPy 2.5, pandas 3.0.

## Phụ lục

[phu-luc.md](phu-luc.md) bổ sung những mục của roadmap mà 5 ngày ở trên chưa phủ: 3 bài P1–P3: feature engineering (đo trên `quality`: logistic regression từ 0.768 lên 0.813, ngang HGB), dữ liệu mất cân bằng (resampling, SMOTE, hiệu chỉnh prior, rò rỉ khi resample trước khi chia), metric hồi quy (MAE/RMSE/R²/MAPE/pinball, train trên $\log y$ và smearing). Thời lượng ~1h45 đọc + quiz 30', có đáp án trong `<details>`. Đọc vào một buổi trống trong tuần, hoặc trước khi sang tuần sau; không bắt buộc cho final tuần này.

## Kiểm tra

| Loại | Khi nào | Thời lượng | Đạt khi |
|---|---|---|---|
| Quiz ngày | Ngay sau khi đọc xong file của ngày đó | 45' (15' quick test + 30' tự luận) | ≥ 21/30 |
| Test bài tập | Sau khi làm bài tập | — | Pass hết |
| Final tuần | Ngày 5 | 180' | ≥ 70/100 |

Quiz và final tuần này **không có câu chứng minh**. Phần tự luận tập trung vào tính tay (metric, chia nút, boosting, K-Means), tìm lỗi trong code, và thiết kế.

## Tuần này nối với các tuần sau thế nào

| Kiến thức Tuần 3 | Được dùng lại ở |
|---|---|
| Chia dữ liệu, rò rỉ, CV | Mọi tuần có đánh giá model; đặc biệt Tuần 8–9 (đánh giá RAG) và Tuần 13 |
| Precision/recall, ngưỡng theo chi phí | Tuần 7 (router chọn model), Tuần 10 (guardrail cho agent), Tuần 13 (evaluation) |
| Pipeline, lưu model | Tuần 13 (đóng gói model `quality` thành service, theo dõi drift) |
| K-Means | Tuần 8 (chỉ mục IVF của vector database; phân cụm câu hỏi người dùng) |
| Gradient boosting | Mọi bài toán dữ liệu bảng; baseline mạnh trước khi nghĩ tới deep learning |
