# Tuần 2 · Toán cho AI: Đại số tuyến tính, Giải tích & Tối ưu, Xác suất & Thông tin

**Mục tiêu tuần:** hiểu và **tự chứng minh, tự tính tay** được phần toán đứng sau mọi model: vector và ma trận, gradient và backprop, gradient descent và Adam, MLE và cross-entropy.
**Sản phẩm cuối tuần:** `project/optlab/`, một phòng thí nghiệm tối ưu hóa viết bằng NumPy thuần. Dùng nó để **kiểm chứng bằng số đo** các định lý của tuần.
**Loại tài liệu: Lý thuyết.** Đây là tuần đầu tiên có toán đầy đủ: định nghĩa chuẩn, mệnh đề kèm chứng minh, bài tính tay. Hãy đọc **với giấy bút**: tự làm lại mọi chứng minh và ví dụ số trước khi đọc lời giải.
**Ứng với roadmap:** Phase 2 (2.1–2.5). Ba phần được để lại cho tuần sau, khi có đủ ngữ cảnh: dropout và early stopping (Tuần 4), vanishing/exploding gradient trong mạng sâu (Tuần 4), t-SNE/UMAP (Tuần 3).

## Lịch 5 ngày (khoảng 5–5.5 giờ mỗi ngày)

| Ngày | Đọc (~2h45–3h) | Quiz (45') | Làm (~2h) |
|---|---|---|---|
| 1 | [ngay-1.md](ngay-1.md): **01** Vector, cosine · **02** Ma trận, bình phương tối thiểu · **03** Trị riêng, SVD, PCA | [ngay-1.quiz.md](ngay-1.quiz.md) | `optlab/linalg.py` |
| 2 | [ngay-2.md](ngay-2.md): **04** Gradient, Hessian · **05** Backprop · **06** Gradient descent · **07** Momentum, Adam | [ngay-2.quiz.md](ngay-2.quiz.md) | `optlab/grad.py`, `optlab/optim.py` |
| 3 | [ngay-3.md](ngay-3.md): **08** Bayes · **09** Phân phối, CLT, khoảng tin cậy · **10** MLE, MAP · **11** Entropy, KL | [ngay-3.quiz.md](ngay-3.quiz.md) | `optlab/stats.py` |
| 4 | — | — | **Project:** [project.md](project.md), viết `train.py` và `cli.py` (4 thí nghiệm) |
| 5 | — | — | `REPORT.md` (2h) · **Final 180'**: [final-exam.md](final-exam.md) |

Ngày 2 và 3 có 4 bài, nặng hơn Ngày 1. Nếu thiếu giờ, hãy đọc tầng 1 của cả 4 bài trước, rồi mới đọc tầng 2.

## Chuẩn bị một lần (10 phút)

```
cd "week-02-math/project"
python -m venv .venv
.venv\Scripts\activate
python -m pip install numpy scipy pandas matplotlib pytest ruff
python -m pytest -q        # lúc đầu sẽ báo lỗi vì chưa có code của bạn — đó là điều bình thường
```

Bài 05 có một ví dụ dùng PyTorch để đối chiếu backprop. Ví dụ này **không bắt buộc**; PyTorch sẽ được cài chính thức ở Tuần 4.

## Cách học mỗi bài

Vẫn **hai tầng** như Tuần 1:
- **Tầng 1 · Trực giác** (10–15'): ví dụ đời thường, ba ý cốt lõi, một ví dụ nhỏ có số.
- **Tầng 2 · Kiến thức chuẩn** (35–45', **bắt buộc**): gồm
  - bảng thuật ngữ kèm nguồn (sách giáo trình chuẩn, bài báo gốc);
  - **mệnh đề và chứng minh**;
  - ví dụ tính tay;
  - các con số cần biết, code chuẩn, cạm bẫy;
  - câu hỏi hay gặp khi phỏng vấn.

**Đề thi chấm theo tầng 2.**

**Gặp chứng minh khó:** đọc phát biểu của mệnh đề, kiểm tra nó bằng ví dụ số ngay bên dưới, rồi mới quay lại đọc chứng minh. Mọi mệnh đề quan trọng đều có ví dụ số đi kèm.

**Ba nguồn chính** của tuần (đều có bản miễn phí hợp pháp):
- Deisenroth, Faisal, Ong, *Mathematics for Machine Learning* (MML).
- Goodfellow, Bengio, Courville, *Deep Learning*, ch. 2–5 và 8.
- Blitzstein & Hwang, *Introduction to Probability*.

**Quy ước:**
- Vector là vector cột.
- $X \in \mathbb{R}^{n \times d}$ là $n$ mẫu, $d$ đặc trưng.
- Loss MSE viết là $\frac{1}{2n}\|X\theta - y\|^2$.
- $\ln$ cho nat, $\log_2$ cho bit.
- Mọi con số đo được (thời gian, số bước, phương sai) đã chạy thật trên NumPy 2.5, SciPy 1.18, PyTorch 2.14 CPU.

## Phụ lục

[phu-luc.md](phu-luc.md) bổ sung những mục của roadmap mà 5 ngày ở trên chưa phủ: 3 bài P1–P3: tối ưu có ràng buộc (Lagrange, KKT, liên hệ ridge/SVM/PCA/softmax), lịch learning rate (step decay, exponential, ReduceLROnPlateau, sàn nhiễu của SGD), plateau (vì sao loss đứng yên rồi tụt đột ngột). Thời lượng ~1h30 đọc + quiz 30', có đáp án trong `<details>`. Đọc vào một buổi trống trong tuần, hoặc trước khi sang tuần sau; không bắt buộc cho final tuần này.

## Kiểm tra

| Loại | Khi nào | Thời lượng | Đạt khi |
|---|---|---|---|
| Quiz ngày | Ngay sau khi đọc xong file của ngày đó | 45' (15' quick test + 30' tự luận) | ≥ 21/30 |
| Test bài tập | Sau khi làm bài tập | — | Pass hết |
| Final tuần | Ngày 5 | 180' | ≥ 70/100 |

Quiz và final của tuần này có **chứng minh** và **tính tay nhiều bước**. Được dùng máy tính cầm tay, không được chạy code (trừ phần C của final).

Không đạt quiz thì đọc lại **đúng phần làm sai** trước khi sang ngày tiếp theo.

## Tuần này nối với các tuần sau thế nào

| Kiến thức Tuần 2 | Được dùng lại ở |
|---|---|
| Cosine, PCA, SVD | Tuần 8 (RAG: tìm kiếm vector, nén embedding), Tuần 11 (LoRA là cập nhật hạng thấp) |
| Backprop, softmax + CE | Tuần 4 (PyTorch, MLP, CNN), Tuần 5 (Transformer) |
| GD, $\kappa$, Adam, warmup, clipping | Tuần 3 (huấn luyện model ML), Tuần 4–5, Tuần 11 (fine-tuning) |
| MLE, MAP, L1/L2 | Tuần 3 (ridge, lasso, logistic regression) |
| Khoảng tin cậy, kiểm định ghép cặp | Tuần 13 (evaluation của LLM và RAG) |
| Entropy, KL, perplexity | Tuần 5 (language model), Tuần 11 (distillation, phạt KL trong RLHF) |
