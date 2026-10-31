# Tuần 4 · Deep Learning và PyTorch

**Mục tiêu tuần:** viết được một vòng huấn luyện PyTorch **đúng** (train/eval, early stopping, checkpoint); hiểu vì sao mạng sâu khó huấn luyện và các công cụ chữa (khởi tạo, normalization, residual, dropout, clipping); tính tay được shape, số tham số và FLOP của MLP, CNN, LSTM; hiểu attention trước Transformer và transfer learning.
**Sản phẩm cuối tuần:** `project/qnet/`, gồm hai mạng neural (MLP và một model attention nhỏ) cho bài toán 👎 của Tuần 3, được so sánh **ghép cặp** với HistGradientBoosting trên cùng split, kèm báo cáo có số liệu.
**Loại tài liệu: Lý thuyết (bản nhẹ).** Có công thức và ví dụ tính tay đầy đủ; **chứng minh chỉ được phác thảo vài dòng**. Bài 10 nghiêng về Kỹ thuật.
**Ứng với roadmap:** Phase 4 (4.1–4.7). Các phần **Tuần 2 và 3 hẹn dời sang** đều có ở đây: dropout và early stopping (bài 06), vanishing/exploding gradient (bài 04), khởi tạo trọng số (bài 04), BatchNorm và LayerNorm (bài 05).

## Lịch 5 ngày (khoảng 5 giờ mỗi ngày)

| Ngày | Đọc (~2h30–3h) | Quiz (45') | Làm (~2h) |
|---|---|---|---|
| 1 | [ngay-1.md](ngay-1.md): **01** Tensor, autograd · **02** `nn.Module`, vòng lặp huấn luyện · **03** MLP, activation, đếm tham số | [ngay-1.quiz.md](ngay-1.quiz.md) | `qnet/loop.py` |
| 2 | [ngay-2.md](ngay-2.md): **04** Vanishing gradient, khởi tạo, residual · **05** BatchNorm, LayerNorm, RMSNorm · **06** Dropout, weight decay, mixed precision | [ngay-2.quiz.md](ngay-2.quiz.md) | `qnet/layers.py` |
| 3 | [ngay-3.md](ngay-3.md): **07** CNN · **08** RNN, LSTM, GRU · **09** Seq2seq, attention, beam search · **10** Transfer learning | [ngay-3.quiz.md](ngay-3.quiz.md) | `qnet/attention.py` |
| 4 | — | — | **Project:** [project.md](project.md), viết `qnet/tabular.py` và `qnet/train.py` |
| 5 | — | — | `REPORT.md` (2h) · **Final 180'**: [final-exam.md](final-exam.md) |

Ngày 3 dài hơn hai ngày đầu (4 bài). Nếu thiếu thời gian, đọc bài 10 vào sáng Ngày 4.

## Chuẩn bị một lần (10 phút)

```
cd "week-04-dl/project"
python -m venv .venv
.venv\Scripts\activate
python -m pip install numpy pandas scikit-learn joblib torch pytest ruff
python -m pytest -q        # lúc đầu sẽ báo lỗi vì chưa có code của bạn — đó là điều bình thường
```

- **Không cần GPU.** Mọi bài tập, thí nghiệm và test đều chạy trên CPU: toàn bộ 28 test mất khoảng 30 giây; project đầy đủ mất khoảng 1.5 phút. Bản `torch` CPU là đủ (trên máy này: `torch 2.14.1+cpu`).
- **Cắm sạc khi chạy.** Trên laptop này, khi chạy pin, CPU tự hạ xung nhịp xuống 1.4 GHz: phép nhân ma trận 2048×2048 chỉ đạt 33 GFLOP/s thay vì 484, và 28 test mất 77 giây thay vì 26. Mọi thời gian trong tài liệu đo khi cắm sạc.
- **Không cần tải dữ liệu.** Log chatbot là dữ liệu tổng hợp (`qnet/data.py`). Thí nghiệm ảnh dùng `sklearn.datasets.load_digits` (chữ số 8×8, có sẵn khi cài scikit-learn). Không dùng `torchvision`.
- Mỗi ngày chỉ chạy test của ngày đó, ví dụ `python -m pytest -q tests/test_loop.py`.
- **Phần tùy chọn cần GPU** (fine-tune ResNet-18 pretrained) nằm ở cuối [project.md](project.md), để chạy trên Kaggle. Không bắt buộc.

## Cách học mỗi bài

Vẫn **hai tầng** như Tuần 3:
- **Tầng 1 · Trực giác:** ví dụ đời thường, ba ý cốt lõi, ví dụ có số.
- **Tầng 2 · Kiến thức chuẩn** (bắt buộc, đề thi chấm theo tầng này): thuật ngữ kèm nguồn, công thức, **ví dụ tính tay**, số đo thật, code chuẩn, cạm bẫy, câu hỏi phỏng vấn.

**Tuần này có nhiều "bất ngờ" được đo thật.** Hãy đọc kỹ những chỗ kết quả đi ngược trực giác, vì chúng hay được hỏi khi phỏng vấn:
- Quên `zero_grad()` thì code vẫn chạy, nhưng val loss dao động 0.40–0.59 thay vì xuống 0.35 (bài 02).
- Adam "cứu" được một mạng 12 lớp khởi tạo quá nhỏ, còn SGD thì kẹt hẳn ở accuracy 10% (bài 04–05).
- Overfit làm val loss tăng gấp đôi nhưng AUC gần như không đổi (bài 06).
- LSTM với init mặc định cũng vanishing gần như RNN; đặt forget bias = 3 thì giải được bài nhớ 40 bước (bài 08).
- Transfer learning có thể **hại** khi miền nguồn hẹp và xa (bài 10).
- Trọng số attention không trùng với mức độ quan trọng thật của đặc trưng (project, câu 5).

**Quy ước:** mọi con số "đo trên máy bạn" chạy bằng PyTorch 2.14 (CPU), NumPy 2.5, scikit-learn 1.9. Kết quả huấn luyện mạng neural có thể lệch nhẹ ở chữ số thứ ba khi đổi số thread hoặc phiên bản.

**Khối "🐢 Đi chậm từng bước"** (thêm ngày 2026-10-27) nằm ngay trước công thức ở ba chỗ khó hình dung nhất: phương sai qua từng lớp khi khởi tạo (mục 4.2), một bước LSTM và gradient trên cell (mục 8.2), q/k/v như tra cứu mềm và lý do chia $\sqrt d$ (mục 9.2). Đọc khối đó trước, rồi mới đọc công thức tóm tắt. Nếu còn chỗ nào khác khó hình dung, hãy đánh dấu lại để bổ sung.

## Phụ lục

[phu-luc.md](phu-luc.md) bổ sung những mục của roadmap mà 5 ngày ở trên chưa phủ, gồm 4 bài:
- **P1 Custom loss:** weighted BCE, focal loss, label smoothing, ổn định số. Đo trên `quality`: `pos_weight` không tăng AUC nhưng đẩy ECE từ 0.025 lên 0.246; hiệu chỉnh odds đưa ECE về 0.020; focal loss làm AUC giảm.
- **P2 `Dataset` và padding:** `collate_fn`, mask, pack, bucketing (đệm từ 41.5% xuống 0.4%), BiLSTM.
- **P3 Object detection cơ bản:** IoU, anchor, NMS, mAP, tính tay và code NumPy.
- **P4 Thiết bị và GPU:** code chạy được trên mọi device, đồng bộ, bộ nhớ; ước lượng cho GPU Kaggle.

Thời lượng ~2h30 đọc + quiz 30', có đáp án trong `<details>`. Đọc vào một buổi trống trong tuần, hoặc trước khi sang tuần sau; không bắt buộc cho final tuần này. Mục *Using HuggingFace pretrained models* (4.7) cần tải model, nên được dời sang **Tuần 5, Ngày 3**.

## Kiểm tra

| Loại | Khi nào | Thời lượng | Đạt khi |
|---|---|---|---|
| Quiz ngày | Ngay sau khi đọc xong file của ngày đó | 45' (15' quick test + 30' tự luận) | ≥ 21/30 |
| Test bài tập | Sau khi làm bài tập | — | Pass hết |
| Final tuần | Ngày 5 | 180' | ≥ 70/100 |
| [Ôn phỏng vấn](on-phong-van.md) | Sau final; ôn lại sau 1 tuần và sau 1 tháng | ~45' (16 câu, nói to, mỗi câu 1') | ≥ 12/16 |

Quiz và final **không có câu chứng minh**. Phần tự luận tập trung vào: tính tay (forward/backward, phương sai khi khởi tạo, BN/LN, shape và tham số CNN, một bước LSTM, attention), chẩn đoán lỗi huấn luyện, và thiết kế. Phần C của final viết `conv2d`, LSTM và beam search bằng NumPy/Python thuần.

## Nhận từ tuần trước, chuyển sang tuần sau

| Từ | Nội dung | Dùng ở bài |
|---|---|---|
| Tuần 2, bài 05 | Backprop, $p - y$ | 01, B2 của final |
| Tuần 2, bài 07 | Adam, lịch learning rate | 02, 06 |
| Tuần 3 | Dữ liệu `quality`, HGB baseline, split, AUC/AP, ECE, `best_threshold` | Project |

**Dời sang Tuần 5:** kiến trúc Transformer đầy đủ (multi-head attention, positional encoding, causal mask, KV-cache) và cách dùng model pretrained của Hugging Face. Bài 09 chỉ dừng ở attention trong seq2seq, nhưng `dot_attention` của bạn chính là lõi của Transformer.

## Tuần này nối với các tuần sau thế nào

| Kiến thức Tuần 4 | Được dùng lại ở |
|---|---|
| `fit`, early stopping, checkpoint (model + optimizer + scheduler) | Tuần 5 (huấn luyện Transformer nhỏ), Tuần 11 (fine-tuning), Tuần 13 (đóng gói model) |
| `dot_attention`, mask, scale $\sqrt d$ | Tuần 5: multi-head attention, causal mask |
| LayerNorm, RMSNorm, residual, GELU/SwiGLU, đếm tham số | Tuần 5: block Transformer; Tuần 12: ước lượng bộ nhớ khi phục vụ LLM |
| Bộ nhớ khi huấn luyện (16 byte/tham số), mixed precision, gradient accumulation | Tuần 11: LoRA/QLoRA trên GPU Kaggle |
| Transfer learning, discriminative LR, đóng băng | Tuần 11 (fine-tuning LLM), Tuần 8 (embedding model cho RAG) |
| Beam search, greedy | Tuần 6 (decoding: temperature, top-p) |
| So sánh ghép cặp nhiều split | Tuần 8–9 và 13 (đánh giá RAG, so sánh model) |
