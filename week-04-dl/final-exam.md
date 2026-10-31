# Final · Tuần 4 · Deep Learning và PyTorch

> ⏱ **180 phút** · **100 điểm** · trung bình – khó · Đạt khi ≥ 70 · Đáp án: [final-solutions.md](final-solutions.md)
> **Được dùng:** giấy nháp và máy tính cầm tay (phần A, B, D); Python + NumPy + PyTorch + pytest (phần C).
> **Không được dùng:** AI, Internet, tài liệu.

| Phần | Nội dung | Điểm | Thời gian gợi ý |
|---|---|---|---|
| A | Quick test | 24 | 30' |
| B | Tính tay | 20 | 40' |
| C | Code (có test sẵn) | 32 | 70' |
| D | Thiết kế | 24 | 40' |

**Bài nào được kiểm tra ở câu nào:**
- 01 Tensor, autograd: A1, B2
- 02 Vòng lặp huấn luyện, scheduler: A2, D3
- 03 MLP, activation, đếm tham số: A3, B1
- 04 Vanishing gradient, khởi tạo, residual, clipping: A4, B3, D3
- 05 BatchNorm, LayerNorm, RMSNorm: A5, B1, B3
- 06 Dropout, weight decay, mixed precision, accumulation: A6, A7, D2, D3
- 07 CNN: A8, A9, B1, C1, D1
- 08 RNN, LSTM, GRU: A10, C2
- 09 Attention, beam search: A11, B4, C3
- 10 Transfer learning: A12, D1, D2

---

## A. Quick test (12 câu × 2đ)

**A1.** $f = x^2 y$ với `x = 3`, `y = 2`, cả hai có `requires_grad=True`. Sau `f.backward()`, `x.grad` và `y.grad` bằng bao nhiêu? Nếu tính lại $f$ và gọi `backward()` thêm một lần (không xóa gradient) thì sao?

**A2.** 50 000 mẫu, batch 128, huấn luyện 10 epoch. Có tổng cộng bao nhiêu bước cập nhật? Với `CosineAnnealingLR` từ $\eta_0 = 10^{-3}$ về 0 trên toàn bộ số bước đó, lr ở bước giữa chừng là bao nhiêu?

**A3.** Đếm số tham số của MLP 20 → 64 → 64 → 1 (có bias).

**A4.** Tính std của He init cho fan-in 1 024. Nếu thay bằng std 0.02 cho MLP ReLU rộng 1 024, hệ số std mỗi lớp là bao nhiêu, và tín hiệu sẽ tắt dần hay bùng nổ?

**A5.** BatchNorm có `momentum = 0.1`, `running_var` đang bằng 1, và một batch có var không lệch bằng 3. `running_var` mới bằng bao nhiêu? Khi suy luận với batch 1 cho một chatbot, bạn chọn BN hay LN? Vì sao?

**A6.** Inverted dropout với $p = 0.25$: một phần tử có giá trị 3 được giữ lại thì đầu ra lúc train là bao nhiêu? Lúc eval là bao nhiêu? Nêu một điểm khác nhau giữa `Adam(weight_decay=λ)` và `AdamW(weight_decay=λ)`.

**A7.** Ước lượng bộ nhớ cho tham số, gradient và trạng thái optimizer khi full fine-tune một model 350 triệu tham số bằng Adam và mixed precision.

**A8.** Ảnh 64×64, conv $k = 5$, $s = 2$, $p = 2$. Tính kích thước đầu ra. Tính số tham số của `nn.Conv2d(32, 64, 5)`.

**A9.** Tính receptive field của chuỗi conv3 → conv3 → maxpool 2 → conv3 (mọi conv stride 1).

**A10.** `nn.LSTM(input_size=32, hidden_size=64)` có bao nhiêu tham số? Nếu thêm `bidirectional=True` thì sao?

**A11.** Điểm attention (chưa softmax) là $(2, 2, 0)$. Tính trọng số. Nếu vị trí thứ hai bị mask thì trọng số thành bao nhiêu?

**A12.** Bạn có 300 ảnh sản phẩm lỗi thuộc 3 lớp, và một ResNet pretrain trên ImageNet có BatchNorm. Chọn chiến lược transfer nào? Khi đóng băng phần thân, cần làm gì với các lớp BN?

---

## B. Tính tay (20đ)

**B1 — CNN: shape, tham số, FLOP (6đ).** Đầu vào `(3, 64, 64)`:

`Conv2d(3, 32, 3, padding=1)` → `BatchNorm2d(32)` → `ReLU` → `MaxPool2d(2)` → `Conv2d(32, 64, 3, padding=1)` → `MaxPool2d(2)` → `Conv2d(64, 128, 3, stride=2, padding=1)` → `AdaptiveAvgPool2d(1)` → `Flatten` → `Linear(128, 10)`

- (a) Ghi shape sau mỗi lớp có thay đổi shape.
- (b) Tính số tham số của từng lớp có tham số, và tổng.
- (c) Tính FLOP của ba lớp conv và tổng.
- (d) Tính receptive field của một điểm ở đầu ra lớp conv cuối.
- (e) Nếu bỏ lớp conv thứ 3 và global pooling, rồi flatten trực tiếp `(64, 16, 16)` vào `Linear(·, 10)`, thì lớp Linear có bao nhiêu tham số? Nhận xét.

**B2 — Backprop qua softmax (5đ).** Lớp Linear $z = Wx + b$ với $x = (1, 2)$, $W = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & -1 \end{pmatrix}$, $b = 0$. Nhãn đúng là lớp **thứ 3** (chỉ số 2). Loss là cross-entropy.
- (a) Tính $z$, softmax $p$ và loss.
- (b) Tính $\nabla_z L$, $\nabla_W L$, $\nabla_b L$ và $\nabla_x L$.
- (c) Thực hiện một bước SGD với lr $= 0.5$, rồi tính lại $p$ và loss.

**B3 — Khởi tạo và chuẩn hóa (5đ).**
- (a) MLP ReLU rộng 512, đầu vào có std 1. Với ba cách khởi tạo: std 0.05, He, Xavier, tính std ở đầu ra của lớp Linear thứ 1 và thứ 10.
- (b) Tính LayerNorm và RMSNorm (không $\gamma, \beta$, $\epsilon = 0$) của vector $(2, -2, 4, 0)$.
- (c) Giải thích ngắn gọn vì sao kết quả của RMSNorm không có trung bình 0, và vì sao điều đó vẫn chấp nhận được trong LLM.

**B4 — Attention (4đ).** Query $q = (1, 1)$; key $k_1 = (1, 0)$, $k_2 = (0, 2)$, $k_3 = (1, 1)$; value $v_1 = (1, 0)$, $v_2 = (0, 1)$, $v_3 = (1, 1)$. Dùng scaled dot-product ($d = 2$).
- (a) Tính điểm số, trọng số và vector đầu ra.
- (b) Mask vị trí 3 rồi tính lại.
- (c) Nếu không chia $\sqrt d$, trọng số thành bao nhiêu? Với $d = 512$, điều gì sẽ xảy ra?

---

## C. Code (32đ)

Viết `code/final_lib.py`. Đứng ở thư mục tuần, chạy `python -m pytest -q code/test_final.py` phải qua hết **13 test**. **Không được sửa test.** Chỉ dùng NumPy cho C1 và C2 (PyTorch chỉ được dùng trong test để đối chiếu).

**C1 — Convolution bằng NumPy (12đ).**
- `conv_out_size(n, k, stride=1, padding=0, dilation=1)`: công thức ở bài 07, mục 7.2.
- `conv2d(x, w, b=None, stride=1, padding=0)`:
  - `x` có shape `(N, C, H, W)`, `w` có shape `(O, C, kh, kw)`, `b` có shape `(O,)`; trả về `(N, O, H_out, W_out)`.
  - Phép toán là **cross-correlation** như `F.conv2d`; padding là số 0.
  - Số kênh không khớp thì raise `ValueError`.
  - Gợi ý: lặp theo **vị trí trong kernel** $(u, v)$, chỉ $k_h k_w$ vòng; mỗi vòng cắt một "lát" của ảnh đã pad với bước `stride`, rồi dùng `np.einsum` hoặc `tensordot` theo chiều kênh.
- `maxpool2d(x, k, stride=None)`: `stride=None` nghĩa là bằng `k`; không padding; khớp `F.max_pool2d`.

**C2 — LSTM bằng NumPy (10đ).** `lstm_forward(x, h0, c0, W_ih, W_hh, b_ih, b_hh) -> (out, (h_T, c_T))`:
- `x` có shape `(T, B, d)`; `h0`, `c0` có shape `(B, H)`; `W_ih` có shape `(4H, d)`; `W_hh` có shape `(4H, H)`; `b_*` có shape `(4H,)`.
- Thứ tự các khối cổng là **i, f, g, o** (giống PyTorch); công thức ở bài 08, mục 8.2.
- `out` có shape `(T, B, H)` và chứa $h_t$ của mọi bước.
- Phải khớp `nn.LSTM` khi lấy đúng trọng số `weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0`, `bias_hh_l0`.

**C3 — Beam search (10đ).** `beam_search(step_logprobs, bos, eos, beam_width=3, max_len=10, alpha=0.0) -> (seq, score)`:
- `step_logprobs(prefix)` nhận một **tuple** token (bắt đầu bằng `bos`) và trả về dict `{token: log-prob}` của token kế tiếp.
- **Mỗi bước:** mở rộng mọi beam đang sống bằng mọi token; xếp **mọi** ứng viên theo **tổng log-prob** giảm dần; giữ `beam_width` ứng viên đầu. Ứng viên nào kết thúc bằng `eos` thì chuyển vào danh sách hoàn thành; số còn lại là các beam sống mới.
- **Dừng** khi không còn beam sống, hoặc sau `max_len` bước; khi đó các beam còn sống cũng được tính là hoàn thành.
- **Điểm** của chuỗi hoàn thành $= \dfrac{\text{tổng log-prob}}{n^{\alpha}}$, với $n$ là số token đã sinh (**tính cả** `eos`, **không tính** `bos`).
- Trả về chuỗi có điểm cao nhất (dạng `list`, **không** gồm `bos`) và điểm của nó (kiểu `float`).

Viết thêm 2 câu: vì sao `beam_width=1` chính là greedy? Vì sao không có chuẩn hóa độ dài (`alpha = 0`) thì beam search thiên về chuỗi ngắn?

---

## D. Thiết kế (24đ) — Phân loại ảnh hiện trường trạm BTS

**Bối cảnh** (mọi con số là **giả định**):
- Kỹ thuật viên chụp ảnh thiết bị trên cột anten bằng điện thoại khi đi kiểm tra định kỳ. Có **12 000 ảnh** (1920×1080) từ **1 500 trạm** (trung bình 8 ảnh/trạm), đã gán 5 nhãn: *bình thường* 80%, *rỉ sét* 8%, *cáp lỏng* 6%, *vật lạ bám trên cột* 4%, *hư anten* 2%.
- Mục tiêu: tự động **gắn cờ** ảnh bất thường để kỹ sư xem lại. Bỏ sót *hư anten* rất đắt (mất sóng cả khu vực); báo nhầm thì tốn 2 phút công kỹ sư.
- **Huấn luyện:** chỉ có Kaggle (GPU T4 16 GB, không hỗ trợ bf16 nhanh; tối đa 12 giờ mỗi phiên; 30 GPU-giờ mỗi tuần; 4 nhân CPU).
- **Triển khai:** on-prem, **chỉ CPU**, khoảng 2 000 ảnh mỗi giờ vào giờ cao điểm.
- Số liệu tham khảo (giả định):
  - ResNet-50 có 25.6 triệu tham số và 4.1 GFLOP mỗi ảnh 224×224; ResNet-18 có 11.7 triệu tham số và 1.8 GFLOP.
  - Trên T4 với fp16, huấn luyện ResNet-50 đạt khoảng 300 ảnh/s, và mỗi ảnh tốn khoảng 50 MB activation.
  - Mỗi nhân CPU giải mã và resize được khoảng 100 ảnh gốc mỗi giây.
  - CPU on-prem chạy conv đạt khoảng 300 GFLOP/s.

**D1 — Dữ liệu và model (8đ).**
- (a) Chia train/val/test thế nào? Vì sao chia ngẫu nhiên **theo ảnh** là sai?
- (b) Chọn 3 phép augmentation hợp lý và 1 phép **không** hợp lý cho bài toán này, kèm lý do.
- (c) Chọn model và chiến lược transfer (bài 10), và giải thích lựa chọn.
- (d) Một model "luôn đoán bình thường" đạt accuracy bao nhiêu? Đề xuất metric chính và cách xử lý mất cân bằng lớp.

**D2 — Ngân sách Kaggle (8đ).**
- (a) Ước lượng bộ nhớ cho tham số, gradient và trạng thái Adam của ResNet-50 khi huấn luyện, và batch size tối đa trên T4 (để dành khoảng 2 GB cho CUDA và framework).
- (b) Với tập train khoảng 70% dữ liệu, một epoch mất bao lâu? 30 epoch mất bao lâu? Một tuần quota cho phép khoảng bao nhiêu lần chạy thử như vậy?
- (c) Ảnh gốc 1920×1080 phải giải mã và resize mỗi epoch. Đây có phải nút thắt không? Đề xuất cách xử lý.
- (d) Phiên 12 giờ có thể bị ngắt giữa chừng. Cần lưu gì để chạy tiếp được, và vì sao chỉ lưu `model.state_dict()` là không đủ?
- (e) Dùng fp16 hay bf16 trên T4? Cần thêm thành phần nào để huấn luyện không bị `nan`?

**D3 — Chẩn đoán và triển khai (8đ).** Ba lần chạy cho ba triệu chứng khác nhau. Với mỗi triệu chứng, nêu nguyên nhân có khả năng nhất và cách sửa:
- (a) Train loss đứng yên ở **1.609** từ đầu. Nếu thay vào đó loss đứng yên ở **0.756** thì khác gì?
- (b) Val accuracy đạt 0.96, nhưng recall của lớp *hư anten* chỉ là 0.10.
- (c) Loss thành `nan` ở khoảng bước 800 khi bật fp16.

Và về triển khai:
- (d) Ước lượng thời gian suy luận mỗi ảnh trên CPU on-prem với ResNet-50 và với ResNet-18. Có đáp ứng 2 000 ảnh/giờ không? Nêu một cách giảm độ trễ nữa nếu cần.
