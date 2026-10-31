# Đáp án · Final Tuần 4

> Mọi con số của phần A, B, D đã được chạy kiểm chứng (phần D dùng các số liệu giả định trong đề). Code phần C đã chạy: 13/13 test pass.

## A. Quick test (24đ, 2đ/câu)

| Câu | Đáp án | Vì sao |
|---|---|---|
| A1 | `x.grad = 12`, `y.grad = 9`; sau lần thứ hai: **24** và **18** | $\partial f/\partial x = 2xy$, $\partial f/\partial y = x^2$; gradient **cộng dồn** vào `.grad` *(1đ mỗi ý)* |
| A2 | $391 \times 10 =$ **3 910** bước; lr giữa chừng $= 5 \times 10^{-4}$ | $\lceil 50\,000/128 \rceil = 391$; cosine tại $t = T/2$ là $\tfrac12\eta_0(1 + \cos\tfrac{\pi}{2})$ |
| A3 | **5 569** | $(20 \cdot 64 + 64) + (64 \cdot 64 + 64) + 65 = 1\,344 + 4\,160 + 65$ |
| A4 | He: **0.0442**; std 0.02 cho hệ số **0.452**, nên **tắt dần** | $\sqrt{2/1024}$; $\sqrt{1024 \cdot 0.0004/2} = \sqrt{0.2048}$ |
| A5 | **1.2**; chọn **LN** | $0.9 \cdot 1 + 0.1 \cdot 3$. BN với batch 1 không có thống kê batch (lúc train còn báo lỗi), và cách chuẩn hóa phụ thuộc vào batch; LN chuẩn hóa từng mẫu riêng *(1đ mỗi ý)* |
| A6 | Train: $3/0.75 =$ **4**; eval: **3** | `Adam(weight_decay)` cộng $\lambda\theta$ vào gradient (L2), nên phần đó bị chia cho $\sqrt{\hat v}$; `AdamW` trừ thẳng $\eta\lambda\theta$ khỏi trọng số |
| A7 | $16 \times 3.5 \times 10^8 =$ **5.6 GB** | 2 (trọng số 16-bit) + 4 (master fp32) + 2 (gradient) + 8 (Adam $m, v$) byte mỗi tham số; chưa tính activation |
| A8 | **32**; **51 264** | $\lfloor (64 + 4 - 5)/2 \rfloor + 1$; $64(32 \cdot 25 + 1)$ |
| A9 | **10** | $1 \to 3 \to 5 \to 6$ (pool, bước nhảy thành 2) $\to 6 + 2 \cdot 2$ |
| A10 | **25 088**; hai chiều: **50 176** | $4 \cdot 64(32 + 64 + 2)$ |
| A11 | $(0.468,\ 0.468,\ 0.063)$; mask vị trí 2: $(0.881,\ 0,\ 0.119)$ | $e^2 = 7.389$, tổng $15.78$; sau mask: $e^2/(e^2 + 1)$ |
| A12 | **Feature extraction** (đóng băng body, train head mới), sau đó có thể fine-tune block cuối với lr nhỏ; luôn so với baseline. BN của phần đóng băng phải ở **chế độ eval** (gọi `body.eval()` sau mỗi lần `model.train()`) để running stats không bị ghi đè | Ít dữ liệu; ảnh sản phẩm khá gần ảnh tự nhiên *(1đ mỗi ý)* |

## B. Tính tay (20đ)

**B1 (6đ)** *(đã kiểm chứng bằng chính model trong PyTorch)*

(a) Shape: *(1.5đ)*
- `(3, 64, 64)` → conv1: `(32, 64, 64)` (BN, ReLU giữ nguyên) → pool: `(32, 32, 32)`
- → conv2: `(64, 32, 32)` → pool: `(64, 16, 16)`
- → conv3 ($\lfloor (16 + 2 - 3)/2 \rfloor + 1 = 8$): `(128, 8, 8)`
- → global avg pool: `(128, 1, 1)` → flatten: `(128,)` → `(10,)`

(b) Tham số: *(1.5đ)*
- conv1: $32(3 \cdot 9 + 1) = 896$
- BN: $2 \cdot 32 = 64$ (thêm 64 buffer, không phải tham số)
- conv2: $64(32 \cdot 9 + 1) = 18\,496$
- conv3: $128(64 \cdot 9 + 1) = 73\,856$
- Linear: $1\,290$
- **Tổng 94 602**

(c) FLOP $= 2\,C_\text{in}k^2C_\text{out}H_\text{out}W_\text{out}$: *(1.5đ)*
- conv1: $2 \cdot 3 \cdot 9 \cdot 32 \cdot 64^2 = 7.08 \times 10^6$
- conv2: $2 \cdot 32 \cdot 9 \cdot 64 \cdot 32^2 = 37.7 \times 10^6$
- conv3: $2 \cdot 64 \cdot 9 \cdot 128 \cdot 8^2 = 9.44 \times 10^6$
- **Tổng ≈ 54.3 MFLOP**

Lớp có ít tham số nhất (conv1) không phải lớp rẻ nhất: chi phí phụ thuộc kích thước feature map.

(d) $r$: conv1 $\to 3$; pool $\to 4$ (bước nhảy 2); conv2 $\to 8$; pool $\to 10$ (bước nhảy 4); conv3 $\to 10 + 2 \cdot 4 =$ **18** (bước nhảy 8). *(0.75đ)*

(e) $64 \cdot 16 \cdot 16 \cdot 10 + 10 =$ **163 850** tham số, nhiều gấp 1.7 lần **toàn bộ** mạng ban đầu. Lớp này cũng gắn chặt vào kích thước ảnh 64×64 và không có bất biến khi dịch ảnh. Global pooling vừa rẻ vừa tổng quát hơn. *(0.75đ)*

**B2 (5đ)** *(đã kiểm chứng bằng autograd)*

(a) $z = (1,\ 2,\ 1 - 2) = (1,\ 2,\ -1)$. $e^z = (2.718,\ 7.389,\ 0.368)$, tổng $10.475$, nên $p = (0.259,\ 0.705,\ 0.035)$. $L = -\ln 0.0351 =$ **3.349**. Model đang rất chắc vào lớp sai. *(1.5đ)*

(b) $\nabla_z L = p - y = (0.259,\ 0.705,\ -0.965)$. *(2đ)*
- $\nabla_W L = (p - y)\,x^\top = \begin{pmatrix} 0.259 & 0.519 \\ 0.705 & 1.411 \\ -0.965 & -1.930 \end{pmatrix}$
- $\nabla_b L = (0.259,\ 0.705,\ -0.965)$
- $\nabla_x L = W^\top(p - y) = (0.259 - 0.965,\ 0.705 + 0.965) = (-0.705,\ 1.670)$

(c) $W \leftarrow W - 0.5\nabla_W$: $\begin{pmatrix} 0.870 & -0.259 \\ -0.353 & 0.295 \\ 1.482 & -0.035 \end{pmatrix}$, $b = (-0.130,\ -0.353,\ 0.482)$. Khi đó $z = (0.222,\ -0.116,\ 1.895)$, $p = (0.142,\ 0.101,\ 0.757)$, $L =$ **0.279** (giảm từ 3.349). Một bước với lr 0.5 là rất mạnh: hàng của lớp đúng được kéo về phía $x$, các hàng khác bị đẩy ra xa. *(1.5đ)*

**B3 (5đ)** *(đã kiểm chứng)*

(a) Lớp 1 có std $\sqrt{512\,\mathrm{Var}(w)}$; hệ số mỗi lớp là $\sqrt{512\,\mathrm{Var}(w)/2}$. *(2.5đ)*

| Init | $\mathrm{Var}(w)$ | Lớp 1 | Hệ số | Lớp 10 |
|---|---|---|---|---|
| std 0.05 | 0.0025 | 1.131 | 0.8 | $1.131 \cdot 0.8^9 =$ **0.152** |
| He | $2/512$ | 1.414 | 1 | **1.414** |
| Xavier | $2/1024$ | 1 | 0.707 | $0.707^9 =$ **0.044** |

(b) *(1.5đ)*
- **LayerNorm:** $\mu = 1$, độ lệch là $(1, -3, 3, -1)$, $\sigma^2 = 20/4 = 5$, $\sigma = 2.236$. Kết quả: **$(0.447,\ -1.342,\ 1.342,\ -0.447)$**.
- **RMSNorm:** $\mathrm{rms} = \sqrt{(4 + 4 + 16 + 0)/4} = \sqrt 6 = 2.449$. Kết quả: **$(0.816,\ -0.816,\ 1.633,\ 0)$**.

(c) RMSNorm chỉ chia cho độ lớn, không trừ trung bình, nên trung bình của kết quả bằng $\mu/\mathrm{rms} = 1/2.449 = 0.41 \ne 0$. Chấp nhận được vì phần quan trọng cho ổn định huấn luyện là **giữ độ lớn** của activation ổn định (*re-scaling invariance*); bước "re-centering" đóng góp ít. Bỏ nó đi thì rẻ hơn mà chất lượng tương đương (Zhang & Sennrich 2019). Vì vậy LLaMA và các LLM mới dùng RMSNorm. *(1đ)*

**B4 (4đ)** *(đã kiểm chứng)*

(a) $q \cdot k = (1,\ 2,\ 2)$, chia $\sqrt 2$ được $(0.707,\ 1.414,\ 1.414)$. Trọng số $(0.198,\ 0.401,\ 0.401)$. Đầu ra $= 0.198(1, 0) + 0.401(0, 1) + 0.401(1, 1) =$ **$(0.599,\ 0.802)$**. *(2đ)*

(b) Mask vị trí 3: softmax của $(0.707, 1.414)$ là $(0.330,\ 0.670,\ 0)$; đầu ra **$(0.330,\ 0.670)$**. *(1đ)*

(c) Không scale: softmax của $(1, 2, 2)$ là $(0.155,\ 0.422,\ 0.422)$, "nhọn" hơn một chút. Với $d = 512$, điểm số có std khoảng $\sqrt{512} \approx 22.6$ (nếu các thành phần có var 1), nên softmax gần như one-hot: đo được trọng số lớn nhất trung bình là 0.953 khi không scale, 0.319 khi có scale (bài 09, mục 9.4). Gradient tới các vị trí khác gần 0, nên khó học. *(1đ)*

## C. Code (32đ)

Chấm theo test: **C1** 7 test (12đ) · **C2** 3 test (10đ) · **C3** 3 test (8đ) + 2 câu trả lời (2đ).

```python
import numpy as np


# ---------- C1 ----------
def conv_out_size(n, k, stride=1, padding=0, dilation=1):
    return (n + 2 * padding - dilation * (k - 1) - 1) // stride + 1


def conv2d(x, w, b=None, stride=1, padding=0):
    """x (N, C, H, W), w (O, C, kh, kw), b (O,) → (N, O, H_out, W_out). Cross-correlation như F.conv2d."""
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    N, C, H, W = x.shape
    O, C2, kh, kw = w.shape
    if C != C2:
        raise ValueError("số kênh của x và w không khớp")
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    Ho, Wo = conv_out_size(H, kh, stride, padding), conv_out_size(W, kw, stride, padding)
    out = np.zeros((N, O, Ho, Wo))
    for u in range(kh):                                            # lặp theo vị trí trong kernel: kh·kw vòng
        for v in range(kw):
            patch = xp[:, :, u:u + stride * Ho:stride, v:v + stride * Wo:stride]   # (N, C, Ho, Wo)
            out += np.einsum("nchw,oc->nohw", patch, w[:, :, u, v])
    if b is not None:
        out += np.asarray(b, dtype=float)[None, :, None, None]
    return out


def maxpool2d(x, k, stride=None):
    x = np.asarray(x, dtype=float)
    s = stride or k
    N, C, H, W = x.shape
    Ho, Wo = conv_out_size(H, k, s), conv_out_size(W, k, s)
    out = np.full((N, C, Ho, Wo), -np.inf)
    for u in range(k):
        for v in range(k):
            out = np.maximum(out, x[:, :, u:u + s * Ho:s, v:v + s * Wo:s])
    return out


# ---------- C2 ----------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def lstm_forward(x, h0, c0, W_ih, W_hh, b_ih, b_hh):
    """x (T, B, d); h0, c0 (B, H); W_ih (4H, d); W_hh (4H, H); b_* (4H,). Thứ tự cổng i, f, g, o."""
    x = np.asarray(x, dtype=float)
    h, c = np.asarray(h0, dtype=float), np.asarray(c0, dtype=float)
    H = h.shape[1]
    outs = []
    for t in range(x.shape[0]):
        z = x[t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh               # (B, 4H)
        i, f = _sigmoid(z[:, :H]), _sigmoid(z[:, H:2 * H])
        g, o = np.tanh(z[:, 2 * H:3 * H]), _sigmoid(z[:, 3 * H:])
        c = f * c + i * g
        h = o * np.tanh(c)
        outs.append(h)
    return np.stack(outs), (h, c)


# ---------- C3 ----------
def beam_search(step_logprobs, bos, eos, beam_width=3, max_len=10, alpha=0.0):
    beams, finished = [((bos,), 0.0)], []
    for _ in range(max_len):
        cands = [(prefix + (tok,), lp + tlp) for prefix, lp in beams for tok, tlp in step_logprobs(prefix).items()]
        cands.sort(key=lambda c: -c[1])                            # xếp theo TỔNG log-prob
        top = cands[:beam_width]
        finished += [c for c in top if c[0][-1] == eos]
        beams = [c for c in top if c[0][-1] != eos]
        if not beams:
            break
    finished += beams                                              # beam bị cắt ở max_len
    scored = [(seq[1:], lp / (len(seq) - 1) ** alpha) for seq, lp in finished]
    seq, sc = max(scored, key=lambda c: c[1])
    return list(seq), float(sc)
```

**Hai câu trả lời của C3:**
1. Với `beam_width = 1`, mỗi bước chỉ giữ **một** ứng viên có tổng log-prob lớn nhất. Vì mọi ứng viên đều kéo dài từ cùng một tiền tố, đó chính là token có log-prob lớn nhất ở bước đó: **greedy**.
2. Mỗi token thêm vào cộng một log-prob **âm**, nên chuỗi dài luôn có tổng nhỏ hơn. Một chuỗi kết thúc sớm (`eos` có xác suất vừa phải) dễ thắng các chuỗi dài hơn dù từng token của chuỗi dài đều hợp lý. Test `test_length_normalization_changes_choice`: với $\alpha = 0$, kết quả là `["</s>"]` ($\ln 0.4 = -0.92$); với $\alpha = 1$, kết quả là `["a", "</s>"]` ($\ln 0.36/2 = -0.51$).

**Lỗi hay gặp:**
- C1: quên padding khi cắt lát (sai shape); dùng `w[:, :, ::-1, ::-1]` (đó là convolution "toán học", không phải cross-correlation của PyTorch); slice `u:u + stride*Ho:stride` bị lệch một.
- C2: sai thứ tự cổng (i, f, **g**, o); quên một trong hai vector bias; dùng `W_ih @ x` sai chiều.
- C3: xếp theo **điểm đã chuẩn hóa** ở mỗi bước (đề yêu cầu tổng log-prob); quên thêm các beam còn sống vào danh sách hoàn thành khi hết `max_len`; tính độ dài có cả `bos`.

## D. Thiết kế (24đ)

**D1 (8đ)**

(a) Chia **theo trạm** (group split): ví dụ 70/15/15 số trạm cho train/val/test, khoảng 8 400 / 1 800 / 1 800 ảnh. Nên chia theo kiểu stratified sao cho mỗi phần đều có đủ các lớp hiếm (test 15% có khoảng 36 ảnh *hư anten*). Ảnh cùng một trạm có **cùng nền**, cùng góc chụp, cùng thiết bị, nhiều khi là cùng một lỗi được chụp nhiều lần. Chia theo ảnh thì model "nhận ra trạm" thay vì nhận ra lỗi, nên điểm test bị thổi phồng (Tuần 3, bài 01 và 03: GroupKFold). Nếu dữ liệu kéo dài nhiều năm thì cân nhắc thêm việc chia theo thời gian. *(2đ)*

(b) *(2đ)*
- **Hợp lý:** lật ngang (thiết bị bên trái hay bên phải cột không đổi nhãn); đổi độ sáng/độ tương phản (nắng, mây, ngược sáng); random resized crop với tỉ lệ vừa phải (người chụp đứng xa gần khác nhau); xoay nhẹ ±10–15° (điện thoại cầm nghiêng); làm mờ nhẹ (rung tay).
- **Không hợp lý:**
  - Jitter **hue/màu mạnh**: rỉ sét được nhận ra chủ yếu nhờ màu nâu đỏ, đổi màu thì mất đúng dấu hiệu của nhãn.
  - **Crop mạnh**: có thể cắt mất vùng lỗi nhỏ (một mối nối cáp lỏng), khiến nhãn "bất thường" thành sai.
  - Lật dọc hoặc xoay 180° cũng không giống bất kỳ ảnh thật nào.

(c) ResNet-50 pretrained trên ImageNet (ảnh tự nhiên, khá gần miền). Có khoảng 8 400 ảnh train, nên đi theo hai giai đoạn *(2đ)*:
1. Feature extraction: đóng băng body (BN ở chế độ eval), train head mới vài epoch.
2. Fine-tune toàn bộ với discriminative LR (body khoảng $10^{-4}$, head khoảng $10^{-3}$), có warmup và early stopping theo metric ở (d).

Lỗi thường nhỏ trong ảnh 1920×1080, nên cân nhắc độ phân giải 384–448 thay vì 224 (đánh đổi bộ nhớ, xem D2). Nếu CPU lúc triển khai là giới hạn, so sánh với ResNet-18 trên cùng split.

(d) "Luôn bình thường" đạt accuracy **0.80** mà không bắt được lỗi nào. *(2đ)*
- **Metric chính:** recall của từng lớp bất thường (đặc biệt *hư anten*) tại một mức precision chấp nhận được, hoặc chi phí kỳ vọng (bỏ sót × giá + báo nhầm × 2 phút công). Báo cáo kèm macro-F1, PR-AUC theo từng lớp, và ma trận nhầm lẫn.
- **Xử lý mất cân bằng:**
  - Class weight trong cross-entropy (`balanced`: 0.25, 2.5, 3.33, 5, 10), hoặc oversampling bằng `WeightedRandomSampler`.
  - Chọn ngưỡng **riêng cho từng lớp** trên val theo chi phí (Tuần 3, bài 04).
  - Có thể thêm một head nhị phân "bất thường / bình thường" để gắn cờ.

**D2 (8đ)**

(a) Tham số + gradient + Adam: $16 \times 25.6 \times 10^6 \approx$ **0.41 GB**. Activation: $(16 - 0.41 - 2)/0.05 \approx 271$ ảnh, nên chọn **batch 128** (hoặc 256 nếu thử thấy vừa) ở 224×224. Ở 384×384, activation tăng $(384/224)^2 \approx 2.9$ lần, nên batch giảm còn khoảng 64–90; nếu cần batch hiệu dụng lớn hơn thì dùng gradient accumulation (ResNet có BN, nên BN vẫn chỉ thấy micro-batch). *(2đ)*

(b) $8\,400/300 = 28$ s mỗi epoch (thêm khoảng 2 s cho val 1 800 ảnh), nên 30 epoch mất khoảng **15 phút**. 30 GPU-giờ mỗi tuần cho khoảng **120 lần chạy** về lý thuyết; thực tế ít hơn (khởi động, nạp dữ liệu, đánh giá), cỡ 50–80 lần. Đủ cho một vòng tune nhỏ (random search khoảng 20–60 cấu hình, Tuần 3, bài 11). *(2đ)*

(c) 4 nhân × 100 ảnh/s = 400 ảnh/s, sát với 300 ảnh/s mà GPU tiêu thụ khi train, và **chậm hơn** GPU khi chạy val/suy luận. Thêm augmentation trên CPU nữa thì GPU sẽ phải chờ. **Xử lý:** resize **một lần** trước khi train (ví dụ cạnh ngắn 448 px), lưu lại thành JPEG nhỏ hoặc mảng `uint8` (12 000 ảnh ở 400 ảnh/s mất khoảng 30 s); dùng `DataLoader(num_workers=4, pin_memory=True, persistent_workers=True)`; có thể làm augmentation trên GPU. *(1.5đ)*

(d) Mỗi epoch (hoặc mỗi N bước), lưu vào `/kaggle/working` *(1.5đ)*:
- `model.state_dict()`;
- `optimizer.state_dict()` (Adam $m, v$ và số bước);
- trạng thái **scheduler** (đang ở đâu trên đường cosine/warmup);
- trạng thái **`GradScaler`** (scale hiện tại);
- số epoch/bước, metric tốt nhất, và trạng thái RNG.

Chỉ lưu model thì khi chạy tiếp, Adam bắt đầu lại từ $m = v = 0$, lr bị đặt lại về đầu lịch (warmup lần hai), và scaler bị đặt lại. Loss sẽ nhảy và kết quả không tái lập được (bài 02, cạm bẫy 6).

(e) **fp16** (T4 có Tensor Core cho fp16, không hỗ trợ bf16 nhanh), kèm **`torch.amp.GradScaler`** (loss scaling) để gradient nhỏ không bị làm tròn về 0 và overflow được phát hiện. Phải `unscale_` trước khi clip. *(1đ)*

**D3 (8đ)**

(a) $1.609 = \ln 5$: model cho ra **phân phối đều** trên 5 lớp, tức chưa học được gì, kể cả tỉ lệ nền. Gradient không tới được head. Nghi phạm số 1 với transfer learning: **tạo optimizer trước khi thay `model.fc`**, nên tham số của head mới không nằm trong optimizer; hoặc đóng băng nhầm cả head; hoặc lr = 0. *(2.5đ)*

$0.756 = H(0.8, 0.08, 0.06, 0.04, 0.02)$: model đã học được **tỉ lệ nền** (bias của head) nhưng không dùng được nội dung ảnh. Nghi phạm:
- Đầu vào hỏng: sai mean/std, ảnh toàn đen, quên chia 255.
- Nhãn bị xáo trộn lệch thứ tự so với ảnh.
- lr quá lớn làm đặc trưng chết, hoặc dropout/regularization quá mạnh.

Cách kiểm tra chung cho cả hai: in `count_params` của các tham số có `requires_grad` và so với các tham số trong `optimizer.param_groups`; chạy **overfit một batch** (quiz Ngày 1, T4).

(b) Mất cân bằng: accuracy bị chi phối bởi lớp *bình thường* (80%). Lớp *hư anten* chỉ có khoảng 170 ảnh train và có thể bị nhầm với *vật lạ*. Sửa bằng *(2đ)*:
- Class weight hoặc oversampling; focal loss (Lin et al., ICCV 2017).
- Ngưỡng riêng cho lớp này chọn trên val theo chi phí (bỏ sót rất đắt, nên ngưỡng thấp).
- Thu thập thêm ảnh lớp hiếm, ưu tiên những ảnh model chưa chắc chắn (active learning).
- Đánh giá bằng recall theo từng lớp chứ không bằng accuracy; xem ma trận nhầm lẫn để biết nhầm với lớp nào.

(c) Overflow của fp16 (giá trị tối đa 65 504): thường do không dùng `GradScaler`, clip khi gradient còn đang bị scale, lr quá lớn hoặc thiếu warmup, hoặc tự tính `log(softmax)` trong fp16. Sửa: dùng `GradScaler`, `unscale_` rồi mới clip, thêm warmup, dùng `F.cross_entropy` nhận logit (autocast tự tính phần này ở fp32), kiểm tra ảnh có giá trị `inf`/`nan` không. *(1.5đ)*

(d) ResNet-50: $4.1 \times 10^9 / 3 \times 10^{11} \approx$ **14 ms** mỗi ảnh về lý thuyết, thực tế khoảng 30–50 ms. ResNet-18: khoảng **6 ms** lý thuyết. Nhu cầu 2 000 ảnh/giờ, tức **0.56 ảnh/s**: dư hàng chục lần, kể cả chỉ dùng một nhân CPU. Phần tốn thời gian thật có thể là **giải mã và resize** ảnh 1920×1080 (khoảng 10 ms/ảnh/nhân theo số liệu đề). *(2đ)*

Nếu cần nhanh hơn (ví dụ phải trả kết quả ngay trên ứng dụng của kỹ thuật viên):
- Export sang ONNX Runtime hoặc OpenVINO.
- Lượng tử hóa INT8.
- Chạy theo batch.
- Distill sang ResNet-18 hoặc MobileNet.
- Giảm độ phân giải đầu vào nếu metric cho phép.
