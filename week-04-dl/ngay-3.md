# Ngày 3 · Kiến trúc: CNN, RNN/LSTM/GRU, attention trước Transformer, transfer learning

> **Loại tài liệu: Lý thuyết (bản nhẹ).** Công thức đầy đủ và ví dụ tính tay. Chứng minh chỉ còn phác thảo vài dòng ("vì sao"). Bài 10 nghiêng về Kỹ thuật.
> **Hôm nay gồm 4 bài:**
> - **07** CNN: convolution, kích thước đầu ra, receptive field, pooling, các kiến trúc kinh điển
> - **08** RNN, LSTM, GRU: xử lý chuỗi và vì sao LSTM nhớ được lâu
> - **09** Seq2seq, attention (Bahdanau/Luong) và beam search: cầu nối sang Transformer
> - **10** Transfer learning: pretrain, feature extraction, fine-tuning
>
> **Thời gian:** đọc ~3h · quiz 45' · bài tập ~2h
> **Ý chính của cả ngày:** mỗi kiến trúc là một **giả định có sẵn** (inductive bias) về dữ liệu. CNN giả định "mẫu hình cục bộ, lặp lại ở mọi vị trí"; RNN giả định "thứ tự quan trọng, xử lý tuần tự"; attention giả định "mỗi vị trí được tự chọn nên nhìn vào đâu". Mọi thí nghiệm chạy trên CPU trong vài giây đến vài phút.

---

## 07 · CNN: convolution, pooling, receptive field

> ⏱ Tầng 1: 15' · Tầng 2: 40'
> **Cần biết trước:** MLP và đếm tham số (bài 03); augmentation (bài 06).

> **📖 Từ mới**
> - **Convolution (tích chập):** trượt một ma trận nhỏ (**kernel/filter**) trên ảnh; tại mỗi vị trí, nhân từng phần tử rồi cộng lại.
> - **Channel (kênh):** chiều "độ sâu" của ảnh (RGB có 3 kênh), hoặc số filter của một lớp.
> - **Padding:** thêm viền số 0 quanh ảnh. **Stride:** bước nhảy của kernel.
> - **Feature map:** ảnh đầu ra của một filter.
> - **Pooling:** thu nhỏ feature map bằng cách lấy max (hoặc trung bình) trong từng ô nhỏ.
> - **Receptive field:** vùng của ảnh **gốc** mà một điểm ở lớp sâu "nhìn thấy".

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Soát một bức ảnh lớn bằng **kính lúp nhỏ**:
- Kính lúp có in sẵn một mẫu hình, ví dụ "cạnh dọc". Bạn trượt nó khắp ảnh; ở đâu khớp thì ghi điểm cao. Đó là **một filter**.
- **Cùng một** kính lúp được dùng ở mọi vị trí: cạnh dọc ở góc trái hay góc phải đều là cạnh dọc. Đó là **chia sẻ trọng số** (weight sharing), và đó là lý do CNN cần **ít tham số** hơn MLP rất nhiều.
- Lớp sau đọc "bản đồ cạnh" của lớp trước để tìm "góc", rồi "mắt", rồi "mặt". Mỗi lớp nhìn được **vùng rộng hơn** của ảnh gốc.

**Ví dụ có số.** Ảnh 4×4 và kernel "cạnh dọc" $K = \begin{pmatrix} 1 & 0 & -1 \\ 1 & 0 & -1 \\ 1 & 0 & -1 \end{pmatrix}$:

$$X = \begin{pmatrix} 1 & 2 & 0 & 1 \\ 0 & 1 & 3 & 1 \\ 2 & 1 & 0 & 0 \\ 1 & 0 & 1 & 2 \end{pmatrix}$$

Ô trên-trái của đầu ra (không padding): cột trái của cửa sổ 3×3 đầu tiên là $(1, 0, 2)$ và cột phải là $(0, 3, 0)$, nên kết quả là $(1 + 0 + 2) - (0 + 3 + 0) = 0$. Làm cho cả 4 vị trí được đầu ra 2×2 $= \begin{pmatrix} 0 & 2 \\ -1 & -1 \end{pmatrix}$ (đã kiểm chứng bằng `F.conv2d`).

**Ba ý cốt lõi:**
1. Convolution = **cục bộ** (mỗi đầu ra chỉ nhìn một vùng nhỏ) + **chia sẻ trọng số** (cùng filter ở mọi vị trí).
2. Kích thước đầu ra phụ thuộc kernel, padding, stride: **phải tính được bằng tay**.
3. Chồng nhiều lớp nhỏ (3×3) thì receptive field lớn dần; pooling/stride làm nó lớn nhanh hơn.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 7.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **2D convolution (cross-correlation)** | $Y_{o,i,j} = b_o + \sum_{c}\sum_{u,v} W_{o,c,u,v}\,X_{c,\,i s + u,\,j s + v}$. Trong DL, "convolution" thực ra là **cross-correlation** (không lật kernel) | Goodfellow §9.1 |
| **Equivariance / invariance** | Dịch ảnh thì feature map **dịch theo** (equivariance của conv); sau pooling hoặc global pooling, đầu ra **gần như không đổi** (invariance, chỉ một phần) | Goodfellow §9.2–9.3 |
| **Pooling** | Max/average trong cửa sổ $k \times k$ với stride $s$. **Global average pooling**: trung bình toàn bộ feature map, còn 1 số cho mỗi kênh | Lin et al., *Network in Network*, 2014 |
| **1×1 convolution** | Tổ hợp tuyến tính các kênh tại từng pixel; dùng để đổi số kênh rẻ | Lin et al. 2014 |
| **Depthwise separable conv** | Mỗi kênh một kernel riêng (`groups=C`), rồi một 1×1 conv để trộn kênh; rẻ hơn nhiều lần | Chollet, *Xception*, CVPR 2017; MobileNet 2017 |
| **Dilation** | Kernel có "lỗ": khoảng cách giữa các phần tử là $d$; tăng receptive field mà không tăng tham số | Yu & Koltun, ICLR 2016 |

#### 7.2 Công thức

**Kích thước đầu ra** (đúng như docs của `nn.Conv2d`):

$$n_\text{out} = \left\lfloor \frac{n + 2p - d(k - 1) - 1}{s} \right\rfloor + 1$$

Với $d = 1$, công thức gọn lại thành $\lfloor (n + 2p - k)/s \rfloor + 1$. "**Same padding**" với $s = 1$ cần $p = (k - 1)/2$ (kernel 3 thì $p = 1$).

**Số tham số:** $C_\text{out}(C_\text{in}k^2 + 1)$. **Không phụ thuộc** kích thước ảnh.

**FLOP:** $2\,C_\text{in}k^2\,C_\text{out}\,H_\text{out}W_\text{out}$.

**Receptive field** qua các lớp (kernel $k_l$, stride $s_l$): $r_l = r_{l-1} + (k_l - 1)\,j_{l-1}$, với $j_l = j_{l-1}s_l$ là "bước nhảy tích lũy" và $r_0 = j_0 = 1$.

*Vì sao VGG chỉ dùng 3×3?* Ba lớp 3×3 có receptive field **7×7**, giống một lớp 7×7, nhưng chỉ cần $3 \cdot 9C^2 = 27C^2$ tham số so với $49C^2$, và có **3** lần phi tuyến thay vì 1 (Simonyan & Zisserman 2015).

#### 7.3 Ví dụ tính tay

**(a) Padding, stride, pooling** trên $X$ ở tầng 1 (đã kiểm chứng):
- Padding 1, stride 1: đầu ra 4×4 $= \begin{pmatrix} -3 & -2 & 1 & 3 \\ -4 & 0 & 2 & 3 \\ -2 & -1 & -1 & 4 \\ -1 & 2 & -1 & 1 \end{pmatrix}$.
- Padding 1, stride 2: $\lfloor (4 + 2 - 3)/2 \rfloor + 1 = 2$, nên đầu ra 2×2 $= \begin{pmatrix} -3 & 1 \\ -2 & -1 \end{pmatrix}$ (lấy các vị trí chẵn của kết quả trên).
- Max-pool 2×2 trên $X$: $\begin{pmatrix} 2 & 3 \\ 2 & 2 \end{pmatrix}$. Average-pool: $\begin{pmatrix} 1 & 1.25 \\ 1 & 0.75 \end{pmatrix}$.

**(b) Kích thước đầu ra:**

| Đầu vào | Lớp | Đầu ra |
|---|---|---|
| 224 | conv $k = 7$, $s = 2$, $p = 3$ (lớp đầu của ResNet) | $\lfloor (224 + 6 - 7)/2 \rfloor + 1 = 112$ |
| 112 | max-pool $k = 3$, $s = 2$, $p = 1$ | 56 |
| 32 | conv $k = 5$, $p = 0$ (LeNet) | 28 |
| 28 | conv $k = 3$, $s = 2$, $p = 1$ | 14 |
| 10 | conv $k = 3$, dilation 2 | $10 - 2 \cdot 2 = 6$ |

**(c) Số tham số** (đã kiểm chứng bằng `numel`):

| Lớp | Phép tính | Tham số |
|---|---|---|
| Conv 3 → 64, 3×3 | $64(3 \cdot 9 + 1)$ | 1 792 |
| Conv 64 → 128, 3×3 | $128(64 \cdot 9 + 1)$ | 73 856 |
| Conv 1×1, 256 → 64 | $64(256 + 1)$ | 16 448 |
| Depthwise 3×3, 64 kênh (`groups=64`) | $64(9 + 1)$ | 640 |
| **Linear** nối ảnh 224×224×3 với 64 neuron | $150\,528 \cdot 64 + 64$ | **9 633 856** |

Một lớp Linear trên ảnh thô tốn gấp **5 000 lần** một lớp conv 3 → 64. Đó là lý do MLP không dùng được cho ảnh lớn.

**(d) LeNet-5** (đầu vào 32×32):
- conv 5×5 (1 → 6) cho 28×28 (156 tham số); avg-pool cho 14×14;
- conv 5×5 (6 → 16) cho 10×10 (2 416); avg-pool cho 5×5;
- flatten 400 → 120 (48 120) → 84 (10 164) → 10 (850).
- **Tổng 61 706**, trong đó 95% nằm ở các lớp Linear.

**(e) FLOP:** conv 3 → 64, 3×3, padding 1 trên ảnh 224×224: $2 \cdot 3 \cdot 9 \cdot 64 \cdot 224 \cdot 224 \approx 1.7 \times 10^8$. Chỉ 1 792 tham số nhưng tốn 0.17 GFLOP: với CNN, **chi phí tính toán** chứ không phải số tham số mới là giới hạn.

**(f) Receptive field.** Ba lớp 3×3 stride 1 cho $1 + 2 + 2 + 2 = 7$. CNN của thí nghiệm dưới (conv3, conv3, maxpool 2, conv3): $r = 1 + 2 + 2 + 1 \cdot 1 + 2 \cdot 2 = 10 \ge 8$, nên mỗi điểm ở lớp cuối nhìn được **toàn bộ** ảnh 8×8.

#### 7.4 Đo trên máy bạn: MLP và CNN trên `load_digits`

1 257 ảnh train, 540 ảnh test, Adam 60 epoch. "Dịch $k$ pixel" là dịch ảnh test sang phải $k$ pixel, điền 0:

| Model | Tham số | Acc test | Dịch 1 px | Dịch 2 px | Thời gian |
|---|---|---|---|---|---|
| MLP 64–128–10 | 9 610 | 0.976 | 0.487 | 0.133 | 3.0 s |
| CNN (conv16, conv32, maxpool, conv32, global avg pool, Linear) | 14 378 | **0.989** | **0.835** | 0.411 | 4.7 s |

- Trên ảnh sạch hai model gần nhau, vì ảnh 8×8 rất nhỏ.
- Khi ảnh bị dịch, MLP sụp đổ: với MLP, pixel $(3, 4)$ và pixel $(3, 5)$ là hai đầu vào **không liên quan**. CNN giữ được nhiều hơn nhờ chia sẻ trọng số và pooling. Nhưng nó cũng **không bất biến hoàn toàn** (0.835, không phải 0.989): pooling chỉ cho bất biến một phần, và viền ảnh bị cắt.
- So với bài 06: augmentation giúp MLP đạt 0.937 khi dịch 1 px. Hai cách tạo bất biến: **dạy** qua dữ liệu, hoặc **cài sẵn** vào kiến trúc.

#### 7.5 Các kiến trúc kinh điển

| Kiến trúc | Năm | Tham số | Ý tưởng chính |
|---|---|---|---|
| **LeNet-5** | 1998 | ~60 nghìn | Conv + pool + FC; đọc chữ số trên séc ngân hàng (LeCun et al., *Proc. IEEE* 1998) |
| **AlexNet** | 2012 | ~61 triệu | ReLU, dropout, huấn luyện trên GPU, augmentation; top-5 error ImageNet 15.3% so với 26% của hạng nhì. Khởi đầu thời kỳ deep learning (Krizhevsky et al., NeurIPS 2012) |
| **VGG-16** | 2014 | ~138 triệu | Chỉ dùng 3×3, rất sâu và đều (Simonyan & Zisserman, ICLR 2015) |
| **ResNet-50** | 2015 | ~25.6 triệu | Residual connection (bài 04), bottleneck 1×1–3×3–1×1, BatchNorm; huấn luyện được 152 lớp (He et al., CVPR 2016). ResNet-18: ~11.7 triệu |
| **EfficientNet / ConvNeXt** | 2019 / 2022 | — | Co giãn đồng thời độ sâu, độ rộng, độ phân giải; ConvNeXt "hiện đại hóa" ResNet theo thiết kế của Transformer |

Ứng dụng: phân loại ảnh; phát hiện vật thể (YOLO, Faster R-CNN: thêm nhánh dự đoán khung bao); phân đoạn (U-Net); và cả âm thanh (phổ tần là một "ảnh").

#### 7.6 Code chuẩn

```python
from torch import nn

cnn = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),     # (B, 1, 8, 8) → (B, 16, 8, 8)
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),                # → (B, 32, 8, 8)
    nn.MaxPool2d(2),                                           # → (B, 32, 4, 4)
    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),                     # → (B, 32): không phụ thuộc kích thước ảnh
    nn.Linear(32, 10),
)
x = torch.randn(4, 1, 8, 8)                                    # PyTorch dùng (N, C, H, W)
print(cnn(x).shape)                                            # torch.Size([4, 10])
```

#### 7.7 Cạm bẫy

1. **Sai thứ tự chiều:** PyTorch dùng `(N, C, H, W)`; ảnh đọc bằng PIL/NumPy là `(H, W, C)`. Phải `permute(2, 0, 1)`.
2. **Tính sai kích thước trước lớp Linear** sau khi flatten: dùng `AdaptiveAvgPool2d` hoặc `nn.LazyLinear`, hoặc chạy thử một tensor giả để in shape.
3. **Quên chuẩn hóa ảnh theo mean/std mà model pretrained đã dùng** (ImageNet: mean $(0.485, 0.456, 0.406)$, std $(0.229, 0.224, 0.225)$), bài 10.
4. **Augmentation không hợp với bài toán:** lật ngang ảnh chữ, xoay ảnh X-quang có quy ước trái/phải.
5. **Tin rằng CNN bất biến hoàn toàn với phép dịch:** thí nghiệm trên cho thấy không (0.835).

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"Vì sao CNN cần ít tham số hơn MLP?"* Kết nối cục bộ và chia sẻ trọng số; số tham số không phụ thuộc kích thước ảnh.
- *"Tính shape đầu ra và số tham số của lớp này."* Gần như luôn có trong phỏng vấn CV: mục 7.2.
- *"Vì sao 1×1 conv có ích?"* Đổi số kênh rẻ (bottleneck của ResNet), và thêm phi tuyến.

#### 7.8 Đọc thêm

- Goodfellow ch. 9 (Convolutional Networks).
- CS231n notes, *Convolutional Neural Networks*.
- V. Dumoulin & F. Visin, *A Guide to Convolution Arithmetic for Deep Learning*, arXiv 2016 (hình minh họa padding/stride rất rõ).
- K. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.

---

## 08 · RNN, LSTM, GRU

> ⏱ Tầng 1: 15' · Tầng 2: 55'
> **Cần biết trước:** vanishing gradient (bài 04); sigmoid làm cổng (bài 03).

> **📖 Từ mới**
> - **Chuỗi (sequence):** dữ liệu có thứ tự: câu chữ, chuỗi thời gian, lịch sử hội thoại.
> - **RNN (recurrent neural network):** đọc từng phần tử một, cập nhật một vector **trạng thái ẩn** $h_t$ (bộ nhớ ngắn) sau mỗi bước.
> - **BPTT (backpropagation through time):** "trải" RNN ra theo thời gian thành một mạng sâu $T$ lớp, rồi backprop như bình thường.
> - **Cổng (gate):** vector trong $(0, 1)$ (qua sigmoid), nhân từng phần tử để quyết định **cho qua bao nhiêu** thông tin.
> - **LSTM:** RNN có thêm **cell state** $c_t$ và ba cổng (forget, input, output). **GRU:** phiên bản gọn hơn, 2 cổng.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Đọc một cuốn tiểu thuyết:
- **RNN thường:** sau mỗi trang, bạn viết lại **toàn bộ** cuốn sổ tóm tắt từ đầu, dựa trên sổ cũ và trang mới. Sau 50 trang, chi tiết của trang 1 đã bị chép đi chép lại 50 lần và mờ hẳn.
- **LSTM:** cuốn sổ là một **băng chuyền** (cell state). Mỗi trang bạn chỉ quyết định ba việc: **xóa** dòng nào (forget gate), **thêm** dòng nào (input gate), và **đọc to** phần nào khi cần trả lời (output gate). Nếu không xóa thì thông tin cứ thế trôi theo băng chuyền, **không bị nhân lại** qua mỗi trang.

**Ví dụ có số (RNN vô hướng).** $h_t = \tanh(0.5\,h_{t-1} + x_t)$, $h_0 = 0$, $x = (1, 0, 0)$:
- $h_1 = \tanh(1) = 0.762$;
- $h_2 = \tanh(0.381) = 0.363$;
- $h_3 = \tanh(0.182) = 0.180$.

Thông tin "$x_1 = 1$" **tắt dần** qua mỗi bước. Gradient cũng vậy: $\partial h_3/\partial h_1 = 0.5(1 - h_2^2) \cdot 0.5(1 - h_3^2) = 0.21$. Mỗi bước nhân thêm một hệ số $\le 0.5$.

**Ba ý cốt lõi:**
1. RNN = **cùng một** lớp áp lặp lại ở mọi bước thời gian (chia sẻ trọng số theo thời gian, giống CNN chia sẻ theo không gian).
2. Gradient qua $T$ bước là **tích của $T$ Jacobian**, nên vanishing/exploding y như mạng sâu $T$ lớp (bài 04).
3. LSTM/GRU dùng **cộng có cổng** ($c_t = f \odot c_{t-1} + \dots$). Khi $f \approx 1$, gradient đi qua gần như nguyên vẹn.

---

### TẦNG 2 · Kiến thức chuẩn

#### 8.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Elman RNN** | $h_t = \tanh(W_{ih}x_t + b_{ih} + W_{hh}h_{t-1} + b_{hh})$ | Elman, *Cognitive Science* 1990 |
| **BPTT / truncated BPTT** | Backprop qua chuỗi đã trải ra; bản truncated chỉ lan ngược tối đa $k$ bước để tiết kiệm bộ nhớ | Werbos 1990; Williams & Peng 1990 |
| **LSTM** | Cell state + các cổng forget, input, output (công thức mục 8.2) | Hochreiter & Schmidhuber, *Neural Computation* 1997; forget gate: Gers et al. 2000 |
| **GRU** | Cổng update $z$ và reset $r$; không có cell state riêng | Cho et al., EMNLP 2014 |
| **Bidirectional RNN** | Một RNN đọc xuôi, một RNN đọc ngược; nối hai trạng thái ẩn. Dùng khi **có cả chuỗi** lúc dự đoán (phân loại, gán nhãn), **không** dùng được khi sinh từng bước | Schuster & Paliwal 1997 |
| **Forget-gate bias** | Khởi tạo bias của cổng forget **dương** (1 hoặc lớn hơn) để mặc định "nhớ" | Gers et al. 2000; Jozefowicz et al., ICML 2015 |

#### 8.2 Công thức LSTM và GRU

**🐢 Đi chậm từng bước (đọc trước công thức).**

*Bước 1 · Hình dung một bước LSTM.*

```
  hàng trên (băng chuyền):
     c_{t-1} ──► (× f) ──► (+) ──────────────► c_t ──► tanh ──► (× o) ──► h_t
                            ▲
                         (i × g)
  hàng dưới (bộ điều khiển):
     [h_{t-1}, x_t] ──► 4 lớp tuyến tính ──► f, i, o ∈ (0, 1) qua sigmoid;  g ∈ (−1, 1) qua tanh
```

- Trên băng chuyền chỉ có hai phép tính: **nhân với $f$** (xóa bớt) và **cộng $i \cdot g$** (ghi thêm).
- Mọi thứ phức tạp (ma trận $W$, sigmoid, tanh) đều nằm ở **hàng dưới**. Chúng chỉ quyết định xóa và ghi **bao nhiêu**, chứ không biến đổi trực tiếp nội dung trên băng chuyền.

*Bước 2 · Bốn "nút vặn", mỗi nút một việc.*

| Ký hiệu | Tên | Miền giá trị | Câu hỏi nó trả lời |
|---|---|---|---|
| $f$ | forget gate | $(0, 1)$ | Giữ lại bao nhiêu phần ô nhớ cũ? |
| $i$ | input gate | $(0, 1)$ | Ghi vào bao nhiêu phần nội dung mới? |
| $g$ | ứng viên | $(-1, 1)$ | Nội dung mới là gì (kể cả dấu)? |
| $o$ | output gate | $(0, 1)$ | Cho bên ngoài thấy bao nhiêu phần ô nhớ? |

$g$ dùng tanh vì nội dung cần có dấu. Ba cổng còn lại dùng sigmoid vì chúng là **tỉ lệ**.

*Bước 3 · Chạy tay 3 bước, với giá trị cổng cho sẵn.*
- Nhiệm vụ: nhớ giá trị đọc được ở bước 1, bỏ qua bước 2, và trả lời ở bước 3.
- Ở đây ta **cho sẵn** giá trị các cổng để thấy rõ vai trò của chúng. Mạng thật tự tính các cổng từ $x_t$ và $h_{t-1}$.

| $t$ | Token | $i$ | $f$ | $g$ | $o$ | $c_t = f\,c_{t-1} + i\,g$ | $h_t = o \tanh c_t$ |
|---|---|---|---|---|---|---|---|
| 1 | thông tin (0.8) | 0.9 | 0.5 | 0.8 | 0.1 | $0.5 \cdot 0 + 0.9 \cdot 0.8 = 0.720$ | $0.1 \cdot 0.617 = 0.062$ |
| 2 | nhiễu | 0.05 | 0.95 | −0.6 | 0.1 | $0.95 \cdot 0.72 + 0.05 \cdot (-0.6) = 0.654$ | $0.1 \cdot 0.574 = 0.057$ |
| 3 | câu hỏi | 0.05 | 0.95 | 0.3 | 0.95 | $0.95 \cdot 0.654 + 0.05 \cdot 0.3 = 0.636$ | $0.95 \cdot 0.562 =$ **0.534** |

Đọc bảng:
- **Bước 1** mở input gate để **ghi** thông tin.
- **Bước 2** đóng input gate và giữ forget gate gần 1, nên nhiễu gần như không lọt vào.
- **Bước 3** mở output gate để **đọc** ra.
- Ở bước 1 và 2, $h$ rất nhỏ vì output gate gần đóng, nhưng ô nhớ $c$ vẫn giữ được khoảng 0.65.

So sánh với RNN thường $h_t = \tanh(0.5\,h_{t-1} + x_t)$ trên cùng chuỗi $(0.8, -0.6, 0.3)$: $h = 0.664 \to -0.262 \to 0.168$. Nhiễu ở bước 2 **ghi đè** lên thông tin, vì RNN không có cách nào để "không ghi".

*Bước 4 · Gradient trên băng chuyền.*
- Theo đường băng chuyền: $c_3 = f_3\,(f_2\,c_1 + i_2 g_2) + i_3 g_3$, nên $\partial c_3/\partial c_1 = f_2 f_3 = 0.95^2 = 0.90$.
- Qua $T$ bước, hệ số này là **tích của các $f$**.
- Khi tín hiệu từ $x$ và $h$ gần 0 thì $f = \sigma(b_f)$, với $b_f$ là bias của forget gate:

| $b_f$ | $f = \sigma(b_f)$ | $f^{20}$ | $f^{50}$ |
|---|---|---|---|
| 0 (mặc định) | 0.5 | $9.5 \times 10^{-7}$ | $8.9 \times 10^{-16}$ |
| 1 | 0.731 | 0.0019 | $1.6 \times 10^{-7}$ |
| 2 | 0.881 | 0.079 | 0.0018 |
| 3 | 0.953 | 0.38 | 0.090 |

Bảng này giải thích "bất ngờ" ở mục 8.4:
- Phép cộng trên băng chuyền chỉ loại bỏ được ma trận $W$ và $\tanh'$ khỏi đường đi của gradient. Tích các $f$ thì **vẫn còn**.
- Với init mặc định $f \approx 0.5$, LSTM vẫn vanishing.
- Số đo ở 8.4(a) không trùng với bảng, vì hai đại lượng khác nhau: 8.4(a) đo $\partial h_T/\partial x_1$ so với $\partial h_T/\partial x_T$, và trong mạng thật $f$ dao động theo từng đầu vào. Nhưng xu hướng thì khớp: $b_f$ càng lớn, tín hiệu từ bước 1 càng sống lâu.

*Bước 5 · Từ số vô hướng lên vector.*
- Với hidden size $H$, các đại lượng $c, h, i, f, g, o$ đều là vector $H$ chiều, và mọi phép nhân là nhân từng phần tử. Như vậy **mỗi ô nhớ có cổng riêng**: ô 1 có thể nhớ lâu ($f \approx 1$) trong khi ô 2 xóa liên tục.
- PyTorch gộp bốn lớp tuyến tính của hàng dưới thành một ma trận $4H \times d$ cho $x$ và một ma trận $4H \times H$ cho $h$, xếp các khối theo thứ tự **i, f, g, o**.
- Đó là nguồn gốc của công thức $4H(d + H + 2)$ tham số, và của dòng `bias_ih_l0[H:2 * H]` ở mục 8.5.

**Công thức (tóm tắt các bước trên).**

**LSTM** ($\sigma$ là sigmoid, $\odot$ là nhân từng phần tử; thứ tự cổng trong PyTorch là **i, f, g, o**):

$$\begin{aligned}
i_t &= \sigma(W_{ii}x_t + W_{hi}h_{t-1} + b_i) &\text{(input gate)}\\
f_t &= \sigma(W_{if}x_t + W_{hf}h_{t-1} + b_f) &\text{(forget gate)}\\
g_t &= \tanh(W_{ig}x_t + W_{hg}h_{t-1} + b_g) &\text{(ứng viên)}\\
o_t &= \sigma(W_{io}x_t + W_{ho}h_{t-1} + b_o) &\text{(output gate)}\\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t)
\end{aligned}$$

*Vì sao (phác thảo) LSTM giảm vanishing:* theo đường đi qua cell, $\partial c_t/\partial c_{t-1} = \mathrm{diag}(f_t)$, **không** có ma trận trọng số và không có $\tanh'$ nhân vào. Nếu mạng học được $f \approx 1$ thì gradient đi qua nhiều bước gần như nguyên vẹn. Đây là "constant error carousel" của Hochreiter & Schmidhuber, cùng tinh thần với residual connection.

**GRU:** $z_t = \sigma(\cdot)$, $r_t = \sigma(\cdot)$, $\tilde h_t = \tanh(W_{in}x_t + b_{in} + r_t \odot (W_{hn}h_{t-1} + b_{hn}))$, và $h_t = (1 - z_t)\odot\tilde h_t + z_t \odot h_{t-1}$.

**Số tham số** (input $d$, hidden $H$; PyTorch có **hai** vector bias cho mỗi cổng):
- RNN: $H(d + H + 2)$.
- LSTM: $4H(d + H + 2)$.
- GRU: $3H(d + H + 2)$.

#### 8.3 Ví dụ tính tay

**(a) Một bước LSTM vô hướng.** $x = 1$, $h_0 = 0$, $c_0 = 0.5$; mọi trọng số theo $x$ bằng 1, theo $h$ bằng 0; bias bằng 0 riêng $b_f = 1$:
- $i = \sigma(1) = 0.731$; $f = \sigma(2) = 0.881$; $g = \tanh(1) = 0.762$; $o = \sigma(1) = 0.731$.
- $c_1 = 0.881 \cdot 0.5 + 0.731 \cdot 0.762 = 0.997$; $h_1 = 0.731 \cdot \tanh(0.997) = 0.556$.

`nn.LSTMCell` cho đúng $h_1 = 0.5559$, $c_1 = 0.9972$ (đã kiểm chứng, gán trọng số theo thứ tự i, f, g, o).

**(b) Đếm tham số** với $d = 10$, $H = 20$ (đã kiểm chứng):

| Model | Phép tính | Tham số |
|---|---|---|
| RNN | $20(10 + 20 + 2)$ | 640 |
| LSTM | $4 \cdot 640$ | 2 560 |
| GRU | $3 \cdot 640$ | 1 920 |
| LSTM hai chiều | $2 \cdot 2\,560$ | 5 120 |
| LSTM 2 lớp | $2\,560 + 4 \cdot 20(20 + 20 + 2)$ | 5 920 |

#### 8.4 Đo trên máy bạn

**(a) Gradient theo thời gian lúc khởi tạo.** $\lVert\partial h_T/\partial x_1\rVert / \lVert\partial h_T/\partial x_T\rVert$, hidden 64, init mặc định, trung vị 5 seed:

| $T$ | RNN | LSTM (mặc định) | GRU | LSTM, forget bias = 1 | LSTM, forget bias = 3 |
|---|---|---|---|---|---|
| 20 | $3.7 \times 10^{-6}$ | $5.9 \times 10^{-5}$ | $4.6 \times 10^{-5}$ | $2.2 \times 10^{-2}$ | 0.89 |
| 50 | $5.9 \times 10^{-14}$ | $8.4 \times 10^{-11}$ | $7.1 \times 10^{-11}$ | $1.7 \times 10^{-4}$ | 1.13 |
| 100 | 0 (underflow) | $1.7 \times 10^{-20}$ | $1.2 \times 10^{-20}$ | $5.8 \times 10^{-8}$ | 1.58 |

**Bất ngờ:** LSTM với init mặc định **cũng** vanishing gần như RNN! Lúc khởi tạo, $f = \sigma(\approx 0) \approx 0.5$, nên qua 50 bước còn $0.5^{50} \approx 10^{-15}$. Cấu trúc của LSTM chỉ cho **khả năng** nhớ lâu; mạng phải **học được** $f \approx 1$, hoặc ta đặt sẵn bias của cổng forget dương.

**(b) Bài toán "nhớ token đầu tiên".** Đọc chuỗi $T$ token ngẫu nhiên trong $\{0..9\}$, rồi đoán token **đầu tiên** (ngẫu nhiên: 0.10). Hidden 64, Adam, clip 1.0, 2 seed:

| $T$ | RNN (2 000 bước) | LSTM mặc định (2 000 bước) | GRU (2 000 bước) | LSTM, forget bias 1 (1 500 bước) | LSTM, forget bias 3 (1 500 bước) |
|---|---|---|---|---|---|
| 10 | 0.90 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | — | — |
| 20 | 0.21 / 0.37 | 0.10 / 0.11 | 0.09 / 0.10 | 0.90 / 1.00 | **1.00 / 1.00** |
| 40 | 0.09 / 0.10 | 0.09 / 0.10 | 0.10 / 0.10 | 0.09 / 0.10 | **1.00 / 1.00** |

Đặt forget bias = 3 (một dòng code) biến bài toán không học được thành học được hoàn hảo, với số bước còn **ít hơn**. Khoảng cách 40 bước là chuyện thường gặp: một câu hỏi của người dùng có thể dài hàng trăm token. Đó là lý do RNN bị Transformer thay thế (bài 09 và Tuần 5): attention nhìn thẳng vào token đầu tiên, không phải chuyền qua 40 bước.

#### 8.5 Code chuẩn

```python
from torch import nn

class SeqClassifier(nn.Module):
    def __init__(self, vocab, d=16, hidden=64, n_cls=10, forget_bias=3.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.rnn = nn.LSTM(d, hidden, batch_first=True)           # batch_first: (B, T, d)
        with torch.no_grad():
            H = hidden
            self.rnn.bias_ih_l0[H:2 * H].fill_(forget_bias)       # thứ tự i, f, g, o → khối thứ 2 là f
            self.rnn.bias_hh_l0[H:2 * H].zero_()
        self.head = nn.Linear(hidden, n_cls)

    def forward(self, x):                                          # x: (B, T) chỉ số token
        out, (h_n, c_n) = self.rnn(self.emb(x))                    # out: (B, T, H); h_n: (1, B, H)
        return self.head(h_n[-1])                                  # trạng thái cuối

# chuỗi độ dài khác nhau: pad rồi pack để RNN bỏ qua phần đệm
packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
```

#### 8.6 Cạm bẫy

1. **Nhầm `batch_first`:** mặc định của `nn.LSTM` là `(T, B, d)`. Truyền `(B, T, d)` mà không đặt `batch_first=True` thì **không có lỗi** nào, nhưng model học sai hoàn toàn.
2. **Lấy `out[:, -1]` khi chuỗi có padding:** vị trí cuối là token đệm. Dùng `pack_padded_sequence`, hoặc lấy đúng vị trí `lengths - 1`. Số đo ở phụ lục P2.4: không pack, bài "nhớ token đầu" rơi từ 1.000 xuống 0.10.
3. **Không clip gradient** với RNN: exploding gradient rất hay gặp (Pascanu et al. 2013). Clip theo chuẩn 1.0 là mặc định hợp lý.
4. **Bidirectional cho bài toán sinh văn bản:** nhánh ngược "nhìn thấy tương lai". Đây là rò rỉ dữ liệu theo đúng nghĩa ở Tuần 3.
5. **Nghĩ rằng LSTM tự động nhớ lâu:** mục 8.4. Hãy kiểm tra init của forget bias.

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"Vì sao LSTM giảm vanishing gradient?"* Đường đi qua cell có Jacobian $\mathrm{diag}(f_t)$, là phép cộng có cổng chứ không phải nhân với $W$ và $\tanh'$. Nêu được **điều kiện** $f \approx 1$ và mẹo forget bias.
- *"GRU khác LSTM thế nào?"* 2 cổng thay vì 3, không có cell riêng, ít hơn 25% tham số; chất lượng thường tương đương (Chung et al. 2014).
- *"Vì sao Transformer thay thế RNN?"* RNN tuần tự (không song song hóa được theo $T$); đường đi giữa hai token xa nhau dài $O(T)$; attention có đường đi $O(1)$.

#### 8.7 Đọc thêm

- S. Hochreiter & J. Schmidhuber, *Long Short-Term Memory*, Neural Computation 1997.
- C. Olah, *Understanding LSTM Networks* (blog, 2015): hình vẽ kinh điển.
- R. Jozefowicz, W. Zaremba, I. Sutskever, *An Empirical Exploration of Recurrent Network Architectures*, ICML 2015.
- K. Cho et al., *Learning Phrase Representations using RNN Encoder–Decoder*, EMNLP 2014.
- A. Karpathy, *The Unreasonable Effectiveness of Recurrent Neural Networks* (blog, 2015).

---

## 09 · Seq2seq, attention và beam search

> ⏱ Tầng 1: 15' · Tầng 2: 55'
> **Cần biết trước:** bài 08; softmax (Tuần 2, bài 04); tích vô hướng và cosine (Tuần 2, bài 01).

> **📖 Từ mới**
> - **Seq2seq (encoder–decoder):** encoder đọc chuỗi vào; decoder sinh chuỗi ra từng token một (dịch máy, tóm tắt).
> - **Bottleneck:** trong seq2seq cổ điển, **toàn bộ** câu vào bị nén vào **một** vector trạng thái cuối của encoder.
> - **Attention:** ở mỗi bước sinh, decoder tính **trọng số** cho từng vị trí của câu vào, rồi lấy **trung bình có trọng số** của chúng (gọi là **context vector**).
> - **Query / key / value:** "câu hỏi" của decoder / "nhãn" của từng vị trí để so khớp / "nội dung" được lấy ra.
> - **Teacher forcing:** lúc train, đưa **token đúng** của bước trước vào decoder thay vì token mà decoder tự đoán.
> - **Beam search:** khi sinh, giữ $k$ chuỗi tốt nhất ở mỗi bước thay vì chỉ một (greedy).

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Phiên dịch một bài phát biểu dài:
- **Không có attention:** nghe hết bài, gấp sổ lại, rồi dịch **chỉ dựa vào trí nhớ**. Câu ngắn thì ổn, câu dài thì quên.
- **Có attention:** được để bài gốc trước mặt. Khi dịch từng câu, liếc lại **đúng đoạn** cần dịch. "Liếc vào đâu" chính là trọng số attention; nó do người dịch **tự quyết** ở mỗi câu.

**Ví dụ có số.** Query $q = (1, 0)$; ba key $k_1 = (2, 0)$, $k_2 = (0, 1)$, $k_3 = (1, 1)$; value $v = (1, 2, 3)$:
1. Điểm số (tích vô hướng): $q \cdot k = (2, 0, 1)$.
2. Softmax: $e^{(2, 0, 1)} = (7.389, 1, 2.718)$, tổng 11.107, nên trọng số $\alpha = (0.665,\ 0.090,\ 0.245)$.
3. Đầu ra: $0.665 \cdot 1 + 0.090 \cdot 2 + 0.245 \cdot 3 =$ **1.579**.

Key giống query nhất ($k_1$) được nhìn nhiều nhất, nhưng không phải duy nhất: attention là **tra cứu mềm** (soft lookup).

**Ba ý cốt lõi:**
1. Attention = **so khớp** (query với từng key) → **softmax** → **trung bình có trọng số** của value.
2. Nó gỡ bottleneck: decoder nhìn được **mọi** trạng thái của encoder, chứ không chỉ trạng thái cuối.
3. Khi sinh, greedy chọn token tốt nhất từng bước nhưng có thể bỏ lỡ chuỗi tốt nhất; beam search giữ nhiều ứng viên.

---

### TẦNG 2 · Kiến thức chuẩn

#### 9.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Seq2seq** | Encoder RNN → vector cố định → decoder RNN | Sutskever, Vinyals, Le, NeurIPS 2014; Cho et al. 2014 |
| **Additive (Bahdanau) attention** | $e_j = v^\top\tanh(W_q q + W_k k_j)$ | Bahdanau, Cho, Bengio, ICLR 2015 |
| **Multiplicative (Luong) attention** | *dot*: $e_j = q^\top k_j$; *general*: $e_j = q^\top W k_j$ | Luong, Pham, Manning, EMNLP 2015 |
| **Scaled dot-product attention** | $\mathrm{softmax}(QK^\top/\sqrt{d})\,V$; dạng ma trận cho mọi query cùng lúc. Là lõi của Transformer (Tuần 5) | Vaswani et al., NeurIPS 2017 |
| **Mask** | Đặt điểm số $= -\infty$ tại vị trí không được nhìn (padding, hoặc tương lai), nên sau softmax trọng số đúng bằng 0 | — |
| **Exposure bias** | Lúc train decoder chỉ thấy token đúng (teacher forcing), lúc sinh lại thấy token **tự sinh**, nên lỗi tích lũy | Bengio et al., *Scheduled Sampling*, NeurIPS 2015 |
| **Beam search** | Ở mỗi bước, mở rộng $k$ chuỗi tốt nhất bằng mọi token, rồi giữ lại $k$ chuỗi có tổng log-xác suất lớn nhất | Graves 2012; Sutskever et al. 2014 |

#### 9.2 Công thức

**🐢 Đi chậm từng bước (đọc trước công thức).**

*Bước 1 · Tra cứu cứng: `dict` của Python.*
- Với `faq = {"thanh toán": "Hướng dẫn A", "kỹ thuật": "Hướng dẫn B"}`, lệnh `faq["thanh toán"]` trả về đúng **một** value.
- Điều đó chỉ xảy ra khi query **trùng khớp tuyệt đối** với một key. Query "cước tháng này cao" sẽ gây `KeyError`.

*Bước 2 · Tra cứu mềm.*
- Thay vì hỏi "có trùng không", ta đo **mức giống nhau** giữa query và **từng** key bằng tích vô hướng.
- Softmax đổi các mức giống đó thành trọng số có tổng bằng 1.
- Kết quả là các value được **trộn** theo trọng số. Không bao giờ có `KeyError`: một query lạ vẫn nhận được câu trả lời pha trộn từ những key gần nó nhất.

Hình dung một tủ có nhiều ngăn:
- **Key là nhãn dán ngoài ngăn**, dùng để so khớp.
- **Value là đồ đựng bên trong**, là thứ được lấy ra.
- Hai vai trò này tách riêng, vì cách **nhận ra** một thứ không nhất thiết giống **nội dung** của nó. Nhãn "hóa đơn" khác với nội dung của tờ hóa đơn.

*Bước 3 · Ví dụ tổng đài.* Key có 2 chiều: trục 1 là "về tiền", trục 2 là "về kỹ thuật". Value là số phút xử lý trung bình của từng nhóm.

| Ngăn | Key $k_j$ | Value $v_j$ (phút) |
|---|---|---|
| thanh toán | $(1, 0)$ | 10 |
| kỹ thuật | $(0, 1)$ | 30 |
| khuyến mãi | $(0.7, 0.7)$ | 5 |

Query "cước tháng này cao" là $q = (0.9,\ 0.1)$:
1. Điểm số: $q \cdot k = (0.9,\ 0.1,\ 0.7)$.
2. Softmax: $e^{(0.9,\ 0.1,\ 0.7)} = (2.460,\ 1.105,\ 2.014)$, tổng bằng 5.579, nên $\alpha = (0.441,\ 0.198,\ 0.361)$.
3. Đầu ra: $0.441 \cdot 10 + 0.198 \cdot 30 + 0.361 \cdot 5 =$ **12.16** phút.

Bây giờ lấy một query **cùng hướng nhưng dài hơn**, $q = (3,\ 0)$:
- Điểm số là $(3,\ 0,\ 2.1)$, nên $\alpha = (0.687,\ 0.034,\ 0.279)$ và đầu ra là **9.29** phút.
- Trọng số dồn mạnh hơn vào ngăn "thanh toán". Như vậy, **độ dài** của query điều khiển độ "cứng" của phép tra cứu.
- Ngăn "khuyến mãi" vẫn nhận 0.279 trọng số, vì key của nó cũng có thành phần "về tiền".

*Bước 4 · Vì sao phải chia cho $\sqrt d$.*
- Với các vector ngẫu nhiên, $q \cdot k = \sum_{i=1}^{d} q_i k_i$ là tổng của $d$ số hạng, mỗi số hạng có phương sai 1. Vậy tổng có phương sai $d$ (lại chính là bước 2 của bài 04).
- Đo trên máy bạn: với $d = 1, 4, 64$, phương sai của $q \cdot k$ là 1.00, 4.01, 64.0. Sau khi chia cho $\sqrt d$, cả ba đều về 1.00.
- Nói cách khác, $d$ càng lớn thì query "tự động dài ra" như ở bước 3. Với $d = 64$, các điểm số chênh nhau cỡ ±8.

Hệ quả cho gradient, xét hai key:

| Điểm số | $\alpha$ | $\partial \alpha_1 / \partial e_1 = \alpha_1(1 - \alpha_1)$ |
|---|---|---|
| $(8,\ 0)$, chưa scale | $(0.99966,\ 0.00034)$ | 0.00034 |
| $(1,\ 0)$, đã chia $\sqrt{64} = 8$ | $(0.731,\ 0.269)$ | 0.197 (lớn gấp khoảng **590** lần) |

Softmax bão hòa thì gần như không học được, giống sigmoid ở vùng phẳng (bài 03).

*Bước 5 · Từ một query lên dạng ma trận.*
- Xếp $L_q$ query thành các hàng của $Q$ ($L_q \times d$), và $L_k$ key thành các hàng của $K$ ($L_k \times d$).
- $QK^\top$ là bảng điểm số $L_q \times L_k$; mỗi hàng là phép tính điểm số ở bước 3 cho một query.
- Softmax lấy theo **từng hàng**.
- Nhân với $V$ ($L_k \times d_v$) là phép trộn value ở bước 3, làm cho mọi query cùng lúc.

Trong seq2seq, query là trạng thái decoder, còn key và value đều là trạng thái encoder. Ở Tuần 5 (self-attention), cả ba được sinh ra từ **cùng** các token, qua ba ma trận chiếu khác nhau $W_Q, W_K, W_V$. Cùng một từ, nhưng "nhãn để được tìm thấy" và "nội dung để đưa ra" được học riêng.

**Công thức (tóm tắt các bước trên).** Với query $q$ và các cặp $(k_j, v_j)$, $j = 1..L$:

$$e_j = \text{score}(q, k_j), \qquad \alpha_j = \frac{e^{e_j}}{\sum_{l} e^{e_l}}, \qquad c = \sum_j \alpha_j v_j.$$

Trong seq2seq của Bahdanau, key = value = trạng thái encoder $h_j$, còn query là trạng thái decoder $s_{t-1}$.

**Vì sao chia cho $\sqrt d$?** Nếu các thành phần của $q$ và $k$ độc lập, mean 0, var 1, thì $q \cdot k$ có **var $= d$**. Với $d = 64$, điểm số có std 8, nên softmax gần như **one-hot** và gradient của các vị trí khác gần 0. Chia cho $\sqrt d$ đưa var về 1.

**Beam search:** score của chuỗi $= \sum_t \log p(y_t \mid y_{<t}, x)$. Vì mỗi số hạng âm, chuỗi ngắn được lợi; thường chuẩn hóa bằng cách chia cho $|y|^\alpha$ với $\alpha \approx 0.6$–$1$ (Wu et al., *GNMT*, 2016).

#### 9.3 Ví dụ tính tay

**(a) Scaled dot-product** với ví dụ ở tầng 1 ($d = 2$): điểm số chia cho $\sqrt 2$ thành $(1.414, 0, 0.707)$, trọng số $(0.576,\ 0.140,\ 0.284)$, đầu ra **1.708**. Trọng số "mềm" hơn so với khi không scale.

**(b) Additive attention.** $W_q = W_k = I$, $v = (1, 1)$, $q = (0, 0)$; ba key $(1, 1)$, $(-1, -1)$, $(0, 0)$:
- $e_j = v^\top\tanh(q + k_j)$, nên $e = (2\tanh 1,\ -2\tanh 1,\ 0) = (1.523,\ -1.523,\ 0)$.
- Softmax cho $\alpha = (0.790,\ 0.038,\ 0.172)$.
- Context $c = \sum_j \alpha_j k_j = (0.753,\ 0.753)$.

Cả (a) và (b) là test của bài tập hôm nay.

**(c) Beam search.** Model sinh 2 token:
- Bước 1: $P(A) = 0.5$, $P(B) = 0.4$, $P(\langle/s\rangle) = 0.1$.
- Bước 2 sau $A$: $P(x) = 0.3$, $P(y) = 0.3$, $P(z) = 0.4$.
- Bước 2 sau $B$: $P(x) = 0.9$, $P(y) = 0.05$, $P(z) = 0.05$.

| Cách | Chuỗi | Xác suất | Log-xác suất |
|---|---|---|---|
| Greedy | chọn $A$ (0.5) rồi $z$ (0.4): **A z** | 0.20 | −1.609 |
| Beam $k = 2$, bước 1 | giữ $A$ (−0.693) và $B$ (−0.916) | | |
| Beam $k = 2$, bước 2 | mở rộng 6 chuỗi: **B x** 0.36 · A z 0.20 · A x 0.15 · A y 0.15 · B y 0.02 · B z 0.02 | | |
| Kết quả beam | **B x** | **0.36** | −1.022 |

Greedy bỏ lỡ chuỗi tốt nhất vì ở bước 1, $A$ chỉ hơn $B$ một chút, nhưng **sau** $B$ model rất chắc chắn. Beam search giữ được $B$ đủ lâu để phát hiện điều đó.

#### 9.4 Đo trên máy bạn: bottleneck

Bài toán **copy**: đọc chuỗi $L$ chữ số ngẫu nhiên, rồi sinh lại đúng chuỗi đó. Encoder GRU 64, decoder GRUCell 64, teacher forcing, Adam, 1 500 bước; giải mã greedy trên 1 000 chuỗi mới:

| $L$ | Không attention: token đúng / **cả chuỗi đúng** | Có attention (dot, scaled): token / cả chuỗi |
|---|---|---|
| 5 | 0.980 / 0.944 | 0.999 / 0.997 |
| 10 | 0.611 / **0.187** | 1.000 / 0.997 |
| 20 | 0.276 / **0.000** | 0.994 / 0.980 |

Không có attention, một vector 64 chiều phải "chứa" 20 chữ số (khoảng 66 bit thông tin) sau khi đã bị trộn qua 20 bước GRU. Model gần như chịu thua. Có attention, ở bước $t$ decoder chỉ cần **tìm đúng vị trí $t$** của câu vào. Đây chính là phát hiện của Bahdanau et al. (2015): chất lượng dịch của seq2seq cổ điển **giảm mạnh theo độ dài câu**, còn bản có attention thì không.

**Vì sao phải chia $\sqrt d$** (đo trên 1 000 lần thử ngẫu nhiên, 10 key): trọng số lớn nhất trung bình là

| $d$ | std điểm số | Không scale | Có scale |
|---|---|---|---|
| 4 | 1.98 | 0.494 | 0.304 |
| 64 | 7.99 | **0.857** | 0.321 |
| 512 | 22.7 | **0.953** | 0.319 |

Không scale thì $d$ càng lớn, softmax càng "cứng". Có scale thì độ "mềm" không phụ thuộc $d$.

#### 9.5 Code chuẩn

```python
import math
import torch

def dot_attention(q, k, v, mask=None):
    """q (..., Lq, d), k (..., Lk, d), v (..., Lk, dv); mask bool, True = được nhìn."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    w = torch.softmax(scores, dim=-1)
    return w @ v, w

# PyTorch có sẵn bản tối ưu (dùng ở Tuần 5):
out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

#### 9.6 Cạm bẫy

1. **Quên mask phần padding:** model chú ý vào token đệm. Câu ngắn trong batch bị "pha loãng".
2. **Hàng bị che toàn bộ:** mọi điểm số bằng $-\infty$ nên softmax cho `nan`. Phải đảm bảo mỗi query có ít nhất một vị trí hợp lệ.
3. **Mask bằng cách nhân trọng số với 0 sau softmax:** tổng không còn bằng 1. Phải mask **trước** softmax.
4. **Ngữ nghĩa mask ngược nhau giữa các API:** `scaled_dot_product_attention` coi `True` là **được nhìn**, còn `nn.MultiheadAttention(key_padding_mask=...)` coi `True` là **bị che**. Luôn đọc docs.
5. **Beam search không chuẩn hóa độ dài:** thiên về câu ngắn, có khi dừng ngay ở $\langle/s\rangle$.
6. **Coi trọng số attention là "lời giải thích"** cho dự đoán: tương quan với importance thật có khi rất yếu (Jain & Wallace, NAACL 2019). Dùng để **gợi ý**, không phải để **chứng minh**.

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"Attention giải quyết vấn đề gì của seq2seq?"* Bottleneck: mọi thông tin phải đi qua một vector cố định; đường đi từ token vào $j$ tới token ra $t$ dài $O(T)$.
- *"Additive khác multiplicative thế nào?"* Additive có thêm một MLP nhỏ, nên linh hoạt hơn khi $d_q \ne d_k$. Multiplicative chỉ là nhân ma trận, nhanh hơn nhiều trên GPU; khi có scale thì chất lượng tương đương.
- *"Greedy hay beam?"* Beam tốt cho dịch máy và tóm tắt (đầu ra "đúng" gần như là duy nhất). Với chat hay viết sáng tạo, LLM dùng **sampling** (temperature, top-p: Tuần 6), vì beam cho văn bản lặp và nhạt (Holtzman et al., ICLR 2020).

#### 9.7 Đọc thêm

- D. Bahdanau, K. Cho, Y. Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015.
- M.-T. Luong, H. Pham, C. Manning, *Effective Approaches to Attention-based Neural Machine Translation*, EMNLP 2015.
- I. Sutskever, O. Vinyals, Q. Le, *Sequence to Sequence Learning with Neural Networks*, NeurIPS 2014.
- J. Alammar, *Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)* (blog).
- S. Jain & B. Wallace, *Attention is not Explanation*, NAACL 2019.

---

## 10 · Transfer learning: pretrain, feature extraction, fine-tuning

> ⏱ Tầng 1: 10' · Tầng 2: 35'
> **Cần biết trước:** CNN (bài 07); BatchNorm khi fine-tune (bài 05, cạm bẫy 3).

> **📖 Từ mới**
> - **Pretraining:** huấn luyện trước một model trên một tập dữ liệu **lớn** (ImageNet, toàn bộ Wikipedia, …), thường cho một nhiệm vụ chung.
> - **Transfer learning:** dùng lại model đã pretrain cho một nhiệm vụ **mới** có ít dữ liệu.
> - **Feature extraction (linear probe):** **đóng băng** toàn bộ phần thân (body), chỉ huấn luyện một lớp đầu (head) mới.
> - **Fine-tuning:** huấn luyện tiếp **cả** model (hoặc một phần) trên dữ liệu mới, với learning rate nhỏ.
> - **Freeze / unfreeze:** tắt / bật `requires_grad` cho tham số.
> - **Discriminative learning rate:** mỗi nhóm lớp một learning rate; lớp gần đầu vào dùng lr nhỏ hơn.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Một bác sĩ nội khoa chuyển sang học chuyên khoa da liễu:
- Không ai bắt họ học lại sinh học từ lớp 1. Kiến thức nền (giải phẫu, sinh lý) **chuyển giao** được.
- Cách 1 (**feature extraction**): giữ nguyên toàn bộ kiến thức cũ, chỉ học thêm "bảng tra" các bệnh da liễu. Nhanh, ít rủi ro.
- Cách 2 (**fine-tuning**): điều chỉnh nhẹ cả cách nhìn nhận cũ cho hợp với da liễu. Tốt hơn khi có đủ ca bệnh, nhưng điều chỉnh mạnh quá thì **quên** kiến thức cũ (*catastrophic forgetting*).
- Nhưng nếu người đó là **kỹ sư cầu đường** thì "kiến thức nền" ít giúp được gì: chuyển giao chỉ có ích khi hai lĩnh vực **gần nhau**.

**Ba ý cốt lõi:**
1. Các lớp đầu học đặc trưng **chung** (cạnh, màu, hình khối; với văn bản là từ và cú pháp); các lớp sau học đặc trưng **riêng** của nhiệm vụ cũ.
2. Ít dữ liệu và miền gần thì dùng feature extraction; nhiều dữ liệu hơn thì fine-tune với lr nhỏ.
3. Transfer chỉ tốt khi dữ liệu pretrain **lớn và liên quan**. Đo trên máy bạn (mục 10.4): nó có thể không giúp gì, thậm chí còn hại.

---

### TẦNG 2 · Kiến thức chuẩn

#### 10.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Transferability của feature** | Đặc trưng ở lớp đầu của CNN chung chung (giống Gabor filter), ở lớp sau thì chuyên biệt; khả năng chuyển giao giảm khi nhiệm vụ nguồn và đích **xa nhau** | Yosinski et al., *How transferable are features in deep neural networks?*, NeurIPS 2014 |
| **ImageNet pretraining** | ResNet/ViT pretrain trên 1.28 triệu ảnh, 1 000 lớp; là điểm khởi đầu mặc định cho mọi bài toán ảnh | Deng et al., CVPR 2009; Kornblith et al., *Do Better ImageNet Models Transfer Better?*, CVPR 2019 |
| **"ImageNet moment" của NLP** | 2018: pretrain một language model trên văn bản lớn rồi fine-tune (ULMFiT, ELMo, GPT, BERT) thắng các model chuyên biệt ở hầu hết nhiệm vụ | Howard & Ruder, ACL 2018; Peters et al. 2018; Devlin et al., NAACL 2019 |
| **Gradual unfreezing, discriminative LR** | Mở băng dần từ lớp cuối về lớp đầu; lr giảm dần theo độ sâu (ví dụ chia cho 2.6 mỗi nhóm) | Howard & Ruder 2018 |
| **Catastrophic forgetting** | Fine-tune mạnh làm mất khả năng ở nhiệm vụ cũ | McCloskey & Cohen 1989; Kirkpatrick et al., PNAS 2017 |
| **Negative transfer** | Pretrain làm kết quả **tệ hơn** so với train từ đầu | Rosenstein et al. 2005; Wang et al., CVPR 2019 |

#### 10.2 Chọn chiến lược

| | Dữ liệu đích ít | Dữ liệu đích nhiều |
|---|---|---|
| **Miền gần** với nguồn | Feature extraction (head mới, body đóng băng) | Fine-tune toàn bộ, lr nhỏ |
| **Miền xa** | Khó: fine-tune vài lớp cuối, hoặc tìm model pretrain gần hơn | Fine-tune toàn bộ, hoặc train từ đầu |

**Quy tắc lr khi fine-tune:** head mới dùng lr cỡ $10^{-3}$; body dùng lr nhỏ hơn 10 đến 100 lần ($10^{-5}$–$10^{-4}$), vì body đã ở gần một điểm tốt và bước lớn sẽ phá hỏng nó.

#### 10.3 Ví dụ tính tay: bộ nhớ khi fine-tune

ResNet-18 (11.7 triệu tham số) trên GPU T4 16 GB, Adam, fp32: $16 \times 11.7 \times 10^6 \approx 0.19$ GB cho tham số, gradient và trạng thái optimizer. Phần lớn bộ nhớ sẽ dành cho **activation** (tỉ lệ với batch × độ phân giải). So với LLM 7B ở bài 06 (112 GB): fine-tune CNN cỡ vừa thì **vừa thoải mái** GPU miễn phí; fine-tune LLM thì không, trừ khi dùng LoRA (Tuần 11).

**Feature extraction tiết kiệm hơn nữa:** body đóng băng không cần gradient và không cần trạng thái Adam. Chỉ cần chạy forward **một lần** để tính feature rồi lưu lại; sau đó huấn luyện head (ví dụ logistic regression) trên feature trong vài giây, kể cả trên CPU.

#### 10.4 Đo trên máy bạn: khi nào transfer giúp, khi nào không

CNN của bài 07, pretrain trên một phần của `load_digits`, rồi chuyển sang bài toán đích chỉ có $k$ ảnh mỗi lớp. Ba cách: **scratch** (train từ đầu), **frozen** (đóng băng body, chỉ train head), **fine-tune** (body lr $10^{-3}$ hoặc $3 \times 10^{-4}$, head lr $10^{-2}$). Mỗi ô là trung bình 5 lần lấy mẫu.

**(A) Miền gần:** nguồn là 628 ảnh sạch, 10 lớp; đích là **cùng 10 chữ số nhưng ảnh bị nhiễu** (Gauss std 0.3):

| $k$ ảnh/lớp | Scratch | Frozen | Fine-tune |
|---|---|---|---|
| 2 | 0.376 | **0.626** | 0.618 |
| 5 | 0.552 | **0.736** | 0.727 |

Với 20 ảnh tổng cộng, transfer tăng accuracy từ 0.38 lên 0.63. Đây là kịch bản điển hình của ImageNet → bài toán ảnh của bạn.

**(B) Miền xa:** nguồn là chữ số **0–4**; đích là chữ số **5–9** (đặc trưng sau global pooling, 32 chiều):

| $k$ ảnh/lớp | Scratch | Frozen | Fine-tune |
|---|---|---|---|
| 5 | **0.880** | 0.631 | 0.716 |
| 50 | **0.984** | 0.793 | 0.939 |

**Negative transfer.** 32 đặc trưng học để phân biệt 0–4 bị "chuyên biệt hóa" và không đủ để phân biệt 5–9. Nguồn chỉ có khoảng 630 ảnh nên đặc trưng không đủ tổng quát. Khi dùng feature **trước** global pooling (512 chiều), khoảng cách thu hẹp lại (5 ảnh/lớp: scratch 0.911, frozen 0.896, fine-tune 0.898) nhưng transfer vẫn không thắng.

**Bài học:** ImageNet hiệu quả vì nó **khổng lồ và đa dạng**, nên feature của nó gần như "chung" cho mọi ảnh tự nhiên. Pretrain trên một tập nhỏ và hẹp thì không có phép màu đó. Luôn so với một baseline train từ đầu.

#### 10.5 Code chuẩn

```python
import copy
from torch import nn

def freeze_(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    return module

pre = ...                                   # model đã pretrain: Sequential(body, head)
body = freeze_(copy.deepcopy(pre[0]))       # feature extraction
model = nn.Sequential(body, nn.Linear(32, n_new_classes))
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)

# Fine-tune với discriminative LR
body = copy.deepcopy(pre[0])
model = nn.Sequential(body, nn.Linear(32, n_new_classes))
opt = torch.optim.AdamW([{"params": body.parameters(), "lr": 3e-4},
                         {"params": model[1].parameters(), "lr": 1e-2}])
```

**Với model pretrained thật** (cần mạng để tải trọng số; chạy trên Kaggle/Colab, **không** chạy trên máy này):

```python
# Ảnh: torchvision (máy bạn chưa cài)
from torchvision.models import resnet18, ResNet18_Weights
m = resnet18(weights=ResNet18_Weights.DEFAULT)
m.fc = nn.Linear(m.fc.in_features, 5)                         # thay head 1000 lớp bằng 5 lớp
preprocess = ResNet18_Weights.DEFAULT.transforms()             # đúng resize/crop/mean/std lúc pretrain

# Văn bản: Hugging Face transformers (đã cài sẵn), MODEL_ID là tên model trên Hub
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained("MODEL_ID")
clf = AutoModelForSequenceClassification.from_pretrained("MODEL_ID", num_labels=2)   # head mới, khởi tạo ngẫu nhiên
```

#### 10.6 Cạm bẫy

1. **Mở băng (unfreeze) mà quên thêm vào optimizer:** nếu lúc tạo optimizer bạn chỉ truyền các tham số có `requires_grad`, thì những lớp mở băng về sau **không bao giờ** được cập nhật, dù gradient vẫn được tính. Khi mở băng dần, hãy thêm nhóm tham số mới bằng `opt.add_param_group(...)`, hoặc tạo lại optimizer.
2. **Đóng băng nhưng BN vẫn ở chế độ train:** running stats vẫn bị cập nhật bởi dữ liệu mới dù trọng số không đổi. Phải gọi `body.eval()` (hoặc ít nhất đặt các lớp BN về eval) **sau mỗi lần** gọi `model.train()`.
3. **Tiền xử lý khác với lúc pretrain** (kích thước, mean/std, tokenizer): feature mất ý nghĩa. Dùng đúng `transforms()` hoặc tokenizer đi kèm model.
4. **Fine-tune với lr lớn:** phá hỏng trọng số pretrain ngay từ những bước đầu. Dùng lr nhỏ cho body, kèm warmup.
5. **Không so với baseline train từ đầu:** mục 10.4(B) cho thấy transfer có thể **hại**.
6. **Giấy phép của model và dữ liệu:** model pretrained có license riêng (dùng thương mại hay không). Với Viettel và on-prem, đây là câu hỏi thật, không phải hình thức.

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"Feature extraction hay fine-tuning?"* Bảng ở mục 10.2. Nêu được hai trục: **lượng dữ liệu** và **độ gần của miền**.
- *"Vì sao lr của body phải nhỏ?"* Body đã ở một điểm tốt; bước lớn làm hỏng các đặc trưng chung (catastrophic forgetting).
- *"ImageNet moment của NLP là gì?"* ULMFiT/ELMo/BERT (2018–2019): pretrain language model trên văn bản lớn, rồi fine-tune. Đó là tiền đề của mọi LLM ngày nay (Tuần 5–6, 11).

#### 10.7 Đọc thêm

- J. Yosinski et al., *How transferable are features in deep neural networks?*, NeurIPS 2014.
- J. Howard & S. Ruder, *Universal Language Model Fine-tuning for Text Classification* (ULMFiT), ACL 2018.
- S. Kornblith, J. Shlens, Q. Le, *Do Better ImageNet Models Transfer Better?*, CVPR 2019.
- CS231n notes, *Transfer Learning*.
- Hugging Face course, chương 3 (*Fine-tuning a pretrained model*).

---

## Bài tập Ngày 3 · `qnet/attention.py` (~2h)

Ba thành phần attention. `AttnPool` sẽ là lõi của model `TabAttn` trong project (Ngày 4): mỗi đặc trưng của log chatbot thành một "token", và model tự chọn nên chú ý vào đặc trưng nào. Lệnh test: `python -m pytest -q tests/test_attention.py` (4 test).

| Thành phần | Yêu cầu |
|---|---|
| `dot_attention(q, k, v, mask=None, scale=True)` | `q (..., Lq, d)`, `k (..., Lk, d)`, `v (..., Lk, dv)`. Điểm số `q @ kᵀ`, chia $\sqrt d$ nếu `scale`. `mask` là tensor bool **broadcast được** tới `(..., Lq, Lk)`, với `True` = được nhìn: đặt các vị trí `False` thành `-inf` **trước** softmax. Trả về `(out (..., Lq, dv), weights (..., Lq, Lk))`. Phải khớp `F.scaled_dot_product_attention(q, k, v, attn_mask=mask)` |
| `AdditiveAttention(d_q, d_k, d_h)` | Ba lớp `W_q = nn.Linear(d_q, d_h, bias=False)`, `W_k = nn.Linear(d_k, d_h, bias=False)`, `v = nn.Linear(d_h, 1, bias=False)` (đúng tên này, vì test gán trọng số). `forward(query (B, d_q), keys (B, L, d_k), mask=None (B, L))` → `(context (B, d_k), weights (B, L))`, với context $= \sum_j \alpha_j\,\text{keys}_j$ |
| `AttnPool(d)` | Tham số `query` (vector $d$ chiều, khởi tạo `randn(d) / sqrt(d)`), `W_k`, `W_v = nn.Linear(d, d, bias=False)`. `forward(tokens (B, T, d))`: dùng `dot_attention` với query mở rộng thành `(B, 1, d)`, key `W_k(tokens)`, value `W_v(tokens)`. Trả về `(pooled (B, d), weights (B, T))` |

**Gợi ý:**
- Test `test_attnpool_permutation_and_gradient` kiểm tra rằng **đổi thứ tự token không đổi kết quả gộp**. Attention không có khái niệm vị trí; đó là lý do Transformer cần *positional encoding* (Tuần 5). Với dữ liệu bảng, tính chất này lại đúng là điều ta muốn: các cột không có thứ tự.
- Dùng `masked_fill(~mask, float("-inf"))`; đừng nhân trọng số với mask sau softmax (cạm bẫy 3, bài 09).
