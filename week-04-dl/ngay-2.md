# Ngày 2 · Huấn luyện mạng sâu: gradient, khởi tạo, chuẩn hóa, regularization

> **Loại tài liệu: Lý thuyết (bản nhẹ).** Công thức đầy đủ và ví dụ tính tay. Chứng minh chỉ còn phác thảo vài dòng ("vì sao").
> **Hôm nay gồm 3 bài:**
> - **04** Vanishing/exploding gradient, khởi tạo trọng số, residual connection, gradient clipping
> - **05** Normalization: BatchNorm, LayerNorm, RMSNorm
> - **06** Regularization và huấn luyện hiệu quả: dropout, weight decay, early stopping, data augmentation, mixed precision, gradient accumulation
>
> **Thời gian:** đọc ~2h45 · quiz 45' · bài tập ~2h
> **Nhận lại từ các tuần trước:** Tuần 2 mới nói $\sigma' \le 1/4$; Tuần 3 hẹn dropout và early stopping. Hôm nay làm rõ cả hai. Thí nghiệm dùng hai bộ dữ liệu: log chatbot của project, và `sklearn.datasets.load_digits` (ảnh chữ số 8×8, có sẵn offline) khi cần một bài toán mà mạng sâu thật sự có ích.

---

## 04 · Vanishing/exploding gradient, khởi tạo, residual, gradient clipping

> ⏱ Tầng 1: 10' · Tầng 2: 50'
> **Cần biết trước:** backpropagation (Tuần 2, bài 05); phương sai của tổng các biến độc lập (Tuần 2, bài 09); activation (bài 03).

> **📖 Từ mới**
> - **Vanishing gradient:** gradient nhỏ dần theo hàm mũ khi đi ngược qua nhiều lớp, nên các lớp đầu gần như **không học**.
> - **Exploding gradient:** gradient lớn dần theo hàm mũ, nên bước cập nhật "văng" đi và loss thành `nan`.
> - **Fan-in / fan-out:** số đầu vào / số đầu ra của một lớp (với `nn.Linear(256, 128)`: fan-in 256, fan-out 128).
> - **Khởi tạo (initialization):** cách chọn giá trị ngẫu nhiên ban đầu cho trọng số.
> - **Residual (skip) connection:** cộng đầu vào của một khối vào đầu ra của nó: $h + f(h)$.
> - **Gradient clipping:** nếu chuẩn của gradient vượt ngưỡng $c$ thì thu nhỏ nó lại cho bằng $c$.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Trò chơi "truyền tin" qua 20 người:
- Mỗi người **nói nhỏ hơn** người trước một chút (nhân âm lượng với 0.7): tới người thứ 20, âm lượng chỉ còn $0.7^{19} \approx 0.001$. Tin nhắn biến mất. Đó là **vanishing**.
- Mỗi người **nói to hơn** một chút (nhân 1.13): tới người thứ 20 thành gào thét, gấp 10 lần ban đầu. Đó là **exploding**.
- Chỉ khi hệ số **đúng bằng 1** thì tin nhắn đi qua nguyên vẹn.

Trong mạng neural, tín hiệu đi xuôi (activation) và gradient đi ngược đều là **tích** của các hệ số qua từng lớp. Chọn trọng số ban đầu sao cho "hệ số mỗi lớp ≈ 1" là toàn bộ ý tưởng của Xavier/He init.

**Ví dụ có số.** Sigmoid có đạo hàm tối đa 1/4. Qua 10 lớp sigmoid, gradient bị nhân với tối đa $(1/4)^{10} \approx 10^{-6}$. Đo trên máy bạn (10 lớp, rộng 64, Xavier init): gradient ở lớp đầu nhỏ hơn lớp cuối khoảng **$1.5 \times 10^{-6}$ lần**, khớp đúng bậc độ lớn.

**Ba ý cốt lõi:**
1. Độ lớn tín hiệu và gradient thay đổi theo **hàm mũ** của độ sâu; phải giữ "hệ số mỗi lớp" gần 1.
2. **He init** ($\mathrm{Var}(w) = 2/\text{fan-in}$) cho ReLU; **Xavier** ($2/(\text{fan-in} + \text{fan-out})$) cho tanh/tuyến tính.
3. **Residual connection** tạo "đường cao tốc" cho gradient; **clipping** chặn những bước cập nhật quá lớn.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 4.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Vanishing/exploding gradient** | Gradient qua $L$ lớp là tích của $L$ ma trận Jacobian; chuẩn của tích co lại hoặc phình ra theo hàm mũ của $L$ | Hochreiter 1991 (luận văn); Bengio, Simard, Frasconi, *IEEE TNN* 1994 |
| **Xavier / Glorot init** | $\mathrm{Var}(w) = \dfrac{2}{n_\text{in} + n_\text{out}}$: cân bằng giữa giữ phương sai lúc forward ($1/n_\text{in}$) và lúc backward ($1/n_\text{out}$) | Glorot & Bengio, AISTATS 2010 |
| **He / Kaiming init** | $\mathrm{Var}(w) = \dfrac{2}{n_\text{in}}$ cho ReLU. Hệ số 2 bù cho việc ReLU bỏ đi một nửa tín hiệu | He et al., ICCV 2015 |
| **LeCun init** | $\mathrm{Var}(w) = 1/n_\text{in}$ | LeCun et al., *Efficient BackProp* 1998 |
| **Residual connection** | $h_{l+1} = h_l + f_l(h_l)$ | He et al., *Deep Residual Learning*, CVPR 2016 |
| **Gradient clipping (theo chuẩn)** | $g \leftarrow g \cdot \min\!\big(1, \tfrac{c}{\lVert g\rVert}\big)$, với $\lVert g\rVert$ là chuẩn L2 của **toàn bộ** gradient gộp lại | Pascanu, Mikolov, Bengio, ICML 2013 |

#### 4.2 Công thức: vì sao chọn $2/n_\text{in}$

**🐢 Đi chậm từng bước (đọc trước công thức).** Ta theo dõi một con số duy nhất là **phương sai** (độ "to" của tín hiệu) khi nó đi qua từng lớp.

*Bước 1 · Một tích $w \cdot x$.*
- Cho $w \in \{-1, +1\}$, mỗi giá trị 50%. Khi đó $\mathbb E[w] = 0$ và $\mathrm{Var}(w) = 1$.
- Cho $x \in \{0, 2\}$, mỗi giá trị 50%. Khi đó $\mathbb E[x] = 1$, $\mathrm{Var}(x) = 1$, $\mathbb E[x^2] = 2$.
- Bốn trường hợp đồng khả năng của $wx$ là $0, -2, 0, 2$. Trung bình bằng 0; phương sai bằng $(0 + 4 + 0 + 4)/4 =$ **2**.
- Kết quả bằng $\mathrm{Var}(w) \cdot \mathbb E[x^2] = 1 \cdot 2$, **không** bằng $\mathrm{Var}(w) \cdot \mathrm{Var}(x) = 1$.
- Lý do: $w$ có dấu ngẫu nhiên, nên nó biến cả phần **trung bình** của $x$ thành dao động. Vì vậy công thức dùng $\mathbb E[x^2] = \mathrm{Var}(x) + (\mathbb E x)^2$.

*Bước 2 · Cộng $n$ tích.*
- $z = \sum_{i=1}^{n} w_i x_i$ gồm $n$ số hạng độc lập, mỗi số hạng có phương sai $\mathrm{Var}(w)\,\mathbb E[x^2]$.
- Phương sai của tổng các biến độc lập bằng tổng các phương sai, nên $\mathrm{Var}(z) = n\,\mathrm{Var}(w)\,\mathbb E[x^2]$.
- Hình dung 256 người cùng đẩy một chiếc xe, mỗi người đẩy theo một hướng ngẫu nhiên. Lực tổng **không** lớn gấp 256 lần lực của một người, mà chỉ gấp $\sqrt{256} = 16$ lần, vì std tăng theo $\sqrt n$.
- Muốn lực tổng giữ nguyên, mỗi người phải đẩy nhẹ đi 16 lần. Tức là std của $w$ phải tỉ lệ với $1/\sqrt n$.

*Bước 3 · ReLU cắt mất một nửa.*
- Cho $z \in \{-2, -1, 1, 2\}$, đồng khả năng. Khi đó $\mathrm{Var}(z) = (4 + 1 + 1 + 4)/4 = 2.5$.
- Qua ReLU được $x = (0, 0, 1, 2)$, nên $\mathbb E[x^2] = (0 + 0 + 1 + 4)/4 = 1.25$, đúng bằng $\tfrac12 \cdot 2.5$.
- Nửa âm bị đặt về 0 và mang theo đúng một nửa "năng lượng" của tín hiệu.

*Bước 4 · Ghép lại thành hệ số mỗi lớp.*
- Từ bước 2 và bước 3: $\mathrm{Var}(z_l) = n\,\mathrm{Var}(w) \cdot \tfrac12\mathrm{Var}(z_{l-1})$.
- Như vậy, **ở mọi lớp**, phương sai đều bị nhân với cùng hệ số $n\,\mathrm{Var}(w)/2$.

Bảng dưới lấy $n = 256$, đầu vào có phương sai 1; các giá trị là **phương sai của $z$**:

| Lớp | std 0.01: hệ số $256 \cdot 10^{-4}/2 = 0.0128$ | He: hệ số $256 \cdot \tfrac{2}{256}/2 = 1$ |
|---|---|---|
| 1 (đầu vào chưa qua ReLU, $\mathbb E[x^2] = 1$) | $256 \cdot 10^{-4} = 0.0256$ | $256 \cdot \tfrac{2}{256} = 2$ |
| 2 | $0.0256 \times 0.0128 = 3.3 \times 10^{-4}$ | 2 |
| 3 | $4.2 \times 10^{-6}$ | 2 |
| 4 | $5.4 \times 10^{-8}$ | 2 |
| **Mô phỏng trên máy bạn** (4 096 mẫu), lớp 1 → 4 | 0.0256 · $3.6 \times 10^{-4}$ · $4.6 \times 10^{-6}$ · $5.1 \times 10^{-8}$ | 2 · 1.85 · 1.90 · 1.99 |

Chỉ sau 4 lớp, init std 0.01 đã làm phương sai co lại hơn $10^{7}$ lần, còn He giữ nguyên. Lấy căn bậc hai của hệ số thì được hệ số std $\sqrt{0.0128} = 0.113$ trong bảng ở mục 4.3.

*Bước 5 · Chiều ngược.*
- Gradient tại một nơ-ron của lớp dưới cũng là một tổng có trọng số: $\sum_{j=1}^{n_\text{out}} w_{ji} \cdot (\text{gradient tại } z_j) \cdot \mathbb 1[z_j > 0]$.
- Ba bước trên áp dụng y hệt, chỉ khác hai chỗ: số số hạng là $n_\text{out}$, và đạo hàm của ReLU (bằng 0 hoặc 1) cắt đi một nửa.
- Vì vậy hệ số theo chiều ngược là $n_\text{out}\,\mathrm{Var}(w)/2$. Với lớp vuông ($n_\text{in} = n_\text{out}$), cả hai chiều cùng ổn định.

**Công thức (tóm tắt các bước trên).** Xét một lớp $z_j = \sum_{i=1}^{n} w_{ji} x_i$, với $w$ và $x$ độc lập, $\mathbb E[w] = 0$. Khi đó:

$$\mathrm{Var}(z_j) = n \cdot \mathrm{Var}(w) \cdot \mathbb E[x_i^2].$$

- Nếu $x$ là đầu ra của ReLU từ một $z$ đối xứng quanh 0, thì $\mathbb E[x^2] = \tfrac12\mathrm{Var}(z_\text{lớp trước})$, vì một nửa giá trị bị đặt về 0.
- Vậy hệ số phương sai mỗi lớp là $\tfrac{n}{2}\mathrm{Var}(w)$. Muốn hệ số này **bằng 1** thì $\mathrm{Var}(w) = 2/n$: đó là **He init**.
- Hệ số **độ lệch chuẩn** mỗi lớp là $\sqrt{n\,\mathrm{Var}(w)/2}$.

*Vì sao (phác thảo) với gradient:* backward qua một lớp là nhân với $W^\top \mathrm{diag}(\phi'(z))$. Lập luận tương tự cho hệ số $\tfrac{n_\text{out}}{2}\mathrm{Var}(w)$. Xavier lấy trung bình hai yêu cầu bằng cách dùng $n_\text{in} + n_\text{out}$. Với sigmoid, $\phi' \le 1/4$, nên **dù chọn $W$ thế nào** thì mỗi lớp vẫn có thêm một hệ số co nhỏ.

**Residual.** $\dfrac{\partial h_{l+1}}{\partial h_l} = I + \dfrac{\partial f_l}{\partial h_l}$. Tích qua nhiều khối luôn chứa số hạng $I$ (đường đi thẳng không qua $f$ nào), nên gradient từ loss **luôn** có đường tới các lớp đầu mà không bị nhân với hàng chục ma trận nhỏ.

#### 4.3 Ví dụ tính tay: hệ số mỗi lớp

MLP ReLU rộng $n = 256$, đầu vào có std 1. Hệ số std mỗi lớp là $\sqrt{256 \cdot \mathrm{Var}(w)/2}$:

| Khởi tạo | $\mathrm{Var}(w)$ | Hệ số std mỗi lớp | Dự đoán std ở lớp 20 | **Đo trên máy bạn** |
|---|---|---|---|---|
| std 0.01 | $10^{-4}$ | 0.113 | $0.16 \cdot 0.113^{19} \approx 1.7 \times 10^{-19}$ | $1.2 \times 10^{-19}$ |
| Xavier | $2/512$ | 0.707 | $1 \cdot 0.707^{19} \approx 0.0014$ | 0.00097 |
| **He** | $2/256$ | **1.0** | $\approx 1.4$ | 0.99 |
| std 0.1 | $10^{-2}$ | 1.131 | $1.6 \cdot 1.131^{19} \approx 17$ | 11.7 |

(Lớp 1 nhận đầu vào chưa qua ReLU, nên std của lớp 1 là $\sqrt{n\,\mathrm{Var}(w)}$, ví dụ $\sqrt{256 \cdot 10^{-4}} = 0.16$.) Dự đoán và đo khớp về **bậc độ lớn**. Sai lệch nhỏ đến từ độ rộng hữu hạn và từ việc ReLU không hoàn toàn bỏ đúng một nửa.

**Gradient clipping.** $g = (3, 4)$, $\lVert g\rVert = 5$:
- Với $c = 1$: $g \leftarrow (3, 4) \cdot 1/5 = (0.6,\ 0.8)$. **Hướng giữ nguyên**, chỉ độ dài bị cắt (đã kiểm chứng bằng `clip_grad_norm_`, hàm này trả về chuẩn gốc 5.0).
- Với $c = 10$: không đổi.

#### 4.4 Đo trên máy bạn

**Gradient lớp đầu / gradient lớp cuối** (10 lớp ẩn rộng 64, trung vị của 5 seed):

| Activation + init | Tỉ lệ | Nhận xét |
|---|---|---|
| Sigmoid + Xavier | $1.5 \times 10^{-6}$ | Cận $(1/4)^{10} = 9.5 \times 10^{-7}$ |
| Tanh + Xavier | 0.66 | $\tanh'(0) = 1$, nên gần 0 thì không co |
| ReLU + He | 0.29 | Giữ được bậc độ lớn |

**Residual: 50 khối** (mỗi khối Linear–ReLU–Linear, cùng init Xavier):

| | Chuẩn gradient của khối đầu / khối cuối | std đầu ra |
|---|---|---|
| Không có residual | $6.9 \times 10^{-9}$ / $8.7 \times 10^{-9}$ | $4.7 \times 10^{-10}$ (tín hiệu đã chết) |
| Có residual $h + f(h)/\sqrt{50}$ | 0.62 / 0.68 | 1.89 |

Hệ số $1/\sqrt{\text{số khối}}$ giữ cho tổng $50$ nhánh không làm phương sai tăng tuyến tính. GPT-2 dùng đúng ý này khi khởi tạo các lớp chiếu về residual (Radford et al. 2019).

**Một sắc thái quan trọng: Adam che giấu một phần vấn đề.** `TabMLP` **12 lớp ẩn** trên dữ liệu project, huấn luyện bằng Adam:

| Init | Train loss epoch 1 | Epoch tốt nhất | AUC test |
|---|---|---|---|
| std 0.01 | 0.694 (đứng yên ở $\ln 2$) | 29 | 0.767 |
| Xavier | 0.523 | 9 | 0.779 |
| He | 0.411 | 3 | 0.769 |

Init nhỏ vẫn học được, chỉ chậm hơn. Lý do: Adam chia gradient cho $\sqrt{v}$ nên **tự phóng to** các gradient nhỏ. Với SGD thì không được như vậy: ở bài 05, một MLP 10 lớp huấn luyện bằng SGD bị **kẹt hoàn toàn** ở accuracy 10%.

#### 4.5 Code chuẩn

```python
from torch import nn

def init_he(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")    # std = sqrt(2 / fan_in)
            nn.init.zeros_(m.bias)

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, h):
        return h + self.f(h)                                          # "pre-norm" residual: kiểu của Transformer

# trong vòng lặp, giữa backward() và step():
total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # trả về chuẩn TRƯỚC khi cắt: nên log lại
```

#### 4.6 Cạm bẫy

1. **Tưởng init mặc định của `nn.Linear` là He.** Thực ra nó là `kaiming_uniform_(a=√5)`, cho $\mathrm{Var}(w) = 1/(3\,n_\text{in})$, **nhỏ hơn He 6 lần**. Với mạng nông thì không sao; với mạng sâu không có normalization thì tín hiệu tắt dần.
2. **Dùng sigmoid/tanh cho lớp ẩn của mạng sâu.** Sigmoid chỉ nên dùng ở **đầu ra** hoặc ở **cổng** (LSTM).
3. **Clip theo giá trị** (`clip_grad_value_`) thay vì theo chuẩn: mỗi thành phần bị cắt riêng, nên **hướng** gradient bị thay đổi.
4. **Không log chuẩn gradient.** Một đỉnh nhọn trong biểu đồ chuẩn gradient thường xuất hiện **trước** khi loss thành `nan`; đó là tín hiệu tốt nhất để chẩn đoán sớm.
5. **Residual với hai shape khác nhau** (ví dụ CNN đổi số kênh): phải thêm phép chiếu (`1×1 conv` hoặc `Linear`) trên nhánh tắt.

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"He khác Xavier thế nào, và vì sao?"* Hệ số 2 bù cho việc ReLU bỏ một nửa tín hiệu (mục 4.2).
- *"Vì sao ResNet huấn luyện được 152 lớp?"* Đường tắt đồng nhất giữ gradient (số hạng $I$); mỗi khối chỉ cần học **phần dư** $f(h) = \text{mục tiêu} - h$, và học $f \approx 0$ là dễ.
- *"Loss thành `nan` thì làm gì?"* Giảm lr, thêm clipping và warmup, kiểm tra dữ liệu có NaN/inf, kiểm tra `log(0)`; trên fp16 thì kiểm tra loss scaling (bài 06).

#### 4.7 Đọc thêm

- X. Glorot & Y. Bengio, *Understanding the Difficulty of Training Deep Feedforward Neural Networks*, AISTATS 2010.
- K. He et al., *Delving Deep into Rectifiers*, ICCV 2015; *Deep Residual Learning for Image Recognition*, CVPR 2016.
- R. Pascanu, T. Mikolov, Y. Bengio, *On the Difficulty of Training Recurrent Neural Networks*, ICML 2013.
- Goodfellow §8.4 (Parameter Initialization Strategies).

---

## 05 · Normalization: BatchNorm, LayerNorm, RMSNorm

> ⏱ Tầng 1: 10' · Tầng 2: 40'
> **Cần biết trước:** chuẩn hóa z-score (Tuần 3, bài 10); bài 04.

> **📖 Từ mới**
> - **Normalization layer:** lớp đưa activation về trung bình 0, độ lệch chuẩn 1, rồi cho phép mạng **co giãn lại** bằng hai tham số học được $\gamma$ (scale) và $\beta$ (shift).
> - **BatchNorm (BN):** chuẩn hóa **mỗi đặc trưng** theo các mẫu **trong batch**.
> - **LayerNorm (LN):** chuẩn hóa **mỗi mẫu** theo các đặc trưng **của chính nó**.
> - **RMSNorm:** như LN nhưng **không trừ trung bình**, chỉ chia cho căn trung bình bình phương.
> - **Running statistics:** trung bình trượt của mean/var của các batch, lưu trong buffer để dùng lúc **suy luận** (BN).

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Chấm điểm bài thi có hai kiểu "chuẩn hóa":
- **BatchNorm**: với **mỗi câu hỏi**, so điểm của từng học sinh với **điểm trung bình của cả lớp** ở câu đó. Kết quả của một học sinh phụ thuộc vào **cả lớp hôm đó** gồm những ai.
- **LayerNorm**: với **mỗi học sinh**, so điểm từng câu với **điểm trung bình các câu của chính em đó**. Kết quả không phụ thuộc vào người khác. Một học sinh thi một mình (batch 1) vẫn chấm được.

Vì sao chuẩn hóa giúp ích? Nếu activation của một lớp lúc thì cỡ 0.001, lúc thì cỡ 1 000, lớp sau phải liên tục thích nghi lại, và learning rate nào cũng sẽ quá nhỏ cho chỗ này hoặc quá lớn cho chỗ kia. Chuẩn hóa giữ mọi lớp ở **cùng một thang**, nên dùng được lr lớn hơn và ít phụ thuộc vào init hơn.

**Ba ý cốt lõi:**
1. **BN** chuẩn hóa theo **chiều batch**: mạnh với CNN và batch lớn, nhưng train và eval **hành xử khác nhau**, và hỏng khi batch nhỏ.
2. **LN** chuẩn hóa theo **chiều đặc trưng**: không phụ thuộc batch. Đây là lựa chọn của Transformer (sequence dài ngắn khác nhau, batch nhỏ, suy luận từng câu).
3. **RMSNorm** bỏ bước trừ trung bình: rẻ hơn, chất lượng tương đương. LLaMA và hầu hết LLM mới dùng nó.

---

### TẦNG 2 · Kiến thức chuẩn

#### 5.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Batch Normalization** | $\hat x_{ij} = \dfrac{x_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}$, $y_{ij} = \gamma_j\hat x_{ij} + \beta_j$, với $\mu_j, \sigma_j^2$ tính **trên batch** (chỉ số $i$) cho từng đặc trưng $j$ | Ioffe & Szegedy, ICML 2015 |
| **Layer Normalization** | Như trên nhưng $\mu_i, \sigma_i^2$ tính **trên các đặc trưng** (chỉ số $j$) của từng mẫu $i$ | Ba, Kiros, Hinton, arXiv 2016 |
| **RMSNorm** | $y_{ij} = \gamma_j \dfrac{x_{ij}}{\sqrt{\frac1d\sum_j x_{ij}^2 + \epsilon}}$; không có $\beta$ | Zhang & Sennrich, NeurIPS 2019 |
| **GroupNorm** | Chia kênh thành nhóm, chuẩn hóa trong từng nhóm của từng mẫu; thay BN khi batch nhỏ (phát hiện vật thể, phân đoạn ảnh) | Wu & He, ECCV 2018 |
| **Internal covariate shift** | Giải thích ban đầu của BN: phân phối đầu vào của mỗi lớp thay đổi khi các lớp trước học. Về sau bị phản biện: BN giúp chủ yếu vì làm **mặt loss trơn hơn** | Santurkar et al., NeurIPS 2018 |
| **Pre-LN vs Post-LN** | Transformer gốc đặt LN **sau** phép cộng residual; các model hiện đại đặt LN **trước** khối $f$ (`h + f(LN(h))`), ổn định hơn và ít cần warmup hơn | Xiong et al., ICML 2020 |

#### 5.2 Train và eval của BatchNorm

- **Train:** dùng $\mu_B$ và $\sigma_B^2$ của batch hiện tại (variance **lệch**, chia $B$). Đồng thời cập nhật buffer:
  $$r_\mu \leftarrow (1 - m)\,r_\mu + m\,\mu_B, \qquad r_{\sigma^2} \leftarrow (1 - m)\,r_{\sigma^2} + m\,s_B^2,$$
  với $m$ là `momentum` (mặc định 0.1) và $s_B^2$ là variance **không lệch** (chia $B - 1$).
- **Eval:** dùng $r_\mu, r_{\sigma^2}$, nên đầu ra của mỗi mẫu **không** phụ thuộc vào các mẫu khác.
- LN và RMSNorm **không có** chế độ riêng cho train/eval.

*Vì sao cần $\gamma, \beta$?* Chuẩn hóa ép mọi đặc trưng về mean 0, std 1, mà có thể đó không phải thang tốt nhất (ví dụ đầu vào của sigmoid cần nằm ở vùng bão hòa). $\gamma, \beta$ cho phép mạng **học lại** thang phù hợp, kể cả đảo ngược hẳn phép chuẩn hóa nếu cần.

#### 5.3 Ví dụ tính tay

Batch 3 mẫu × 4 đặc trưng (bỏ qua $\epsilon$; $\gamma = 1$, $\beta = 0$):

$$H = \begin{pmatrix} 1 & 2 & 3 & 6 \\ 3 & 2 & 1 & 2 \\ 2 & 2 & 5 & 4 \end{pmatrix}$$

**BatchNorm** (theo cột):
- Cột 1 $(1, 3, 2)$: $\mu = 2$, $\sigma^2 = 2/3$, nên kết quả là $(-1.225,\ 1.225,\ 0)$.
- Cột 2 $(2, 2, 2)$: $\mu = 2$, $\sigma^2 = 0$, nên $\tfrac{0}{0}$! Với $\epsilon = 10^{-5}$ thì kết quả là $(0, 0, 0)$. **Đây là lý do $\epsilon$ tồn tại.**
- Cột 3 $(3, 1, 5)$: $\mu = 3$, $\sigma^2 = 8/3$, nên kết quả là $(0,\ -1.225,\ 1.225)$.

**LayerNorm** (theo hàng). Hàng 2 $(3, 2, 1, 2)$: $\mu = 2$, $\sigma^2 = (1 + 0 + 1 + 0)/4 = 0.5$, nên kết quả là $(1.414,\ 0,\ -1.414,\ 0)$.

**RMSNorm**, hàng 2: $\mathrm{rms} = \sqrt{(9 + 4 + 1 + 4)/4} = \sqrt{4.5} = 2.121$, nên kết quả là $(1.414,\ 0.943,\ 0.471,\ 0.943)$. **Không** có trung bình 0: RMSNorm chỉ chuẩn hóa độ lớn.

**LayerNorm, ví dụ thứ hai** (cũng là test của bài tập hôm nay): $x = (2, 4, 4, 4, 5, 5, 7, 9)$ có $\mu = 5$; độ lệch là $(-3, -1, -1, -1, 0, 0, 2, 4)$, nên $\sigma^2 = 32/8 = 4$ và $\sigma = 2$. Kết quả: $(-1.5,\ -0.5,\ -0.5,\ -0.5,\ 0,\ 0,\ 1,\ 2)$.

**Running stats của BN** (`momentum` 0.1, khởi đầu $r_\mu = 0$, $r_{\sigma^2} = 1$). Hai lần liên tiếp đưa vào batch $(1, 3)$, có $\mu_B = 2$ và $s_B^2 = 2$:
- Sau lần 1: $r_\mu = 0.9 \cdot 0 + 0.1 \cdot 2 = 0.2$; $r_{\sigma^2} = 0.9 + 0.2 = 1.1$.
- Sau lần 2: $r_\mu = 0.38$; $r_{\sigma^2} = 1.19$.

Running stats tiến về giá trị thật **rất chậm**. Vì vậy sau vài batch đầu, đầu ra lúc eval khác hẳn lúc train (đã kiểm chứng bằng `nn.BatchNorm1d`).

#### 5.4 Đo trên máy bạn

**(a) Mạng sâu + SGD: normalization quyết định học được hay không.** MLP 10 lớp ẩn rộng 128 trên `load_digits` (init mặc định của PyTorch, SGD momentum 0.9, 30 epoch, 2 seed):

| | lr = 0.01: train loss epoch 30 / acc test | lr = 0.05: train loss epoch 30 / acc test |
|---|---|---|
| Không chuẩn hóa | 2.303 / **0.10** (kẹt ở đoán ngẫu nhiên, $\ln 10 = 2.303$) | 2.305 / 0.10 |
| BatchNorm | 0.02–0.03 / **0.978–0.980** | 0.06–0.08 / 0.972–0.978 |
| LayerNorm | 0.03–0.14 / 0.941–0.954 | **phân kỳ** / 0.10 |

- Không chuẩn hóa: tín hiệu tắt dần qua 10 lớp (bài 04, cạm bẫy 1), nên SGD không nhúc nhích.
- BN chịu được lr lớn hơn LN trên bài toán này.

**(b) BN với batch nhỏ.** MLP 4 lớp có BN, Adam, 20 epoch:

| Batch | BN, accuracy test | LN, accuracy test |
|---|---|---|
| 2 | **0.711** | 0.961 |
| 8 | 0.980 | 0.982 |
| 64 | 0.991 | 0.983 |

Với batch 2, mean và var của batch chỉ dựa trên **2 mẫu** nên rất nhiễu, và running stats lúc eval không khớp với cái mạng đã "quen" lúc train. LN không bị ảnh hưởng.

**(c) Trên dữ liệu bảng của project**, ảnh hưởng nhỏ và nhiễu. `TabMLP` 8 lớp, Adam, lr $10^{-3}$: không chuẩn hóa 0.760; BN 0.722; LN 0.770 (AUC test, một seed). Với lr $10^{-2}$: 0.762 / 0.756 / 0.783. Mạng nông (2 lớp) mặc định của project **không cần** normalization. Đừng thêm một lớp chỉ vì "mạng nào cũng có".

#### 5.5 Code chuẩn

```python
from torch import nn

nn.BatchNorm1d(64)        # đầu vào (N, 64) hoặc (N, 64, L)
nn.BatchNorm2d(64)        # CNN: (N, C, H, W), chuẩn hóa theo (N, H, W) cho từng kênh C
nn.LayerNorm(512)         # chuẩn hóa chiều CUỐI: (B, T, 512) → mỗi token một lần
nn.RMSNorm(4096, eps=1e-6)

# Linear ngay trước BN không cần bias: BN trừ trung bình nên bias bị triệt tiêu
block = nn.Sequential(nn.Linear(128, 64, bias=False), nn.BatchNorm1d(64), nn.ReLU())

loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)   # tránh batch cuối 1 mẫu khi có BN
```

#### 5.6 Cạm bẫy

1. **Batch cuối có 1 mẫu khi train với BN:** `ValueError: Expected more than 1 value per channel when training` (đã gặp khi chạy thí nghiệm (b): 1 257 mẫu, batch 2). Sửa bằng `drop_last=True`.
2. **Quên `model.eval()`:** BN dùng thống kê của batch test, nên dự đoán của một mẫu phụ thuộc vào các mẫu khác trong batch. Với batch 1 thì còn lỗi luôn.
3. **Fine-tune model có BN trên batch nhỏ:** running stats bị ghi đè bởi thống kê nhiễu. Thường phải **đóng băng BN** (để BN ở chế độ eval, không cập nhật $\gamma, \beta$) khi fine-tune (bài 10).
4. **Gradient accumulation không giúp được BN:** BN vẫn chỉ "thấy" micro-batch (bài 06).
5. **Weight decay lên $\gamma, \beta$ và bias:** thường nên loại chúng khỏi weight decay (mục 6.5).
6. **Huấn luyện phân tán:** mỗi GPU tính BN trên phần batch của nó; muốn dùng thống kê chung phải dùng `SyncBatchNorm`.

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"Vì sao Transformer dùng LayerNorm chứ không dùng BatchNorm?"* Các câu có độ dài khác nhau và có padding nên thống kê theo batch bị nhiễu; batch thường nhỏ; suy luận sinh từng token với batch 1; và LN không tạo ra sự phụ thuộc giữa các mẫu.
- *"RMSNorm khác LayerNorm thế nào?"* Bỏ trừ trung bình và bỏ $\beta$, nên ít phép tính hơn: bài báo gốc đo được nhanh hơn khoảng 7–64% tùy model, chất lượng tương đương.
- *"BN có tác dụng regularization không?"* Có, một chút: nhiễu từ thống kê batch hoạt động như nhiễu ngẫu nhiên. Vì vậy dùng BN thì thường cần dropout ít hơn.

#### 5.7 Đọc thêm

- S. Ioffe & C. Szegedy, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*, ICML 2015.
- J. L. Ba, J. R. Kiros, G. Hinton, *Layer Normalization*, arXiv 2016.
- B. Zhang & R. Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019.
- S. Santurkar et al., *How Does Batch Normalization Help Optimization?*, NeurIPS 2018.
- R. Xiong et al., *On Layer Normalization in the Transformer Architecture*, ICML 2020.

---

## 06 · Regularization và huấn luyện hiệu quả

> ⏱ Tầng 1: 15' · Tầng 2: 45'
> **Cần biết trước:** bias–variance và regularization (Tuần 3, bài 02); Adam (Tuần 2, bài 07); bài 02.

> **📖 Từ mới**
> - **Dropout:** lúc train, **tắt ngẫu nhiên** mỗi neuron với xác suất $p$; lúc eval, giữ tất cả.
> - **Weight decay:** mỗi bước, kéo trọng số về gần 0 một chút.
> - **Early stopping:** dừng huấn luyện khi điểm validation không còn cải thiện, rồi lấy lại trọng số tốt nhất.
> - **Data augmentation:** tạo thêm mẫu train bằng các biến đổi **không đổi nhãn** (lật, dịch, cắt ảnh; thay từ đồng nghĩa trong văn bản).
> - **Mixed precision:** tính phần lớn phép toán ở 16 bit (nhanh, ít bộ nhớ) nhưng giữ bản trọng số 32 bit để cập nhật.
> - **Gradient accumulation:** cộng gradient của nhiều batch nhỏ rồi mới cập nhật một lần, để giả lập một batch lớn.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- **Dropout:** một đội bóng mà mỗi buổi tập có vài người vắng ngẫu nhiên. Không ai được phép "ỷ" vào một ngôi sao; mỗi người phải tự làm được việc, và chiến thuật phải bền với việc thiếu người. Khi thi đấu thật thì đủ cả đội.
- **Early stopping:** ôn thi bằng một bộ đề. Lúc đầu làm lại đề giúp hiểu bài; tới một lúc thì bạn chỉ còn **thuộc đáp án**. Hãy dừng ở lúc điểm thi thử (validation) cao nhất.
- **Augmentation:** học nhận mặt chữ "7" từ ảnh hơi lệch, hơi nghiêng, hơi mờ, để nhận được chữ "7" viết tay thật.

**Ví dụ có số (inverted dropout).** $p = 0.5$, đầu vào $(2, 4, 6, 8)$, mặt nạ ngẫu nhiên $(1, 0, 1, 1)$. Đầu ra là $(2, 0, 6, 8) / 0.5 = (4,\ 0,\ 12,\ 16)$. Chia cho $1 - p$ để **kỳ vọng** của đầu ra bằng đầu vào; nhờ vậy lúc eval không cần sửa gì.

**Ba ý cốt lõi:**
1. Mọi regularization đều đổi một chút bias lấy nhiều variance hơn (Tuần 3, bài 02).
2. Early stopping là regularization **rẻ nhất** và gần như luôn nên dùng.
3. Mixed precision và gradient accumulation không làm model tốt hơn; chúng giúp model **vừa vào** phần cứng.

---

### TẦNG 2 · Kiến thức chuẩn

#### 6.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Dropout** | $y = x \odot m/(1 - p)$, $m_i \sim \text{Bernoulli}(1 - p)$ lúc train; $y = x$ lúc eval. Có thể xem như huấn luyện trung bình của $2^n$ mạng con dùng chung trọng số | Srivastava et al., JMLR 2014 |
| **L2 regularization** | Thêm $\tfrac{\lambda}{2}\lVert\theta\rVert^2$ vào loss | Goodfellow §7.1 |
| **Decoupled weight decay (AdamW)** | $\theta \leftarrow \theta - \eta\big(\hat m/(\sqrt{\hat v} + \epsilon) + \lambda\theta\big)$: phần decay **không** bị chia cho $\sqrt{\hat v}$. Với SGD, L2 và weight decay là một; với Adam thì **khác nhau** | Loshchilov & Hutter, ICLR 2019 |
| **Early stopping** | Theo dõi metric validation, giữ checkpoint tốt nhất, dừng sau `patience` epoch không cải thiện. Với model tuyến tính và loss bậc hai, nó tương đương xấp xỉ L2 | Goodfellow §7.8; Prechelt 1998 |
| **Data augmentation** | Biến đổi đầu vào mà không đổi nhãn, để model học **tính bất biến** (invariance) | Shorten & Khoshgoftaar, *J. Big Data* 2019 |
| **Label smoothing** | Thay nhãn one-hot bằng $(1 - \varepsilon)\cdot$one-hot $+ \varepsilon/K$, để model bớt quá tự tin | Szegedy et al., CVPR 2016 |
| **Mixed precision** | Forward/backward ở fp16 hoặc bf16; giữ **master weights** fp32; với fp16 thì nhân loss với một hệ số lớn (*loss scaling*) để gradient nhỏ không bị làm tròn về 0 | Micikevicius et al., ICLR 2018 |
| **Gradient accumulation** | Chia batch $B$ thành $k$ micro-batch; backward `loss / k` cho từng micro-batch, rồi `step` một lần | — |

#### 6.2 Các con số cần biết: bộ nhớ khi huấn luyện

Mỗi tham số khi huấn luyện bằng Adam cần (Rajbhandari et al., *ZeRO*, SC 2020):

| Thành phần | fp32 thuần | Mixed precision |
|---|---|---|
| Trọng số | 4 byte | 2 (fp16/bf16) + 4 (master fp32) |
| Gradient | 4 | 2 |
| Adam $m$, $v$ | 8 | 8 (fp32) |
| **Tổng** | **16 byte** | **16 byte** |

Mixed precision **không** giảm bộ nhớ cho tham số. Nó giảm bộ nhớ **activation** (lưu ở 16 bit) và tăng tốc nhờ Tensor Core. **Ước lượng nhanh:** full fine-tune một model 7B bằng Adam cần $16 \times 7 \times 10^9 = 112$ GB, chưa kể activation. Không vừa GPU T4 16 GB của Kaggle. Đó là lý do có LoRA và QLoRA (Tuần 11).

**Weight decay mạnh cỡ nào?** Mỗi bước AdamW nhân trọng số với $(1 - \eta\lambda)$:
- $\eta = 10^{-3}$, $\lambda = 0.01$, sau 10 000 bước: $(1 - 10^{-5})^{10\,000} \approx e^{-0.1} = 0.905$.
- Thí nghiệm ở mục 6.4: $\lambda = 0.1$, 750 bước, nên hệ số là $0.928$. Quá yếu để thấy khác biệt. Đây là lý do weight decay gần như không đổi kết quả trong bảng dưới.

#### 6.3 Ví dụ tính tay: gradient accumulation

Batch 128 chia thành 4 micro-batch 32. Loss trung bình của batch lớn là $\tfrac{1}{128}\sum_{i=1}^{128}\ell_i = \tfrac14\sum_{k=1}^{4}\Big(\tfrac{1}{32}\sum_{i \in k}\ell_i\Big)$. Vì vậy mỗi micro-batch phải backward **`loss_k / 4`**, và tổng gradient cộng dồn đúng bằng gradient của batch 128. Đo trên máy bạn:
- Chênh lệch tối đa giữa hai cách là $2.4 \times 10^{-8}$ (sai số làm tròn float32).
- Nếu **quên chia cho 4**, gradient lớn gấp đúng **4.0 lần**, tương đương vô tình tăng lr lên 4 lần.

#### 6.4 Đo trên máy bạn

**(a) Overfit trên dữ liệu nhỏ.** `TabMLP` 2 lớp rộng 128, chỉ train trên **600** mẫu, chạy đủ 150 epoch (không early stopping):

| Dropout | Weight decay | Val loss tốt nhất (tại epoch) | Train loss epoch 150 | Val loss epoch 150 | AUC test (epoch 150) |
|---|---|---|---|---|---|
| 0 | 0 | 0.393 (6) | **0.026** | **0.857** | 0.707 |
| 0.3 | 0 | 0.393 (12) | 0.244 | 0.493 | 0.714 |
| 0.5 | 0 | 0.392 (34) | 0.320 | 0.426 | 0.722 |
| 0 | 0.1 | 0.393 (6) | 0.029 | 0.811 | 0.709 |

Đọc bảng:
- Không regularization: train loss về gần 0 (học thuộc 600 mẫu), val loss **tăng gấp đôi** so với điểm tốt nhất. Đó là overfit điển hình.
- Dropout **làm chậm** overfit (điểm tốt nhất dời từ epoch 6 sang 34) và giữ val loss cuối thấp hơn nhiều.
- **Một bất ngờ đáng nhớ:** AUC gần như không đổi (0.707–0.722) dù val loss tăng từ 0.39 lên 0.86. Overfit làm model **quá tự tin** (xác suất bị đẩy về 0 hoặc 1, nên log-loss tăng vọt khi sai), nhưng **thứ hạng** giữa các mẫu thì hỏng ít hơn nhiều. Bài học: early stopping theo **val loss** bảo vệ **calibration** (Tuần 3, bài 04); chọn metric dừng theo đúng thứ bạn cần.

**(b) Augmentation.** MLP 64–128–10 trên `load_digits`; augmentation là dịch vòng ảnh ngẫu nhiên $\{-1, 0, 1\}$ pixel theo mỗi chiều:

| | Train loss cuối | Acc test | Acc test khi ảnh bị dịch 1 pixel |
|---|---|---|---|
| Không augmentation | 0.008 | **0.976** | 0.489 |
| Có augmentation | 0.264 | 0.932 | **0.937** |

MLP không "biết" rằng ảnh dịch đi một pixel vẫn là cùng chữ số; augmentation **dạy** cho nó điều đó, đổi lại một ít accuracy trên ảnh sạch (cùng 60 epoch, bài toán khó hơn nên cần train lâu hơn). Bài 07: CNN có sẵn tính chất này một phần nhờ kiến trúc.

**(c) Mixed precision trên CPU.** Bài 01 đã đo: bf16 trên CPU này **chậm hơn fp32 khoảng 850 lần**. Mixed precision là kỹ thuật của **GPU**. Trên Kaggle T4 (không hỗ trợ bf16 nhanh), dùng fp16 kèm `GradScaler`.

#### 6.5 Code chuẩn

```python
# 1) AdamW, không decay bias và tham số của norm
decay = [p for n, p in model.named_parameters() if p.ndim >= 2]          # ma trận trọng số, embedding
no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]        # bias, γ/β của LayerNorm/BatchNorm
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                         {"params": no_decay, "weight_decay": 0.0}], lr=3e-4)

# 2) Mixed precision + gradient accumulation (trên GPU)
scaler = torch.amp.GradScaler("cuda")
accum = 4
for i, (xb, yb) in enumerate(loader):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = loss_fn(model(xb), yb) / accum                             # chia cho số micro-batch
    scaler.scale(loss).backward()                                         # loss scaling cho fp16
    if (i + 1) % accum == 0:
        scaler.unscale_(opt)                                              # bỏ scale TRƯỚC khi clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad()
```

#### 6.6 Cạm bẫy

1. **Dropout khi eval:** quên `model.eval()` (bài 02, mục 2.4b).
2. **Augmentation làm đổi nhãn:** xoay 180° biến "6" thành "9"; lật ngang ảnh chữ viết; thay từ phủ định trong văn bản cảm xúc.
3. **Augmentation trên tập validation/test:** chỉ augment **train**. (Ngoại lệ có chủ đích: *test-time augmentation*, tức lấy trung bình dự đoán trên nhiều phiên bản biến đổi.)
4. **Dùng `Adam(weight_decay=...)` và tưởng là AdamW:** `Adam` cộng $\lambda\theta$ vào gradient (L2), nên phần đó bị chia cho $\sqrt{\hat v}$: tham số có gradient lớn thì hầu như không bị decay. Dùng `AdamW`.
5. **Early stopping trên tập test:** đó là tune trên test (Tuần 3, bài 01).
6. **Clip khi gradient còn đang bị scale** (fp16): phải `unscale_` trước, nếu không ngưỡng clip mất ý nghĩa.
7. **Đặt dropout ngay trước BatchNorm:** variance lúc train và lúc eval lệch nhau (*variance shift*, Li et al., CVPR 2019). Thường chỉ dùng một trong hai, hoặc đặt dropout sau lớp BN cuối.

**Sắc thái hay bị hỏi khi phỏng vấn:**
- *"Vì sao gọi là inverted dropout?"* Phép chia cho $1 - p$ được làm **lúc train** thay vì nhân $1 - p$ lúc eval, nên code suy luận không cần biết dropout tồn tại.
- *"Adam + L2 khác AdamW thế nào?"* Mục 6.1 và cạm bẫy 4.
- *"Không đủ VRAM để dùng batch 64 thì làm gì?"* Gradient accumulation (nhớ chia loss), mixed precision, gradient checkpointing, batch nhỏ hơn kèm lr nhỏ hơn. **Lưu ý:** accumulation không giúp BatchNorm.
- *"fp16 khác bf16 chỗ nào?"* bf16 có 8 bit mũ như fp32 (cùng dải giá trị, không cần loss scaling) nhưng chỉ 7 bit mantissa; fp16 có dải hẹp (tối đa 65 504) nên cần loss scaling.

#### 6.7 Đọc thêm

- N. Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, JMLR 2014.
- I. Loshchilov & F. Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019.
- P. Micikevicius et al., *Mixed Precision Training*, ICLR 2018.
- S. Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, SC 2020.
- Goodfellow ch. 7 (Regularization for Deep Learning).
- PyTorch docs, *Automatic Mixed Precision* (`torch.amp`).

---

## Bài tập Ngày 2 · `qnet/layers.py` (~2h)

Tự viết các lớp của hôm nay; test đối chiếu từng lớp với `torch.nn`. Các lớp này sẽ được `TabMLP` và `TabAttn` của project dùng thật. Lệnh test: `python -m pytest -q tests/test_layers.py` (8 test).

| Thành phần | Yêu cầu |
|---|---|
| `Dropout(p=0.5)` | `nn.Module`. Nếu $p \notin [0, 1)$ thì raise `ValueError`. Khi `self.training` và $p > 0$: mặt nạ `torch.rand_like(x) >= p`, trả về `x * mask / (1 - p)`. Còn lại trả về `x` nguyên vẹn |
| `LayerNorm(d, eps=1e-5)` | Tham số `weight` (khởi tạo bằng 1) và `bias` (bằng 0). Chuẩn hóa theo **chiều cuối**, dùng variance **lệch**. Phải khớp `nn.LayerNorm` cả forward lẫn gradient |
| `RMSNorm(d, eps=1e-6)` | Chỉ có `weight`; công thức ở mục 5.1. Khớp `nn.RMSNorm(d, eps=...)` |
| `BatchNorm1d(d, eps=1e-5, momentum=0.1)` | Tham số `weight`, `bias`; **buffer** `running_mean` (khởi tạo 0) và `running_var` (khởi tạo 1), đăng ký bằng `register_buffer`. Train: chuẩn hóa bằng mean/var (lệch) của batch theo chiều 0; cập nhật running stats như mục 5.2 (var **không lệch**) trong `torch.no_grad()`. Eval: dùng running stats |
| `init_(module, scheme="he")` | Với **mọi** `nn.Linear` bên trong `module`: trọng số $\sim \mathcal N(0, \text{std}^2)$ với `"he"`: $\sqrt{2/\text{fan-in}}$, `"xavier"`: $\sqrt{2/(\text{fan-in} + \text{fan-out})}$, `"small"`: 0.01; bias bằng 0. Scheme lạ thì raise `ValueError`. Trả về `module` |
| `activation_stds(model, x)` | Dùng `register_forward_hook` trên **mọi** `nn.Linear` để ghi `out.std()` theo thứ tự forward, chạy `model(x)` trong `no_grad`, rồi **gỡ hết hook** (kể cả khi có lỗi: `try/finally`). Trả về list float |
| `layer_grad_norms(model, loss)` | `model.zero_grad()`, `loss.backward()`, rồi trả về chuẩn L2 của `weight.grad` của từng `nn.Linear` (theo thứ tự `model.modules()`) |

**Gợi ý:**
- Lấy shape của `nn.Linear`: `fan_out, fan_in = m.weight.shape`.
- Test `test_vanishing_gradient_sigmoid_vs_relu` là phiên bản thu nhỏ của bảng ở mục 4.4. Hãy in ra các tỉ lệ để tự thấy.
