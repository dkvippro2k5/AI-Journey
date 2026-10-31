# Quiz · Ngày 3 · CNN, RNN, attention, transfer learning (bài 07–10)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> Làm trên giấy, được dùng máy tính cầm tay. **Không chạy code.** Đáp án ở cuối file.

## Phần 1 — Quick test (14 câu, câu 14 được 2đ, các câu khác 1đ)

1. Ảnh 32×32, conv $k = 3$, $p = 1$. Kích thước đầu ra khi $s = 1$ và khi $s = 2$?
2. `nn.Conv2d(16, 32, kernel_size=5)` có bao nhiêu tham số?
3. Chồng 4 lớp conv 3×3 stride 1. Receptive field của một điểm ở lớp cuối là bao nhiêu?
4. Cùng lớp `nn.Conv2d(3, 64, 3)`, áp lên ảnh 32×32 và ảnh 1024×1024. Số tham số thay đổi thế nào? Số FLOP thay đổi thế nào?
5. Vì sao MLP trên `load_digits` sụp từ 0.976 xuống 0.487 khi ảnh test bị dịch 1 pixel, còn CNN chỉ giảm xuống 0.835?
6. `nn.LSTM(input_size=8, hidden_size=16)` có bao nhiêu tham số?
7. Với cùng input và hidden, số tham số GRU bằng bao nhiêu phần số tham số LSTM?
8. Nếu forget gate luôn bằng 0.9, sau 20 bước còn lại bao nhiêu phần của $c_0$ theo đường đi qua cell? Nếu forget gate bằng 0.5 thì sao?
9. Bidirectional LSTM dùng được cho bài toán nào: (A) phân loại cảm xúc của cả câu; (B) sinh token tiếp theo?
10. Điểm attention $(3, 1, 1)$. Tính trọng số sau softmax.
11. $q, k \in \mathbb R^{256}$ với các thành phần độc lập, mean 0, var 1. Tính độ lệch chuẩn của $q \cdot k$. Vì sao phải chia cho $\sqrt d$?
12. Beam search với beam width $k = 1$ tương đương với phương pháp nào?
13. Có 200 ảnh X-quang được gắn nhãn, và một model pretrain trên ImageNet. Bắt đầu bằng feature extraction hay fine-tune toàn bộ? Vì sao?
14. *(2đ)* Đầu vào `(1, 3, 32, 32)` đi qua: `Conv2d(3, 16, 3, padding=1)` → `MaxPool2d(2)` → `Conv2d(16, 32, 3)` → `MaxPool2d(2)` → `Flatten` → `Linear(?, 10)`. Ghi shape sau mỗi lớp, số đầu vào của Linear, và tổng số tham số.

## Phần 2 — Tự luận (4 câu, 30 phút)

**T1 · Convolution tính tay (4đ).** Ảnh $X$ (5×5) và kernel Laplacian $K$:

$$X = \begin{pmatrix} 1 & 2 & 3 & 0 & 1 \\ 0 & 1 & 2 & 3 & 1 \\ 1 & 0 & 1 & 2 & 2 \\ 2 & 1 & 0 & 1 & 0 \\ 1 & 2 & 1 & 0 & 1 \end{pmatrix}, \qquad K = \begin{pmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{pmatrix}$$

- (a) Tính đầu ra (không padding, stride 1).
- (b) Đầu ra với stride 2 (không padding) là gì? Có cần tính lại từ đầu không?
- (c) Một lớp `Conv2d(64, 128, 3, padding=1)` trên feature map 56×56: tính số tham số và số FLOP.
- (d) Tính receptive field của chuỗi: conv3 → conv3 → maxpool 2 → conv3 → conv3 → maxpool 2 (mọi conv stride 1).

**T2 · Một bước LSTM (4đ).** LSTM vô hướng với $x = 2$, $h_0 = 0.5$, $c_0 = -1$:

| Cổng | Trọng số theo $x$ | Trọng số theo $h$ | Bias |
|---|---|---|---|
| $i$ | 0.5 | 1 | 0 |
| $f$ | 0 | 1 | 1 |
| $g$ | 1 | −1 | 0 |
| $o$ | 1 | 0 | −1 |

- (a) Tính $i, f, g, o$, rồi $c_1$ và $h_1$.
- (b) Tính $\partial c_1/\partial c_0$. Nếu forget gate giữ giá trị này trong 30 bước, đường đi qua cell còn lại bao nhiêu?
- (c) So với RNN thường $h_t = \tanh(w h_{t-1} + u x_t)$, vì sao đạo hàm qua cell của LSTM "sạch" hơn?

**T3 · Phân tích (3đ).** Một bạn huấn luyện seq2seq GRU **không attention** cho bài toán copy chuỗi chữ số và thu được bảng dưới (accuracy cả chuỗi). Nhóm của bạn ấy định tăng hidden size từ 64 lên 512 để giải quyết.

| Độ dài | 5 | 10 | 20 |
|---|---|---|---|
| Không attention | 0.944 | 0.187 | 0.000 |

- (a) Giải thích nguyên nhân bằng khái niệm **bottleneck**. Tăng hidden size giúp được tới đâu, và vì sao đó không phải lời giải tốt?
- (b) Mô tả cách thêm attention: query, key, value là gì ở mỗi bước decoder? Chi phí thêm cho mỗi bước là bao nhiêu FLOP với $L = 20$, $d = 64$ (tính phần điểm số và tổng có trọng số)?
- (c) Khi giải mã, bạn ấy dùng greedy. Nêu một tình huống beam search cho kết quả tốt hơn, và một tình huống không nên dùng beam search.

**T4 · Code (4đ).** Một batch chuỗi đã được **pad bằng 0 ở cuối**, đưa qua `nn.LSTM(batch_first=True)` cho ra `out` có shape `(B, T, H)`; `lengths` là tensor `(B,)` chứa độ dài thật. Viết:
- `last_hidden(out, lengths) -> (B, H)`: trạng thái ẩn ở vị trí thật cuối cùng của mỗi chuỗi;
- `masked_mean(out, lengths) -> (B, H)`: trung bình **chỉ trên các vị trí thật**.

Giải thích vì sao `out[:, -1]` sai, và cho biết hàm nào của PyTorch giúp tránh vấn đề này ngay từ đầu.

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Quick test**

1. $s = 1$: $\lfloor (32 + 2 - 3)/1 \rfloor + 1 =$ **32**. $s = 2$: $\lfloor 31/2 \rfloor + 1 =$ **16**.
2. $32(16 \cdot 25 + 1) =$ **12 832**.
3. $1 + 4 \cdot 2 =$ **9**.
4. Số tham số **không đổi** (1 792). FLOP tỉ lệ với $H_\text{out}W_\text{out}$, nên lớn gấp khoảng $(1022/30)^2 \approx$ **1 160 lần**.
5. Với MLP, mỗi pixel là một đầu vào riêng có trọng số riêng; dịch ảnh làm **mọi** đầu vào đổi chỗ. CNN chia sẻ trọng số theo vị trí (equivariance) và có pooling, nên đặc trưng chỉ "dịch theo" và được gộp lại. Bất biến chỉ là một phần, nên vẫn giảm xuống 0.835.
6. $4 \cdot 16(8 + 16 + 2) =$ **1 664**.
7. **3/4**.
8. $0.9^{20} =$ **0.122**; $0.5^{20} =$ **$9.5 \times 10^{-7}$**.
9. **A**. Với B, nhánh ngược nhìn thấy tương lai (rò rỉ).
10. $e^3 = 20.09$, $e^1 = 2.718$; tổng 25.52, nên trọng số là **$(0.787,\ 0.107,\ 0.107)$**.
11. Var $= 256$, nên std = **16**. Không chia thì softmax gần như one-hot (bão hòa), và gradient tới các vị trí khác gần 0. Chia $\sqrt d$ đưa std về 1.
12. **Greedy decoding**.
13. **Feature extraction** trước (ít dữ liệu); rồi thử fine-tune vài lớp cuối với lr nhỏ, và so với nhau trên validation. Miền X-quang khá xa ImageNet, nên fine-tune một phần thường thắng khi có đủ dữ liệu. Luôn so với baseline.
14. Shape: `(1, 16, 32, 32)` → `(1, 16, 16, 16)` → `(1, 32, 14, 14)` → `(1, 32, 7, 7)` → `(1, 1568)` → `(1, 10)`. Linear nhận **1 568** đầu vào. Tham số: $448 + 4\,640 + 15\,690 =$ **20 778**. *(Shape 1đ · tham số 1đ)*

**T1** *(đã chạy kiểm chứng bằng `F.conv2d`)*

(a) Mỗi ô = tổng của 4 hàng xóm trên/dưới/trái/phải − 4 × ô ở giữa. Ví dụ ô $(0, 0)$ có tâm $X_{1,1} = 1$ và các hàng xóm $2, 0, 0, 2$, nên kết quả là $4 - 4 = 0$. *(1.5đ)*

$$\begin{pmatrix} 0 & 0 & -7 \\ 4 & 0 & -1 \\ 0 & 4 & -2 \end{pmatrix}$$

Laplacian cho ra 0 ở vùng "phẳng" và giá trị lớn ở chỗ có điểm sáng/tối đột ngột (ô $-7$ ứng với $X_{1,3} = 3$, sáng hơn hẳn xung quanh).

(b) Stride 2 lấy các vị trí $(0, 0), (0, 2), (2, 0), (2, 2)$ của kết quả (a): $\begin{pmatrix} 0 & -7 \\ 0 & -2 \end{pmatrix}$. **Không cần** tính lại: stride chỉ là lấy mẫu thưa từ kết quả stride 1 (cùng padding). *(0.5đ)*

(c) Tham số: $128(64 \cdot 9 + 1) =$ **73 856**. FLOP: $2 \cdot 64 \cdot 9 \cdot 128 \cdot 56 \cdot 56 \approx$ **$4.6 \times 10^8$**. *(1đ)*

(d) $r$: $1 \to 3 \to 5 \to 6$ (pool, rồi $j = 2$) $\to 10 \to 14 \to 16$ (pool). Vậy **16**. *(1đ)*

**T2** *(đã chạy kiểm chứng bằng `nn.LSTMCell`)*

(a) Tiền kích hoạt: $z_i = 1 + 0.5 = 1.5$; $z_f = 0.5 + 1 = 1.5$; $z_g = 2 - 0.5 = 1.5$; $z_o = 2 - 1 = 1$. *(2đ)*
- $i = f = \sigma(1.5) = 0.818$; $g = \tanh(1.5) = 0.905$; $o = \sigma(1) = 0.731$.
- $c_1 = 0.818 \cdot (-1) + 0.818 \cdot 0.905 =$ **−0.078**.
- $h_1 = 0.731 \cdot \tanh(-0.078) =$ **−0.057**.

(b) $\partial c_1/\partial c_0 = f =$ **0.818** (autograd cho đúng 0.8176). Qua 30 bước: $0.818^{30} \approx$ **0.0024**. Vẫn vanishing, chỉ chậm hơn. Muốn nhớ 30 bước thì $f$ phải gần 1 hơn nhiều (ví dụ $f = 0.99$ cho $0.74$), và đó là lý do khởi tạo forget bias dương. *(1đ)*

(c) RNN: $\partial h_t/\partial h_{t-1} = \mathrm{diag}(1 - h_t^2)\,W$, tức nhân với ma trận trọng số **và** đạo hàm tanh ở mỗi bước, nên tích chuẩn của chúng co hoặc giãn theo hàm mũ. LSTM (theo đường đi qua cell): $\partial c_t/\partial c_{t-1} = \mathrm{diag}(f_t)$, không có $W$, không có $\tanh'$, và mạng **điều khiển** được giá trị này qua cổng. *(1đ)*

**T3**

(a) Toàn bộ chuỗi vào phải được nén vào **một** vector cuối của encoder, sau khi đã bị trộn qua $L$ bước tuần tự (bottleneck). Chuỗi dài 20 cần khoảng 66 bit; về lý thuyết 64 hay 512 số thực đều đủ chứa, nhưng vấn đề là **học được** cách nén và giải nén qua $L$ bước (gradient phải đi qua $O(L)$ bước). Tăng hidden size có thể giúp một ít với $L$ vừa phải, nhưng tốn tham số theo $H^2$, và vẫn hỏng khi $L$ tăng tiếp. *(1đ)*

(b) Ở mỗi bước $t$ của decoder: **query** là trạng thái decoder $s_t$; **key = value** là mọi trạng thái encoder $h_1..h_L$. Tính $\alpha = \mathrm{softmax}(h_j \cdot s_t/\sqrt d)$ và context $c_t = \sum_j\alpha_j h_j$, rồi đưa $[s_t; c_t]$ vào lớp dự đoán (và có thể vào bước sau). Chi phí: điểm số $2Ld$ cộng tổng có trọng số $2Ld$, tức $4 \cdot 20 \cdot 64 = 5\,120$ FLOP mỗi bước. Rất rẻ so với một bước GRU (khoảng $2 \cdot 3 \cdot 2 \cdot 64^2 \approx 49\,000$ FLOP). Đo trên máy bạn: thêm attention đưa độ chính xác cả chuỗi ở $L = 20$ từ 0.000 lên 0.980. *(1đ)*

(c) *(1đ)*
- **Beam tốt hơn:** khi bước đầu có hai lựa chọn gần ngang nhau nhưng một lựa chọn dẫn tới phần tiếp theo rất chắc chắn (ví dụ ở mục 9.3: greedy cho 0.20, beam cho 0.36). Điều này hay gặp ở dịch máy và tóm tắt.
- **Không nên:** sinh văn bản mở hoặc chat (beam cho văn bản lặp và nhạt; nên dùng sampling); hoặc khi độ trễ rất quan trọng (beam tốn gấp $k$ lần).
- Với bài copy, model đã gần như chắc chắn ở mỗi bước, nên beam gần như không đổi kết quả.

**T4** *(đã chạy kiểm chứng: `last_hidden` khớp đúng `h_n` của `pack_padded_sequence`; `out[:, -1]` thì không)*

```python
def last_hidden(out, lengths):
    return out[torch.arange(out.shape[0]), lengths - 1]           # mỗi hàng lấy vị trí lengths[b] - 1

def masked_mean(out, lengths):
    mask = (torch.arange(out.shape[1])[None, :] < lengths[:, None]).unsqueeze(-1).to(out.dtype)  # (B, T, 1)
    return (out * mask).sum(1) / lengths[:, None].to(out.dtype)
```

- `out[:, -1]` lấy vị trí $T - 1$. Với chuỗi ngắn hơn $T$, đó là trạng thái **sau khi đã đọc thêm các token đệm 0**: không phải biểu diễn của câu thật, và khác nhau tùy chuỗi dài hay ngắn.
- Tránh từ đầu: `nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)`. Khi đó LSTM bỏ qua phần đệm và `h_n` chính là trạng thái thật cuối cùng.

*(Chấm: mỗi hàm 1.5đ; giải thích được 1đ.)*

</details>
