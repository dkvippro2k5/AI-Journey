# Quiz · Ngày 1 · PyTorch và MLP (bài 01–03)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> Làm trên giấy, được dùng máy tính cầm tay. **Không chạy code.** Đáp án ở cuối file.

## Phần 1 — Quick test (14 câu, câu 14 được 2đ, các câu khác 1đ)

1. `torch.from_numpy(np.ones(3))` có dtype gì? Đưa tensor đó vào một `nn.Linear` vừa tạo thì chuyện gì xảy ra?
2. `w = torch.tensor(1.0, requires_grad=True)`. Gọi `(w * 5).backward()` **hai lần** liên tiếp thì `w.grad` bằng bao nhiêu?
3. $f = (ab + c)^2$ với $a = 2$, $b = 3$, $c = -4$. Tính $\partial f/\partial a$ và $\partial f/\partial c$.
4. Trong vòng lặp huấn luyện, viết `total_loss += loss` (thay vì `loss.item()`) gây ra vấn đề gì?
5. Đúng/Sai: `model.eval()` tắt việc tính gradient.
6. $n = 10\,000$ mẫu, batch 256. Mỗi epoch có bao nhiêu bước cập nhật?
7. `CosineAnnealingLR` với $\eta_0 = 0.1$, $\eta_{\min} = 0$, $T = 100$. Tính $\eta$ tại bước 50 và bước 25.
8. Dữ liệu có 20% mẫu dương. BCE của model "luôn đoán $p = 0.2$" bằng bao nhiêu?
9. `nn.Linear(100, 50)` có bao nhiêu tham số?
10. MLP 784 → 256 → 128 → 10 (có bias, ReLU ở giữa) có bao nhiêu tham số?
11. Vì sao chồng 10 lớp `nn.Linear` **không có** activation ở giữa cũng không mạnh hơn một lớp `nn.Linear`?
12. Cho biết $\sigma'(0)$, $\tanh'(0)$ và $\mathrm{ReLU}'(-3)$.
13. `nn.Embedding(1000, 16)` có bao nhiêu tham số? Nó tương đương với phép toán nào trên one-hot?
14. *(2đ)* Đoạn code dưới đây (dùng `loss_fn = nn.BCEWithLogitsLoss()`) có **3 lỗi**. Chỉ ra và sửa.
    ```python
    model.train()
    total = 0
    for xb, yb in loader:
        loss = loss_fn(torch.sigmoid(model(xb)), yb)
        loss.backward()
        optimizer.step()
        total += loss
    ```

## Phần 2 — Tự luận (4 câu, 30 phút)

**T1 · Forward, backward và một bước SGD (4đ).** Mạng 2 → 2 → 1:
- $x = (1, 2)$, $W_1 = \begin{pmatrix} 1 & -1 \\ 0.5 & 0.5 \end{pmatrix}$, $b_1 = (0, -1)$, ReLU;
- $w_2 = (2, -1)$, $b_2 = 0.5$, đầu ra là logit $z$; nhãn $y = 1$; loss là BCE với logit.

(a) Tính $a = W_1 x + b_1$, $h$, $z$, $p = \sigma(z)$ và $L$.
(b) Tính gradient theo $w_2$, $b_2$, $W_1$ và $b_1$. Tham số nào có gradient bằng 0, và vì sao?
(c) Thực hiện một bước SGD với lr $= 0.1$, rồi tính lại $z$ và $L$. Loss có giảm không?

**T2 · Chẩn đoán (4đ).** Một bạn huấn luyện `TabMLP` trên dữ liệu 15.5% dương và gặp ba tình huống. Với mỗi tình huống, nêu **nguyên nhân có khả năng nhất** và **cách kiểm tra**.
- (a) Train loss đứng yên ở **0.432** từ epoch 1 tới epoch 50.
- (b) Val loss lúc 0.36, lúc 0.41, lúc 0.38 **dù trọng số không đổi** (chạy `evaluate` hai lần liên tiếp cho hai kết quả khác nhau).
- (c) Loss bằng `nan` sau khoảng 20 bước với lr $= 1.0$.

**T3 · Thiết kế và đếm (3đ).** Bài toán dự đoán thuê bao rời mạng có 30 cột số và 3 cột phân loại: `tinh` (63 giá trị), `goi_cuoc` (200), `dong_may` (500). Dùng `TabMLP` với embedding chiều $\min(50, \lfloor (K + 1)/2 \rfloor)$ cho mỗi cột ($K$ là số giá trị, cộng thêm chỉ số 0 cho giá trị lạ), lớp ẩn (128, 64), head ra 1 logit.
- (a) Tính số tham số của phần embedding, của phần Linear, và tổng.
- (b) Ước lượng FLOP để huấn luyện **một epoch** trên 1 triệu dòng (dùng quy tắc "6 × số trọng số Linear × số mẫu", bỏ qua embedding). Trên CPU đạt khoảng 100 GFLOP/s thì mất bao lâu? Vì sao thực tế sẽ chậm hơn?

**T4 · Code (4đ).** Viết `overfit_one_batch(model, batch, loss_fn, steps=200, lr=1e-2) -> list[float]`: lặp lại `steps` bước Adam trên **cùng một** batch `(*xs, y)` và trả về danh sách loss. Sau đó trả lời:
- Dùng hàm này để làm gì?
- Với `TabMLP` trên 32 mẫu, bạn kỳ vọng loss cuối cùng cỡ bao nhiêu?
- Nếu lỡ thêm `torch.sigmoid` vào cuối `forward`, loss cuối sẽ thế nào?

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Quick test**

1. **`torch.float64`**. Đưa vào `nn.Linear` (float32) sẽ báo `RuntimeError: mat1 and mat2 must have the same dtype`.
2. **10**: gradient cộng dồn $5 + 5$.
3. $ab + c = 2$, nên $f = 4$. $\partial f/\partial a = 2(ab + c)\,b =$ **12**; $\partial f/\partial c = 2(ab + c) =$ **4**. (Thêm: $\partial f/\partial b = 8$.)
4. `loss` vẫn giữ đồ thị tính toán; cộng dồn nghĩa là **giữ lại đồ thị của mọi batch**, nên bộ nhớ tăng dần theo từng bước.
5. **Sai**. `eval()` chỉ đổi hành vi của dropout/BatchNorm; muốn tắt gradient phải dùng `torch.no_grad()`.
6. $\lceil 10\,000/256 \rceil =$ **40**.
7. Bước 50: $0.05(1 + \cos(\pi/2)) =$ **0.05**. Bước 25: $0.05(1 + \cos(\pi/4)) =$ **0.0854**.
8. $H(0.2) = -(0.2\ln 0.2 + 0.8\ln 0.8) =$ **0.500**.
9. $100 \cdot 50 + 50 =$ **5 050**.
10. $200\,960 + 32\,896 + 1\,290 =$ **235 146**.
11. $W_{10}\cdots W_1 x + \text{(tổng các bias)}$ vẫn là **một** phép biến đổi affine $Wx + b$: tích các ma trận vẫn là một ma trận.
12. $\sigma'(0) = 0.25$; $\tanh'(0) = 1$; $\mathrm{ReLU}'(-3) = 0$.
13. **16 000**. Tương đương `one_hot(i) @ W` với $W \in \mathbb R^{1000 \times 16}$ (Linear không bias), tức lấy hàng $i$ của $W$.
14. Ba lỗi: *(mỗi lỗi kèm cách sửa được 2/3đ)*
    - Thiếu `optimizer.zero_grad()` ở đầu vòng lặp.
    - `torch.sigmoid` trước `BCEWithLogitsLoss` là sigmoid hai lần; phải truyền logit `model(xb)` trực tiếp.
    - `total += loss` giữ đồ thị; phải dùng `loss.item()` (và nhân với `len(yb)` nếu muốn lấy trung bình theo mẫu).

**T1** *(đã chạy kiểm chứng bằng autograd)*

(a) $a = (1 - 2 + 0,\ 0.5 + 1 - 1) = (-1,\ 0.5)$; $h = (0,\ 0.5)$; $z = 2 \cdot 0 - 1 \cdot 0.5 + 0.5 = 0$; $p = 0.5$; $L = -\ln 0.5 =$ **0.6931**. *(1đ)*

(b) $\partial L/\partial z = p - y = -0.5$. *(2đ)*
- $\nabla_{w_2} = -0.5\,h = (0,\ -0.25)$; $\partial L/\partial b_2 = -0.5$.
- $\partial L/\partial h = -0.5\,w_2 = (-1,\ 0.5)$; nhân với mặt nạ ReLU $(0, 1)$ được $\partial L/\partial a = (0,\ 0.5)$.
- $\nabla_{W_1} = (\partial L/\partial a)\,x^\top = \begin{pmatrix} 0 & 0 \\ 0.5 & 1 \end{pmatrix}$; $\nabla_{b_1} = (0,\ 0.5)$.
- **Hàng 1 của $W_1$, $b_1[0]$ và $w_2[0]$ có gradient 0**: neuron 1 có $a_1 = -1 < 0$, ReLU "đóng" nên không có tín hiệu nào đi qua (và $h_1 = 0$ nên $w_2[0]$ cũng không ảnh hưởng tới $z$).

(c) Tham số mới: $w_2 = (2,\ -0.975)$, $b_2 = 0.55$, hàng 2 của $W_1$ là $(0.45,\ 0.4)$, $b_1 = (0,\ -1.05)$. Khi đó $a = (-1,\ 0.2)$, $h = (0,\ 0.2)$, $z = -0.975 \cdot 0.2 + 0.55 =$ **0.355**, $p = 0.588$, $L =$ **0.531** (giảm từ 0.693). *(1đ)*

**T2** *((a) 1.5đ · (b) 1.25đ · (c) 1.25đ; mỗi ý gồm nguyên nhân và cách kiểm tra)*

(a) $0.432 = H(0.155)$ đúng bằng loss của **model hằng số**: model chỉ học được tỉ lệ nền. Các khả năng:
- Gradient không tới được tham số: quên `optimizer.step()`, optimizer được tạo với **danh sách tham số sai** (ví dụ tạo trước khi thay lớp mới, hoặc dùng `list` thay vì `ModuleList`), hoặc lr quá nhỏ.
- Đầu vào hỏng: toàn 0 hoặc NaN đã bị điền.
- Dead ReLU toàn bộ.

Kiểm tra: in `count_params`; in chuẩn gradient của từng lớp sau `backward`; chạy `overfit_one_batch` (T4).

(b) Đang đánh giá ở chế độ `train()`, nên **dropout** vẫn bật (và nếu có BatchNorm thì nó dùng thống kê của batch). Kiểm tra `model.training`; gọi `model.eval()` trước khi đánh giá. Hai lần `evaluate` khi đó phải cho kết quả **giống hệt**.

(c) lr quá lớn: bước cập nhật làm logit bùng nổ, rồi xuất hiện `inf - inf` hoặc tràn số (nếu dùng `sigmoid` + `log` thủ công). Kiểm tra: giảm lr 10 lần; theo dõi chuẩn gradient; thêm `clip_grad_norm_`; dùng `torch.autograd.set_detect_anomaly(True)` để tìm phép toán sinh ra NaN; kiểm tra dữ liệu có NaN/inf không.

**T3** *(đã chạy kiểm chứng)*

(a) Chiều embedding: `tinh` $\min(50, 32) = 32$; `goi_cuoc` 50; `dong_may` 50. *(2đ)*
- Embedding: $64 \cdot 32 + 201 \cdot 50 + 501 \cdot 50 = 2\,048 + 10\,050 + 25\,050 =$ **37 148**.
- Đầu vào lớp 1: $30 + 32 + 50 + 50 = 162$ chiều.
- Linear: $(162 \cdot 128 + 128) + (128 \cdot 64 + 64) + 65 = 20\,864 + 8\,256 + 65 =$ **29 185**.
- Tổng: **66 333**. Hơn một nửa nằm ở embedding.

(b) Trọng số Linear (không tính bias): $162 \cdot 128 + 128 \cdot 64 + 64 = 28\,992$. FLOP mỗi epoch $\approx 6 \times 28\,992 \times 10^6 \approx 1.74 \times 10^{11}$, tức khoảng **1.7 giây** ở 100 GFLOP/s. *(1đ)*

Thực tế chậm hơn nhiều vì:
- Ma trận nhỏ không tận dụng hết CPU.
- Mỗi bước tốn overhead Python và phải khởi chạy kernel.
- Phải nạp và chuyển dữ liệu (`DataLoader`).
- Còn phép tra embedding, optimizer step và tính loss.

Mục 2.4 đo được 0.09 s/epoch cho 3 840 mẫu, tức khoảng 23 s cho 1 triệu dòng ở batch 128.

**T4** *(đã chạy kiểm chứng trên 32 mẫu của project)*

```python
def overfit_one_batch(model, batch, loss_fn, steps=200, lr=1e-2):
    *xs, y = batch
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(*xs), y)
        loss.backward()
        opt.step()
        hist.append(loss.item())
    return hist
```

- **Mục đích:** kiểm tra nhanh rằng mô hình, loss, gradient và optimizer **nối đúng với nhau**. Một mạng đủ lớn phải **học thuộc** được 32 mẫu. Nếu loss không về gần 0 thì chắc chắn có bug, chưa cần nghĩ tới hyperparameter. *(1đ)*
- **Kỳ vọng:** `TabMLP` không dropout đi từ 0.887 xuống khoảng **$4 \times 10^{-5}$**; với dropout 0.5 thì chỉ xuống khoảng $10^{-3}$, vì dropout vẫn gây nhiễu. Khi chạy kiểm tra này nên tắt dropout. *(0.5đ)*
- **Lỡ thêm sigmoid:** logit bị ép vào $(0, 1)$, nên BCE với logit không thể nhỏ hơn khoảng $\ln(1 + e^{-1}) = 0.31$ cho mẫu dương và $\ln 2 = 0.69$ cho mẫu âm. Loss **kẹt ở khoảng 0.6–0.7**, không bao giờ về 0. Đây là triệu chứng điển hình. *(0.5đ)*

*(Chấm: code đúng được 2đ.)*

</details>
