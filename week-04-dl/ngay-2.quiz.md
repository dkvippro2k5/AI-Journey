# Quiz · Ngày 2 · Huấn luyện mạng sâu (bài 04–06)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> Làm trên giấy, được dùng máy tính cầm tay. **Không chạy code.** Đáp án ở cuối file.

## Phần 1 — Quick test (14 câu, câu 14 được 2đ, các câu khác 1đ)

1. Tính std của He init cho `nn.Linear(512, 256)`.
2. Tính std của Xavier init (dạng chuẩn) cho `nn.Linear(300, 100)`.
3. MLP ReLU rộng 100, trọng số khởi tạo với std 0.1. Hệ số std mỗi lớp bằng bao nhiêu? Qua nhiều lớp, tín hiệu sẽ tắt dần hay bùng nổ?
4. Bỏ qua trọng số, gradient đi qua 6 lớp sigmoid bị nhân với **tối đa** bao nhiêu?
5. Gradient $g = (6, 8)$, `clip_grad_norm_(max_norm=5)`. Gradient sau khi cắt là gì? Còn với `max_norm=20`?
6. Với $h_{l+1} = h_l + f(h_l)$, tính $\partial h_{l+1}/\partial h_l$. Vì sao dạng này giúp gradient đi qua mạng sâu?
7. BatchNorm có `momentum = 0.1`, `running_mean` ban đầu là 0. Hai batch liên tiếp đều có mean 5. `running_mean` bằng bao nhiêu sau mỗi batch?
8. Tính LayerNorm (không $\gamma, \beta$, $\epsilon = 0$) của $(1, 3)$, và RMSNorm của $(3, 4)$.
9. Trong ba lớp BN, LN, RMSNorm, lớp nào cho kết quả **phụ thuộc vào các mẫu khác** trong batch lúc train?
10. Inverted dropout với $p = 0.2$. Một phần tử có giá trị 5 được giữ lại thì đầu ra bằng bao nhiêu? Lúc eval thì bằng bao nhiêu?
11. Đúng/Sai: `torch.optim.Adam(params, weight_decay=0.01)` tương đương `torch.optim.AdamW(params, weight_decay=0.01)`.
12. Huấn luyện đầy đủ một model 1.3 tỉ tham số bằng Adam và mixed precision cần bao nhiêu GB cho tham số, gradient và trạng thái optimizer? Có vừa GPU 16 GB không?
13. Gradient accumulation với 8 micro-batch, nhưng quên chia loss cho 8. Điều này tương đương với việc gì?
14. *(2đ)* Một MLP 4 lớp có BatchNorm đạt accuracy 0.99 khi train với batch 64, nhưng chỉ đạt 0.71 khi train với batch 2. Giải thích vì sao, và đưa ra hai cách sửa.

## Phần 2 — Tự luận (4 câu, 30 phút)

**T1 · Khởi tạo (4đ).** MLP ReLU, mọi lớp rộng 400, đầu vào có std 1. Bỏ qua bias.
- (a) Với ba cách khởi tạo: He, std 0.1, và Xavier, tính std ở đầu ra của lớp Linear thứ 1 và thứ 5.
- (b) Với Xavier, tới lớp thứ mấy thì std nhỏ hơn $10^{-3}$?
- (c) Nếu đổi activation thành tanh thì nên dùng init nào? Vì sao?

**T2 · BatchNorm và LayerNorm tính tay (4đ).** Batch 2 mẫu × 3 đặc trưng: $X = \begin{pmatrix} 1 & 4 & 2 \\ 3 & 0 & 2 \end{pmatrix}$, với $\gamma = 1$, $\beta = 0$.
- (a) Tính đầu ra của BatchNorm ở chế độ train (bỏ qua $\epsilon$ trừ khi cần). Cột 3 có gì đặc biệt?
- (b) Tính đầu ra của LayerNorm.
- (c) Sau batch này, `running_mean` và `running_var` bằng bao nhiêu (momentum 0.1, khởi đầu 0 và 1, var của running là var **không lệch**)?
- (d) Ở chế độ eval, mẫu $(2, 2, 2)$ cho ra kết quả gì? So với việc mẫu này nằm trong một batch train, bạn rút ra điều gì?

**T3 · Early stopping (3đ).** Val loss theo epoch (đếm từ 0):

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Val loss | 0.500 | 0.420 | 0.390 | 0.385 | 0.3845 | 0.386 | 0.383 | 0.388 | 0.391 | 0.395 |

Quy tắc: "cải thiện" khi `val < best − min_delta`, với `min_delta = 0.001`.
- (a) Với `patience = 3`: epoch tốt nhất là epoch nào, và huấn luyện dừng sau epoch nào?
- (b) Với `patience = 2` thì sao? Điều gì đã bị bỏ lỡ?
- (c) Train loss ở epoch 9 là 0.02. Đề xuất hai thay đổi để giảm khoảng cách giữa train và val.

**T4 · Code (4đ).** Viết `accumulate_step(model, micro_batches, loss_fn, optimizer, clip_norm=None) -> float`. Trong đó `micro_batches` là list các tuple `(*xs, y)` có thể **khác kích thước** nhau. Hàm thực hiện **một** bước cập nhật với gradient **đúng bằng** gradient của loss trung bình trên toàn bộ các mẫu gộp lại, và trả về loss trung bình đó. Giải thích vì sao cách "chia loss cho số micro-batch $k$" chỉ đúng khi các micro-batch bằng nhau.

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Quick test**

1. $\sqrt{2/512} =$ **0.0625**.
2. $\sqrt{2/400} =$ **0.0707**.
3. $\sqrt{100 \cdot 0.01/2} =$ **0.707** < 1, nên tín hiệu **tắt dần** (He cho rộng 100 cần std $\sqrt{0.02} = 0.141$).
4. $(1/4)^6 =$ **$2.44 \times 10^{-4}$**.
5. $\lVert g\rVert = 10$, nên sau khi cắt được $(6, 8) \cdot 5/10 =$ **$(3, 4)$**. Với `max_norm=20`: **không đổi**.
6. $I + \partial f/\partial h_l$. Tích các Jacobian qua nhiều khối luôn chứa số hạng đồng nhất $I$, tức một đường đi thẳng từ loss về lớp đầu, không bị nhân với các ma trận nhỏ.
7. Sau batch 1: $0.9 \cdot 0 + 0.1 \cdot 5 =$ **0.5**. Sau batch 2: $0.9 \cdot 0.5 + 0.5 =$ **0.95**.
8. LN của $(1, 3)$: $\mu = 2$, $\sigma = 1$, nên được **$(-1, 1)$**. RMSNorm của $(3, 4)$: $\mathrm{rms} = \sqrt{12.5} = 3.536$, nên được **$(0.849,\ 1.131)$**.
9. **BatchNorm** (thống kê tính trên batch).
10. Lúc train: $5/0.8 =$ **6.25**. Lúc eval: **5**.
11. **Sai**. `Adam` cộng $\lambda\theta$ vào gradient (L2), nên phần này bị chia cho $\sqrt{\hat v}$. `AdamW` decay trực tiếp lên trọng số (Loshchilov & Hutter 2019).
12. $16 \times 1.3 \times 10^9 =$ **20.8 GB**: **không** vừa GPU 16 GB, dù chưa tính activation.
13. Gradient lớn gấp 8 lần, nên tương đương **tăng learning rate lên 8 lần** (với SGD; với Adam thì phần lớn được tự chuẩn hóa, nhưng epsilon và clipping vẫn bị ảnh hưởng).
14. Với batch 2, mean/var của batch chỉ dựa trên 2 mẫu nên **rất nhiễu**; mạng học với các thống kê nhiễu đó, còn running stats dùng lúc eval thì khác hẳn. Hai cách sửa *(1đ mỗi cách)*:
    - Dùng **LayerNorm** hoặc GroupNorm (không phụ thuộc batch). Đo trên máy bạn: LN với batch 2 đạt 0.961.
    - Tăng batch (nếu thiếu bộ nhớ thì gradient accumulation **không** giúp được BN), hoặc dùng `SyncBatchNorm` trên nhiều GPU, hoặc bỏ BN.

**T1** *(đã chạy kiểm chứng bằng `activation_stds` trên 2 000 mẫu)*

Hệ số std mỗi lớp là $\sqrt{400\,\mathrm{Var}(w)/2}$; std của lớp 1 là $\sqrt{400\,\mathrm{Var}(w)}$. *(2.5đ)*

| Init | $\mathrm{Var}(w)$ | Lớp 1 | Hệ số | Lớp 5 | Mô phỏng (lớp 1 / lớp 5) |
|---|---|---|---|---|---|
| He | $2/400$ | 1.414 | 1 | **1.414** | 1.413 / 1.41 |
| std 0.1 | 0.01 | 2 | 1.414 | $2 \cdot 1.414^4 =$ **8** | 2.00 / 8.48 |
| Xavier | $2/800$ | 1 | 0.707 | $0.707^4 =$ **0.25** | 1.00 / 0.22 |

(b) $0.707^{L-1} < 10^{-3} \iff L - 1 > \ln 10^{-3}/\ln 0.707 = 19.93$, nên **$L = 21$**. Mô phỏng: lớp 20 là 0.00105, lớp 21 là 0.00071. *(1đ)*

(c) **Xavier** (hoặc LeCun). Tanh đối xứng quanh 0 và có $\tanh'(0) = 1$, nên gần 0 nó gần như tuyến tính và **không** bỏ đi nửa tín hiệu như ReLU. Hệ số 2 của He sẽ làm tín hiệu tăng, đẩy tanh vào vùng bão hòa. *(0.5đ)*

**T2** *(đã chạy kiểm chứng bằng `nn.BatchNorm1d` và `layer_norm`)*

(a) *(1đ)*
- Cột 1 $(1, 3)$: $\mu = 2$, $\sigma^2 = 1$, nên được $(-1, 1)$.
- Cột 2 $(4, 0)$: $\mu = 2$, $\sigma^2 = 4$, nên được $(1, -1)$.
- Cột 3 $(2, 2)$: $\sigma^2 = 0$. Không có $\epsilon$ thì là $0/0$; có $\epsilon$ thì cho ra **$(0, 0)$**.
- Với batch 2, BN luôn cho ra $\pm 1$ ở mọi cột không hằng số: **mọi thông tin về độ lớn bị xóa**. Đó là một lý do BN tệ ở batch nhỏ.

(b) *(1đ)*
- Hàng 1 $(1, 4, 2)$: $\mu = 7/3 = 2.333$, $\sigma^2 = 14/9 = 1.556$, $\sigma = 1.247$. Kết quả: $(-1.069,\ 1.336,\ -0.267)$.
- Hàng 2 $(3, 0, 2)$: $\mu = 5/3$, cùng $\sigma^2 = 1.556$. Kết quả: $(1.069,\ -1.336,\ 0.267)$.

(c) Mean của batch là $(2, 2, 2)$, nên `running_mean` là $(0.2,\ 0.2,\ 0.2)$. Var **không lệch** là $(2, 8, 0)$, nên `running_var` là $0.9 + 0.1\cdot(2, 8, 0) = (1.1,\ 1.7,\ 0.9)$. *(1đ)*

(d) $(2 - 0.2)/\sqrt{(1.1,\ 1.7,\ 0.9)} = (1.716,\ 1.381,\ 1.897)$. Nếu mẫu này nằm trong batch train $X$ thì nó đúng bằng mean của batch, nên sẽ ra $\approx (0, 0, 0)$. Sau vài batch đầu, running stats còn **rất xa** thống kê thật, nên eval khác train rất nhiều. Phải huấn luyện đủ lâu để running stats hội tụ (hoặc tăng `momentum`). *(1đ)*

**T3** *(đã chạy kiểm chứng bằng đúng quy tắc của `fit`)*

(a) Lần lượt cải thiện ở epoch 0, 1, 2, 3 (giảm 0.005 > 0.001). Epoch 4: chỉ giảm 0.0005 < `min_delta`, nên **không** tính là cải thiện. Epoch 5: không. Epoch 6: $0.383 < 0.385 - 0.001$, nên **cải thiện**. Epoch 7, 8, 9: không. Vậy **best = epoch 6 (0.383), dừng sau epoch 9**. *(1.5đ)*

(b) `patience = 2`: sau best ở epoch 3, epoch 4 và 5 không cải thiện, nên **dừng sau epoch 5 với best = epoch 3 (0.385)**. Đã bỏ lỡ điểm tốt hơn ở epoch 6. `patience` quá nhỏ thì dễ dừng vì **nhiễu** của val loss. *(1đ)*

(c) Hai trong số các cách sau *(0.5đ)*:
- Tăng dropout (bài 06: điểm tốt nhất dời muộn hơn, val cuối thấp hơn).
- Tăng weight decay (đủ mạnh: $\eta\lambda \times$ số bước phải đáng kể).
- Thu nhỏ model.
- Thêm dữ liệu hoặc augmentation.
- Giữ early stopping.

**T4** *(đã chạy kiểm chứng: với micro-batch 40/30/30 mẫu, chênh lệch với gradient của cả batch là $1.5 \times 10^{-8}$; còn cách "chia cho $k$" lệch tới $0.02$)*

```python
def accumulate_step(model, micro_batches, loss_fn, optimizer, clip_norm=None):
    optimizer.zero_grad()
    n = sum(len(mb[-1]) for mb in micro_batches)
    total = 0.0
    for *xs, y in micro_batches:
        loss = loss_fn(model(*xs), y)                      # trung bình trên micro-batch
        (loss * len(y) / n).backward()                     # trọng số = tỉ lệ mẫu của micro-batch
        total += loss.item() * len(y)
    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)   # clip SAU khi cộng đủ
    optimizer.step()
    return total / n
```

Giải thích: loss trung bình của cả batch là $\tfrac1n\sum_i \ell_i = \sum_k \tfrac{n_k}{n}\bar\ell_k$. Mỗi micro-batch phải có trọng số $n_k/n$. Cách chia cho $k$ ngầm giả định $n_k/n = 1/k$, chỉ đúng khi mọi $n_k$ bằng nhau. Micro-batch cuối thường nhỏ hơn, nên sẽ bị tính **nặng hơn** thực tế.

*(Chấm: code đúng được 3đ; giải thích được 1đ. Clip trước khi cộng đủ gradient thì trừ 0.5đ.)*

</details>
