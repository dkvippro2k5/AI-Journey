# Phụ lục Tuần 4 · Custom loss, Dataset và padding, object detection, GPU

> **Loại tài liệu:** P1–P3 là **Lý thuyết (bản nhẹ)**: công thức và ví dụ tính tay đầy đủ, chứng minh chỉ phác thảo vài dòng. P4 là **Kỹ thuật**: chỉ có bảng ước lượng nhanh.
> **Vì sao có phụ lục này:** so với Phase 4 của roadmap, 5 ngày ở trên còn thiếu hoặc mới chỉ nhắc tên những mục sau:
> - *custom loss functions* (4.6): chưa dạy;
> - *custom `Dataset`* (4.6): mới có một đoạn văn, chưa có code và `collate_fn`;
> - *bidirectional RNN* (4.4): mới có một dòng trong bảng thuật ngữ;
> - *object detection* (4.3): mới nhắc tên;
> - *moving to GPU* (4.6): mới có một dòng `device = ...`.
>
> **Gồm 4 bài:**
> - **P1** Custom loss: weighted BCE, focal loss, label smoothing, ổn định số
> - **P2** `Dataset` tự viết, `collate_fn`, padding, mask, BiLSTM
> - **P3** Object detection cơ bản: box, IoU, anchor, NMS, mAP
> - **P4** Thiết bị và GPU: code chạy được trên mọi device, đồng bộ, bộ nhớ
>
> **Thời gian:** đọc ~2h30 · quiz 30' (cuối file). Không có bài tập project riêng; mỗi bài có một đoạn "Tự làm" 10–20'.
> **Không có trong phụ lục:** dùng model pretrained của Hugging Face (4.7). Mục này cần tải model, nên được dạy ở **Tuần 5, Ngày 3**, cùng với tokenizer.
> **Môi trường kiểm chứng:** PyTorch 2.14 (CPU), NumPy 2.5, scikit-learn 1.9, máy bạn đang cắm sạc. P1 và P2 dùng code tham chiếu của `qnet` và dữ liệu `quality` (6 000 dòng, 15.5% 👎), cùng 5 split như project. Máy bạn không có GPU, nên các đoạn code GPU ở P4 **chỉ mô tả, không chạy**; mọi khẳng định chạy được trên CPU thì đã chạy.

---

## P1 · Custom loss: weighted BCE, focal loss, label smoothing, ổn định số

> ⏱ Tầng 1: 10' · Tầng 2: 30'
> **Cần biết trước:** BCE và cross-entropy, gradient $p - y$ (Tuần 2, bài 05); vòng lặp huấn luyện, `BCEWithLogitsLoss` (bài 02); hiệu chỉnh prior và ECE (Tuần 3, phụ lục P2).

> **📖 Từ mới**
> - **Loss function:** hàm đo mức "sai" của dự đoán so với nhãn. Optimizer chỉ biết làm loss nhỏ đi, nên loss **là** định nghĩa của "tốt" đối với model.
> - **Reduction:** cách gộp loss của từng mẫu thành một số: `mean` (trung bình), `sum` (tổng), hoặc `none` (giữ nguyên từng mẫu).
> - **Logit:** đầu ra **trước** sigmoid/softmax, nằm trong $(-\infty, \infty)$.
> - **$p_t$:** xác suất model gán cho **lớp đúng**: $p_t = p$ nếu $y = 1$, và $p_t = 1 - p$ nếu $y = 0$.
> - **Ổn định số (numerical stability):** công thức đúng trên giấy nhưng cho `inf`/`nan` trên máy, vì số thực dấu phẩy động có giới hạn.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Loss là **barem chấm điểm**; model là học sinh chỉ học để được điểm cao theo barem. Muốn học sinh chú ý hơn tới câu khó, bạn sửa barem chứ không sửa học sinh:
- **Weighted BCE:** câu về lớp hiếm được nhân điểm lên $w$ lần.
- **Focal loss:** câu học sinh đã làm đúng chắc chắn thì gần như không tính điểm nữa; điểm dồn vào câu còn sai.
- **Label smoothing:** không đòi học sinh trả lời "chắc chắn 100%", chỉ cần "khoảng 93%".

Nhưng học sinh học theo barem nào thì **"tự tin"** theo barem đó. Barem đã bị sửa thì con số tự tin (xác suất) không còn là xác suất thật nữa.

**Ví dụ có số: focal loss "tắt tiếng" mẫu dễ.** Focal loss nhân cross-entropy với $(1 - p_t)^\gamma$. Với $\gamma = 2$:

| Mẫu | $p_t$ | CE $= -\ln p_t$ | Hệ số $(1 - p_t)^2$ | Focal |
|---|---|---|---|---|
| Dễ | 0.9 | 0.105 | 0.01 | 0.00105 |
| Vừa | 0.6 | 0.511 | 0.16 | 0.0817 |
| Khó | 0.1 | 2.303 | 0.81 | 1.865 |

Với CE, mẫu khó "nặng" gấp **22 lần** mẫu dễ. Với focal loss, con số đó là **1 770 lần**. Khi có hàng chục nghìn mẫu dễ (nền ảnh trong object detection, xem P3), focal loss ngăn chúng lấn át vài mẫu khó.

**Ba ý cốt lõi:**
1. Custom loss trong PyTorch chỉ là một hàm (hoặc một `nn.Module`) tính ra **một tensor vô hướng** từ các phép toán tensor. Autograd tự lo phần backward.
2. Hãy viết loss trên **logit**, dùng các hàm ổn định có sẵn (`binary_cross_entropy_with_logits`, `log_softmax`). Đừng tự viết `log(sigmoid(z))`.
3. Mọi loss "cân lại" (trọng số lớp, focal) đều **làm lệch thang xác suất**. Nếu cần xác suất thật, hãy hiệu chỉnh, hoặc giữ BCE và chọn ngưỡng.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### P1.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Weighted BCE (`pos_weight`)** | $\ell = -\big[w\,y\log\sigma(z) + (1 - y)\log(1 - \sigma(z))\big]$: nhân loss của mẫu dương với $w$ | PyTorch docs, `BCEWithLogitsLoss`; King & Zeng, *Political Analysis* 2001 |
| **Class weights (đa lớp)** | `CrossEntropyLoss(weight=...)`: mỗi lớp một trọng số | PyTorch docs |
| **Focal loss** | $\ell = -\alpha_t(1 - p_t)^\gamma\log p_t$; $\gamma = 0$, không có $\alpha$ thì đúng bằng CE | Lin, Goyal, Girshick, He, Dollár, *Focal Loss for Dense Object Detection*, ICCV 2017 |
| **Label smoothing** | Nhãn đích $(1 - \varepsilon)\cdot\text{one-hot} + \varepsilon/K$ | Szegedy et al., *Rethinking the Inception Architecture*, CVPR 2016; Müller, Kornblith, Hinton, NeurIPS 2019 |
| **Huber / Smooth L1** | Bình phương khi sai số nhỏ, tuyệt đối khi sai số lớn; dùng cho hồi quy box (P3) | Huber 1964 (Tuần 3, phụ lục P3); Girshick, *Fast R-CNN*, ICCV 2015 |
| **`ignore_index`** | Nhãn đặc biệt (mặc định −100) bị bỏ qua trong `CrossEntropyLoss`; dùng cho token đệm | PyTorch docs |
| **Log-sum-exp trick** | Tính $\log\sum e^{z_i}$ bằng $m + \log\sum e^{z_i - m}$ với $m = \max z$, để không tràn số | Tuần 2, bài 04 |

#### P1.2 Ba câu hỏi khi viết một custom loss

1. **Loss nhận gì?** Logit hay xác suất? Shape `(B,)` hay `(B, 1)`, `(B, K)` hay `(B, T, K)`? Nhãn có dtype `float` (BCE) hay `long` (CE)?
2. **Gộp thế nào?** `mean` theo **mẫu thật** (bỏ phần đệm), hay `sum`? `sum` làm gradient lớn lên $B$ lần: với SGD, điều đó tương đương nhân learning rate với $B$; với Adam thì gần như không đổi gì, vì Adam chia cho $\sqrt v$ (đã kiểm chứng: 50 bước Adam với `mean` và `sum` cho **cùng** trọng số tới 4 chữ số).
3. **Hằng số tối ưu là gì?** Với một điểm có xác suất thật $\pi$, loss đạt cực tiểu ở dự đoán $p^\star$ nào? Đây là câu hỏi quyết định xác suất đầu ra **có nghĩa** gì.

**Câu 3 với `pos_weight` $= w$ (phác thảo).** Kỳ vọng loss tại một điểm là $-\big[w\pi\log p + (1 - \pi)\log(1 - p)\big]$. Đạo hàm theo $p$ bằng 0 cho
$$p^\star = \frac{w\pi}{w\pi + 1 - \pi}, \qquad \text{tức là}\quad \text{odds}(p^\star) = w \cdot \text{odds}(\pi).$$
Model học được odds bị **phóng to đúng $w$ lần**. Muốn có xác suất thật thì chia odds cho $w$. Đây là cùng một phép hiệu chỉnh prior như Tuần 3, phụ lục P2.3.

#### P1.3 Ví dụ tính tay

**(a) `pos_weight`.** Logit $z = 0$, $y = 1$, $w = 5$: $\ell = -5\log\sigma(0) = 5\ln 2 =$ **3.466**. Logit $z = 2$, $y = 0$: trọng số không áp dụng, $\ell = -\log(1 - \sigma(2)) =$ **2.127**. (Khớp `BCEWithLogitsLoss(pos_weight=torch.tensor(5.))`.)

**(b) `pos_weight` cân bằng lớp.** Dữ liệu project có $\pi = 0.155$. Chọn $w = (1 - \pi)/\pi = 5.45$ ("cân bằng"): tại một điểm có $\pi(x)$ đúng bằng tỉ lệ nền, $p^\star = \dfrac{5.45 \times 0.155}{5.45 \times 0.155 + 0.845} =$ **0.5**. Với $w = 3$: $p^\star = 0.465/1.310 =$ **0.355**.

**(c) Focal loss** cho một mẫu, $\gamma = 2$, không có $\alpha$:

| Logit $z$ | $y$ | BCE | $p_t = e^{-\text{BCE}}$ | Focal $= (1 - p_t)^2 \cdot \text{BCE}$ |
|---|---|---|---|---|
| 2 | 1 | 0.1269 | 0.881 | **0.0018** |
| 2 | 0 | 2.1269 | 0.119 | **1.650** |
| 0 | 1 | 0.6931 | 0.5 | **0.173** |

Mẹo $p_t = e^{-\text{BCE}}$ đúng cho cả $y = 0$ lẫn $y = 1$, vì $\text{BCE} = -\log p_t$. Nhờ vậy focal loss dùng lại được hàm BCE ổn định.

**(d) Label smoothing**, $K = 3$, $\varepsilon = 0.1$, lớp đúng là 0:
- Nhãn đích: $(0.9 + 0.1/3,\ 0.1/3,\ 0.1/3) = (0.933,\ 0.033,\ 0.033)$.
- Logit $(2, 0.5, -1)$ cho `log_softmax` $= (-0.241,\ -1.741,\ -3.241)$.
- CE thường: **0.241**. CE có smoothing: $0.933 \cdot 0.241 + 0.033 \cdot 1.741 + 0.033 \cdot 3.241 =$ **0.391**. Khớp `F.cross_entropy(..., label_smoothing=0.1)`.
- Loss có smoothing **không bao giờ về 0**: model bị phạt nếu quá tự tin, nên logit không bị đẩy ra vô cực.

**(e) Ổn định số.** Logit $z = -200$, $y = 1$:
- Tự viết: `-(y * torch.log(torch.sigmoid(z)))` cho **`inf`**, vì trong float32 $\sigma(z) = 0$ khi $z \le -89$ (đo trên máy bạn: $\sigma(-88) = 6 \times 10^{-39}$, $\sigma(-89) = 0$).
- `binary_cross_entropy_with_logits` cho **200.0**, đúng giá trị: nó tính $\log\sigma(z) = -\log(1 + e^{-z})$ bằng log-sum-exp.

**(f) Sigmoid hai lần.** Model đã trả về $\sigma(z)$ rồi bạn lại đưa vào `BCEWithLogitsLoss`. Với $z = 5$, $y = 1$: loss đúng là 0.0067, loss sai là **0.315**. Loss sai không thể xuống dưới $-\log\sigma(1) = 0.313$ (vì đầu vào $\le 1$), nên model mãi bị đẩy và không bao giờ hội tụ đúng. Code vẫn chạy và không báo lỗi.

#### P1.4 Đo trên máy bạn: loss khác nhau trên `quality`

`TabMLP` mặc định (5 417 tham số), AdamW, early stopping theo val loss **của chính loss đó**, 5 split như project. $w = 5.45$.

| Loss | AUC | Chênh AUC với BCE (số split thắng) | AP | ECE | $\bar p$ (thật: 0.155) | Epoch tốt nhất |
|---|---|---|---|---|---|---|
| BCE | 0.785 | — | **0.488** | **0.025** | 0.156 | 20.8 |
| BCE, `pos_weight` = 5.45 | **0.788** | +0.002 (3/5) | 0.481 | 0.246 | 0.400 | 17.0 |
| … + chia odds cho 5.45 | 0.788 | | 0.481 | **0.020** | 0.154 | |
| Focal, $\gamma = 2$ | 0.779 | −0.007 (1/5) | 0.466 | 0.169 | 0.313 | 13.4 |
| Focal, $\gamma = 2$, $\alpha = 0.25$ | 0.780 | −0.006 (0/5) | 0.472 | 0.105 | 0.238 | 19.4 |

**Đọc bảng:**
- **`pos_weight` không làm model xếp hạng tốt hơn** (+0.002 AUC, thắng 3/5 split: nhiễu), và AP còn giảm nhẹ. Nó chỉ dời thang xác suất: $\bar p$ tăng từ 0.156 lên 0.400.
- **Phép hiệu chỉnh ở P1.2 hoạt động:** chia odds cho $w$ đưa ECE từ 0.246 về **0.020** (cả 5/5 split về dưới 0.026), và $\bar p$ về 0.154.
- **Focal loss làm AUC giảm** trên 4/5 split, và **không có công thức hiệu chỉnh đơn giản**. Lý do: focal loss không "phạt" mẫu đã đúng, nên model không cần đẩy xác suất ra hai đầu; các xác suất bị **nén về phía giữa**. Phần lớn mẫu là âm, xác suất bị kéo lên, nên $\bar p$ tăng lên 0.313. $\alpha = 0.25$ giảm trọng số mẫu dương nên kéo $\bar p$ xuống lại một phần.
- **Bài học:** focal loss được thiết kế cho object detection, nơi tỉ lệ nền/vật thể cỡ 1 000 : 1 và hầu hết mẫu nền là **cực dễ** (Lin et al. 2017). Với dữ liệu bảng 15% dương, nó không giúp gì. Đây là cùng kết luận với Tuần 3, phụ lục P2: "sửa mất cân bằng" ở loss không cải thiện xếp hạng mà làm hỏng xác suất.

#### P1.5 Code chuẩn

```python
import torch
import torch.nn.functional as F
from torch import nn

class FocalLoss(nn.Module):
    """Focal loss nhị phân (Lin et al. 2017) trên LOGIT. gamma = 0 và alpha = None thì đúng bằng BCE."""
    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")   # ổn định số
        p_t = torch.exp(-bce)                                                    # xác suất của lớp đúng
        loss = (1 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            loss = loss * (self.alpha * y + (1 - self.alpha) * (1 - y))
        return loss.mean()

def masked_mean(per_item: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Trung bình loss chỉ trên các vị trí thật (mask True); dùng cho chuỗi có đệm (P2)."""
    mask = mask.to(per_item.dtype)
    return (per_item * mask).sum() / mask.sum().clamp_min(1)

def correct_pos_weight(p, w):
    """Đổi xác suất của model train bằng pos_weight = w về thang thật (P1.2)."""
    odds = p / (1 - p) / w
    return odds / (1 + odds)

# Dùng như mọi loss có sẵn: fit(model, train_loader, val_loader, FocalLoss(2.0), opt)
```

**Kiểm tra một custom loss trước khi dùng** (đã chạy):
- Trường hợp suy biến: `FocalLoss(gamma=0)` bằng `F.binary_cross_entropy_with_logits` tới 16 chữ số.
- Gradient: `torch.autograd.gradcheck(lambda z: FocalLoss()(z, y), (z,))` với `z` dtype `float64`, `requires_grad=True` trả về `True`. `gradcheck` so gradient của autograd với sai phân hữu hạn; phải dùng `float64` vì float32 không đủ chính xác cho sai phân.
- Giá trị tính tay: bảng P1.3(c).

**Khi nào cần `torch.autograd.Function`?** Hầu như không bao giờ. Chỉ khi bạn cần một backward **khác** với đạo hàm mà autograd tự suy ra: ổn định số, tiết kiệm bộ nhớ, hoặc "straight-through estimator" cho phép làm tròn trong quantization (Tuần 11).

#### P1.6 Cạm bẫy

1. **Sigmoid hai lần** (P1.3(f)): model trả về xác suất, rồi loss lại là `BCEWithLogitsLoss`. Không có lỗi, chỉ có model tệ. Quy ước: model trả về **logit**, sigmoid chỉ áp lúc dự đoán.
2. **Tự viết `log(sigmoid(z))`** hoặc `log(softmax(z))`: `inf`/`nan` khi logit lớn. Dùng `F.logsigmoid`, `F.log_softmax`, hoặc loss "with logits".
3. **Shape không khớp và bị broadcast:** logit `(B, 1)` với nhãn `(B,)`. `BCEWithLogitsLoss` báo lỗi shape, nhưng một loss tự viết như `((z - y) ** 2).mean()` sẽ âm thầm broadcast thành ma trận `(B, B)`.
4. **Đổi `mean` thành `sum`** khi dùng SGD: gradient lớn lên $B$ lần, nên learning rate cũ không còn đúng.
5. **Tính loss trên token đệm:** dùng `ignore_index=-100` cho nhãn ở vị trí đệm, hoặc `masked_mean`.
6. **Dùng xác suất của model train bằng `pos_weight`/focal như xác suất thật:** ECE 0.246 thay vì 0.025 (P1.4).
7. **Cộng dồn `total += loss` để log:** `total` giữ lại **toàn bộ đồ thị tính toán** của mọi batch, nên bộ nhớ tăng mãi (đã kiểm chứng: `total.grad_fn` khác `None`). Phải dùng `total += loss.item()`, hoặc `loss.detach()`.

**Sắc thái phỏng vấn:**
- *"Dữ liệu mất cân bằng, có nên dùng focal loss không?"* Nêu nguồn gốc (dense detection, tỉ lệ nền 1 000 : 1, mẫu dễ áp đảo). Với phân loại thường, hãy thử BCE và chọn ngưỡng trước, rồi so `pos_weight`/focal **theo cặp split** bằng AP. Nhắc tới tác động lên calibration.
- *"Viết custom loss trong PyTorch thế nào?"* Một hàm thuần hoặc `nn.Module` dùng toán tử tensor; autograd lo backward; kiểm tra bằng trường hợp suy biến và `gradcheck`; làm việc trên logit.

#### P1.7 Tự làm (15')

1. Viết `weighted_bce(logits, y, w)` bằng `F.logsigmoid`, không dùng `pos_weight`. Kiểm tra khớp `BCEWithLogitsLoss(pos_weight=...)` trên tensor ngẫu nhiên, và không ra `inf` với logit ±200.
2. Gõ lại `FocalLoss`, chạy hai phép kiểm tra ở P1.5.
3. Huấn luyện `TabMLP` bằng `fit(...)` của bạn với `FocalLoss(2.0)` trên split seed 0, so AUC và ECE với BCE.

#### P1.8 Đọc thêm

- T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, *Focal Loss for Dense Object Detection*, ICCV 2017.
- R. Müller, S. Kornblith, G. Hinton, *When Does Label Smoothing Help?*, NeurIPS 2019.
- PyTorch docs: `BCEWithLogitsLoss`, `CrossEntropyLoss` (`weight`, `ignore_index`, `label_smoothing`), *Extending PyTorch* (`torch.autograd.Function`), `torch.autograd.gradcheck`.
- J. Mukhoti et al., *Calibrating Deep Neural Networks using Focal Loss*, NeurIPS 2020: một góc nhìn khác, cho thấy focal loss có thể **giảm** overconfidence của mạng lớn trên ảnh. Kết quả phụ thuộc bài toán, nên luôn đo ECE.

---

## P2 · `Dataset` tự viết, `collate_fn`, padding, mask, BiLSTM

> ⏱ Tầng 1: 10' · Tầng 2: 35'
> **Cần biết trước:** `TensorDataset`/`DataLoader` (bài 02); LSTM và `pack_padded_sequence` (bài 08); mask trong attention (bài 09).

> **📖 Từ mới**
> - **Map-style `Dataset`:** lớp có `__len__` (bao nhiêu mẫu) và `__getitem__(i)` (lấy mẫu thứ $i$).
> - **`collate_fn`:** hàm nhận **một list mẫu** và ghép chúng thành **một batch** tensor.
> - **Padding (đệm):** thêm token đặc biệt `PAD` vào cuối các chuỗi ngắn để mọi chuỗi trong batch dài bằng nhau.
> - **Mask:** tensor bool cùng shape với batch, `True` ở vị trí thật, `False` ở vị trí đệm.
> - **Packing:** đóng gói batch đã đệm kèm độ dài thật (`PackedSequence`), để RNN dừng đúng ở token thật cuối cùng của từng chuỗi.
> - **Bucketing:** gom các chuỗi có độ dài gần nhau vào cùng batch để bớt padding.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Xếp tin nhắn vào khay để gửi đi xử lý. Khay hình chữ nhật, nhưng tin nhắn dài ngắn khác nhau:
- **Padding:** chèn giấy trắng vào sau các tin ngắn cho đủ chiều dài khay.
- **Mask:** dán kèm một tờ ghi chú "từ ô này trở đi là giấy trắng". Người xử lý phải đọc ghi chú này; nếu không, họ sẽ đọc cả giấy trắng.
- **Bucketing:** xếp các tin dài gần bằng nhau vào chung một khay, để không phải chèn nhiều giấy trắng.

**Ví dụ có số.** Ba chuỗi token `[5, 3, 8]`, `[7]`, `[2, 2, 9, 4]`:

```
x    = [[5, 3, 8, 0],          mask = [[T, T, T, F],
        [7, 0, 0, 0],                  [T, F, F, F],
        [2, 2, 9, 4]]                  [T, T, T, T]]
```

- 12 ô, 8 ô thật: **33%** là đệm, tức là phí 33% phép tính.
- Mean-pooling **có mask** trên giá trị $h$ = `[[1, 2, 3, 0], [4, 0, 0, 0], [1, 1, 1, 1]]` cho $(2, 4, 1)$. **Không mask** thì được $(1.5, 1, 1)$: chuỗi `[7]` bị "pha loãng" còn một phần tư.

**Ba ý cốt lõi:**
1. `Dataset` trả về **một** mẫu; `collate_fn` quyết định **ghép** chúng ra sao. Chuỗi dài ngắn khác nhau bắt buộc phải có `collate_fn` riêng.
2. Đệm xong thì mọi phép **gộp theo thời gian** (lấy vị trí cuối, trung bình, attention, loss) đều phải dùng **mask** hoặc **độ dài thật**.
3. BiLSTM đọc chuỗi theo **hai chiều**. Trạng thái cuối của chiều ngược đã "vừa đọc xong" token đầu tiên. Chỉ dùng được BiLSTM khi lúc dự đoán **đã có cả chuỗi**.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### P2.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Map-style / iterable-style dataset** | `__getitem__` + `__len__` (truy cập ngẫu nhiên) / `__iter__` (luồng dữ liệu, ví dụ đọc file lớn tuần tự) | PyTorch docs, `torch.utils.data` |
| **Sampler / batch sampler** | Quyết định **thứ tự chỉ số** (và cách gom thành batch) mà DataLoader lấy | như trên |
| **`collate_fn`, `default_collate`** | Ghép list mẫu thành batch; mặc định là `torch.stack` từng trường, nên đòi các mẫu cùng shape | như trên |
| **`pad_sequence`** | Đệm list tensor độ dài khác nhau thành tensor `(B, T_max, ...)` | `torch.nn.utils.rnn` |
| **`pack_padded_sequence`** | Đóng gói batch đã đệm theo độ dài thật; RNN bỏ qua phần đệm | như trên |
| **`padding_idx`** | Chỉ số trong `nn.Embedding` có vector luôn bằng 0 và không nhận gradient | PyTorch docs |
| **Bidirectional RNN** | Hai RNN độc lập đọc xuôi và đọc ngược; đầu ra nối lại thành $2H$ chiều | Schuster & Paliwal, *IEEE Trans. Signal Processing* 1997 |
| **Bucketing** | Gom chuỗi độ dài gần nhau vào cùng batch | Khomenko et al. 2016; dùng phổ biến trong dịch máy |

#### P2.2 Shape của BiLSTM (bảng tra)

`nn.LSTM(d, H, num_layers=L, bidirectional=True, batch_first=True)` với đầu vào `(B, T, d)`:

| Đầu ra | Shape | Nội dung |
|---|---|---|
| `out` | `(B, T, 2H)` | Tại mỗi vị trí: `out[..., :H]` là chiều xuôi, `out[..., H:]` là chiều ngược, **của lớp cuối** |
| `h_n` | `(2L, B, H)` | Trạng thái cuối của mọi lớp và chiều, xếp `[lớp 1 xuôi, lớp 1 ngược, …, lớp L xuôi, lớp L ngược]` |
| `h_n[-2]` | `(B, H)` | Lớp cuối, chiều xuôi: đã đọc **hết chuỗi** từ trái sang phải; bằng `out[:, -1, :H]` khi không có đệm |
| `h_n[-1]` | `(B, H)` | Lớp cuối, chiều ngược: đã đọc từ phải về tới **token đầu tiên**; bằng `out[:, 0, H:]` |

(Hai đẳng thức ở cột cuối đã kiểm chứng bằng `torch.allclose`.) **Số tham số** gấp đôi LSTM một chiều: $2 \cdot 4H(d + H + 2)$. Với $d = 16$, $H = 32$: $2 \cdot 4 \cdot 32 \cdot 50 =$ **12 800**.

**Cạm bẫy shape kinh điển.** Lấy `out[:, -1]` của BiLSTM để làm đặc trưng câu: nửa xuôi thì đúng (khi không có đệm), nhưng **nửa ngược ở vị trí cuối mới chỉ đọc đúng một token**. Phải dùng `torch.cat([h_n[-2], h_n[-1]], -1)`.

#### P2.3 Ví dụ tính tay: chi phí padding

**(a)** Độ dài 2, 5, 3, 6 trong một batch: đệm tới 6, nên có $4 \times 6 = 24$ ô cho 16 token thật. Phần đệm chiếm $8/24 =$ **33%**.

**(b) Bucketing.** 10 000 chuỗi có độ dài ngẫu nhiên đều trong $[5, 30]$, batch 64 (đo trên máy bạn):
- Batch ngẫu nhiên: **41.5%** ô là đệm. Gần như mọi batch đều có một chuỗi dài gần 30.
- Sắp theo độ dài rồi cắt batch: **0.4%** ô là đệm.

Khi dùng bucketing, nên **xáo thứ tự các batch** (không xáo các mẫu bên trong). Nếu không, model sẽ thấy toàn chuỗi ngắn trước rồi mới tới chuỗi dài, và gradient bị lệch theo thứ tự đó.

#### P2.4 Đo trên máy bạn: đệm, pack và hai chiều

Chuỗi token ngẫu nhiên trong $\{1..10\}$, độ dài đều trong $[5, 30]$, PAD = 0 (`padding_idx=0`). Embedding 16, hidden 32, Adam 3e-3, clip 1.0, batch 64, 2 seed; accuracy trên 2 000 chuỗi mới (ngẫu nhiên: 0.10). Hai bài:
- **Token cuối:** đoán token thật cuối cùng. Thông tin nằm ngay trước chỗ đọc ra.
- **Token đầu:** đoán token đầu tiên. Phải nhớ qua 5–30 bước, như bài 08.

Ba cách lấy biểu diễn câu:
- **Không pack:** LSTM chạy qua cả phần đệm, lấy `out[:, -1]`.
- **Pack:** dùng `h_n[-1]`, dừng đúng ở token thật cuối của từng chuỗi.
- **BiLSTM + pack:** `cat(h_n[-2], h_n[-1])`.

| Bài (1 500 bước) | Không pack, `out[:, -1]` | Pack, `h_n[-1]` | BiLSTM + pack |
|---|---|---|---|
| Token cuối | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| Token đầu | **0.095 / 0.103** | 1.000 / 1.000 | 1.000 / 1.000 |

**Bất ngờ 1: không pack vẫn làm được bài "token cuối", và làm rất bền.** Token đệm luôn là cùng một vector 0, nên mạng học được cách đi qua nó mà **không đổi trạng thái** (cổng forget ≈ 1, cổng input ≈ 0 khi gặp đệm). Đem model đó test với chuỗi được đệm thêm tới độ dài 60, 120, rồi 300: accuracy vẫn **1.000** ở cả 2 seed. Vậy "lấy `out[:, -1]` trên batch có đệm" không phải lúc nào cũng làm model sai ngay. Nhưng nó vẫn là lỗi, vì lý do ở bất ngờ 2.

**Bất ngờ 2: không pack thì bài "token đầu" thất bại hoàn toàn, còn pack thì giải được.** Bài này khó y như ở bài 08, nhưng kết quả khác hẳn nhau. Giả thuyết: khi pack, chuỗi dài 5 chỉ chạy 5 bước, nên các chuỗi ngắn đóng vai một **"giáo trình dễ"** giúp mạng học cách nhớ, rồi mới tới chuỗi dài. Khi không pack, chuỗi nào cũng bị kéo dài thành khoảng 30 bước (vì gần như batch nào cũng có một chuỗi dài gần 30), nên giáo trình dễ biến mất. Kiểm chứng giả thuyết:

| Thí nghiệm (token đầu) | Accuracy (2 seed) |
|---|---|
| Pack, độ dài 5–30, 200 bước | 0.180 / 0.201 |
| Pack, độ dài 5–30, 500 bước | 0.836 / 0.780 |
| Pack, mọi chuỗi dài **đúng 30**, 1 500 bước | **0.090 / 0.105** |
| BiLSTM + pack, độ dài 5–30, **200 bước** | **1.000 / 1.000** |
| BiLSTM + pack, mọi chuỗi dài đúng 30, 500 bước | 1.000 / 1.000 |

- Bỏ các chuỗi ngắn đi (mọi chuỗi dài 30) thì **pack cũng thất bại**, đúng như dự đoán. Như vậy, không pack đã âm thầm **biến mọi chuỗi ngắn thành chuỗi dài**.
- **BiLSTM giải bài "token đầu" gần như ngay lập tức**, kể cả khi mọi chuỗi dài 30. Chiều ngược đọc từ phải sang trái và **kết thúc** ở token đầu tiên, nên với nó, "token đầu" là thông tin vừa mới đọc, không cần nhớ gì. Cùng lý do đó, BiLSTM mạnh cho phân loại và gán nhãn: mỗi vị trí có ngữ cảnh "gần" từ cả hai phía.

#### P2.5 Code chuẩn

```python
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

PAD = 0

class TokenDataset(Dataset):
    """Mỗi mẫu: (tensor id token độ dài thay đổi, nhãn). Token đã được mã hóa sẵn, id 0 dành cho PAD."""
    def __init__(self, seqs: list[list[int]], labels: list[int]):
        self.seqs = [torch.tensor(s, dtype=torch.long) for s in seqs]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int):
        return self.seqs[i], self.labels[i]            # việc nặng (đọc file, tách từ) nên làm ở __init__ hoặc trước đó

def collate(batch):
    """Hàm cấp module (không dùng lambda) để DataLoader gửi được sang worker process."""
    seqs, ys = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    x = pad_sequence(seqs, batch_first=True, padding_value=PAD)           # (B, T_max)
    mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]           # (B, T_max), True = token thật
    return x, lengths, mask, torch.stack(ys)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab: int, n_cls: int, d: int = 64, H: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=PAD)
        self.rnn = nn.LSTM(d, H, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * H, n_cls)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(self.emb(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.rnn(packed)
        return self.head(torch.cat([h_n[-2], h_n[-1]], dim=-1))          # xuôi đọc hết câu; ngược về tới token đầu

if __name__ == "__main__":                                                # bắt buộc trên Windows khi num_workers > 0
    loader = DataLoader(TokenDataset(seqs, labels), batch_size=64, shuffle=True, collate_fn=collate,
                        num_workers=2, generator=torch.Generator().manual_seed(0))
```

**Lý do thiết kế:**
- `collate` trả về cả `lengths` lẫn `mask`: RNN cần `lengths` để pack; mean-pooling, attention (bài 09) và loss theo token cần `mask`.
- `lengths.cpu()`: `pack_padded_sequence` đòi độ dài nằm trên CPU, kể cả khi batch ở GPU (P4).
- `padding_idx=PAD` giữ vector của PAD luôn bằng 0 và không cập nhật nó. Token đệm vì vậy không mang nội dung gì, nhưng nếu không pack thì RNN **vẫn chạy qua** nó và trạng thái vẫn thay đổi (P2.4).

#### P2.6 Cạm bẫy

1. **Không có `collate_fn` cho chuỗi dài ngắn khác nhau:** `RuntimeError: stack expects each tensor to be equal size, but got [3] at entry 0 and [5] at entry 1` (thông báo thật trên máy bạn).
2. **`collate_fn=lambda ...` với `num_workers > 0`:** `PicklingError: Can't pickle local object <lambda>` trên Windows. Từ Python 3.14, Linux cũng mặc định dùng `forkserver` thay cho `fork` (cảnh báo của PyTorch trên máy bạn nói rõ điều này), nên lỗi này giờ xảy ra cả trên Linux. Hãy dùng hàm cấp module.
3. **Thiếu `if __name__ == "__main__":`** khi `num_workers > 0` trên Windows: mỗi worker import lại script và lại tạo DataLoader.
4. **Gộp theo thời gian mà không mask, hoặc RNN không pack:** `out[:, -1]`, `h.mean(1)`, attention, loss theo token.
   - RNN có thể tự học cách đi qua đệm (P2.4, bất ngờ 1), nhưng mọi chuỗi ngắn bị biến thành chuỗi dài, và bài toán cần nhớ xa có thể trở nên không học được (bất ngờ 2: từ 1.000 xuống 0.10).
   - Với `h.mean(1)` hay attention không mask, **cùng một câu** cho ra biểu diễn khác nhau tùy câu dài nhất trong batch, nên dự đoán phụ thuộc vào việc câu đó được xếp chung batch với câu nào.
5. **`out[:, -1]` của BiLSTM:** nửa ngược mới đọc một token (P2.2).
6. **BiLSTM cho bài toán sinh** hoặc khi dự đoán theo luồng (streaming): nhánh ngược nhìn thấy tương lai (bài 08, cạm bẫy 4).
7. **Việc nặng trong `__getitem__` khi `num_workers=0`:** GPU (hoặc CPU) đứng chờ dữ liệu. Tiền xử lý trước, hoặc tăng `num_workers`.

**Sắc thái phỏng vấn:**
- *"Batch có câu dài câu ngắn, bạn xử lý thế nào?"* Đệm động theo batch (không đệm theo độ dài lớn nhất của toàn bộ dữ liệu); mask hoặc pack; bucketing để giảm đệm; cắt chuỗi quá dài; `ignore_index` cho loss.
- *"Khi nào dùng BiLSTM?"* Khi có cả chuỗi lúc dự đoán: phân loại, gán nhãn từng token (NER, tách từ tiếng Việt). Không dùng cho sinh văn bản. Ngày nay encoder Transformer hai chiều (BERT, Tuần 5) thay thế nó ở hầu hết bài toán.

#### P2.7 Tự làm (20')

1. Gõ lại `TokenDataset`, `collate`, `BiLSTMClassifier`. Kiểm tra: batch 3 chuỗi ở Tầng 1 cho đúng `x` và `mask`; `h_n[-1]` bằng `out[:, 0, H:]` khi không pack.
2. Viết `masked_mean_pool(h, mask)` và kiểm tra với ví dụ $(2, 4, 1)$ ở Tầng 1.
3. Viết `BucketBatchSampler(lengths, batch_size, seed)`: sắp chỉ số theo độ dài, cắt thành batch, **xáo thứ tự các batch**, rồi truyền vào `DataLoader(batch_sampler=...)`. Đo tỉ lệ đệm như P2.3(b).

#### P2.8 Đọc thêm

- PyTorch docs: *Datasets & DataLoaders* tutorial; `torch.utils.data` (mục *Memory Pinning*, *Multi-process data loading*, *Platform-specific behaviors*); `torch.nn.utils.rnn`.
- M. Schuster & K. Paliwal, *Bidirectional Recurrent Neural Networks*, IEEE Transactions on Signal Processing 45 (1997).
- Z. Huang, W. Xu, K. Yu, *Bidirectional LSTM-CRF Models for Sequence Tagging*, arXiv 2015.

---

## P3 · Object detection cơ bản: box, IoU, anchor, NMS, mAP

> ⏱ Tầng 1: 10' · Tầng 2: 35'
> **Cần biết trước:** CNN, stride, feature map (bài 07); transfer learning (bài 10); precision/recall, AP (Tuần 3, bài 04); P1 (focal loss, Smooth L1).

> **📖 Từ mới**
> - **Bounding box:** hình chữ nhật bao quanh vật thể. Dạng `xyxy` là $(x_1, y_1, x_2, y_2)$ (góc trên-trái, góc dưới-phải); dạng `cxcywh` là (tâm, rộng, cao).
> - **IoU (Intersection over Union):** diện tích phần giao chia cho diện tích phần hợp của hai box; bằng 1 khi trùng khít, bằng 0 khi không chạm nhau.
> - **Anchor box:** box "mẫu" đặt sẵn tại mỗi ô của feature map, với vài kích thước và tỉ lệ. Model chỉ học cách **chỉnh** anchor cho khớp vật thể.
> - **NMS (non-maximum suppression):** bỏ các box trùng lặp, chỉ giữ box có điểm cao nhất trong mỗi cụm chồng lấn.
> - **mAP:** trung bình AP qua các lớp (và có khi qua nhiều ngưỡng IoU). Đây là metric chuẩn của detection.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Nhân viên an ninh xem camera trạm BTS:
- **Phân loại ảnh** (bài 07): "Trong khung hình có người không?" Một câu trả lời cho cả ảnh.
- **Object detection:** "Có **mấy** người, **ở đâu**?" Khoanh từng người bằng một hình chữ nhật, kèm nhãn và độ tin cậy.

Cách làm của hầu hết detector: phủ lên ảnh **hàng chục nghìn** khung đoán sẵn (anchor), hỏi từng khung "có vật thể không, nếu có thì dịch khung đi bao nhiêu". Nhiều khung cạnh nhau sẽ cùng báo một người, nên cuối cùng NMS gộp chúng lại.

**Ví dụ có số: IoU.** Box $A = (0, 0, 4, 4)$ và $B = (2, 2, 6, 6)$:
- Phần giao: $x \in [2, 4]$, $y \in [2, 4]$, nên diện tích là $2 \times 2 = 4$.
- Phần hợp: $16 + 16 - 4 = 28$.
- IoU $= 4/28 =$ **0.143**. Hai box trông "chồng nhau khá nhiều" nhưng IoU rất thấp. Ngưỡng 0.5 thông dụng là khắt khe hơn bạn nghĩ.

**Ba ý cốt lõi:**
1. Đầu ra là một **danh sách có độ dài thay đổi** gồm (box, lớp, điểm). Model dự đoán **dày đặc** (mỗi anchor một dự đoán), rồi lọc theo điểm và NMS.
2. Loss gồm hai phần: **phân loại** cho mọi anchor (hầu hết là nền, nên dùng focal loss) và **hồi quy box** chỉ cho các anchor khớp vật thể (Smooth L1 hoặc GIoU).
3. Đánh giá: một dự đoán là đúng nếu IoU với box thật ≥ ngưỡng; quét ngưỡng điểm để có đường precision–recall, rồi lấy **AP**.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### P3.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **IoU (Jaccard index)** | $\lvert A \cap B\rvert / \lvert A \cup B\rvert$ | Jaccard 1901; dùng làm tiêu chí khớp trong PASCAL VOC (Everingham et al., IJCV 2010) |
| **Two-stage detector** | Bước 1 đề xuất vùng (region proposal), bước 2 phân loại và chỉnh từng vùng | R-CNN (Girshick et al., CVPR 2014); Faster R-CNN (Ren et al., NeurIPS 2015) |
| **One-stage detector** | Dự đoán lớp và box trực tiếp tại mọi vị trí của feature map | YOLO (Redmon et al., CVPR 2016); SSD (Liu et al., ECCV 2016); RetinaNet (Lin et al., ICCV 2017) |
| **Anchor box** | Box mẫu tại mỗi vị trí, nhiều kích thước × tỉ lệ (Faster R-CNN dùng 3 × 3 = 9) | Ren et al. 2015 |
| **FPN (Feature Pyramid Network)** | Dự đoán ở nhiều mức stride (8, 16, 32, …): mức mịn cho vật nhỏ, mức thô cho vật lớn | Lin et al., CVPR 2017 |
| **NMS** | Giữ box điểm cao nhất, bỏ mọi box còn lại có IoU > ngưỡng với nó, lặp lại | Dùng từ các detector cổ điển; chuẩn trong Faster R-CNN, YOLO |
| **AP / mAP** | Diện tích dưới đường precision–recall (đã nội suy) của một lớp / trung bình qua các lớp. VOC: IoU 0.5. COCO: trung bình 10 ngưỡng IoU 0.50:0.05:0.95 | Everingham et al. 2010; Lin et al., *Microsoft COCO*, ECCV 2014 |
| **DETR** | Transformer dự đoán trực tiếp một **tập** box; ghép với box thật bằng Hungarian matching, nên **không cần NMS** và anchor | Carion et al., ECCV 2020 |
| **Segmentation** | *Semantic:* gán lớp cho từng pixel (FCN, U-Net). *Instance:* tách từng vật thể ở mức pixel (Mask R-CNN) | Long et al., CVPR 2015; Ronneberger et al., MICCAI 2015; He et al., ICCV 2017 |

#### P3.2 Pipeline của một detector one-stage

```
ảnh 640×640 ─► backbone (CNN pretrained, bài 07 + 10)
            ─► neck (FPN): feature map stride 8, 16, 32  →  80×80, 40×40, 20×20 ô
            ─► head: tại MỖI ô, với MỖI anchor (3 anchor/ô):
                    4 số chỉnh box + 1 điểm "có vật thể" + C điểm lớp
            ─► giải mã box ─► lọc theo điểm (ví dụ > 0.25) ─► NMS theo từng lớp ─► danh sách (box, lớp, điểm)
```

**Gán nhãn khi train** (kiểu RetinaNet): anchor có IoU ≥ 0.5 với một box thật là **dương** (học lớp và học chỉnh box); IoU < 0.4 là **nền**; ở giữa thì **bỏ qua**. Hầu hết anchor là nền dễ, đúng là bối cảnh mà focal loss (P1) được sinh ra.

#### P3.3 Ví dụ tính tay

**(a) IoU.**

| Box 1 | Box 2 | Giao | Hợp | IoU |
|---|---|---|---|---|
| $(0, 0, 4, 4)$ | $(2, 2, 6, 6)$ | 4 | 28 | **0.143** |
| $(0, 0, 4, 4)$ | $(1, 0, 5, 4)$ (dịch phải 1) | 12 | 20 | **0.6** |
| $(0, 0, 4, 4)$ | $(0, 0, 2, 2)$ (nằm trong) | 4 | 16 | **0.25** |
| $(0, 0, 4, 4)$ | $(4, 4, 6, 6)$ (chạm góc) | 0 | 20 | **0** |

Dịch một box cạnh 4 đi **1 đơn vị** (25% cạnh) đã làm IoU giảm từ 1 xuống 0.6. Vật thể nhỏ vì vậy rất khó đạt IoU cao.

**(b) NMS.** Bốn box cùng lớp, sắp theo điểm:

| Box | Tọa độ | Điểm |
|---|---|---|
| A | $(0, 0, 10, 10)$ | 0.9 |
| B | $(1, 1, 11, 11)$ | 0.8 |
| C | $(20, 20, 30, 30)$ | 0.7 |
| D | $(5, 0, 15, 10)$ | 0.6 |

IoU: $A$–$B = 81/119 = 0.681$; $A$–$D = 50/150 = 0.333$; $B$–$D = 0.370$; $C$ không chạm box nào.
- **Ngưỡng 0.5:** giữ $A$; bỏ $B$ (0.681 > 0.5); giữ $C$; $D$ có IoU 0.333 với $A$ và 0 với $C$ nên được giữ. Kết quả: **A, C, D**.
- **Ngưỡng 0.3:** $D$ bị bỏ (0.333 > 0.3). Kết quả: **A, C**.

Ngưỡng NMS thấp thì gộp mạnh tay: ở chỗ đông người, hai người đứng sát nhau có thể bị gộp thành một.

**(c) Mã hóa box so với anchor** (Faster R-CNN). Anchor có tâm $(50, 50)$, kích thước $40 \times 40$. Box thật có tâm $(60, 45)$, kích thước $80 \times 40$:

$$t_x = \frac{x - x_a}{w_a} = 0.25,\quad t_y = \frac{y - y_a}{h_a} = -0.125,\quad t_w = \ln\frac{w}{w_a} = \ln 2 = 0.693,\quad t_h = \ln 1 = 0.$$

Model học dự đoán $(t_x, t_y, t_w, t_h)$, không học tọa độ tuyệt đối. Chia cho kích thước anchor làm cho vật to và vật nhỏ có cùng thang số; dùng $\ln$ cho kích thước để "gấp đôi" và "giảm một nửa" đối xứng nhau. Giải mã ngược lại được đúng $(60, 45, 80, 40)$.

**(d) Đếm dự đoán.** Ảnh $640 \times 640$, FPN ở stride 8, 16, 32, 3 anchor mỗi ô:
- Số ô: $80^2 + 40^2 + 20^2 = 8\,400$; số dự đoán: $8\,400 \times 3 =$ **25 200**.
- Với $C = 80$ lớp (COCO), mỗi anchor cần $4 + 1 + 80 = 85$ số, nên head có $3 \times 85 =$ **255** kênh đầu ra.

Trong 25 200 dự đoán, một ảnh thường chỉ có vài chục vật thể. Tỉ lệ nền : vật thể cỡ 1 000 : 1.

**(e) AP cho một lớp.** Có 3 vật thể thật. 5 dự đoán, sắp theo điểm giảm dần, đã khớp với box thật bằng IoU ≥ 0.5 (mỗi box thật chỉ khớp một lần):

| Hạng | Đúng/sai | TP tích lũy | Precision | Recall |
|---|---|---|---|---|
| 1 | TP | 1 | 1.000 | 0.333 |
| 2 | FP | 1 | 0.500 | 0.333 |
| 3 | TP | 2 | 0.667 | 0.667 |
| 4 | FP | 2 | 0.500 | 0.667 |
| 5 | TP | 3 | 0.600 | 1.000 |

Nội suy: precision tại mỗi mức recall $r$ là **precision lớn nhất ở mọi recall ≥ $r$**. Ba mức recall 1/3, 2/3, 1 có precision nội suy 1.0, 0.667, 0.6. AP $= \tfrac13(1.0 + 0.667 + 0.6) =$ **0.756** (khớp `average_precision_score` trong trường hợp này). Nếu có 4 vật thể thật mà vẫn chỉ 3 TP, recall tối đa chỉ là 0.75 và AP bị giới hạn tương ứng: **bỏ sót vật thể bị phạt qua recall**.

#### P3.4 Code chuẩn (NumPy)

```python
import numpy as np

def box_iou(a, b):
    """a (N, 4), b (M, 4) dạng xyxy → ma trận IoU (N, M)."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])          # góc trên-trái của phần giao
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])          # góc dưới-phải của phần giao
    wh = np.clip(rb - lt, 0, None)                           # không giao thì cạnh = 0
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter)

def nms(boxes, scores, iou_thr=0.5):
    """Trả về chỉ số các box được giữ, theo thứ tự điểm giảm dần."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.array(keep, dtype=np.int64)

def batched_nms(boxes, scores, classes, iou_thr=0.5):
    """NMS riêng cho từng lớp: dịch box của mỗi lớp ra một vùng không chồng lên lớp khác."""
    offset = classes[:, None] * (boxes.max() + 1)
    return nms(boxes + offset, scores, iou_thr)
```

Đã kiểm chứng: `box_iou` và `nms` khớp bản vòng lặp thuần Python trên 200 bộ box ngẫu nhiên (1–40 box); ví dụ P3.3(b) cho `[0, 2, 3]` ở ngưỡng 0.5 và `[0, 2]` ở ngưỡng 0.3. Trong thực tế, dùng `torchvision.ops.box_iou`, `nms`, `batched_nms` (máy bạn chưa cài `torchvision`).

**Lý do thiết kế của `batched_nms`:** hai box trùng nhau nhưng khác lớp (một người đang cầm một chiếc điện thoại) không được loại nhau. Mẹo cộng một khoảng dịch theo lớp biến bài toán "NMS cho từng lớp" thành **một** lần gọi NMS.

#### P3.5 Cạm bẫy

1. **Lẫn định dạng box** (`xyxy`, `xywh`, `cxcywh`; pixel hay chuẩn hóa về $[0, 1]$): IoU vô nghĩa, mà không có lỗi nào. Ghi rõ định dạng trong tên biến hoặc docstring.
2. **Augmentation không biến đổi box:** lật ngang ảnh thì $x$ của box cũng phải thành $W - x$; cắt hoặc resize ảnh thì box cũng phải cắt và co theo.
3. **NMS gộp mọi lớp với nhau:** dùng `batched_nms`.
4. **So mAP của hai bài báo dùng định nghĩa khác nhau:** mAP@0.5 (VOC) thường cao hơn nhiều so với mAP@[.5:.95] (COCO).
5. **Vật thể nhỏ hơn stride:** vật $10 \times 10$ pixel trên feature map stride 32 chỉ chiếm 1/3 ô. Cần mức FPN mịn, hoặc tăng độ phân giải ảnh vào.
6. **Ngưỡng điểm quá cao khi đánh giá:** cắt mất phần recall cao của đường PR, nên AP thấp giả tạo. Khi tính mAP, giữ cả các dự đoán điểm thấp.

**Sắc thái phỏng vấn:**
- *"NMS làm gì, vì sao DETR không cần?"* Detector dày đặc cho nhiều box trùng một vật, NMS giữ box tốt nhất. DETR học **một-một** giữa dự đoán và box thật bằng Hungarian matching lúc train, nên tự nó không sinh box trùng.
- *"mAP tính thế nào?"* Khớp theo IoU, sắp theo điểm, đường PR, nội suy, diện tích; trung bình qua lớp (và qua ngưỡng IoU với COCO).
- *"Detection cho camera trạm BTS chạy trên thiết bị biên?"* Chọn detector one-stage nhỏ; giảm độ phân giải ảnh; chỉ dùng những mức FPN cần cho kích thước vật thể thật; INT8 quantization (Tuần 12); đo latency trên đúng thiết bị.

#### P3.6 Tự làm (15')

1. Gõ lại `box_iou` (dạng vector, không vòng lặp) và kiểm tra bốn dòng của bảng P3.3(a).
2. Gõ lại `nms`, kiểm tra ví dụ P3.3(b) ở hai ngưỡng.
3. Viết `encode(gt, anchor)` / `decode(t, anchor)` cho dạng `cxcywh` và kiểm tra rằng `decode(encode(gt, a), a) == gt` với ví dụ P3.3(c).

#### P3.7 Đọc thêm

- S. Ren, K. He, R. Girshick, J. Sun, *Faster R-CNN*, NeurIPS 2015.
- J. Redmon et al., *You Only Look Once*, CVPR 2016.
- T.-Y. Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017; *Focal Loss for Dense Object Detection*, ICCV 2017.
- N. Carion et al., *End-to-End Object Detection with Transformers* (DETR), ECCV 2020.
- COCO detection evaluation (cocodataset.org, mục *Detection Evaluation*); torchvision docs, *Object Detection Finetuning Tutorial* (chạy trên Kaggle).

---

## P4 · Thiết bị và GPU: code chạy được trên mọi device, đồng bộ, bộ nhớ

> ⏱ Tầng 1: 10' · Tầng 2: 25' · **Loại: Kỹ thuật**
> **Cần biết trước:** tensor, device (bài 01); vòng lặp huấn luyện, checkpoint (bài 02); bộ nhớ khi huấn luyện 16 byte/tham số, mixed precision (bài 06).

> **📖 Từ mới**
> - **Device:** nơi tensor nằm và nơi phép tính chạy: `cpu`, `cuda` (GPU NVIDIA), `mps` (GPU của Apple).
> - **VRAM:** bộ nhớ riêng của GPU. Dữ liệu phải được **chép** từ RAM sang VRAM trước khi GPU dùng được.
> - **Bất đồng bộ (asynchronous):** khi gọi một phép tính trên GPU, Python chỉ **xếp lệnh vào hàng đợi** rồi chạy tiếp ngay, không chờ GPU làm xong.
> - **Đồng bộ (synchronize):** bắt CPU chờ cho tới khi GPU làm xong mọi lệnh đã xếp.
> - **Pinned memory:** vùng RAM bị "ghim" (không bị hệ điều hành chuyển ra đĩa), nên chép sang GPU nhanh hơn và chép được bất đồng bộ.
> - **OOM (out of memory):** hết VRAM.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** CPU là vài **đầu bếp giỏi**, làm được mọi món phức tạp. GPU là một **bếp công nghiệp** với hàng nghìn phụ bếp, mỗi người chỉ làm một động tác đơn giản, nhưng làm cùng lúc. Bếp công nghiệp nằm ở **tòa nhà khác**:
- Nguyên liệu phải được **chở sang** (chép RAM → VRAM). Mọi nguyên liệu của một món phải ở **cùng một bếp**; nếu không, món đó không nấu được (lỗi device).
- Bạn **gửi phiếu order** rồi đi làm việc khác ngay (bất đồng bộ). Muốn biết bếp nấu mất bao lâu thì phải **đứng chờ** món ra (synchronize). Mỗi lần gọi điện hỏi "xong chưa?" (`.item()`, `print`) là bạn bắt mình phải chờ.
- Nếu xe chở nguyên liệu chậm, bếp công nghiệp **ngồi chơi**. Rất hay gặp: GPU chỉ bận 30% vì DataLoader không kịp cấp dữ liệu.

**Ba ý cốt lõi:**
1. Model, dữ liệu, và **mọi tensor tạo ra trong `forward`** phải ở cùng một device. Code nên lấy device từ input (`x.device`) thay vì viết cứng.
2. GPU chạy bất đồng bộ: đo thời gian phải `synchronize`, và hạn chế các điểm đồng bộ ngầm trong vòng lặp.
3. Nút cổ chai thường là **dữ liệu** (đọc, giải mã, chép) hoặc **bộ nhớ**, không phải phép tính.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### P4.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **`Tensor.to(device)`** | Trả về **bản sao** trên device mới (hoặc chính nó nếu đã ở đó). Tensor gốc không đổi | PyTorch docs |
| **`Module.to(device)`** | Chuyển **tại chỗ** mọi parameter và buffer, trả về chính module đó | PyTorch docs |
| **Buffer (`register_buffer`)** | Tensor không học nhưng thuộc trạng thái của module: nằm trong `state_dict`, đi theo `.to()` | bài 02 |
| **CUDA semantics: asynchronous execution** | Kernel GPU được xếp vào stream và chạy bất đồng bộ với CPU | PyTorch docs, *CUDA semantics* |
| **`pin_memory`, `non_blocking=True`** | DataLoader trả batch trong pinned memory; `.to(device, non_blocking=True)` chép mà không chặn CPU | PyTorch docs, *Memory Pinning* |
| **`torch.cuda.synchronize()`** | Chờ GPU hoàn tất mọi lệnh | PyTorch docs |
| **`torch.cuda.max_memory_allocated()`** | Đỉnh VRAM do tensor chiếm, dùng để đo bộ nhớ thật | PyTorch docs, *CUDA semantics: Memory management* |
| **`map_location`** | Tham số của `torch.load`: nạp checkpoint vào device chỉ định | PyTorch docs, *Saving and Loading Models* |
| **DDP (DistributedDataParallel)** | Nhiều GPU, mỗi GPU một process, đồng bộ gradient bằng all-reduce. Thay cho `DataParallel` cũ | PyTorch docs; Li et al., VLDB 2020 |

#### P4.2 Ước lượng nhanh: GPU Kaggle so với máy bạn

Thông số theo datasheet của NVIDIA (đỉnh lý thuyết); máy bạn là số **đo được** ở bài 06.

| | Máy bạn (CPU, cắm sạc) | T4 (Kaggle, có bản 2 × T4) | P100 (Kaggle) |
|---|---|---|---|
| Bộ nhớ | RAM hệ thống | 16 GB | 16 GB |
| Băng thông bộ nhớ | — | 320 GB/s | 732 GB/s |
| FP32 | **0.48 TFLOP/s** đo được (matmul 2048²) | 8.1 TFLOP/s | 9.3 TFLOP/s |
| FP16 (tensor core / mixed) | bf16 chậm ~850 lần so với fp32 | 65 TFLOP/s | 18.7 TFLOP/s |
| Kết nối với CPU | — | PCIe 3.0 ×16, ~16 GB/s lý thuyết | PCIe 3.0 ×16 |

**Ước lượng một bước huấn luyện ảnh** (batch 256, ảnh $3 \times 224 \times 224$):
- **Chép dữ liệu:** float32 là $256 \cdot 3 \cdot 224 \cdot 224 \cdot 4 = 154$ MB, ở 16 GB/s mất khoảng **10 ms**. Nếu gửi `uint8` (38.5 MB) rồi chuẩn hóa trên GPU thì còn khoảng **2.4 ms**.
- **Tính toán:** ResNet-18 khoảng 1.8 tỉ phép nhân–cộng mỗi ảnh (He et al. 2016), tức khoảng 3.6 GFLOP. Forward + backward khoảng gấp 3 lần, nên một bước mất $256 \times 3.6 \times 3 \approx 2.8$ TFLOP. Ở mức giả định 30% của đỉnh FP16 trên T4 (~20 TFLOP/s), mất khoảng **0.14 s**. Trên máy bạn ở 0.48 TFLOP/s là khoảng **6 s**, chậm hơn khoảng 40 lần.
- **Giải mã JPEG trên CPU:** giả định mỗi core giải mã và resize được 400 ảnh/s; 256 ảnh trên 1 core mất **0.64 s**, lâu hơn phần tính toán 4 lần. Đó là lý do phải có `num_workers`, và vì sao máy GPU trên cloud ít CPU core thường bị nghẽn ở khâu dữ liệu.

**Bộ nhớ** (bài 06): ResNet-18 có 11.7 triệu tham số, huấn luyện bằng Adam fp32 cần $11.7\text{M} \times 16 \approx 187$ MB cho trọng số, gradient và trạng thái optimizer. Phần lớn VRAM thật sự bị chiếm bởi **activation**, tăng tuyến tính theo batch size. Nếu bị OOM, giảm batch và dùng gradient accumulation (bài 06), hoặc dùng mixed precision.

#### P4.3 Code chuẩn: chạy được trên CPU, CUDA và MPS

```python
import time
import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")                       # máy bạn: cpu (đã kiểm tra: cuda và mps đều False)

class Net(torch.nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.register_buffer("pos", torch.arange(T))     # buffer: đi theo .to(device), nằm trong state_dict
    def forward(self, x):
        noise = torch.randn(x.shape, device=x.device)    # tensor mới: lấy device từ input
        ...

device = get_device()
model = Net(T=128).to(device)                        # Module.to: tại chỗ, trả về chính nó
opt = torch.optim.AdamW(model.parameters(), lr=1e-3) # tạo SAU khi .to (khuyến nghị của docs torch.optim)

loader = DataLoader(ds, batch_size=256, shuffle=True, collate_fn=collate, num_workers=4,
                    pin_memory=(device.type == "cuda"), persistent_workers=True)

for x, lengths, mask, y in loader:
    x, mask, y = (t.to(device, non_blocking=True) for t in (x, mask, y))   # PHẢI gán lại
    loss = loss_fn(model(x, lengths), y)                                   # lengths giữ ở CPU cho pack (P2)
    ...

# Checkpoint lưu trên GPU, nạp trên máy không có GPU
state = torch.load("ckpt.pt", map_location="cpu", weights_only=True)

# Đo thời gian trên GPU
torch.cuda.synchronize(); t0 = time.perf_counter()
out = model(x)
torch.cuda.synchronize(); dt = time.perf_counter() - t0
```

**Kiểm chứng được trên CPU** (dùng đổi dtype thay cho đổi device, vì cả hai đi qua cùng cơ chế `.to()`):
- Module có `register_buffer("buf", ...)` và một thuộc tính thường `self.plain = torch.zeros(2)`. Sau `model.to(torch.float64)`: `buf` thành float64, `plain` **vẫn là float32**; `state_dict` chỉ có `buf`. Trên GPU, `plain` sẽ nằm lại ở CPU và gây lỗi device trong `forward`.
- `x.to(torch.float64)` mà không gán lại: `x` vẫn là float32.
- Sau `lin.to(...)`, `lin.weight` **vẫn là cùng một object** Parameter, nên optimizer tạo trước đó vẫn trỏ đúng. Docs vẫn khuyên tạo optimizer **sau** khi `.to()` cho chắc chắn.
- `torch.load(..., map_location="cpu", weights_only=True)` nạp được `state_dict`.

#### P4.4 Cạm bẫy

1. **Lỗi device:** `RuntimeError: Expected all tensors to be on the same device ...`. Thường do:
   - `x.to(device)` mà không gán lại;
   - tensor tạo trong `forward` không có `device=` (mặc định là CPU);
   - tensor lưu bằng thuộc tính thường thay vì `register_buffer`;
   - `nn.Module` con lưu trong `list` Python thay vì `nn.ModuleList` (bài 03).
2. **Đo thời gian không `synchronize`:** chỉ đo được thời gian **xếp lệnh**, có thể nhanh hơn thực tế hàng chục lần.
3. **Điểm đồng bộ ngầm trong vòng lặp:** `loss.item()`, `print(tensor)`, `.cpu()`, `.numpy()`, `if tensor > 0:`. Mỗi lần gọi, CPU đứng chờ GPU. Gọi `.item()` mỗi bước thì chấp nhận được; gọi hàng chục lần mỗi bước (log từng metric) thì không.
4. **GPU chỉ bận 30%:** DataLoader không kịp. Kiểm tra `num_workers`, `pin_memory`, việc nặng trong `__getitem__`; giải mã và resize ảnh trước; gửi `uint8`.
5. **OOM khi đánh giá:** quên `torch.no_grad()` (hoặc `torch.inference_mode()`), nên autograd giữ activation của mọi batch. Hoặc cộng dồn `total += loss` (P1, cạm bẫy 7).
6. **`torch.cuda.empty_cache()` không chữa được OOM:** nó chỉ trả lại phần bộ nhớ **đã rảnh** trong cache; tensor còn được tham chiếu vẫn chiếm chỗ.
7. **Kết quả trên GPU không lặp lại được hoàn toàn**, kể cả với cùng seed: một số kernel dùng phép cộng nguyên tử (atomic add) theo thứ tự không cố định, và `cudnn.benchmark=True` có thể chọn thuật toán khác nhau giữa các lần chạy. Cần thì bật `torch.use_deterministic_algorithms(True)` (chậm hơn).

**Sắc thái phỏng vấn:**
- *"Training trên GPU chậm, GPU utilization thấp, bạn chẩn đoán thế nào?"* Đo thời gian chờ dữ liệu so với thời gian một bước (`torch.profiler`, `nvidia-smi`); tăng `num_workers`, bật `pin_memory`, tiền xử lý trước; tăng batch size; bỏ các điểm đồng bộ ngầm; bật mixed precision để dùng tensor core.
- *"Viết code để chạy được cả trên laptop lẫn server GPU?"* Một hàm `get_device()`; device lấy từ input; buffer thay vì thuộc tính thường; `map_location` khi nạp checkpoint; không viết cứng `.cuda()`.

#### P4.5 Tự làm (15', thêm 30' nếu chạy trên Kaggle)

1. Thêm tham số `device` cho `train_one_epoch`, `evaluate`, `predict` trong `qnet/loop.py` của bạn (chuyển từng batch sang `device`). Chạy lại toàn bộ test với `device="cpu"`: phải pass như cũ.
2. Viết một module có một thuộc tính tensor thường, gọi `.to(torch.float64)`, và quan sát lỗi khi `forward` cộng nó với input float64. Sửa bằng `register_buffer`.
3. *(Tùy chọn, Kaggle T4.)* Chạy `python -m qnet.train` với `device="cuda"`, so thời gian với máy bạn. Dự đoán trước: MLP 5 417 tham số với batch 128 là quá nhỏ để GPU phát huy, nên có thể **không nhanh hơn**, vì chi phí khởi chạy kernel và chép dữ liệu lấn át phần tính toán. Kiểm tra dự đoán đó.

#### P4.6 Đọc thêm

- PyTorch docs: *CUDA semantics* (asynchronous execution, memory management), *Reproducibility*, `torch.utils.data` (*Memory Pinning*), *Saving and Loading Models* (`map_location`).
- PyTorch Tutorials: *Performance Tuning Guide*; *Getting Started with Distributed Data Parallel*.
- NVIDIA T4 và Tesla P100 datasheet.
- S. Li et al., *PyTorch Distributed: Experiences on Accelerating Data Parallel Training*, VLDB 2020.

---

## Quiz phụ lục Tuần 4

> ⏱ **30 phút** · 10 câu nhanh (1đ) + 2 câu tự luận (5đ) · **20 điểm** · Đạt khi ≥ 14
> Làm trên giấy, được dùng máy tính cầm tay. **Không có câu chứng minh.**

**Phần 1 — Câu nhanh**

1. Train bằng `pos_weight` $w = 4$. Tại một điểm có xác suất thật 0.2, model sẽ học ra xác suất bao nhiêu?
2. Focal loss $\gamma = 2$, không có $\alpha$, với $p_t = 0.8$: tính CE và focal loss.
3. Vì sao không nên viết loss là `-(y * torch.log(torch.sigmoid(z)))`? Với $z = -200$, $y = 1$, kết quả là gì, và hàm nào cho kết quả đúng?
4. Label smoothing $\varepsilon = 0.1$, $K = 5$ lớp, lớp đúng là lớp thứ 3. Viết vector nhãn đích.
5. Đổi `reduction` từ `mean` sang `sum` với batch 64. Bước cập nhật thay đổi thế nào với SGD, và với Adam?
6. Một batch có các chuỗi dài 2, 5, 3, 6, được đệm tới độ dài lớn nhất. Bao nhiêu phần trăm ô là đệm?
7. `nn.LSTM(8, 16, bidirectional=True, batch_first=True)`, đầu vào `(4, 10, 8)`. Cho số tham số, shape của `out` và của `h_n`.
8. Tính IoU của $(0, 0, 6, 4)$ và $(2, 0, 8, 4)$.
9. Anchor tâm $(100, 100)$, kích thước $50 \times 100$; model dự đoán $t = (0.2,\ -0.1,\ \ln 2,\ 0)$. Box giải mã ra (tâm, rộng, cao) là gì?
10. Một bạn viết `self.mask = torch.ones(T)` trong `__init__`, rồi gọi `model.to("cuda")`. `forward` báo lỗi device. Vì sao, và sửa thế nào?

**Phần 2 — Tự luận**

**T1 · Chọn loss cho phát hiện cuộc gọi lừa đảo (5đ).** Tổng đài có 2% cuộc gọi lừa đảo. Đội vận hành cần **xác suất đúng** để chọn ngưỡng theo chi phí: chặn nhầm một cuộc gọi sạch tốn $c_{FP} = 1$, bỏ lọt một cuộc gọi lừa đảo tốn $c_{FN} = 20$. Một đồng nghiệp train bằng `BCEWithLogitsLoss(pos_weight=49)`.
- (a) Model này cho một cuộc gọi xác suất 0.5. Xác suất thật là bao nhiêu?
- (b) Tính ngưỡng tối ưu theo chi phí trên xác suất thật, rồi đổi nó về thang đầu ra của model `pos_weight`.
- (c) Đồng nghiệp đề xuất đổi sang focal loss $\gamma = 2$ "cho tốt hơn". Dựa vào bảng P1.4, bạn trả lời thế nào? Đề xuất quy trình so sánh.

**T2 · Gỡ lỗi pipeline phân loại tin nhắn (5đ).** Đoạn code sau chạy trên Windows để phân loại intent của tin nhắn khách hàng (độ dài khác nhau), bằng BiLSTM:

```python
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4,
                    collate_fn=lambda b: (pad_sequence([s for s, _ in b], batch_first=True),
                                          torch.stack([y for _, y in b])))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 64)
        self.rnn = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.head = nn.Linear(256, n_cls)
    def forward(self, x):
        out, _ = self.rnn(self.emb(x))
        return self.head(out.mean(dim=1))

for x, y in loader:
    x.to(device); y.to(device)
    ...
```

Tìm **4 lỗi**. Với mỗi lỗi, nêu hậu quả và cách sửa.

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Phần 1**

1. odds $= 4 \times (0.2/0.8) = 1$, nên $p =$ **0.5**.
2. CE $= -\ln 0.8 =$ **0.223**; hệ số $(1 - 0.8)^2 = 0.04$; focal $=$ **0.0089**.
3. Trong float32, $\sigma(-200) = 0$ (thực tế đã bằng 0 từ $z \le -89$), nên $\log 0 = -\infty$ và loss là **`inf`**. `F.binary_cross_entropy_with_logits` (hoặc `-F.logsigmoid(z)`) cho đúng **200**.
4. $(0.02,\ 0.02,\ 0.92,\ 0.02,\ 0.02)$: lớp đúng nhận $0.9 + 0.1/5$, các lớp khác nhận $0.1/5$.
5. Gradient lớn lên **64 lần**. Với SGD: bước cập nhật lớn lên 64 lần, tương đương nhân learning rate với 64. Với Adam: **gần như không đổi**, vì cả $m$ lẫn $\sqrt v$ cùng nhân với 64 nên tỉ số giữ nguyên (chỉ $\varepsilon$ làm lệch rất nhỏ).
6. $4 \times 6 = 24$ ô, 16 token thật, nên đệm chiếm $8/24 =$ **33.3%**.
7. Tham số: $2 \cdot 4 \cdot 16 \cdot (8 + 16 + 2) =$ **3 328**. `out`: **(4, 10, 32)**. `h_n`: **(2, 4, 16)**.
8. Giao: $x \in [2, 6]$, $y \in [0, 4]$, nên $4 \times 4 = 16$. Hợp: $24 + 24 - 16 = 32$. IoU $=$ **0.5**.
9. Tâm $x = 100 + 0.2 \cdot 50 = 110$; tâm $y = 100 - 0.1 \cdot 100 = 90$; rộng $= 50 \cdot 2 = 100$; cao $= 100 \cdot e^0 = 100$. Box: **tâm (110, 90), 100 × 100**.
10. Thuộc tính tensor thường **không** được `Module.to()` chuyển, nên `mask` nằm lại ở CPU trong khi input ở GPU. Sửa: `self.register_buffer("mask", torch.ones(T))`, hoặc tạo trong `forward` với `device=x.device`.

**T1 (5đ)**

(a) 1.5đ. Chia odds cho $w$: odds $= 1/49$, nên $p = (1/49)/(1 + 1/49) = 1/50 =$ **0.02**. "50/50" của model chỉ bằng đúng tỉ lệ nền.

(b) 2đ. $t^\star = c_{FP}/(c_{FP} + c_{FN}) = 1/21 =$ **0.0476** trên xác suất thật. Đổi về thang của model: odds $= (1/21)/(20/21) = 0.05$; nhân với 49 được 2.45; $p = 2.45/3.45 =$ **0.710**. Model `pos_weight` phải cho ≥ 0.71 thì mới chặn.

(c) 1.5đ. Trên dữ liệu `quality` (P1.4), focal loss làm AUC **giảm** (−0.007, chỉ thắng 1/5 split), AP giảm, ECE tăng lên 0.169 và **không có công thức hiệu chỉnh đơn giản** như `pos_weight`. Focal loss được thiết kế cho detection với tỉ lệ nền 1 000 : 1 và mẫu dễ áp đảo. Đề xuất: baseline là BCE thường cộng ngưỡng theo chi phí; so BCE, `pos_weight` (đã hiệu chỉnh) và focal **theo cặp split** (hoặc theo cặp fold, chia theo thời gian) bằng AP và chi phí kỳ vọng ở ngưỡng chọn; kiểm tra ECE; chỉ đổi sang focal nếu nó thắng rõ ràng trên đa số split.

**T2 (5đ, mỗi lỗi 1.25đ: nêu đúng lỗi, hậu quả, cách sửa)**

1. **`collate_fn` là lambda với `num_workers=4` trên Windows:** `PicklingError: Can't pickle ... <lambda>` (từ Python 3.14 thì cả Linux cũng vậy). Sửa: hàm `collate` cấp module, và đặt code chạy trong `if __name__ == "__main__":`. Nên trả thêm `lengths` và `mask`.
2. **`out.mean(dim=1)` trung bình cả token đệm:** câu ngắn bị pha loãng, và biểu diễn phụ thuộc vào độ dài của câu **dài nhất trong batch**. Hậu quả: cùng một tin nhắn có thể bị phân loại khác nhau tùy nó nằm chung batch với tin nào. Sửa: mean-pooling có mask, hoặc pack rồi dùng `torch.cat([h_n[-2], h_n[-1]], -1)`. Thêm `padding_idx=0` cho `Embedding`.
3. **BiLSTM không pack:** nhánh xuôi chạy qua đệm ở cuối, còn nhánh ngược của mọi câu ngắn **bắt đầu bằng một loạt token đệm**. Mọi câu ngắn thực chất bị kéo dài bằng câu dài nhất trong batch; P2.4 cho thấy điều này có thể biến một bài học được (1.000) thành không học được (0.10). Sửa: `pack_padded_sequence(..., lengths, enforce_sorted=False)` và truyền `lengths` ra từ `collate`.
4. **`x.to(device)` không gán lại:** `x` vẫn ở CPU, nên gặp lỗi device ở lớp đầu tiên (nếu `model` ở GPU). Sửa: `x, y = x.to(device), y.to(device)`.

(Chấp nhận các lỗi hợp lý khác thay cho một trong bốn lỗi trên, ví dụ thiếu `padding_idx`, hoặc không truyền `generator` nên không tái lập được thứ tự batch, nếu nêu đúng hậu quả.)

</details>
