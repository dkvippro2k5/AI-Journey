# Project Tuần 2 · `optlab` — phòng thí nghiệm tối ưu hóa bằng NumPy

> **Ngày 4:** viết vòng lặp huấn luyện và 4 thí nghiệm (~5h).
> **Ngày 5:** chạy thí nghiệm, viết `REPORT.md` (2h), sau đó làm final (3h).
> **Mục tiêu:** dùng **chính code của bạn** để kiểm chứng các định lý của tuần. Lý thuyết nói gì, thí nghiệm phải cho thấy đúng như vậy; chỗ nào lệch thì giải thích được vì sao.
> **Liên hệ với hệ thống chung:** `linalg.py` (PCA, cosine) được dùng lại khi nén embedding ở Tuần 8; `stats.py` (`mean_ci`, `wilson_interval`) được dùng lại cho evaluation ở Tuần 13.

## 1. Bài toán

```
data.py (cho sẵn)            linalg.py (Ngày 1)        grad.py + optim.py (Ngày 2)     stats.py (Ngày 3)
make_regression ──┐          standardize, lstsq,       mse_loss, logistic_loss,        mean_ci
make_classification├──────▶  hessian_eigs_mse ───────▶  SGD/Momentum/RMSProp/Adam ───▶ (5 seed → mean ± CI)
rosenbrock ───────┘                                            │
                                                     train.py (Ngày 4): vòng lặp chung
                                                               │
                                       cli.py (Ngày 4): 4 thí nghiệm → results/*.csv + *.png
```

| Thí nghiệm | Câu hỏi | Bài lý thuyết |
|---|---|---|
| `lr-sweep` | Có đúng là GD phân kỳ khi và chỉ khi $\alpha > 2/L$ không? | Mệnh đề 6.3 |
| `scaling` | Số bước của GD có tỉ lệ với $\kappa$ không? Chuẩn hóa giúp được bao nhiêu? | Hệ quả 6.4, Mệnh đề 6.5 |
| `optimizers` | Bốn optimizer khác nhau thế nào trên logistic regression, và khác biệt đó có ý nghĩa thống kê không? | Bài 07, bài 09 |
| `rosenbrock` | Trong một thung lũng hẹp và cong, momentum và Adam hơn GD ở đâu? | Bài 07 |

## 2. Thư mục project

```
project/
├── pyproject.toml
├── optlab/
│   ├── __init__.py
│   ├── data.py          # cho sẵn: make_regression, make_classification, rosenbrock
│   ├── linalg.py        # Ngày 1
│   ├── grad.py          # Ngày 2
│   ├── optim.py         # Ngày 2
│   ├── stats.py         # Ngày 3
│   ├── train.py         # Ngày 4
│   └── cli.py           # Ngày 4
├── tests/               # 45 test cho cả 4 ngày (cho sẵn)
└── _reference/          # lời giải tham khảo — chỉ mở khi kẹt quá 1h
```

## 3. Ngày 4 — `optlab/train.py`

Kiểm tra bằng `tests/test_train.py` (6 test).

```python
@dataclass
class History:
    losses: list[float]        # loss trên TOÀN BỘ dữ liệu: trước epoch 1 và sau mỗi epoch
    dists: list[float]         # ‖θ − θ*‖ ở cùng các thời điểm (rỗng nếu không có theta_star)
    thetas: list[np.ndarray]   # bản sao θ ở cùng các thời điểm (dùng để vẽ quỹ đạo)
    steps: int                 # tổng số lần gọi opt.step
    diverged: bool
    reached_tol: bool

def train(loss_fn, theta0, X, y, opt, *, epochs, batch_size=None, seed=0,
          clip=None, theta_star=None, tol=None, max_loss=1e6) -> tuple[np.ndarray, History]
```

- `loss_fn(theta, X, y) -> (loss, grad)`. Với Rosenbrock thì `X = y = None`.
- `batch_size=None`, hoặc `X is None`, hoặc `batch_size >= n`: **full batch**, 1 bước mỗi epoch. Ngược lại: mỗi epoch xáo trộn bằng `rng.permutation(n)` rồi chia thành các batch liên tiếp (batch cuối có thể nhỏ hơn). `rng = np.random.default_rng(seed)` được tạo **một lần** trước vòng lặp.
- `clip`: gọi `clip_by_norm` lên gradient trước khi `opt.step`.
- **Ghi lại** (`losses`, `thetas`, và `dists` nếu có `theta_star`) một lần **trước** epoch đầu và một lần **sau mỗi epoch**. Sau mỗi lần ghi, kiểm tra hai điều kiện dừng:
  - loss không hữu hạn hoặc lớn hơn `max_loss`: `diverged = True`, dừng;
  - có `tol` và `dists[-1] < tol`: `reached_tol = True`, dừng.

## 4. Ngày 4 — `optlab/cli.py`

Kiểm tra bằng `tests/test_cli.py` (5 test). Mỗi hàm thí nghiệm trả về `(DataFrame, curves)`; `curves` là dict dùng để vẽ hình.

**`exp_lr_sweep(seed=0)`**
- Dữ liệu: `X, y, _ = make_regression(500, 2, scales=[1, 3], seed=seed)`, rồi `Xb = add_bias(X)` (**không** chuẩn hóa, để $\kappa \approx 9$).
- `L = hessian_eigs_mse(Xb)[-1]`, `theta_star = lstsq(Xb, y)`.
- Với mỗi `m` trong `(0.1, 0.5, 1.0, 1.9, 2.1)`: `train(mse_loss, zeros, Xb, y, SGD(m / L), epochs=2000, theta_star=theta_star, tol=1e-6)`.
- Các cột: `lr_over_L, lr, steps, reached_tol, final_loss, diverged`. `curves[f"lr={m}/L"] = h.losses`.

**`exp_scaling(seed=0)`**
- Dữ liệu: `make_regression(500, 2, scales=[1, 100], offsets=[0, 50], seed=seed)`.
- Hai cấu hình: `"raw"` = `add_bias(X)` và `"standardized"` = `add_bias(standardize(X)[0])`.
- Với mỗi cấu hình: `ev = hessian_eigs_mse(M)`, $\kappa$ `= ev[-1] / ev[0]`, rồi chạy `SGD(1 / ev[-1])` với `epochs=5000`, `tol=1e-3`, `theta_star = lstsq(M, y)`.
- `predicted_steps = ceil(ln(dists[0] / 1e-3) / -ln(1 - 1/κ))` (Hệ quả 6.4); nếu `1 - 1/κ <= 0` thì lấy 1.
- Các cột: `setting, kappa, steps, reached_tol, final_dist, predicted_steps`. `curves[setting] = h.dists`.

**`exp_optimizers(seeds=range(5), epochs=5)`**
- Với mỗi seed `s`: `X, y = make_classification(400, seed=s)`; loss tối ưu `best[s]` lấy bằng `train(logistic_loss, zeros(3), add_bias(X), y, SGD(1.0), epochs=2000)` (full batch; hàm lồi nên hội tụ về tối ưu toàn cục).
- Learning rate: `{"sgd": 0.1, "momentum": 0.1, "rmsprop": 0.01, "adam": 0.01}`. Mỗi optimizer chạy `batch_size=32`, `seed=s`, rồi tính `excess = h.losses[-1] - best[s]`.
- `mean_ci(excess)` cho ra các cột `optimizer, lr, mean_excess, ci_lo, ci_hi`. `curves[name]` là `losses` của seed đầu tiên.

**`exp_rosenbrock(epochs=3000)`**
- Xuất phát từ `(-1.5, 2.0)`, `theta_star = (1, 1)`, full batch.
- Learning rate: `{"sgd": 1e-3, "momentum": 1e-3, "rmsprop": 1e-2, "adam": 2e-2}`.
- Các cột: `optimizer, lr, final_loss, final_dist, diverged`. `curves[name] = np.array(h.thetas)` (quỹ đạo).

**`main(argv=None)`**
- Lệnh: `python -m optlab.cli [all|lr-sweep|scaling|optimizers|rosenbrock] [--out results] [--no-plot]`.
- Ghi `<out>/<tên>.csv`. Nếu không có `--no-plot` thì ghi thêm `<out>/<tên>.png` (dùng `matplotlib.use("Agg")`): loss hoặc khoảng cách theo epoch, **trục y dạng log**; riêng Rosenbrock vẽ đường đồng mức kèm quỹ đạo.
- In từng bảng ra màn hình. Cấu hình `logging` tại đây.

## 5. Chạy thử

```
cd project
python -m pytest -q                       # 45 test của cả 4 ngày
python -m optlab.cli --out results        # ~4 giây
```

**Kết quả mong đợi** (đo trên máy bạn bằng lời giải tham khảo):

```
== lr-sweep ==
 lr_over_L      lr  steps  reached_tol  final_loss  diverged
       0.1 0.01187   1320         True    0.005086     False
       0.5 0.05935    259         True    0.005086     False
         1  0.1187    126         True    0.005086     False
       1.9  0.2255    121         True    0.005086     False
       2.1  0.2493     77        False   1.122e+06      True

== scaling ==
     setting     kappa  steps  reached_tol  final_dist  predicted_steps
         raw 1.531e+04   5000        False       1.009           111153
standardized     1.013      2         True   0.0002363                2

== optimizers ==
optimizer   lr  mean_excess     ci_lo    ci_hi
      sgd  0.1      0.03133   0.01666  0.04601
 momentum  0.1     0.001134 0.0002978 0.001969
  rmsprop 0.01      0.08035   0.05122   0.1095
     adam 0.01       0.1029   0.07099   0.1349

== rosenbrock ==
optimizer    lr  final_loss  final_dist  diverged
      sgd 0.001     0.04867      0.4507     False
 momentum 0.001   3.319e-12   4.077e-06     False
  rmsprop  0.01     0.02264     0.06061     False
     adam  0.02   3.148e-11   1.256e-05     False
```

## 6. Ngày 5 — `REPORT.md`: giải thích từng con số bằng lý thuyết

Trả lời **bằng phép tính**, không chỉ bằng lời.

1. **`lr-sweep`.**
   - In ra `hessian_eigs_mse(Xb)` để có $\mu$, $L$, $\kappa$.
   - Vì sao `2.1/L` phân kỳ còn `1.9/L` thì không?
   - Dùng $\rho(\alpha) = \max(\lvert 1 - \alpha\mu\rvert, \lvert 1 - \alpha L\rvert)$ để dự đoán số bước cho `1.0/L` và `1.9/L`, rồi so với 126 và 121. Vì sao hai con số gần nhau như vậy, dù learning rate gần gấp đôi?
2. **`scaling`.**
   - Cột `raw` dự đoán cần khoảng 111 000 bước, nên ngân sách 5 000 bước không đủ. Ước lượng thời gian nếu chạy đủ.
   - Vì sao cột có **trung bình 50** cũng làm $\kappa$ tăng, chứ không chỉ cột có độ lệch chuẩn 100? (Gợi ý: cosine giữa cột đó và cột bias.)
   - Lưu ý: $\theta^\star$ của hai cấu hình là **hai tham số hóa khác nhau** của cùng một mô hình.
3. **`optimizers`.**
   - Optimizer nào tốt hơn **có ý nghĩa thống kê**? So sánh các khoảng tin cậy, và giải thích vì sao excess loss được **ghép cặp theo seed** (trừ đi `best[s]` của chính seed đó).
   - Vì sao Adam với `lr = 0.01` chậm nhất? Gợi ý: $\theta^\star \approx (-0.05,\ 1.75,\ 1.91)$; Adam dịch mỗi tham số tối đa khoảng `lr` mỗi bước (Mệnh đề 7.3); mỗi epoch có $\lceil 400/32\rceil = 13$ bước, 5 epoch là 65 bước.
   - Thử lại với Adam `lr = 0.1` và báo cáo kết quả.
   - Momentum với `lr = 0.1` có learning rate hiệu dụng bằng bao nhiêu (Mệnh đề 7.2)?
4. **`rosenbrock`.**
   - Nhìn `rosenbrock.png`: mô tả quỹ đạo của SGD.
   - Tính Hessian của Rosenbrock tại $(1, 1)$, rồi tính trị riêng và $\kappa$ của nó. Điều này giải thích gì về tốc độ của SGD gần đích?
   - Vì sao momentum vượt trội trong thung lũng hẹp?
5. **(Mở rộng) Nhiễu của mini-batch.** Viết một hàm nhỏ đo phương sai của gradient mini-batch trên bài logistic với $B = 1, 8, 64$, và kiểm tra tỉ lệ $1/B$ (Mệnh đề 6.6).

**Checklist hoàn thành:**
- [ ] `python -m pytest -q`: cả 45 test pass.
- [ ] `python -m ruff check .` không còn lỗi.
- [ ] `results/` có đủ 4 file CSV và 4 hình PNG; `results/` nằm trong `.gitignore`.
- [ ] `REPORT.md` trả lời đủ câu 1–4, mỗi câu có ít nhất một phép tính đối chiếu lý thuyết với số đo.

**Mở rộng (không bắt buộc):**
- Thêm `AdamW` (quiz Ngày 2, T4) và lịch `cosine_lr` vào `train` (tham số `schedule`), rồi so sánh trên `optimizers`.
- Thêm thí nghiệm `pca`: sinh 2 000 vector 64 chiều có cấu trúc hạng thấp, nén về $k$ chiều, đo xem top-10 theo cosine trước và sau khi nén trùng nhau bao nhiêu (chuẩn bị cho Tuần 8).
