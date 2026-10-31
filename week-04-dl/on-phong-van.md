# Ôn phỏng vấn · Tuần 4 · Deep Learning và PyTorch

> **Dùng khi nào:** sau final, hoặc ôn lại vào các tuần sau (khuyến nghị: lần đầu ngay sau Tuần 4, rồi ôn lại sau 1 tuần và sau 1 tháng).
> **Cách luyện (khoảng 45'):**
> 1. Đọc câu hỏi, **chưa mở đáp án**.
> 2. Hẹn giờ **1 phút**, **nói to** câu trả lời như đang ngồi phỏng vấn (ghi âm càng tốt).
> 3. Mở đáp án, đối chiếu với **"Ý phải có"**. Thiếu ý nào thì đánh dấu, và nói lại lần nữa.
> 4. Câu nào thiếu từ 2 ý trở lên thì đọc lại bài được dẫn ở cuối câu.
>
> **Mẹo chung:** trả lời theo khung **định nghĩa ngắn → vì sao → ví dụ có số → khi nào dùng / cạm bẫy**. Một con số cụ thể bạn tự đo được ("trên project của mình, …") thuyết phục hơn mọi định nghĩa.

---

## A. PyTorch và huấn luyện

**1. Backpropagation là gì? Vì sao deep learning dùng reverse-mode autodiff?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Backprop là cách tính gradient của loss theo mọi tham số bằng chain rule, đi **ngược** từ loss về đầu vào. Ở mỗi nút, gradient bằng *upstream gradient* nhân *local gradient*; nếu một biến dùng ở nhiều chỗ thì các gradient **cộng** lại. Deep learning dùng reverse mode vì bài toán có **một** đầu ra vô hướng (loss) và **rất nhiều** đầu vào (tham số). Reverse mode cho toàn bộ gradient chỉ với chi phí cỡ vài lần một lượt forward, còn forward mode phải chạy một lần cho mỗi tham số. Cái giá là phải **lưu activation** của forward pass, nên huấn luyện tốn bộ nhớ hơn suy luận nhiều.

**Ý phải có:** chain rule đi ngược · upstream × local · một đầu ra, nhiều đầu vào · tốn bộ nhớ activation.
**Nếu bị hỏi tiếp:** *"Gradient của softmax + cross-entropy theo logit?"* → $p - y$.
**Xem lại:** bài 01; Tuần 2, bài 05.
</details>

**2. Vì sao phải gọi `optimizer.zero_grad()`? `model.eval()` có tắt gradient không?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** PyTorch **cộng dồn** gradient vào `.grad` chứ không ghi đè. Đó là thiết kế có chủ đích để hỗ trợ gradient accumulation, nên mỗi bước phải tự xóa gradient. Quên bước này thì code vẫn chạy và không báo lỗi; trên project của mình, val loss dao động 0.40–0.59 thay vì xuống 0.35. `model.eval()` **không** tắt gradient. Nó chỉ đổi hành vi của dropout (tắt) và BatchNorm (dùng running stats). Khi đánh giá phải dùng cả `model.eval()` lẫn `torch.no_grad()`.

**Ý phải có:** cộng dồn · lỗi "im lặng" · `eval` chỉ đổi dropout/BN · cần thêm `no_grad`.
**Xem lại:** bài 01, 02 (mục 2.4).
</details>

**3. Model không học (loss đứng yên) hoặc loss thành `nan`. Bạn debug thế nào?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Đầu tiên mình **overfit một batch nhỏ**, khoảng 30 mẫu: một mạng đủ lớn phải đưa loss về gần 0; nếu không được thì chắc chắn có bug, chưa cần nghĩ tới hyperparameter. Sau đó mình so giá trị loss với các mốc dễ đoán. Ví dụ phân loại 5 lớp mà loss kẹt ở $\ln 5 = 1.609$ nghĩa là model đang đoán đều, gradient không tới được tham số; một lỗi hay gặp là tạo optimizer **trước khi** thay lớp cuối. Còn kẹt ở entropy của tỉ lệ lớp nghĩa là model chỉ học được tỉ lệ nền, nên phải kiểm tra đầu vào và nhãn. Với `nan`: giảm learning rate, thêm warmup và gradient clipping, kiểm tra dữ liệu có NaN/inf không; nếu dùng fp16 thì phải có `GradScaler`. Mình luôn log chuẩn gradient, vì nó thường vọt lên **trước** khi loss thành `nan`.

**Ý phải có:** overfit một batch · so với mốc $\ln K$ / entropy của tỉ lệ nền · lr, warmup, clipping · fp16 cần GradScaler · log chuẩn gradient.
**Xem lại:** quiz Ngày 1 T2, T4; final D3.
</details>

**4. Batch size ảnh hưởng thế nào tới huấn luyện? Tăng batch thì đổi learning rate ra sao?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Batch lớn cho gradient ít nhiễu hơn (phương sai tỉ lệ khoảng $1/B$) và tận dụng phần cứng tốt hơn, nhưng **số bước cập nhật mỗi epoch giảm**. Với cùng số epoch, model được cập nhật ít lần hơn. Trên project của mình, dùng cả tập train làm một batch thì chỉ có 1 bước mỗi epoch, và AUC tụt từ 0.78 xuống 0.73. Quy tắc kinh nghiệm: tăng batch $k$ lần thì tăng lr tuyến tính $k$ lần, kèm warmup (Goyal et al. 2017); quy tắc này hết đúng khi batch quá lớn. Batch nhỏ thì nhiễu hơn, đôi khi giúp tổng quát hóa, nhưng làm hỏng BatchNorm.

**Ý phải có:** nhiễu gradient · số bước cập nhật · linear scaling + warmup · batch nhỏ hại BN.
**Xem lại:** bài 02 (mục 2.4c), bài 05.
</details>

## B. Huấn luyện mạng sâu

**5. Vanishing/exploding gradient là gì? Nêu các cách khắc phục.**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Gradient đi ngược qua $L$ lớp là **tích** của $L$ Jacobian. Nếu "hệ số" mỗi lớp nhỏ hơn 1 thì gradient co lại theo hàm mũ, ngược lại thì phình ra. Ví dụ sigmoid có đạo hàm tối đa 1/4: qua 10 lớp còn khoảng $10^{-6}$; mình đo được tỉ lệ gradient giữa lớp đầu và lớp cuối là $1.5 \times 10^{-6}$. Các cách khắc phục:
- Dùng **ReLU/GELU** cho lớp ẩn.
- **Khởi tạo đúng** (He cho ReLU) để hệ số mỗi lớp xấp xỉ 1.
- **Normalization** (BN/LN).
- **Residual connection**: đạo hàm có dạng $I + \partial f/\partial h$, nên gradient luôn có đường đi thẳng.
- **Gradient clipping** chống bùng nổ.
- Với chuỗi: **LSTM/GRU** thay cho RNN thường.

**Ý phải có:** tích Jacobian theo hàm mũ · ví dụ sigmoid · ít nhất 4 trong 6 cách chữa, có giải thích residual.
**Xem lại:** bài 04.
</details>

**6. He init khác Xavier init thế nào? Vì sao có hệ số 2?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Mục tiêu là giữ phương sai của activation không đổi qua các lớp. Với $z = \sum_{i=1}^{n} w_i x_i$, ta có $\mathrm{Var}(z) = n\,\mathrm{Var}(w)\,\mathbb E[x^2]$. ReLU đặt một nửa giá trị về 0, nên $\mathbb E[x^2]$ chỉ còn một nửa phương sai của lớp trước. Muốn bù lại thì cần $\mathrm{Var}(w) = 2/n_\text{in}$: đó là **He init**. **Xavier** dùng $2/(n_\text{in} + n_\text{out})$, cân bằng giữa forward và backward, và hợp với tanh hoặc tuyến tính (không bỏ một nửa tín hiệu). Mình đo trên MLP 20 lớp rộng 256: với He, std ở lớp 20 vẫn khoảng 1; với Xavier + ReLU, std còn 0.001; với std 0.01, std còn $10^{-19}$.

**Ý phải có:** công thức phương sai · ReLU bỏ một nửa → hệ số 2 · Xavier cho tanh · một con số minh họa.
**Nếu bị hỏi tiếp:** *"Init mặc định của `nn.Linear` là gì?"* → `kaiming_uniform_(a=√5)`, phương sai $1/(3n_\text{in})$, nhỏ hơn He 6 lần.
**Xem lại:** bài 04 (mục 4.2–4.3).
</details>

**7. BatchNorm khác LayerNorm thế nào? Vì sao Transformer dùng LayerNorm (hoặc RMSNorm)?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** BatchNorm chuẩn hóa **mỗi đặc trưng theo các mẫu trong batch**: lúc train dùng thống kê của batch, lúc eval dùng running mean/var, nên train và eval hành xử khác nhau và kết quả phụ thuộc batch. LayerNorm chuẩn hóa **mỗi mẫu theo các đặc trưng của chính nó**: không phụ thuộc batch, train và eval như nhau. Transformer dùng LN vì các câu có độ dài khác nhau và có padding, batch thường nhỏ, khi sinh văn bản thì batch có thể chỉ là 1, và LN không tạo phụ thuộc giữa các mẫu. Mình đo trên `load_digits`: BN với batch 2 chỉ đạt 0.71 accuracy, LN đạt 0.96. RMSNorm bỏ bước trừ trung bình và tham số $\beta$, nên rẻ hơn mà chất lượng tương đương; LLaMA dùng nó.

**Ý phải có:** chiều chuẩn hóa · train/eval của BN · lý do cho Transformer · RMSNorm.
**Xem lại:** bài 05.
</details>

**8. Dropout hoạt động thế nào? "Inverted dropout" là gì?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Lúc train, mỗi neuron bị tắt ngẫu nhiên với xác suất $p$, nên mạng không thể dựa vào vài neuron riêng lẻ. Có thể xem như huấn luyện trung bình của rất nhiều mạng con dùng chung trọng số. Lúc eval thì giữ đủ mọi neuron. "Inverted" nghĩa là phép chia cho $1 - p$ được làm **lúc train**, để kỳ vọng của đầu ra không đổi; nhờ vậy lúc suy luận không cần sửa gì. Cạm bẫy: quên `model.eval()` thì dự đoán bị ngẫu nhiên. Trên project, cùng một mẫu, hai lần dự đoán chênh nhau tới 0.58 xác suất.

**Ý phải có:** tắt ngẫu nhiên lúc train · chia $1 - p$ để giữ kỳ vọng · eval giữ nguyên · cạm bẫy quên `eval`.
**Xem lại:** bài 06; bài 02 (mục 2.4b).
</details>

**9. Adam với `weight_decay` khác AdamW thế nào?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Với SGD, L2 regularization và weight decay là một. Với Adam thì khác. `Adam(weight_decay=λ)` cộng $\lambda\theta$ vào **gradient**, nên phần này cũng bị chia cho $\sqrt{\hat v}$: tham số có gradient lớn gần như không bị decay. AdamW **tách rời** weight decay: trừ thẳng $\eta\lambda\theta$ khỏi trọng số, không qua phép chia đó (Loshchilov & Hutter, ICLR 2019). Vì vậy AdamW là mặc định khi huấn luyện Transformer. Thêm một chi tiết thực tế: thường không decay bias và các tham số của LayerNorm/BatchNorm.

**Ý phải có:** L2 ≡ weight decay với SGD · Adam chia cho $\sqrt v$ · AdamW tách rời · không decay bias/norm.
**Xem lại:** bài 06.
</details>

**10. GPU không đủ bộ nhớ để huấn luyện. Bạn làm gì? Full fine-tune một model 7B cần bao nhiêu bộ nhớ?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Đầu tiên mình ước lượng. Huấn luyện bằng Adam tốn khoảng **16 byte mỗi tham số** (trọng số, gradient, $m$ và $v$ của Adam; với mixed precision thì có thêm master weights fp32), chưa kể activation. Model 7B cần $16 \times 7 \times 10^9 = 112$ GB, nên không vừa một GPU 16 GB. Các cách xử lý:
- **Mixed precision**: activation nhỏ hơn và nhanh hơn trên GPU có Tensor Core; fp16 cần loss scaling.
- **Gradient accumulation**: batch nhỏ nhưng cộng gradient nhiều bước. Nhớ chia loss, và nó không giúp được BatchNorm.
- **Gradient checkpointing**: tính lại activation thay vì lưu.
- Giảm batch hoặc độ dài chuỗi.
- Với LLM: dùng **LoRA/QLoRA** thay cho full fine-tune.

**Ý phải có:** 16 byte/tham số · phép tính 112 GB · ít nhất 3 kỹ thuật · lưu ý accumulation và BN.
**Xem lại:** bài 06 (mục 6.2–6.3).
</details>

## C. Kiến trúc

**11. Vì sao dùng CNN chứ không dùng MLP cho ảnh? Tính shape và số tham số của một lớp conv.**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** CNN có hai giả định phù hợp với ảnh: **kết nối cục bộ** (mỗi đầu ra chỉ nhìn một vùng nhỏ) và **chia sẻ trọng số** (cùng một filter dùng ở mọi vị trí). Nhờ vậy số tham số không phụ thuộc kích thước ảnh, và model gần như bất biến khi ảnh bị dịch. Một lớp Linear nối ảnh 224×224×3 với 64 neuron tốn 9.6 triệu tham số; conv 3×3 từ 3 lên 64 kênh chỉ tốn 1 792. Mình đo trên `load_digits`: khi dịch ảnh 1 pixel, MLP tụt từ 0.98 xuống 0.49 accuracy, còn CNN vẫn giữ 0.84. Công thức kích thước đầu ra là $\lfloor (n + 2p - k)/s \rfloor + 1$; số tham số là $C_\text{out}(C_\text{in}k^2 + 1)$. Ví dụ conv đầu tiên của ResNet ($k = 7$, $s = 2$, $p = 3$) biến ảnh 224 thành 112.

**Ý phải có:** cục bộ + chia sẻ trọng số · ví dụ so sánh số tham số · hai công thức · tính được một ví dụ ngay tại chỗ.
**Nếu bị hỏi tiếp:** *"Vì sao VGG dùng 3×3?"* → ba lớp 3×3 có receptive field 7×7 nhưng chỉ cần $27C^2$ tham số so với $49C^2$, và có thêm phi tuyến.
**Xem lại:** bài 07.
</details>

**12. Vì sao LSTM giảm được vanishing gradient? GRU khác LSTM thế nào?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** RNN thường viết lại toàn bộ trạng thái ở mỗi bước, nên gradient qua $T$ bước là tích của $T$ lần nhân với ma trận trọng số và đạo hàm tanh. LSTM có **cell state** được cập nhật bằng phép **cộng có cổng**: $c_t = f_t \odot c_{t-1} + i_t \odot g_t$. Đạo hàm theo đường đi qua cell chỉ là $f_t$, không có ma trận trọng số, nên khi forget gate gần 1 thì gradient đi qua nhiều bước gần như nguyên vẹn. Điều kiện $f \approx 1$ là quan trọng. Lúc khởi tạo $f \approx 0.5$, nên LSTM mặc định **vẫn** bị vanishing. Mình thử bài "nhớ token đầu tiên" với 40 bước: LSTM mặc định chỉ đoán ngẫu nhiên, còn đặt forget bias bằng 3 thì đạt 100%. GRU có 2 cổng, không có cell riêng, ít hơn 25% tham số, và chất lượng thường tương đương.

**Ý phải có:** cộng có cổng · đạo hàm qua cell bằng $f$ · điều kiện $f \approx 1$ và mẹo forget bias · so sánh với GRU.
**Xem lại:** bài 08.
</details>

**13. Attention giải quyết vấn đề gì của seq2seq? Query, key, value là gì? Vì sao chia cho $\sqrt d$?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Seq2seq cổ điển nén **cả câu vào** thành **một** vector cuối của encoder: đó là bottleneck, và câu càng dài thì càng tệ. Attention cho decoder, ở mỗi bước, nhìn lại **mọi** trạng thái của encoder và tự chọn nên nhìn vào đâu. Query là "câu hỏi" của decoder; key là "nhãn" của từng vị trí để so khớp; value là nội dung được lấy ra. Trọng số bằng softmax của điểm khớp $q \cdot k$, và đầu ra là trung bình có trọng số của các value: một kiểu **tra cứu mềm**. Mình thử bài copy chuỗi dài 20: không có attention thì accuracy cả chuỗi là 0%, có attention là 98%. Chia cho $\sqrt d$ vì nếu $q$ và $k$ có các thành phần var 1 thì $q \cdot k$ có var $d$. Không chia thì softmax gần như one-hot và gradient tới các vị trí khác gần bằng 0.

**Ý phải có:** bottleneck · q/k/v và tra cứu mềm · một ví dụ có số · lý do $\sqrt d$ bằng phương sai.
**Nếu bị hỏi tiếp:** *"Additive khác dot-product attention?"* → additive có thêm một MLP nhỏ, linh hoạt khi $d_q \ne d_k$; dot-product nhanh hơn trên GPU và có scale thì chất lượng tương đương.
**Xem lại:** bài 09.
</details>

**14. Greedy, beam search và sampling: khi nào dùng cái nào?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Greedy chọn token có xác suất cao nhất ở từng bước: nhanh, nhưng có thể bỏ lỡ chuỗi tốt nhất. Beam search giữ $k$ chuỗi tốt nhất theo tổng log-prob ở mỗi bước. Ví dụ: bước đầu, $A$ là 0.5 và $B$ là 0.4, nhưng sau $B$ model rất chắc chắn, nên chuỗi tốt nhất bắt đầu bằng $B$ (0.36 so với 0.20 của greedy). Beam cần **chuẩn hóa độ dài**, nếu không sẽ thiên về câu ngắn. Beam hợp với nhiệm vụ có đáp án gần như duy nhất (dịch máy, tóm tắt). Với chat và viết sáng tạo, LLM dùng **sampling** (temperature, top-p), vì beam cho văn bản lặp và nhạt.

**Ý phải có:** định nghĩa ba cách · ví dụ greedy sai · chuẩn hóa độ dài · beam cho dịch, sampling cho chat.
**Xem lại:** bài 09 (mục 9.3c); final C3.
</details>

**15. Transfer learning: feature extraction hay fine-tuning? Khi nào transfer không giúp gì?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Mình xét hai trục: **lượng dữ liệu** của bài toán đích và **độ gần** giữa miền nguồn và miền đích. Ít dữ liệu và miền gần thì dùng feature extraction: đóng băng phần thân, chỉ train head mới. Nhiều dữ liệu hơn thì fine-tune toàn bộ với lr nhỏ cho phần thân (nhỏ hơn head 10–100 lần), để không phá hỏng các đặc trưng đã học. Khi đóng băng một model có BatchNorm thì phải để BN ở chế độ eval. Transfer không tự động có lợi: mình thử pretrain trên chữ số 0–4 rồi chuyển sang 5–9, kết quả **tệ hơn** train từ đầu (0.63 so với 0.88). Khi miền gần (ảnh sạch → ảnh nhiễu, cùng các lớp), với 2 ảnh mỗi lớp, accuracy tăng từ 0.38 lên 0.63. ImageNet hiệu quả vì nó lớn và đa dạng. Vì vậy luôn phải so với baseline train từ đầu.

**Ý phải có:** hai trục quyết định · lr nhỏ cho phần thân · BN khi đóng băng · có thể bị negative transfer, nêu ví dụ.
**Xem lại:** bài 10.
</details>

## D. Kể về project (câu gần như chắc chắn bị hỏi)

**16. "Kể về một project deep learning bạn đã làm."** *(luyện bản 2 phút)*

<details><summary>Khung trả lời và bản mẫu</summary>

**Khung:** bối cảnh → mục tiêu đo được → cách làm → **kết quả có số** → điều học được hoặc điều bất ngờ → sẽ làm gì tiếp.

**Bản mẫu (project `qnet`).**
- **Bối cảnh:** mình có một model dự đoán câu trả lời của chatbot sẽ bị người dùng chấm 👎. Câu hỏi đặt ra là mạng neural có thắng gradient boosting trên dữ liệu bảng này không.
- **Cách làm:** mình viết hai mạng bằng PyTorch. Một MLP có embedding cho biến phân loại. Một model attention nhỏ: mỗi cột thành một token, gộp bằng attention pooling. Cả hai dùng chung một vòng huấn luyện tự viết, có early stopping và checkpoint.
- **Đánh giá:** để so sánh công bằng, mình dùng **cùng split** với HistGradientBoosting, và lặp lại trên 5 split, so sánh **ghép cặp**.
- **Kết quả:** HGB đạt AUC 0.802, model attention 0.792, MLP 0.785, logistic regression 0.756. Mạng neural không thắng, đúng với kết quả của Grinsztajn et al. (2022) cho dữ liệu bảng.
- **Bất ngờ:** trọng số attention **không** trùng với mức độ quan trọng thật. `retrieval_score` đứng thứ hai theo permutation importance nhưng chỉ thứ chín theo trọng số attention. Vì vậy mình không dùng attention làm lời giải thích cho model.
- **Nếu làm tiếp:** thử ensemble HGB với MLP, và thử encoding số dạng piecewise-linear cho MLP.

**Câu hỏi đào sâu hay gặp** (chuẩn bị sẵn):
- *"Vì sao lặp nhiều split?"* Trên một split, khoảng tin cậy của AUC rộng khoảng ±0.04, nên chênh 0.01 không nói lên gì.
- *"So sánh có công bằng không?"* Hai mạng chỉ thấy 80% dữ liệu train (20% dành cho early stopping). Mình đã thử huấn luyện lại MLP trên toàn bộ train, và khoảng cách thu hẹp nhưng không mất.
- *"Vì sao cây thắng trên dữ liệu bảng?"* Quan hệ dạng ngưỡng và tương tác (ví dụ `latency > 5 s`); có các cột vô dụng; dữ liệu không bất biến khi xoay đặc trưng.

Thay các chi tiết bằng **số của chính bạn** trong `REPORT.md`.
</details>

---

**Tự chấm:** mỗi câu được 1 điểm nếu nói đủ "Ý phải có" trong khoảng 1 phút (câu 16 trong khoảng 2 phút). Đạt **≥ 12/16** là sẵn sàng cho phần DL cơ bản của vòng phỏng vấn kỹ thuật.
