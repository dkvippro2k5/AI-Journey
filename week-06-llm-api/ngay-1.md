# Ngày 1 · LLM ở quy mô lớn, context window, huấn luyện LLM

> **Loại tài liệu: Kết hợp.** Công thức chỉ dành cho cơ chế lõi: đếm tham số, $6ND$, bộ nhớ KV, ALiBi, Bradley–Terry, DPO. Mọi phép tính khác ở mức ước lượng nhanh.
> **Hôm nay gồm 3 bài:**
> - **01** LLM ở quy mô lớn: tham số, FLOP, scaling laws, MoE, bảng các LLM
> - **02** Context window: vị trí, sliding window, prefill và decode, token tiếng Việt
> - **03** Huấn luyện LLM: pretrain → SFT → RLHF → Constitutional AI → DPO; chat template
>
> **Thời gian:** đọc ~2h50 · quiz 45' · bài tập ~2h
> **Model dùng để đo (tùy chọn):** một model instruct khoảng 0.5B tham số (`MODEL_ID`, đã tải về máy bạn), chạy fp32 trên CPU 16 thread.

---

## 01 · LLM ở quy mô lớn: tham số, FLOP, scaling laws, MoE, bảng các LLM

> ⏱ Tầng 1: 15' · Tầng 2: 45'
> **Cần biết trước:** đếm tham số một block Transformer (Tuần 5, bài 16); số byte theo dtype, 16 byte/tham số khi train với Adam (Tuần 4, bài 06); GQA (Tuần 5, bài 16).

> **📖 Từ mới**
> - **LLM (large language model):** mô hình ngôn ngữ decoder-only có từ khoảng 1 tỉ tham số trở lên, train trên hàng nghìn tỉ token.
> - **Tham số (N):** số trọng số của model. Quyết định **bộ nhớ** cần để chứa model.
> - **Token huấn luyện (D):** tổng số token model đã đọc khi pretrain.
> - **FLOP:** một phép tính số thực (cộng hoặc nhân). **FLOP/s:** tốc độ tính toán của phần cứng.
> - **Scaling law:** quy luật loss giảm theo N, D, và compute, có dạng hàm lũy thừa.
> - **Compute-optimal (Chinchilla):** với một ngân sách compute cố định, chia giữa N và D sao cho loss thấp nhất.
> - **MoE (mixture of experts):** mỗi lớp có nhiều FFN ("expert"); một **router** chọn vài expert cho mỗi token. **Tham số tổng** (phải nằm trong bộ nhớ) khác **tham số kích hoạt** (tính toán mỗi token).
> - **Dense:** model thường, mọi tham số đều tham gia vào mọi token.
> - **Open-weight / closed:** tải được trọng số về để tự host, hay chỉ gọi được qua API.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Một LLM giống một nhân viên đã đọc cả thư viện. Thuê người đọc nhiều hơn và nhớ nhiều hơn (model lớn hơn) thì trả lời tốt hơn, nhưng **mỗi câu trả lời** cũng tốn hơn: người đó phải "lật lại" toàn bộ trí nhớ cho từng chữ viết ra. Với LLM, chi phí mỗi token gần như tỉ lệ thuận với số tham số. Vì vậy câu hỏi của kỹ sư không phải "model nào giỏi nhất", mà là "model **nhỏ nhất** nào đủ giỏi cho việc này".

**Ví dụ có số (đo trên máy bạn).** Model 0.5B tham số, fp32, CPU 16 thread:

| Đại lượng | Giá trị |
|---|---|
| Tham số | 494 032 768 (27.6% nằm ở bảng embedding) |
| Bộ nhớ trọng số | 1.98 GB (fp32), 0.99 GB nếu fp16 |
| FLOP mỗi token khi chạy (≈ 2 × tham số không tính embedding) | 0.72 GFLOP |
| Tốc độ nhân ma trận đo được | 204 GFLOP/s |
| Prefill (đọc prompt) | khoảng 170 token/s, tức khoảng 120 GFLOP/s (60% tốc độ nhân ma trận) |
| Decode (sinh từng token) | khoảng 12 token/s |

Một model 8B lớn gấp 16 lần. Trên cùng CPU, decode sẽ chỉ còn dưới 1 token/s. Đó là lý do phục vụ LLM cần GPU, hoặc gọi API.

**Ba ý cốt lõi:**
1. **Biết cấu hình là biết kích thước.** Với decoder kiểu LLaMA, mỗi lớp có khoảng $2d^2 + 2d \cdot d_{kv} + 3d \cdot d_\text{ff}$ tham số. Bộ nhớ = tham số × số byte mỗi tham số.
2. **Compute:** train tốn khoảng $6ND$ FLOP, chạy tốn khoảng $2N$ FLOP mỗi token. Chinchilla: compute-optimal khi $D \approx 20N$. Model hiện nay cố ý train **lâu hơn nhiều** (Llama 3 8B: khoảng 1 900 token/tham số), vì chi phí **phục vụ** mới là chi phí lớn.
3. **MoE tách bộ nhớ khỏi compute:** Mixtral 8x7B phải chứa 46.7B tham số, nhưng mỗi token chỉ tính qua 12.9B.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 01.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Scaling laws** | Loss giảm theo lũy thừa của N, D, C: $L(N) \approx (N_c/N)^{\alpha_N}$, $\alpha_N \approx 0.076$ | Kaplan et al., *Scaling Laws for Neural Language Models*, arXiv 2020 |
| **Compute-optimal (Chinchilla)** | $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$; tối ưu khi N và D cùng tăng theo $C^{0.5}$, khoảng 20 token/tham số | Hoffmann et al., *Training Compute-Optimal LLMs*, NeurIPS 2022 |
| **$C \approx 6ND$** | Forward khoảng $2ND$ (mỗi tham số một phép nhân + một phép cộng cho mỗi token), backward khoảng gấp đôi forward | Kaplan et al. 2020, mục 2.1 |
| **In-context learning** | Model lớn làm được việc mới chỉ từ ví dụ trong prompt, không cập nhật trọng số | Brown et al., GPT-3, NeurIPS 2020 |
| **MoE** | Lớp gồm E expert và một router chọn top-k cho mỗi token | Shazeer et al., ICLR 2017; Fedus et al., *Switch Transformer*, JMLR 2022 |
| **Mixtral** | MoE 8 expert, top-2, 46.7B tham số tổng, 12.9B kích hoạt | Jiang et al., *Mixtral of Experts*, 2024 |
| **GQA** | Nhiều head query dùng chung một head K/V; giảm KV-cache | Ainslie et al., EMNLP 2023 |
| **SwiGLU** | FFN $W_2(\mathrm{SiLU}(W_1x) \odot W_3x)$: 3 ma trận thay vì 2 | Shazeer, *GLU Variants Improve Transformer*, 2020 |
| **RMSNorm** | Chuẩn hóa chỉ chia cho RMS, không trừ trung bình, chỉ có trọng số | Zhang & Sennrich, NeurIPS 2019 |
| **MFU** (model FLOPs utilization) | FLOP/s thực tế của model chia cho FLOP/s đỉnh của phần cứng | Chowdhery et al., PaLM, 2022 |

#### 01.2 🐢 Đi chậm từng bước: đếm tham số một lớp decoder (đọc trước công thức)

Lấy cấu hình công bố của **Llama-3-8B**: $d = 4096$, 32 lớp, 32 head query, 8 head KV, $d_\text{head} = 128$, $d_\text{ff} = 14\,336$, vocab $128\,256$, không bias, embedding và lm_head **không** dùng chung.

*Bước 1 · Attention.* Query có 32 head × 128 = 4 096 chiều; K và V chỉ có 8 × 128 = **1 024** chiều (GQA).

| Ma trận | Shape | Tham số |
|---|---|---|
| $W_Q$ | 4096 × 4096 | 16 777 216 |
| $W_K$ | 4096 × 1024 | 4 194 304 |
| $W_V$ | 4096 × 1024 | 4 194 304 |
| $W_O$ | 4096 × 4096 | 16 777 216 |
| **Cộng** | | **41 943 040** |

*Bước 2 · FFN SwiGLU.* Ba ma trận (gate, up, down), mỗi ma trận $4096 \times 14\,336$: $3 \times 58\,720\,256 =$ **176 160 768**.

*Bước 3 · Hai RMSNorm.* Mỗi cái có $d$ trọng số: **8 192**.

*Bước 4 · Một lớp.* $41\,943\,040 + 176\,160\,768 + 8\,192 =$ **218 112 000**. Nhân 32 lớp: **6.98B**.

*Bước 5 · Embedding và đầu ra.* $128\,256 \times 4\,096 = 525\,336\,576$, nhân 2 (không dùng chung) $= 1.05$B. Thêm RMSNorm cuối (4 096).

*Bước 6 · Tổng.* $6.98\text{B} + 1.05\text{B} =$ **8 030 261 248**, khớp đúng con số công bố.

**Nhận xét.** FFN chiếm 81% tham số mỗi lớp. GQA làm $W_K$, $W_V$ nhỏ đi 4 lần. Với model **nhỏ**, embedding chiếm tỉ lệ lớn: model 0.5B của bạn có 27.6% tham số nằm ở bảng embedding 151 936 × 896 (vocab lớn để phủ nhiều ngôn ngữ).

**Công thức (tóm tắt các bước trên).** Với $d_{kv} = h_{kv} \cdot d_\text{head}$:
$$N_\text{lớp} = \underbrace{2d^2 + 2d \cdot d_{kv}}_{\text{attention}} + \underbrace{3d \cdot d_\text{ff}}_{\text{SwiGLU}} + 2d, \qquad N = L \cdot N_\text{lớp} + V d\,(1 \text{ hoặc } 2) + d.$$
Quy tắc cũ $N \approx 12Ld^2$ (MHA, FFN $4d$ hai ma trận) cho $12 \times 32 \times 4096^2 = 6.44$B, hụt khoảng 8% so với phần không tính embedding (6.98B), vì SwiGLU với $d_\text{ff} = 3.5d$ có $10.5d^2$ thay vì $8d^2$.

#### 01.3 FLOP và bộ nhớ

- **Chạy (inference):** mỗi token đi qua mỗi tham số một lần, với một phép nhân và một phép cộng, nên khoảng $2N$ FLOP mỗi token. Phần attention thêm khoảng $4 \cdot L \cdot T \cdot d$ FLOP mỗi token ở độ dài $T$. Phần này đáng kể khi context dài (bài 02, mục 02.4).
- **Train:** $C \approx 6ND$.
- **Bộ nhớ trọng số:** $N \times$ byte/tham số: fp32 4, fp16/bf16 2, int8 1, int4 0.5. Train bằng Adam với mixed precision cần khoảng **16 byte/tham số** (Tuần 4, bài 06), chưa tính activation.

| Model (cấu hình công bố) | Tham số | Trọng số fp16 | KV-cache mỗi token, fp16 (bài 02) | FLOP mỗi token ($2N$) |
|---|---|---|---|---|
| Qwen2.5-0.5B (model đo của bạn) | 0.494B | 0.99 GB | 12 KiB | 1.0 GFLOP |
| Mistral-7B | 7.24B | 14.5 GB | 128 KiB | 14.5 GFLOP |
| Qwen2.5-7B | 7.62B | 15.2 GB | 56 KiB | 15.2 GFLOP |
| Llama-3-8B | 8.03B | 16.1 GB | 128 KiB | 16.1 GFLOP |
| Llama-3-70B | 70.6B | 141 GB | 320 KiB | 141 GFLOP |
| Mixtral 8x7B (MoE) | 46.7B tổng, 12.9B kích hoạt | 93 GB | 128 KiB | 25.8 GFLOP |

(Mọi số tham số trong bảng được tính bằng `param_count` của bạn và khớp với số công bố.)

**Ví dụ tính tay.** Llama-3-8B được train trên khoảng 15T token: $C = 6 \times 8.03 \times 10^9 \times 15 \times 10^{12} \approx 7.2 \times 10^{23}$ FLOP. Một GPU chạy hiệu dụng 400 TFLOP/s (khoảng 40% đỉnh bf16 của GPU trung tâm dữ liệu đời 2023) cần $7.2 \times 10^{23} / 4 \times 10^{14} = 1.8 \times 10^9$ GPU-giây, tức **khoảng 500 000 GPU-giờ**. Đây là lý do gần như không ai tự pretrain: AI engineer dùng model có sẵn, rồi fine-tune (Tuần 11).

#### 01.4 Scaling laws và Chinchilla

**Kaplan et al. (2020):** khi không bị giới hạn bởi hai yếu tố còn lại, loss giảm theo lũy thừa: $L(N) \propto N^{-0.076}$, $L(D) \propto D^{-0.095}$, $L(C) \propto C^{-0.050}$. Hệ quả: tăng N gấp 10 lần giảm loss khoảng $1 - 10^{-0.076} \approx 16\%$ (phần loss có thể giảm được).

**Hoffmann et al. (2022, "Chinchilla"):** khớp hàm $L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$ với $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$:
- $E$ là loss không thể giảm (entropy của văn bản).
- Hai số hạng còn lại là phần "thiếu model" và "thiếu dữ liệu".
- Tối ưu dưới ràng buộc $C = 6ND$ cho $N_\text{opt} \propto C^{0.5}$, $D_\text{opt} \propto C^{0.5}$, tức khoảng **20 token mỗi tham số**.

**Ví dụ tính tay: Chinchilla so với Gopher** (compute gần bằng nhau, khoảng $5 \times 10^{23}$ FLOP):

| Model | N | D | $A/N^\alpha$ | $B/D^\beta$ | $L$ dự đoán |
|---|---|---|---|---|---|
| Gopher | 280B | 300B | 0.052 | 0.251 | 1.993 |
| Chinchilla | 70B | 1.4T | 0.083 | 0.163 | **1.937** |

Model nhỏ hơn 4 lần nhưng đọc nhiều dữ liệu hơn 4.7 lần thì loss thấp hơn, và còn **rẻ hơn 4 lần khi phục vụ**. Ngược lại, với ngân sách $C = 5.76 \times 10^{23}$: $N = \sqrt{C/120} = 69$B, $D = 20N = 1.4$T.

**"Train quá mức" có chủ đích.** Chinchilla tối ưu chi phí **train**. Nếu model sẽ phục vụ hàng nghìn tỉ token, mỗi token tốn $2N$, thì nên train model **nhỏ hơn**, lâu hơn: Llama 3 8B dùng khoảng 1 900 token/tham số, Qwen2.5-0.5B khoảng 36 000 token/tham số (18T token). Với AI engineer, điều đó có nghĩa: model nhỏ đời mới thường mạnh hơn model lớn đời cũ.

#### 01.5 Mixture of Experts

Mỗi lớp MoE thay FFN bằng E expert. Với mỗi token $x$, router tính logit $g = W_r x$, chọn top-k, chuẩn hóa softmax **trên các expert được chọn**, rồi cộng có trọng số:
$$y = \sum_{i \in \text{top-}k(g)} \frac{e^{g_i}}{\sum_{j \in \text{top-}k} e^{g_j}}\, E_i(x).$$

**Ví dụ tính tay.** 4 expert, logit router $g = (2.0, 1.0, 0.5, -1.0)$, top-2: chọn expert 1, 2; trọng số $\mathrm{softmax}(2.0, 1.0) = (0.731, 0.269)$. Đầu ra $y = 0.731\,E_1(x) + 0.269\,E_2(x)$. Expert 3, 4 không tính gì cho token này.

**Tham số tổng và kích hoạt của Mixtral 8x7B** (cấu hình như Mistral-7B, 8 expert, top-2):
- Một expert: $3 \times 4096 \times 14\,336 = 176$M.
- Tổng: embedding 0.26B + 32 × (attention 42M + **8** expert + router) = **46.7B**.
- Kích hoạt: thay 8 expert bằng **2**: **12.9B**.

Mixtral cần bộ nhớ như một model 47B (93 GB fp16), nhưng tốn compute mỗi token như một model 13B.

**Vấn đề của MoE:**
- **Cân bằng tải:** router hay dồn token vào vài expert. Switch Transformer thêm loss phụ $\alpha \cdot E \sum_i f_i P_i$, với $f_i$ là tỉ lệ token được gửi tới expert $i$ và $P_i$ là xác suất router trung bình dành cho expert $i$.
- **Phục vụ:** cần chứa mọi expert; khi batch nhỏ, mỗi expert chỉ nhận vài token nên hiệu suất GPU thấp.

DeepSeek-V3 (2024) là MoE 671B tham số tổng, 37B kích hoạt.

#### 01.6 Bảng các LLM (roadmap 5.7, dời từ Tuần 5)

Bảng này ghi theo paper và model card **tại thời điểm viết**. Họ model thay đổi theo quý, nên hãy học **các trục so sánh**, đừng học thuộc tên.

| Họ model | Loại | Kích thước | Context | Điểm kiến trúc đáng nhớ | Trọng số |
|---|---|---|---|---|---|
| GPT-4o / dòng o (OpenAI) | Decoder, đa phương thức | Không công bố | 128K | Dòng "o" là model suy luận (sinh chuỗi suy nghĩ ẩn trước khi trả lời) | Đóng |
| Claude (Anthropic) | Decoder, đa phương thức | Không công bố | 200K | Constitutional AI (bài 03) | Đóng |
| Gemini (Google) | Đa phương thức từ đầu | Không công bố | tới 1M–2M | Context rất dài, context caching | Đóng |
| Llama 3 / 3.1 (Meta) | Decoder | 8B, 70B, 405B | 8K (3), 128K (3.1) | GQA 8 head KV, vocab 128K, RoPE $\theta = 500\,000$, 15T token | Mở (giấy phép riêng) |
| Mistral 7B | Decoder | 7.2B | 8K–32K | Sliding window 4 096, GQA (bài 02) | Mở (Apache 2.0) |
| Mixtral 8x7B | MoE | 46.7B / 12.9B | 32K | 8 expert, top-2 | Mở (Apache 2.0) |
| Qwen2.5 (Alibaba) | Decoder | 0.5B–72B | 32K–128K | Đa ngôn ngữ (có tiếng Việt), 18T token, bias ở QKV | Mở (phần lớn Apache 2.0) |
| Gemma 2 (Google) | Decoder | 2B, 9B, 27B | 8K | Xen kẽ lớp sliding window và lớp toàn cục, vocab 256K | Mở (giấy phép riêng) |
| DeepSeek-V3 | MoE | 671B / 37B | 128K | Multi-head latent attention (nén KV) | Mở |
| Model cho tiếng Việt | Decoder | vài tỉ tham số | — | PhoGPT (VinAI), Vistral (dựa trên Mistral), SeaLLM (Đông Nam Á) | Mở |

**Chọn model cho một hệ thống tự host (kiểu Viettel)**, theo thứ tự:
1. **Giấy phép** có cho dùng thương mại không.
2. **Chất lượng tiếng Việt** trên bộ đánh giá của **chính bạn** (Tuần 5, bài 19), không phải điểm benchmark tiếng Anh.
3. **Có vừa phần cứng không:** trọng số + KV-cache (bài 02) ≤ bộ nhớ GPU.
4. **Context** đủ cho bài toán.
5. **Độ trễ và chi phí** mỗi token.

#### 01.7 Đo trên máy bạn

| Đại lượng | Đo được |
|---|---|
| Nhân ma trận 2048 × 2048, fp32 | 204 GFLOP/s |
| Băng thông bộ nhớ (cộng một mảng 1 GiB, chỉ đọc) | khoảng 70 GB/s |
| Prefill, model 0.5B, prompt 128 → 4 096 token | 160–179 token/s, tức 114–128 GFLOP/s (khoảng **60%** tốc độ nhân ma trận) |
| Decode, prompt 128 token | 12.5 token/s, tức 80 ms/token |

Phép ước lượng ngược: mỗi token decode phải đọc 1.98 GB trọng số. Với 12.5 token/s, đó là khoảng 25 GB/s, khoảng một phần ba băng thông đọc tối đa đo được. Phần còn lại là overhead của các phép nhân nhỏ và của Python giữa các lớp. Decode bị chặn bởi **bộ nhớ** (và overhead), không phải bởi compute (bài 02, mục 02.2).

**Số đo trên laptop dao động.** Đo lại vào một lúc khác: nhân ma trận 490–620 GFLOP/s, prefill 200–260 token/s, decode 12–16 token/s. Xung nhịp đổi theo nhiệt độ và tải, nên chênh lệch ±30% giữa các lượt là bình thường. Mọi số trong bài này lấy từ **cùng một lượt đo** để so sánh được với nhau.

#### 01.8 Code chuẩn

```python
def param_count(vocab, d, n_layers, n_heads, n_kv_heads, d_ff, *, tied=True, qkv_bias=False):
    hd = d // n_heads
    kv = n_kv_heads * hd
    attn = d * d + 2 * d * kv + d * d + (d + 2 * kv if qkv_bias else 0)
    layer = attn + 3 * d * d_ff + 2 * d                                # SwiGLU 3 ma trận + 2 RMSNorm
    return vocab * d * (1 if tied else 2) + n_layers * layer + d

def train_flops(n_params, n_tokens):
    return 6 * n_params * n_tokens

def chinchilla_optimal(compute, tokens_per_param=20.0):
    n = math.sqrt(compute / (6 * tokens_per_param))                  # C = 6·N·(k·N)
    return n, tokens_per_param * n
```

#### 01.9 Cạm bẫy

1. **Nhầm tham số tổng và tham số kích hoạt của MoE.** Bộ nhớ theo tổng, compute theo kích hoạt.
2. **Quên embedding khi ước lượng model nhỏ.** Với model 0.5B, bỏ embedding là hụt 27%.
3. **Dùng $6ND$ cho inference.** Inference là $2N$ mỗi token (cộng phần attention khi context dài).
4. **So loss giữa các model khác tokenizer** (Tuần 5, bài 17): không so được.
5. **"Model lớn hơn luôn tốt hơn."** Model nhỏ đời mới, train lâu, thường thắng model lớn đời cũ, và rẻ hơn nhiều khi phục vụ.
6. **Chỉ tính trọng số khi chọn GPU.** Phải cộng KV-cache của mọi request đồng thời (bài 02).

**Sắc thái phỏng vấn:**
- *"Model 7B cần bao nhiêu VRAM?"* Khoảng 14 GB fp16 chỉ cho trọng số, khoảng 3.5–4 GB nếu int4. Cộng KV-cache theo độ dài và số request đồng thời, cộng overhead. Train full fine-tune với Adam khoảng 16 byte/tham số, tức hơn 100 GB.
- *"Chinchilla nói gì? Vì sao Llama 3 không theo?"* Khoảng 20 token/tham số là tối ưu chi phí train. Llama 3 train lâu hơn nhiều để có model nhỏ mà mạnh, vì chi phí phục vụ chiếm phần lớn.
- *"MoE có lợi gì, hại gì?"* Lợi: compute mỗi token như model nhỏ, năng lực như model lớn. Hại: bộ nhớ như model lớn, cân bằng tải, khó phục vụ với batch nhỏ.

#### 01.10 Đọc thêm

- J. Kaplan et al., *Scaling Laws for Neural Language Models*, 2020.
- J. Hoffmann et al., *Training Compute-Optimal Large Language Models*, NeurIPS 2022.
- A. Jiang et al., *Mixtral of Experts*, 2024.
- Meta, *The Llama 3 Herd of Models*, 2024 (mục kiến trúc và hạ tầng).
- A. Karpathy, *State of GPT* (Microsoft Build 2023), video.

---

## 02 · Context window: vị trí, sliding window, prefill và decode, token tiếng Việt

> ⏱ Tầng 1: 15' · Tầng 2: 45'
> **Cần biết trước:** sinusoidal PE, RoPE, causal mask (Tuần 5, bài 15); KV-cache, GQA, FlashAttention (Tuần 5, bài 16); BPE, byte-level BPE (Tuần 5, bài 14).

> **📖 Từ mới**
> - **Context window:** số token tối đa (prompt **cộng** câu trả lời) model xử lý trong một lần gọi.
> - **Prefill:** pha đọc toàn bộ prompt trong một lần forward song song, tạo KV-cache.
> - **Decode:** pha sinh từng token, mỗi bước một lần forward.
> - **TTFT (time to first token):** thời gian từ lúc gửi tới lúc nhận token đầu tiên, gần bằng thời gian prefill (cộng mạng và hàng đợi).
> - **TPOT (time per output token) / ITL (inter-token latency):** thời gian giữa hai token liên tiếp khi decode.
> - **Compute-bound / memory-bound:** bị chặn bởi tốc độ tính toán, hay bởi tốc độ đọc bộ nhớ.
> - **Relative position:** model dùng **khoảng cách** giữa hai token thay vì vị trí tuyệt đối.
> - **ALiBi:** trừ vào điểm attention một lượng tỉ lệ với khoảng cách.
> - **Sliding window attention:** mỗi token chỉ nhìn $W$ token gần nhất.
> - **Lost in the middle:** model dùng thông tin ở đầu và cuối context tốt hơn thông tin ở giữa.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Một nhân viên nhận hồ sơ khiếu nại dày 50 trang:
- **Đọc hồ sơ (prefill):** đọc lướt cả 50 trang một lượt, ghi chú lại (KV-cache). Việc này làm song song được, nhanh trên mỗi trang.
- **Viết thư trả lời (decode):** viết **từng chữ**. Trước mỗi chữ, anh ta phải lật lại sổ quy định dày cộp (trọng số model) và lướt ghi chú. Sổ quy định thì dày như nhau dù viết chữ nào, nên viết chậm hơn đọc rất nhiều.

Hồ sơ càng dày thì đọc càng lâu, ghi chú càng nhiều, và chi tiết nằm ở giữa hồ sơ dễ bị quên nhất.

**Ví dụ có số (đo trên máy bạn, model 0.5B):**

| Độ dài prompt | Prefill (≈ TTFT) | Decode | KV-cache (fp32) |
|---|---|---|---|
| 128 token | 0.8 s | 12.5 token/s | 3.8 MiB |
| 1 024 token | 5.7 s | 10.7 token/s | 24.8 MiB |
| 4 096 token | **25.8 s** | 7.7 token/s | 96.8 MiB |

Prompt dài gấp 32 lần thì TTFT tăng gấp 32 lần. Decode chậm dần vì mỗi bước phải đọc KV-cache dài hơn.

**Ba ý cốt lõi:**
1. **Token đầu vào và đầu ra có chi phí khác nhau.** Prefill song song và compute-bound; decode tuần tự và memory-bound. Vì vậy các nhà cung cấp tính giá token đầu ra cao hơn token đầu vào nhiều lần, và TTFT tăng theo độ dài prompt.
2. **Context window bị giới hạn bởi ba thứ:** cách mã hóa vị trí có ngoại suy được không (RoPE, ALiBi), bộ nhớ KV ($\propto T$), và chi phí attention ($\propto T^2$). Các kỹ thuật: GQA, sliding window, RoPE scaling, FlashAttention.
3. **Context dài không có nghĩa là dùng tốt context dài.** Đặt thông tin quan trọng ở đầu hoặc cuối prompt.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 02.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **TTFT, TPOT** | Hai chỉ số độ trễ chuẩn của hệ thống phục vụ LLM; độ trễ tổng ≈ TTFT + (số token ra − 1) × TPOT | Tài liệu vLLM, TensorRT-LLM; Kwon et al., *PagedAttention*, SOSP 2023 |
| **Arithmetic intensity / roofline** | FLOP trên mỗi byte đọc từ bộ nhớ; thấp hơn tỉ lệ (FLOP/s đỉnh ÷ băng thông) thì bị chặn bởi bộ nhớ | Williams, Waterman, Patterson, *Roofline*, CACM 2009 |
| **Relative position representations** | Embedding học được cho khoảng cách (đã cắt) giữa hai token, cộng vào key/value | Shaw, Uszkoreit, Vaswani, NAACL 2018 |
| **T5 relative bias** | Một số vô hướng học được cho mỗi "ô" khoảng cách (chia theo log), cộng vào điểm attention | Raffel et al., T5, JMLR 2020 |
| **ALiBi** | Điểm attention trừ $m_h \cdot (i - j)$; $m_h$ theo cấp số nhân, không có tham số học | Press, Smith, Lewis, *Train Short, Test Long*, ICLR 2022 |
| **Sliding window attention** | Mỗi token chỉ attend $W$ token trước; tầm nhìn tăng theo số lớp | Beltagy et al., Longformer, 2020; Jiang et al., Mistral 7B, 2023 |
| **Position interpolation, YaRN** | Mở rộng context của model RoPE bằng cách co vị trí lại rồi train tiếp ngắn | Chen et al., 2023; Peng et al., ICLR 2024 |
| **Lost in the middle** | Độ chính xác theo vị trí thông tin trong context có dạng chữ U | Liu et al., TACL 2024 |

#### 02.2 🐢 Đi chậm từng bước: vì sao decode chậm hơn prefill rất nhiều (đọc trước công thức)

Model 0.5B, fp32: trọng số 1.98 GB nằm trong RAM. CPU tính được khoảng 204 GFLOP/s, đọc RAM được tối đa khoảng 70 GB/s.

*Bước 1 · Một lần forward luôn phải đọc toàn bộ trọng số từ bộ nhớ.* Dù xử lý 1 token hay 512 token, mỗi ma trận vẫn phải đi từ RAM vào bộ đệm của CPU một lần (ít nhất).

*Bước 2 · Prefill 512 token.*
- Tính toán: $2 \times 358\text{M (không tính embedding)} \times 512 \approx 367$ GFLOP.
- Đọc bộ nhớ: khoảng 2 GB.
- Tỉ lệ: khoảng **180 FLOP mỗi byte**. Một lần đọc ma trận được dùng cho 512 token. CPU bận **tính**, nên prefill bị chặn bởi compute. Đo được: 2.95 s, tức khoảng 124 GFLOP/s.

*Bước 3 · Decode 1 token.*
- Tính toán: khoảng 0.72 GFLOP.
- Đọc bộ nhớ: vẫn khoảng 2 GB.
- Tỉ lệ: khoảng **0.36 FLOP mỗi byte**. CPU phần lớn thời gian **chờ dữ liệu**. Thời gian một bước ≈ 2 GB ÷ băng thông: trần lý thuyết $2 / 70 pprox 28$ ms. Đo được 80 ms mỗi token, vì còn overhead (bài 01, mục 01.7).

*Bước 4 · Đường ranh giới.* CPU cần khoảng $204 / 70 \approx 3$ FLOP mỗi byte mới dùng hết khả năng tính. Prefill vượt xa mức đó; decode thấp hơn hàng chục lần.

*Bước 5 · Hệ quả.*
- **Decode nhanh lên khi gom batch:** 32 người dùng decode cùng lúc thì một lần đọc trọng số phục vụ 32 token. Đây là ý tưởng của continuous batching (Tuần 12).
- **Lượng tử hóa làm decode nhanh lên:** int4 đọc ít byte hơn 8 lần so với fp32.
- **Context dài làm decode chậm đi:** mỗi bước còn phải đọc KV-cache. Đo được 12.5 → 7.7 token/s khi prompt tăng từ 128 lên 4 096 token.

**Công thức (tóm tắt các bước trên).**
$$t_\text{prefill} \approx \frac{2N \cdot T}{\text{FLOP/s hiệu dụng}}, \qquad t_\text{decode/token} \approx \frac{\text{byte trọng số} + \text{byte KV}}{\text{băng thông}}, \qquad \text{độ trễ} \approx \text{TTFT} + n_\text{out} \cdot \text{TPOT}.$$

**Ví dụ tính tay.** Llama-3-8B, fp16 (16 GB), trên một GPU băng thông 2 TB/s: trần decode $2 \times 10^{12} / 1.6 \times 10^{10} \approx$ **125 token/s** cho một người dùng. Trên CPU của bạn (fp32, 32 GB): trần $70/32 pprox 2.2$ token/s, thực tế khoảng 0.8 token/s nếu hiệu suất như model 0.5B (một phần ba), chưa kể RAM có đủ hay không.

#### 02.3 🐢 Đi chậm từng bước: ALiBi và sliding window (đọc trước công thức)

**ALiBi.** Không cộng vector vị trí vào embedding. Thay vào đó, trừ trực tiếp vào điểm attention một khoản tỉ lệ với khoảng cách.

*Bước 1 · Hệ số mỗi head.* Với $H = 8$ head: $m_h = 2^{-8h/H}$, tức $\tfrac12, \tfrac14, \dots, \tfrac1{256}$. Head có $m$ lớn chỉ nhìn gần; head có $m$ nhỏ nhìn xa.

*Bước 2 · Một ví dụ.* Token hiện tại có điểm $q \cdot k / \sqrt{d}$ với 4 token trước là $(2.0, 1.0, 1.0, 1.0)$. Bốn token đó cách token hiện tại lần lượt $10, 2, 1, 0$ vị trí.

| Head | Điểm sau khi trừ $m \times$ khoảng cách | Trọng số attention sau softmax |
|---|---|---|
| Không có ALiBi | (2.0, 1.0, 1.0, 1.0) | (0.475, 0.175, 0.175, 0.175) |
| $m = 1/2$ | (−3.0, 0.0, 0.5, 1.0) | (0.009, 0.185, 0.304, **0.502**) |
| $m = 1/16$ | (1.375, 0.875, 0.9375, 1.0) | (**0.340**, 0.206, 0.220, 0.234) |

Head $m = 1/2$ gần như bỏ qua token cách 10 vị trí dù nó có điểm cao nhất. Head $m = 1/16$ vẫn nhìn được nó.

*Bước 3 · Vì sao ngoại suy được.* Khoản trừ chỉ phụ thuộc khoảng cách, và tăng tuyến tính. Gặp khoảng cách dài hơn lúc train, model chỉ thấy "rất xa, trọng số rất nhỏ", chứ không gặp một vector vị trí lạ. Paper ALiBi: train ở 1 024 token, đánh giá ở 2 048 token vẫn tốt.

**Sliding window.**

*Bước 1.* Ở mỗi lớp, token $i$ chỉ attend token $i - W + 1 \dots i$. Mistral 7B dùng $W = 4\,096$.

*Bước 2 · Tầm nhìn tăng theo lớp.* Ở lớp 2, token $i$ nhìn token $i - 4\,095$, mà token đó ở lớp 1 đã nhìn thêm 4 096 token trước nó. Sau $L$ lớp, tầm nhìn lý thuyết là $L \cdot W = 32 \times 4\,096 \approx 131$K token.

*Bước 3 · Bộ nhớ.* KV-cache mỗi lớp chỉ cần giữ $W$ token gần nhất (bộ đệm vòng). Bộ nhớ không tăng sau $W$.

*Bước 4 · Cái giá.* Thông tin ở xa phải "chuyền" qua nhiều lớp, nên bị mờ đi. Gemma 2 xen kẽ lớp sliding window với lớp toàn cục để bù lại.

**Bảng so sánh các cách mã hóa vị trí:**

| Cách | Đưa vị trí vào đâu | Ngoại suy ra ngoài độ dài train | Dùng ở |
|---|---|---|---|
| Learned absolute | Cộng vào embedding | Không (không có vector cho vị trí mới) | BERT, GPT-2 |
| Sinusoidal | Cộng vào embedding | Kém | Transformer gốc |
| Relative (Shaw, T5) | Cộng vào điểm attention / key | Khá (khoảng cách bị cắt hoặc chia ô) | T5 |
| RoPE | Xoay q, k | Kém nếu không scaling; tốt với PI/YaRN + train tiếp | LLaMA, Qwen, Mistral |
| ALiBi | Trừ vào điểm attention | Tốt | BLOOM, MPT |

#### 02.4 Ngân sách context và bộ nhớ KV

- **Ngân sách:** token prompt + token trả lời ≤ context window. Đặt `max_tokens` = 1 000 trên model 8K thì prompt chỉ còn 7K.
- **Bộ nhớ KV mỗi request:** $2 \cdot L \cdot h_{kv} \cdot d_\text{head} \cdot T \cdot$ byte.

**Ví dụ tính tay.**
- Llama-3-8B, fp16: $2 \times 32 \times 8 \times 128 \times 2 = 131\,072$ byte = **128 KiB mỗi token**.
- Context 8K: 1 GiB mỗi request. Với 32 request đồng thời: **32 GiB**, nhiều gấp đôi trọng số (16 GB).
- Nếu không có GQA (32 head KV): 512 KiB mỗi token, gấp 4 lần.
- Model 0.5B của bạn: $2 \times 24 \times 2 \times 64 = 6\,144$ giá trị mỗi token, tức **24 576 byte fp32**. Đo được đúng **24 576 B/token**.

**Attention so với phần còn lại.** Mỗi token tốn khoảng $2N$ FLOP cho các ma trận, cộng khoảng $4LTd$ FLOP cho attention. Với Llama-3-8B:

| Độ dài $T$ | Attention mỗi token | $2N$ | Tỉ lệ |
|---|---|---|---|
| 2K | 1.1 GFLOP | 16.1 GFLOP | 7% |
| 8K | 4.3 GFLOP | 16.1 GFLOP | 27% |
| 128K | 68.7 GFLOP | 16.1 GFLOP | **430%** |

Ở context rất dài, attention chiếm phần lớn chi phí. Đó là lúc FlashAttention (giảm đọc ghi bộ nhớ), sliding window và các biến thể attention tuyến tính có ý nghĩa.

#### 02.5 Token tiếng Việt và đếm token (đo trên máy bạn)

20 câu tổng đài và bản dịch tiếng Anh tương ứng (172 âm tiết tiếng Việt, 166 từ tiếng Anh):

| Tokenizer | VI có dấu | VI bỏ dấu | EN | VI / EN | Token mỗi âm tiết VI | Ký tự mỗi token VI |
|---|---|---|---|---|---|---|
| `o200k_base` (tiktoken, đời mới) | 233 | 239 | 192 | **1.21** | 1.35 | 3.17 |
| `cl100k_base` (tiktoken, đời trước) | 395 | 279 | 193 | **2.05** | 2.30 | 1.87 |
| Tokenizer của model 0.5B (BPE, 151K) | 225 | 279 | 193 | 1.17 | 1.31 | 3.28 |

- **Tokenizer đời cũ phạt tiếng Việt gấp đôi:** `cl100k_base` tách "hủy gói cước" thành 9 token (`h`, `ủ`, `y`, ` g`, `ó`, `i`, ` c`, `ướ`, `c`); `o200k_base` chỉ cần 6.
- **Bỏ dấu không tiết kiệm** với tokenizer mới (233 → 239; với tokenizer của model 0.5B: 225 → 279), mà còn làm mất nghĩa (Tuần 5, bài 11).
- Tin nhắn tổng đài trung bình 8.5 âm tiết, khoảng **13 token** với `o200k_base`.

**Đếm token trước khi gửi:**
- Với OpenAI: `tiktoken`, cộng khoảng 3 token khung cho mỗi message (bài tập hôm nay).
- Anthropic và Gemini có endpoint đếm token riêng.
- Mỗi nhà cung cấp dùng tokenizer khác nhau, nên **cùng một prompt có số token khác nhau** giữa các nhà cung cấp. Khi tính ngân sách, để lề an toàn 5–10%.

#### 02.6 Lost in the middle (đo trên máy bạn)

Đặt một dòng "mã khuyến mãi tháng này là KM-xxxx" vào đầu, giữa hoặc cuối một log 60 tin nhắn (khoảng 1 225 token), rồi hỏi mã đó. 20 lần thử cho mỗi vị trí:

| Vị trí | Đúng |
|---|---|
| Đầu | 20/20 |
| Giữa | 20/20 |
| Cuối | 20/20 |

**Không thấy hiệu ứng** ở thí nghiệm này: model 0.5B tìm đúng mã ở cả ba vị trí. Lý do: context chỉ khoảng 1 200 token, và dòng cần tìm rất nổi bật ("Ghi chú nội bộ: mã khuyến mãi…" giữa các tin nhắn khách hàng), nên đây là một bài toán tìm kiếm chuỗi dễ. Hiệu ứng xuất hiện khi context dài hơn nhiều và thông tin phải được **suy luận** từ một trong nhiều tài liệu trông giống nhau. Bài học phụ: một thí nghiệm "không thấy gì" vẫn có giá trị nếu bạn hiểu vì sao; đừng khái quát từ một bài kiểm tra quá dễ.

Liu et al. (TACL 2024) đo trên các model lớn với 20–30 tài liệu và thấy đường chữ U: thông tin ở giữa có thể thấp hơn hơn 20 điểm phần trăm so với ở đầu. **Thực hành:** đặt chỉ dẫn và thông tin quan trọng nhất ở **đầu** hoặc **cuối** prompt. Ở RAG (Tuần 8): sắp tài liệu liên quan nhất ở hai đầu, và đừng nhồi quá nhiều tài liệu.

#### 02.7 Mở rộng context

- **Position interpolation (PI):** model RoPE train ở 4K, muốn chạy 16K, thì co vị trí lại 4 lần ($m \to m/4$), rồi train tiếp vài trăm bước.
- **NTK-aware scaling, YaRN:** co các tần số thấp nhiều hơn tần số cao, giữ được độ phân giải ở khoảng cách gần.
- **Ghi nhớ khi chọn model:** "context 128K" trên model card nghĩa là model **chạy được** ở 128K, không có nghĩa là nó **dùng tốt** cả 128K (mục 02.6).

#### 02.8 Code chuẩn

```python
def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per_value=2):
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_value

def latency_s(input_tokens, output_tokens, prefill_tps, decode_tps, overhead_s=0.0):
    ttft = overhead_s + input_tokens / prefill_tps
    return ttft, ttft + output_tokens / decode_tps

def decode_tps_upper_bound(n_params, bytes_per_param, mem_bandwidth_gbs):
    return mem_bandwidth_gbs * 1e9 / (n_params * bytes_per_param)    # batch 1, memory-bound

enc = tiktoken.get_encoding("o200k_base")
n = len(enc.encode(text, disallowed_special=()))                    # chuỗi "<|endoftext|>" do người dùng gõ = văn bản
```

#### 02.9 Cạm bẫy

1. **Đếm token bằng tokenizer của nhà cung cấp khác.** Lệch có thể tới 2 lần với tiếng Việt.
2. **Quên token đầu ra khi tính ngân sách context.** Prompt vừa khít context thì model không còn chỗ trả lời.
3. **Nhầm context window với giới hạn đầu ra.** Nhiều API giới hạn token **đầu ra** thấp hơn nhiều so với context window.
4. **Cho rằng TTFT không đổi.** TTFT tỉ lệ với độ dài prompt; prompt 50K token có thể mất vài giây ngay cả trên GPU mạnh.
5. **Chỉ tính bộ nhớ trọng số khi chọn GPU.** KV-cache của mọi request đồng thời có thể lớn hơn trọng số.
6. **Bỏ dấu để tiết kiệm token.** Với tokenizer mới, không tiết kiệm được mà còn mất nghĩa.
7. **Để `encode` ném lỗi khi người dùng gõ chuỗi giống special token.** Dùng `disallowed_special=()`, và đừng bao giờ để văn bản người dùng được hiểu là special token (bài 06).

**Sắc thái phỏng vấn:**
- *"Vì sao token đầu ra đắt hơn token đầu vào?"* Prefill song song, compute-bound; decode tuần tự, memory-bound. Mỗi token ra tốn một lần đọc toàn bộ trọng số và KV-cache.
- *"Làm sao giảm TTFT?"* Prompt ngắn hơn; prompt/prefix caching (bài 07); model nhỏ hơn; streaming để người dùng thấy chữ sớm (bài 08); đặt server gần người dùng.
- *"RoPE và ALiBi khác nhau thế nào?"* RoPE xoay q, k để tích vô hướng phụ thuộc khoảng cách; ALiBi trừ một lượng tuyến tính theo khoảng cách. ALiBi ngoại suy tốt hơn khi không train tiếp; RoPE phổ biến hơn và mở rộng bằng PI/YaRN.

#### 02.10 Đọc thêm

- O. Press, N. Smith, M. Lewis, *Train Short, Test Long: Attention with Linear Biases*, ICLR 2022.
- A. Jiang et al., *Mistral 7B*, 2023 (sliding window, rolling buffer cache).
- N. Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2024.
- W. Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*, SOSP 2023 (đọc trước cho Tuần 12).
- OpenAI Cookbook, *How to count tokens with tiktoken*.

---

## 03 · Huấn luyện LLM: pretrain → SFT → RLHF → Constitutional AI → DPO; chat template

> ⏱ Tầng 1: 15' · Tầng 2: 45'
> **Cần biết trước:** causal LM và cross-entropy, `ignore_index` (Tuần 5, bài 17; Tuần 4, phụ lục P1); sigmoid, log-likelihood (Tuần 2, bài 05); KL divergence (Tuần 2, bài 10); special token (Tuần 5, bài 14).

> **📖 Từ mới**
> - **Base model:** model chỉ mới pretrain; nó **viết tiếp** văn bản, không "trả lời".
> - **SFT (supervised fine-tuning) / instruction tuning:** train tiếp trên các cặp (chỉ dẫn, câu trả lời mẫu).
> - **Chat template:** cách xếp các lượt hội thoại thành một chuỗi token, có special token đánh dấu vai trò (ví dụ ChatML: `<|im_start|>user … <|im_end|>`).
> - **Dữ liệu ưu tiên (preference data):** bộ ba (prompt, câu trả lời tốt hơn $y_w$, câu trả lời kém hơn $y_l$).
> - **Reward model (RM):** model chấm điểm một câu trả lời, train từ dữ liệu ưu tiên.
> - **RLHF:** tối ưu model bằng học tăng cường để tăng điểm RM, có phạt KL để không đi quá xa model ban đầu.
> - **Reward hacking:** model tìm cách tăng điểm RM mà không thật sự tốt hơn (ví dụ trả lời dài hơn).
> - **Constitutional AI (CAI) / RLAIF:** dùng một bộ nguyên tắc và chính AI để phê bình, sửa, và tạo nhãn ưu tiên.
> - **DPO:** tối ưu thẳng trên dữ liệu ưu tiên bằng một hàm loss phân loại, không cần RM riêng hay vòng RL.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Đào tạo một nhân viên tổng đài:
1. **Pretrain:** người đó đọc cả thư viện. Biết rất nhiều, nhưng hỏi "hủy gói thế nào?" thì có thể viết tiếp thành một bài báo về gói cước.
2. **SFT:** cho học vài chục nghìn đoạn hội thoại mẫu: khách hỏi, nhân viên trả lời. Giờ người đó biết **định dạng** một cuộc trò chuyện và giọng điệu.
3. **RLHF / DPO:** quản lý đọc hai câu trả lời và chỉ ra câu nào tốt hơn. Người đó học được **thế nào là tốt hơn**, kể cả những điều khó viết thành câu mẫu: lịch sự, ngắn gọn, biết từ chối.
4. **Constitutional AI:** đưa cho người đó một bộ quy tắc ứng xử, và để người đó **tự** phê bình, tự sửa câu trả lời của mình theo quy tắc.

**Ví dụ có số.**
- **InstructGPT** (Ouyang et al. 2022): người chấm **thích câu trả lời của model 1.3B** đã qua SFT + RLHF hơn câu trả lời của GPT-3 175B chưa qua bước nào. Model nhỏ hơn hơn 100 lần, chỉ nhờ đúng định dạng và đúng ý người dùng.
- **Đo trên máy bạn:** model 0.5B instruct, 135 tin nhắn tổng đài, yêu cầu trả lời đúng một nhãn intent. Gửi qua **chat template** thì **96.3%** câu trả lời là đúng một nhãn hợp lệ; gửi dạng văn bản thô (không có template) thì **0%**: model viết tiếp lan man và không dừng.

**Ba ý cốt lõi:**
1. Model instruct đã học **một định dạng**. Gửi prompt phải dùng **đúng** chat template đó; SDK của nhà cung cấp làm việc này cho bạn, nhưng khi tự host thì bạn phải tự lo.
2. SFT dạy **định dạng và hành vi** (khoảng vài nghìn đến vài trăm nghìn ví dụ). Preference tuning (RLHF, DPO) dạy **so sánh tốt và kém hơn**.
3. DPO tối ưu cùng một mục tiêu với RLHF (tăng phần thưởng, phạt KL), nhưng bằng một loss phân loại trên cặp câu trả lời. Đơn giản và ổn định hơn PPO.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 03.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Instruction tuning** | Fine-tune trên nhiều tác vụ diễn đạt bằng chỉ dẫn; model làm tốt hơn tác vụ mới ở chế độ zero-shot | Wei et al., FLAN, ICLR 2022 |
| **RLHF** | Học từ so sánh của người, rồi tối ưu chính sách bằng RL | Christiano et al., NeurIPS 2017; Ouyang et al., InstructGPT, NeurIPS 2022 |
| **Bradley–Terry** | $P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$ | Bradley & Terry, Biometrika 1952 |
| **PPO** | Thuật toán policy gradient có cắt tỉ lệ cập nhật | Schulman et al., 2017 |
| **Constitutional AI** | SL: tự phê bình và sửa theo nguyên tắc; RL: dùng nhãn ưu tiên do AI tạo (RLAIF) | Bai et al., *Constitutional AI*, Anthropic 2022 |
| **DPO** | Loss: $-\log\sigma\big(\beta[\log\frac{\pi(y_w)}{\pi_\text{ref}(y_w)} - \log\frac{\pi(y_l)}{\pi_\text{ref}(y_l)}]\big)$ | Rafailov et al., NeurIPS 2023 |
| **LIMA** | 1 000 ví dụ SFT chọn lọc kỹ đã cho model chat khá tốt: chất lượng quan trọng hơn số lượng | Zhou et al., NeurIPS 2023 |
| **Chat template** | Mẫu Jinja trong `tokenizer_config.json` xếp message thành chuỗi | Tài liệu Hugging Face, *Chat templates* |
| **RL với phần thưởng kiểm chứng được** | RL trên bài toán có đáp án kiểm tra được (toán, code); tạo model suy luận | DeepSeek-AI, *DeepSeek-R1*, 2025 (thuật toán GRPO) |

#### 03.2 🐢 Đi chậm từng bước: chat template và SFT chỉ tính loss trên câu trả lời

*Bước 1 · Hội thoại thành chuỗi.* Model 0.5B của bạn dùng định dạng ChatML. Với system "Bạn là trợ lý." và user "hủy gói", `apply_chat_template` cho đúng chuỗi sau (đã kiểm tra bằng code):
```
<|im_start|>system\nBạn là trợ lý.<|im_end|>\n<|im_start|>user\nhủy gói<|im_end|>\n<|im_start|>assistant\n
```
Dòng cuối `<|im_start|>assistant\n` (generation prompt) báo cho model "đến lượt bạn".

*Bước 2 · Một điều bất ngờ.* Nếu **không** có message system, template của model này **tự chèn** một system prompt mặc định bằng tiếng Anh ("You are Qwen, created by Alibaba Cloud…"). Muốn kiểm soát hành vi thì luôn gửi system prompt của bạn.

*Bước 3 · Ví dụ SFT.* Thêm câu trả lời "Dạ, soạn HUY gửi 1234 ạ.<|im_end|>". Chuỗi token gồm phần system và user (khoảng 20 token) và phần assistant (khoảng 12 token).

*Bước 4 · Nhãn.* Loss chỉ tính trên phần assistant. Nhãn của mọi token system và user đặt bằng −100 (`ignore_index`):

| Token | `<|im_start|>` | system | … | user | hủy | gói | … | assistant | Dạ | , | … | `<|im_end|>` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Nhãn | −100 | −100 | −100 | −100 | −100 | −100 | −100 | −100 | Dạ | , | … | `<|im_end|>` |

*Bước 5 · Vì sao.* Nếu tính loss cả trên lượt user, model học cách **giả làm khách hàng**: sinh tiếp câu hỏi thay vì trả lời. Token `<|im_end|>` **có** nhãn: model phải học khi nào thì **dừng**. Model không biết dừng sẽ viết lan man cho tới hết `max_tokens`.

#### 03.3 🐢 Đi chậm từng bước: reward model, PPO và DPO (tính tay)

**(a) Reward model.** Bradley–Terry: xác suất người chấm chọn $y_w$ là $\sigma(r_w - r_l)$. Loss của RM là $-\log\sigma(r_w - r_l)$.
- Ví dụ: $r_w = 1.2$, $r_l = 0.4$. $\sigma(0.8) = 0.690$, loss $= 0.371$.
- Nếu RM chấm ngược ($r_w < r_l$), loss lớn hơn $\ln 2 = 0.693$.

**(b) PPO với phạt KL.** Mục tiêu: $\max_\pi \mathbb{E}[r(x, y)] - \beta\,\mathrm{KL}(\pi \,\|\, \pi_\text{ref})$. Với một mẫu, phần thưởng thực nhận là $r - \beta(\log\pi(y) - \log\pi_\text{ref}(y))$.
- Ví dụ: $r = 2.0$; $\log\pi(y) = -12.0$; $\log\pi_\text{ref}(y) = -15.0$; $\beta = 0.1$. Phạt $= 0.1 \times 3.0 = 0.3$, phần thưởng thực nhận $= 1.7$.
- Ý nghĩa: model mới gán xác suất cho $y$ cao gấp $e^3 \approx 20$ lần model gốc, và bị trừ điểm vì đi quá xa. Không có phạt KL, model sẽ tìm ra những câu "lạ" khai thác điểm yếu của RM (reward hacking).

**(c) DPO.** Rafailov et al. chứng minh rằng mục tiêu ở (b) có nghiệm dạng đóng, trong đó phần thưởng **ngầm** là $\hat r(y) = \beta\log\frac{\pi(y)}{\pi_\text{ref}(y)}$. Thay vào Bradley–Terry ở (a) được loss chỉ cần log-prob của hai model.

*Bước 1.* Log-prob (tổng trên các token của câu trả lời):

| | Model đang train $\pi$ | Model tham chiếu $\pi_\text{ref}$ (bản SFT, đóng băng) | Chênh |
|---|---|---|---|
| $y_w$ (tốt hơn) | −20 | −22 | **+2** |
| $y_l$ (kém hơn) | −25 | −24 | **−1** |

*Bước 2.* Hiệu hai chênh: $2 - (-1) = 3$. Nhân $\beta = 0.1$: $0.3$.

*Bước 3.* Loss $= -\log\sigma(0.3) = -\log 0.574 =$ **0.554**.

*Bước 4 · Gradient làm gì.* Loss giảm khi khoảng cách $\beta \cdot 3$ tăng: đẩy $\log\pi(y_w)$ lên, $\log\pi(y_l)$ xuống, **so với model tham chiếu**. Gradient có hệ số $1 - \sigma(0.3) = 0.426$: cặp nào model đã phân biệt tốt thì hệ số nhỏ, học ít đi.

*Bước 5 · $\beta$ là gì.* $\beta$ nhỏ: model được phép đi xa model gốc hơn (giống phạt KL yếu). $\beta$ lớn: bám sát model gốc. Thường dùng $\beta = 0.1$.

**Công thức (tóm tắt).**
$$\mathcal L_\text{RM} = -\log\sigma(r_w - r_l), \qquad \mathcal L_\text{DPO} = -\log\sigma\Big(\beta\big[\log\tfrac{\pi(y_w)}{\pi_\text{ref}(y_w)} - \log\tfrac{\pi(y_l)}{\pi_\text{ref}(y_l)}\big]\Big).$$

**So sánh RLHF (PPO) và DPO:**

| | RLHF với PPO | DPO |
|---|---|---|
| Số model trong bộ nhớ khi train | 4 (chính sách, tham chiếu, reward, value) | 2 (chính sách, tham chiếu) |
| Sinh câu trả lời trong lúc train | Có (on-policy) | Không (dữ liệu cố định) |
| Độ ổn định, số siêu tham số | Khó chỉnh | Như fine-tune thường |
| Điểm mạnh | Học được từ câu trả lời mới do chính model sinh ra | Đơn giản, rẻ; thực hành ở Tuần 11 |

#### 03.4 Constitutional AI

Bai et al. (2022) có hai pha:
1. **SL-CAI:** model sinh câu trả lời cho các prompt có hại → tự phê bình theo một nguyên tắc ("câu trả lời có gây hại không?") → tự viết lại → fine-tune trên các bản đã sửa.
2. **RL-CAI (RLAIF):** model so sánh cặp câu trả lời theo nguyên tắc để tạo nhãn ưu tiên, train RM từ nhãn đó, rồi chạy RL.

**Lợi ích:** giảm việc người phải đọc và gán nhãn nội dung độc hại; các nguyên tắc được viết ra rõ ràng, kiểm tra được. **Đánh đổi:** cân bằng giữa vô hại và hữu ích. Model quá thận trọng sẽ từ chối cả câu hỏi bình thường.

#### 03.5 Đo trên máy bạn: chat template

Model 0.5B instruct, 135 tin nhắn (15 mỗi intent), yêu cầu trả lời đúng một nhãn trong 9 nhãn, greedy, tối đa 10 token:

| Cách gửi | Trả lời đúng một nhãn hợp lệ | Có chứa một nhãn | Accuracy |
|---|---|---|---|
| Văn bản thô, không chat template | **0%** | 100% | 0.259 |
| Qua chat template (zero-shot) | **96.3%** | 97.0% | 0.230 |

- **Văn bản thô:** model hành xử như base model, viết tiếp: `loi_mang\n\nTrả lời: …`, `tra_cuoc\n\nTrả lời đúng: …`, cho tới hết 10 token. Vì câu trả lời thường **bắt đầu** bằng một nhãn, regex vẫn nhặt được nhãn, nên accuracy (0.259) còn cao hơn zero-shot. Nhưng không câu nào dùng được trực tiếp trong một hệ thống tự động, và mọi câu đều chạy tới giới hạn token (tốn tiền, tốn thời gian).
- **Qua template:** đúng định dạng gần như luôn luôn, nhưng **nội dung** kém: model 0.5B đoán `dang_ky_goi` cho gần hết tin nhắn. Định dạng do chat template và SFT quyết định; độ chính xác do năng lực model và prompt quyết định (bài 04).

**Chi phí của template.** Mỗi message tốn thêm 4–5 token khung (`<|im_start|>`, vai trò, xuống dòng, `<|im_end|>`, xuống dòng). Các API tính cả số token này. Bài tập hôm nay dùng công thức "3 token mỗi message + 3 token mở đầu câu trả lời" của OpenAI cookbook.

#### 03.6 Điều này có nghĩa gì với AI engineer

- **Base hay instruct?** Gọi API hay làm chatbot: luôn dùng instruct/chat. Base chỉ dùng khi tự fine-tune theo định dạng riêng (Tuần 11).
- **System prompt có trọng lượng** vì model được train để ưu tiên nó (instruction hierarchy, bài 06).
- **Các thói quen sinh ra từ cách train:**
  - Ảo giác: pretrain thưởng cho việc viết tiếp **trôi chảy**, không thưởng cho việc nói đúng.
  - Nịnh người dùng (sycophancy): người chấm RLHF hay thích câu trả lời đồng tình với mình.
  - Câu trả lời dài: RM hay thiên vị câu dài.
  - Mốc kiến thức (knowledge cutoff): model không biết gì sau thời điểm dữ liệu pretrain.
- **Model suy luận** (RL với phần thưởng kiểm chứng được) sinh rất nhiều token "suy nghĩ" trước câu trả lời. Chúng mạnh ở toán và code, nhưng chậm và đắt. Đừng dùng cho phân loại intent.

#### 03.7 Code chuẩn

```python
# SFT: nhãn -100 cho mọi token không thuộc câu trả lời
prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True)      # system + user + "assistant\n"
answer_ids = tok.encode(answer + "<|im_end|>")
input_ids = prompt_ids + answer_ids
labels = [-100] * len(prompt_ids) + answer_ids
# (dịch một vị trí khi tính loss, như causal LM của Tuần 5)

# DPO loss cho một batch, đầu vào là tổng log-prob trên token câu trả lời
def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -torch.nn.functional.logsigmoid(margin).mean()
```

#### 03.8 Cạm bẫy

1. **Gửi prompt thô cho model instruct tự host,** hoặc dùng sai template của họ model khác: đầu ra lan man hoặc không dừng.
2. **Thêm BOS hai lần** (một lần trong template, một lần do tokenizer tự thêm).
3. **SFT tính loss trên cả lượt user:** model học giả làm khách hàng.
4. **Quên token kết thúc trong nhãn SFT:** model không biết dừng.
5. **DPO với model tham chiếu khác điểm khởi đầu của model đang train:** phần thưởng ngầm sai ngay từ đầu.
6. **Đọc "điểm RM tăng" như "model tốt lên"** mà không kiểm tra bằng người hoặc bằng bộ đánh giá độc lập (reward hacking).

**Sắc thái phỏng vấn:**
- *"RLHF gồm những bước nào?"* SFT → thu dữ liệu so sánh → train RM (Bradley–Terry) → PPO tối đa hóa điểm RM với phạt KL so với model SFT.
- *"DPO khác RLHF thế nào?"* Cùng mục tiêu (phần thưởng trừ KL), nhưng DPO dùng nghiệm dạng đóng để biến thành loss phân loại trên cặp câu trả lời. Không cần RM, không sinh mẫu khi train, chỉ 2 model trong bộ nhớ.
- *"Constitutional AI là gì?"* Dùng một bộ nguyên tắc viết sẵn để model tự phê bình, sửa câu trả lời (SL), và tạo nhãn ưu tiên (RLAIF); giảm nhãn của người cho nội dung độc hại.

#### 03.9 Đọc thêm

- L. Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback*, NeurIPS 2022.
- Y. Bai et al., *Constitutional AI: Harmlessness from AI Feedback*, 2022.
- R. Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023.
- C. Zhou et al., *LIMA: Less Is More for Alignment*, NeurIPS 2023.
- Hugging Face, *Chat Templates* (tài liệu `transformers`).

---

## Bài tập Ngày 1 · `gateway/tokens.py` (~2h)

Lệnh test: `python -m pytest -q tests/test_tokens.py` (7 test).

| Hàm | Yêu cầu |
|---|---|
| `count_tokens(text, encoding="o200k_base")` | Số token theo `tiktoken`. Chuỗi trông như special token (`"<|endoftext|>"`) phải được đếm như văn bản thường, không ném lỗi (`disallowed_special=()`) |
| `count_chat_tokens(messages, encoding=..., *, per_message=3, per_name=1, reply_priming=3)` | `reply_priming` + với mỗi message: `per_message` + token của **mọi** giá trị (role, content, name); có khóa `name` thì cộng thêm `per_name` |
| `render_chatml(messages, add_generation_prompt=True)` | Mỗi message thành `<|im_start|>{role}\n{content}<|im_end|>\n`; nếu `add_generation_prompt` thì thêm `<|im_start|>assistant\n` |
| `Price(input, output, cached_input=None)` và `cost_usd(input_tokens, output_tokens, price, cached_tokens=0)` | Giá theo USD / 1 triệu token. Token trúng cache tính theo `cached_input` (không có thì theo `input`). `cached_tokens` ngoài $[0, \text{input\_tokens}]$ → `ValueError` |
| `param_count(vocab, d, n_layers, n_heads, n_kv_heads, d_ff, *, tied=True, qkv_bias=False)` | Công thức mục 01.2; `qkv_bias` thêm $d + 2d_{kv}$ mỗi lớp |
| `kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per_value=2)` | Mục 02.4 |
| `train_flops(n_params, n_tokens)`, `chinchilla_optimal(compute, tokens_per_param=20)` | $6ND$; trả về $(N, D)$ với $D = kN$, $6ND = C$ |
| `latency_s(input_tokens, output_tokens, prefill_tps, decode_tps, overhead_s=0)` | Trả về `(ttft, total)` theo mục 02.2 |
| `decode_tps_upper_bound(n_params, bytes_per_param, mem_bandwidth_gbs)` | Băng thông (GB/s × $10^9$) chia cho số byte trọng số |

**Gợi ý:**
- `test_param_count_matches_published_models` dùng cấu hình thật của hai model mở. Nếu lệch, hãy so từng thành phần với bảng ở mục 01.2.
- `get_encoding` nên được cache (`functools.lru_cache`): tạo encoding mất khoảng 0.1 s.
- Đếm token **không** gọi mạng (sau lần tải encoding đầu tiên). Đừng gọi API chỉ để đếm token.
