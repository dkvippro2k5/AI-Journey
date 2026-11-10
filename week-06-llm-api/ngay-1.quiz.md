# Quiz · Ngày 1 · LLM ở quy mô lớn, context window, huấn luyện LLM (bài 01–03)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> Làm trên giấy, được dùng máy tính cầm tay. **Không chạy code.** Đáp án ở cuối file.
> Quy ước: 1 GB = $10^9$ byte, 1 GiB = $2^{30}$ byte, 1 KiB = 1 024 byte.

## Phần 1 — Quick test (14 câu, câu 14 được 2đ, các câu khác 1đ)

1. Một lớp decoder kiểu LLaMA: $d = 2\,048$, 16 head query, 4 head KV, không bias. Phần attention ($W_Q, W_K, W_V, W_O$) có bao nhiêu tham số?
2. Model 7B tham số cần bao nhiêu bộ nhớ cho trọng số ở fp16? Ở int4?
3. Khi chạy, model 7B tốn khoảng bao nhiêu FLOP cho mỗi token (bỏ qua phần attention)?
4. Train model 1B tham số trên 100B token tốn khoảng bao nhiêu FLOP?
5. Theo Chinchilla, model 3B tham số nên được train trên khoảng bao nhiêu token? Vì sao các model 3B hiện nay thường được train trên nhiều hơn thế rất nhiều?
6. Mixtral 8x7B có 46.7B tham số tổng và 12.9B tham số kích hoạt. Con số nào quyết định bộ nhớ GPU, con số nào quyết định FLOP mỗi token?
7. KV-cache mỗi token ở fp16 của model có 24 lớp, 2 head KV, $d_\text{head} = 64$ là bao nhiêu byte?
8. Giải thích trong một câu: vì sao nhà cung cấp tính giá token đầu ra cao hơn token đầu vào?
9. Prompt 2 000 token, prefill 4 000 token/s, overhead mạng 0.3 s, decode 50 token/s. Tính TTFT và tổng thời gian để nhận 200 token trả lời.
10. ALiBi với $H = 4$ head dùng hệ số $m_h = 2^{-8h/H}$, $h = 1..4$. Viết 4 hệ số.
11. Mistral 7B dùng sliding window $W = 4\,096$ và có 32 lớp. Tầm nhìn lý thuyết của token cuối là bao nhiêu token? KV-cache của mỗi lớp cần giữ tối đa bao nhiêu token?
12. Theo "lost in the middle", nên đặt chỉ dẫn và thông tin quan trọng nhất ở đâu trong một prompt dài?
13. Khi SFT một model chat, vì sao nhãn của các token thuộc lượt system/user được đặt bằng −100, còn token `<|im_end|>` cuối câu trả lời thì **có** nhãn?
14. *(2đ)* Reward model chấm câu tốt hơn $r_w = 0.5$ và câu kém hơn $r_l = 1.5$. Theo Bradley–Terry, xác suất RM cho rằng $y_w$ tốt hơn là bao nhiêu? Loss của cặp này bằng bao nhiêu? Cho $e^{-1} = 0.368$.

## Phần 2 — Tự luận (4 câu, 30 phút)

**T1 · Ước lượng triển khai (4đ).** Bạn cần tự host một model có cấu hình: $d = 4\,096$, $L = 32$, 32 head query, 8 head KV ($d_\text{head} = 128$), $d_\text{ff} = 14\,336$ (SwiGLU), vocab $128\,256$, embedding và lm_head **không** dùng chung, không bias, 2 RMSNorm mỗi lớp + 1 RMSNorm cuối.
- (a) Tính số tham số và bộ nhớ trọng số fp16 (GB).
- (b) Tính KV-cache mỗi token (fp16), và tổng KV-cache cho 16 request đồng thời, mỗi request 4 096 token (GiB).
- (c) GPU có 40 GB. Bỏ qua overhead, tối đa bao nhiêu request 4 096 token chạy đồng thời được?
- (d) GPU có băng thông bộ nhớ 2 TB/s. Ước lượng trần tốc độ decode cho **một** người dùng. Vì sao phục vụ nhiều người cùng lúc lại không làm tốc độ mỗi người giảm tương ứng?

**T2 · Đọc số đo prefill/decode (4đ).** Số đo trên CPU của bạn với model 0.5B:

| Prompt | Prefill | Decode |
|---|---|---|
| 128 token | 0.80 s | 12.5 token/s |
| 1 024 token | 5.72 s | 10.7 token/s |
| 4 096 token | 25.76 s | 7.7 token/s |

- (a) Tính tốc độ prefill (token/s) ở cả ba độ dài. Nhận xét.
- (b) Vì sao decode chậm hơn prefill khoảng 14 lần? Vì sao decode chậm dần khi prompt dài ra?
- (c) Ước lượng tổng thời gian cho một prompt 2 000 token và câu trả lời 100 token (dùng tốc độ prefill trung bình khoảng 172 token/s, decode khoảng 10.5 token/s).
- (d) Đề xuất hai cách giảm TTFT mà không đổi phần cứng.

**T3 · Chẩn đoán chat template (3đ).** Một đồng nghiệp tự host model instruct (template ChatML) và gửi chuỗi `"Phân loại tin nhắn sau thành một nhãn: hủy gói GX30"` thẳng vào `model.generate`, không qua `apply_chat_template`, không có system prompt. Model trả lời lan man, nhiều khi bằng tiếng Anh, và không dừng cho tới `max_new_tokens`.
- (a) Nêu ít nhất hai nguyên nhân.
- (b) Viết lại cách gửi đúng (mô tả chuỗi token cuối cùng model nhận được).
- (c) Đồng nghiệp sau đó sửa bằng cách bỏ system prompt và để template tự lo. Điều gì xảy ra với họ model này, và vì sao nên tự đặt system prompt?

**T4 · DPO tính tay (4đ).** Log-prob (tổng trên các token câu trả lời):

| | Model đang train $\pi$ | Model tham chiếu $\pi_\text{ref}$ |
|---|---|---|
| $y_w$ (người chấm chọn) | −41 | −40 |
| $y_l$ (người chấm loại) | −35 | −37 |

Cho $\sigma(0.3) = 0.574$, $\sigma(1.5) = 0.818$.
- (a) Tính phần thưởng ngầm $\hat r = \beta\log\frac{\pi}{\pi_\text{ref}}$ của mỗi câu với $\beta = 0.1$, và loss DPO.
- (b) Diễn giải: so với model tham chiếu, model đang train đã đi sai hướng ở đâu? Gradient sẽ đẩy hai log-prob theo hướng nào?
- (c) Tính lại loss với $\beta = 0.5$. $\beta$ lớn hơn có nghĩa gì về việc bám sát model tham chiếu?
- (d) Nêu hai khác biệt về chi phí/vận hành giữa DPO và RLHF với PPO.

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Phần 1**

1. $d_\text{head} = 2048/16 = 128$, $d_{kv} = 4 \times 128 = 512$. Attention $= 2d^2 + 2d \cdot d_{kv} = 8\,388\,608 + 2\,097\,152 =$ **10 485 760**.
2. fp16: $7 \times 10^9 \times 2 =$ **14 GB**. int4: $7 \times 10^9 \times 0.5 =$ **3.5 GB** (chưa tính hệ số scale của lượng tử hóa).
3. $2N =$ **14 GFLOP** mỗi token.
4. $6ND = 6 \times 10^9 \times 10^{11} =$ **$6 \times 10^{20}$ FLOP**.
5. $20 \times 3\text{B} =$ **60B token**. Model hiện nay train lâu hơn nhiều (hàng nghìn token/tham số) vì chi phí **phục vụ** ($2N$ mỗi token, nhân với hàng nghìn tỉ token) lớn hơn chi phí train: model nhỏ mà mạnh thì rẻ khi chạy.
6. **Bộ nhớ:** 46.7B (mọi expert phải nằm trong bộ nhớ). **FLOP mỗi token:** 12.9B (mỗi token chỉ đi qua 2 expert).
7. $2 \times 24 \times 2 \times 64 \times 2 =$ **12 288 byte** (12 KiB).
8. Prefill xử lý cả prompt song song (compute-bound, mỗi lần đọc trọng số dùng cho nhiều token); decode sinh **từng** token, mỗi token phải đọc lại toàn bộ trọng số và KV-cache (memory-bound), nên mỗi token ra tốn tài nguyên GPU hơn nhiều.
9. TTFT $= 0.3 + 2000/4000 =$ **0.8 s**. Tổng $= 0.8 + 200/50 =$ **4.8 s**.
10. $m = 2^{-2}, 2^{-4}, 2^{-6}, 2^{-8}$ = **0.25, 0.0625, 0.0156, 0.0039**.
11. Tầm nhìn $L \cdot W = 32 \times 4096 =$ **131 072** token. Mỗi lớp chỉ cần giữ **4 096** token gần nhất (bộ đệm vòng).
12. Ở **đầu** hoặc **cuối** prompt (đường chữ U); tránh để ở giữa một context dài.
13. Loss trên system/user sẽ dạy model **sinh ra lượt của người dùng** (giả làm khách), không phải trả lời. Token `<|im_end|>` có nhãn để model học **khi nào dừng**; thiếu nó, model viết lan man tới hết `max_tokens`.
14. *(2đ)* $P = \sigma(0.5 - 1.5) = \sigma(-1) = 1/(1 + e^{1}) = 0.368/1.368 =$ **0.269** (1đ). Loss $= -\ln 0.269 =$ **1.313** (1đ). RM đang chấm ngược, nên loss lớn hơn $\ln 2 = 0.693$.

**T1 (4đ)**

(a) *(1đ)* Mỗi lớp: attention $2 \times 4096^2 + 2 \times 4096 \times 1024 = 41\,943\,040$; FFN $3 \times 4096 \times 14\,336 = 176\,160\,768$; norm 8 192, tổng **218 112 000**. 32 lớp: 6 979 584 000. Embedding + lm_head: $2 \times 128\,256 \times 4\,096 = 1\,050\,673\,152$. Norm cuối 4 096. Tổng **8 030 261 248** (khoảng 8.03B). fp16: **16.06 GB**.

(b) *(1đ)* $2 \times 32 \times 8 \times 128 \times 2 =$ **131 072 byte = 128 KiB** mỗi token. Mỗi request 4 096 token: $128 \text{ KiB} \times 4096 = 512$ MiB = 0.5 GiB. 16 request: **8 GiB**.

(c) *(1đ)* Còn trống $40 - 16.06 = 23.94$ GB $= 22.3$ GiB. Chia cho 0.5 GiB mỗi request: khoảng **44 request**. (Thực tế ít hơn vì activation, overhead của framework, phân mảnh bộ nhớ; PagedAttention ở Tuần 12 giảm phân mảnh.)

(d) *(1đ)* Mỗi token decode phải đọc 16.06 GB trọng số: $2 \times 10^{12} / 1.606 \times 10^{10} \approx$ **125 token/s**. Phục vụ nhiều người cùng lúc thì **một** lần đọc trọng số được dùng cho token của tất cả mọi người trong batch. Decode bị chặn bởi bộ nhớ, nên batch lớn hơn gần như không làm mỗi bước chậm đi, cho tới khi chạm giới hạn compute hoặc bộ nhớ KV.

**T2 (4đ)**

(a) *(1đ)* 128/0.80 = **160**; 1 024/5.72 = **179**; 4 096/25.76 = **159** token/s. Gần như **không đổi**: prefill compute-bound, thời gian tỉ lệ tuyến tính với số token. Ở 4 096 token hơi chậm lại vì phần attention ($\propto T^2$) bắt đầu đáng kể.

(b) *(1đ)* Prefill: một lần đọc trọng số (khoảng 2 GB fp32) phục vụ cả nghìn token, nên CPU bận tính. Decode: mỗi token cần một lần đọc trọng số, nên bị chặn bởi băng thông bộ nhớ (khoảng 0.36 FLOP/byte, thấp hơn nhiều so với mức CPU cần để dùng hết khả năng tính). Prompt dài thì KV-cache dài, mỗi bước decode phải đọc thêm KV và tính attention trên nhiều vị trí hơn, nên chậm dần (12.5 → 7.7 token/s).

(c) *(1đ)* Prefill $2000/172 \approx 11.6$ s; decode $100/10.5 \approx 9.5$ s; tổng khoảng **21 s**. TTFT khoảng 11.6 s là không chấp nhận được cho chat.

(d) *(1đ, mỗi ý 0.5đ)* Rút ngắn prompt (bớt few-shot, cắt lịch sử, chỉ đưa tài liệu liên quan); dùng lại KV-cache của phần đầu prompt cố định (prefix/prompt caching); dùng model nhỏ hơn hoặc lượng tử hóa; streaming để giảm độ trễ cảm nhận (không giảm TTFT thật nhưng người dùng thấy chữ ngay khi có).

**T3 (3đ)**

(a) *(1đ)* (i) Không có chat template: model instruct được train trên chuỗi ChatML, gặp văn bản thô thì hành xử như base model, **viết tiếp** văn bản. (ii) Không có `<|im_start|>assistant\n` nên model không biết đến lượt mình trả lời. (iii) Không có `<|im_end|>` làm điểm dừng, nên sinh tới hết `max_new_tokens`. (iv) Không có system prompt định rõ nhiệm vụ, ngôn ngữ, định dạng đầu ra.

(b) *(1đ)* Dùng `tok.apply_chat_template(messages, add_generation_prompt=True)` với system ("Bạn phân loại… chỉ trả lời đúng một nhãn trong: …") và user (tin nhắn). Chuỗi cuối cùng: `<|im_start|>system\n…<|im_end|>\n<|im_start|>user\nhủy gói GX30<|im_end|>\n<|im_start|>assistant\n`. Dừng khi gặp `<|im_end|>`.

(c) *(1đ)* Template của họ model này **tự chèn** một system prompt mặc định bằng tiếng Anh khi không có system message. Hành vi lúc đó không do bạn kiểm soát (dễ trả lời tiếng Anh, sai định dạng). Hãy luôn tự đặt system prompt: nhiệm vụ, ngôn ngữ, định dạng, giới hạn.

**T4 (4đ)**

(a) *(1.5đ)* $\hat r_w = 0.1 \times (-41 + 40) =$ **−0.1**; $\hat r_l = 0.1 \times (-35 + 37) =$ **+0.2**. Margin $= -0.1 - 0.2 = -0.3$. Loss $= -\log\sigma(-0.3) = -\log(1 - 0.574) = -\log 0.426 =$ **0.854**.

(b) *(1đ)* So với model tham chiếu, model đang train đã **giảm** xác suất câu được chọn (−41 so với −40) và **tăng** xác suất câu bị loại (−35 so với −37): sai cả hai hướng. Gradient đẩy $\log\pi(y_w)$ **lên** và $\log\pi(y_l)$ **xuống**, với hệ số $1 - \sigma(-0.3) = 0.574$ (lớn, vì model đang sai rõ).

(c) *(1đ)* Margin $= 0.5 \times (-3) = -1.5$; loss $= -\log(1 - 0.818) = -\log 0.182 =$ **1.70**. $\beta$ lớn làm mọi sai lệch so với model tham chiếu bị "phóng to" trong loss, tức là ràng buộc giữ model **bám sát** tham chiếu chặt hơn (giống hệ số phạt KL lớn).

(d) *(0.5đ)* DPO chỉ cần 2 model trong bộ nhớ (PPO cần 4: chính sách, tham chiếu, reward, value); DPO không phải sinh câu trả lời trong lúc train và không có vòng RL, nên rẻ và ổn định hơn, ít siêu tham số hơn. (Đổi lại, PPO học được từ câu trả lời mới do chính model sinh ra.)

</details>
