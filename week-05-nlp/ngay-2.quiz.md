# Quiz · Ngày 2 · Tokenization và Transformer (bài 14–16)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> Làm trên giấy, được dùng máy tính cầm tay. **Không chạy code.** Đáp án ở cuối file.

## Phần 1 — Quick test (14 câu, câu 14 được 2đ, các câu khác 1đ)

1. Corpus: "ban" ×4, "nan" ×2. BPE (có `</w>`, hòa thì chọn cặp nhỏ nhất theo thứ tự chuỗi) ghép cặp nào **đầu tiên**? Vì sao?
2. Với hai merge đã học `(a, n)` rồi `(an, </w>)`, tokenize hai từ "ban" và "nana".
3. WordPiece với vocab `{goi, go, dang, ##ky, ##n, ##i, [UNK]}`: tokenize "goin" và "dangky".
4. Vì sao byte-level BPE không bao giờ cần token `<unk>`?
5. Từ "đổi" chiếm bao nhiêu byte UTF-8? (Gợi ý: `đ` là U+0111, `ổ` là U+1ED5.)
6. `MultiHeadAttention(d_model=512, n_heads=8)` có bias: tính $d_\text{head}$ và tổng số tham số.
7. Batch 4 câu, mỗi câu 10 token, 8 head. Ma trận trọng số attention có shape gì?
8. Causal mask $4 \times 4$ có bao nhiêu ô `True`?
9. Viết vector sinusoidal PE ở vị trí 0 với $d = 4$. Vì sao vị trí 0 không phải là vector 0?
10. Với RoPE, so sánh $q_5^\top k_2$ và $q_8^\top k_5$ (cùng vector $q$, $k$ trước khi xoay).
11. Một lớp Transformer với $d = 512$, FFN $4d$, có bias và 2 LayerNorm: bao nhiêu tham số?
12. Decoder có $L = 24$, $d = 2\,048$ (đủ head KV), fp16, ngữ cảnh 2 048 token. KV-cache của **một** câu là bao nhiêu GiB? Của batch 8 câu?
13. Router intent cho tổng đài nên dùng encoder hai chiều hay decoder causal? Cho 2 lý do.
14. *(2đ)* Đoạn code multi-head attention sau có **3 lỗi**. Chỉ ra và sửa.
    ```python
    def forward(self, x, mask):                      # mask: True = được nhìn
        B, T, d = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, self.d_head).transpose(1, 2)
        w = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d), dim=-1)
        w = w * mask
        out = (w @ v).transpose(1, 2).view(B, T, d)
        return self.out_proj(out)
    ```

## Phần 2 — Tự luận (4 câu, 30 phút)

**T1 · BPE bằng tay (4đ).** Corpus: "nap" ×4, "nam" ×2, "lap" ×3. Mỗi từ được tách ký tự và thêm `</w>`.
- (a) Đếm mọi cặp kề nhau (có nhân tần suất). Học **3** merge đầu tiên (hòa thì chọn cặp nhỏ nhất theo thứ tự chuỗi; `"</w>"` đứng trước mọi chữ cái thường).
- (b) Tokenize "nap", "map", "lam" bằng các merge đã học.
- (c) Vì sao "map" có token chung với "nap", còn "lam" lại bị tách hết thành ký tự?

**T2 · Attention bằng tay (4đ).** Ba token, $d_\text{head} = 2$, một head:
$$Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix},\quad K = \begin{pmatrix} 1 & 1 \\ 2 & 0 \\ 0 & 0 \end{pmatrix},\quad V = \begin{pmatrix} 1 & 0 \\ 0 & 2 \\ 3 & 1 \end{pmatrix}.$$
- (a) Tính ma trận điểm $QK^\top/\sqrt 2$.
- (b) Với **causal mask**, tính trọng số và đầu ra của token 2 và token 3.
- (c) Bây giờ là encoder (không causal), nhưng token 2 là **token đệm**. Tính trọng số và đầu ra của token 3. Vì sao đầu ra của token 1 và token 2 giống hệt nhau?

**T3 · Chọn kiến trúc và ước lượng (3đ).** Một tổng đài cần router intent chạy trên **CPU**, độ trễ p95 dưới 50 ms, 200 tin nhắn/giây. Có ba phương án: (A) TF-IDF + logistic regression; (B) encoder 2 lớp tự train (125k tham số, như project); (C) gọi một LLM decoder 7B tự host, prompt dài 300 token. So sánh ba phương án về độ trễ, bộ nhớ, khả năng khái quát sang cách diễn đạt mới, và chi phí. Đề xuất phương án và cách kết hợp.

**T4 · KV-cache (4đ).**
- (a) Viết hàm `attend_step(q, k_new, v_new, K_cache, V_cache)` cho **một** head: `q`, `k_new`, `v_new` có shape `(d,)`; `K_cache`, `V_cache` có shape `(t, d)` (có thể `t = 0`). Hàm trả về `(out, K_cache_mới, V_cache_mới)`.
- (b) Giải thích vì sao lặp `attend_step` qua các vị trí $1..T$ cho đúng kết quả của causal attention trên cả chuỗi, và vì sao không cần mask trong `attend_step`.
- (c) Vì sao cache chỉ lưu $K$ và $V$, không lưu $Q$?

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Phần 1**

1. Cặp: `b a` 4, `a n` 6, `n </w>` 6, `n a` 2. Hòa ở 6; vì `"a" < "n"` nên ghép **`a n`** trước.
2. "ban": **`b` `an</w>`**. "nana": `n a n a </w>` → ghép `a n` (hạng 0) được `n an a </w>`; `an a` và `a </w>` chưa học, nên kết quả là **`n` `an` `a` `</w>`**.
3. "goin": `goi` khớp dài nhất, còn `n`, thành **`goi` `##n`**. "dangky": **`dang` `##ky`**.
4. Vocab gốc gồm đủ **256 byte**, và mọi văn bản UTF-8 đều là dãy byte, nên mọi đầu vào đều biểu diễn được (trong trường hợp xấu nhất, mỗi byte một token).
5. `đ` (U+0111) 2 byte + `ổ` (U+1ED5) 3 byte + `i` 1 byte $=$ **6 byte**.
6. $d_\text{head} = 64$; tham số $= 4(512^2 + 512) =$ **1 050 624** (không phụ thuộc số head).
7. **(4, 8, 10, 10)**.
8. $1 + 2 + 3 + 4 =$ **10**.
9. $(\sin 0, \cos 0, \sin 0, \cos 0) =$ **(0, 1, 0, 1)**. Nhờ các chiều cos, vị trí 0 vẫn có tín hiệu riêng; cặp (sin, cos) cho phép biểu diễn dịch vị trí bằng một phép xoay.
10. **Bằng nhau**: tích vô hướng sau RoPE chỉ phụ thuộc hiệu vị trí, mà $5 - 2 = 8 - 5 = 3$.
11. $12 \cdot 512^2 + 13 \cdot 512 = 3\,145\,728 + 6\,656 =$ **3 152 384**.
12. $2 \times 24 \times 2048 \times 2048 \times 2$ byte $=$ **0.375 GiB** mỗi câu; batch 8 là **3 GiB**.
13. **Encoder hai chiều.** (i) Phân loại cần ngữ cảnh cả hai phía ("không … nữa"); (ii) không sinh văn bản nên không cần causal mask và KV-cache; (iii) nhỏ, nhanh trên CPU; một lần forward cho ra kết quả. (Bất kỳ 2 ý.)
14. *(mỗi lỗi khoảng 0.67đ)*
    - Chia cho $\sqrt d$ thay vì $\sqrt{d_\text{head}}$: phải là `math.sqrt(self.d_head)`.
    - Mask bằng cách **nhân sau softmax**: tổng trọng số không còn bằng 1, và vị trí bị che vẫn ảnh hưởng tới mẫu số. Phải `scores.masked_fill(~mask, -inf)` **trước** softmax.
    - `.view(B, T, d)` sau `transpose`: tensor không liền bộ nhớ, nên lỗi. Dùng `.reshape(B, T, d)` (hoặc `.contiguous().view(...)`).

**T1 (4đ)**

(a) *(2đ)* Đếm cặp: `a p` = 4 + 3 = **7**; `p </w>` = **7**; `n a` = 4 + 2 = 6; `l a` = 3; `a m` = 2; `m </w>` = 2.
1. Hòa ở 7: `("a", "p") < ("p", "</w>")`, nên ghép **`a p` → `ap`**.
2. Giờ có `n ap </w>` (4), `l ap </w>` (3), `n a m </w>` (2). Cặp `ap </w>` = 7, lớn nhất, nên ghép **`ap </w>` → `ap</w>`**.
3. `n ap</w>` = 4, `l ap</w>` = 3, `n a` = 2, `a m` = 2, `m </w>` = 2. Ghép **`n ap</w>` → `nap</w>`**.

(b) *(1đ)* "nap": **`nap</w>`**. "map": `m a p </w>` → `m ap </w>` → **`m` `ap</w>`**. "lam": `l a m </w>`: không cặp nào đã học (`a m` chưa được ghép), nên **`l` `a` `m` `</w>`**.

(c) *(1đ)* Merge chỉ được học cho những cặp **phổ biến trong corpus**. Vần "ap" cuối từ xuất hiện 7 lần, nên "map" (từ mới) dùng lại được `ap</w>`. Cặp `a m` chỉ có 2 lần, nên chưa kịp được học trong 3 merge đầu.

**T2 (4đ)**

(a) *(1đ)*
$$\frac{QK^\top}{\sqrt 2} = \frac{1}{\sqrt 2}\begin{pmatrix} 1 & 2 & 0 \\ 1 & 0 & 0 \\ 2 & 2 & 0 \end{pmatrix} = \begin{pmatrix} 0.707 & 1.414 & 0 \\ 0.707 & 0 & 0 \\ 1.414 & 1.414 & 0 \end{pmatrix}.$$

(b) *(1.5đ)*
- Token 2 thấy vị trí 1–2: $\mathrm{softmax}(0.707, 0) = (0.670,\ 0.330)$; đầu ra $= 0.670\,(1, 0) + 0.330\,(0, 2) =$ **(0.670, 0.660)**.
- Token 3 thấy cả ba: $e^{1.414} = 4.113$, nên trọng số $= (4.113, 4.113, 1)/9.226 = (0.446,\ 0.446,\ 0.108)$; đầu ra $= 0.446\,(1, 0) + 0.446\,(0, 2) + 0.108\,(3, 1) =$ **(0.771, 1.000)**.

(c) *(1.5đ)* Token 3 chỉ thấy vị trí 1 và 3: $\mathrm{softmax}(1.414, 0) = (0.804,\ 0.196)$; đầu ra $= 0.804\,(1, 0) + 0.196\,(3, 1) =$ **(1.391, 0.196)**.
Token 1 và token 2 cho đầu ra giống hệt nhau, vì hai hàng điểm của chúng **trên các cột được nhìn** (cột 1 và 3) giống nhau: $(0.707, 0)$. Cả hai cho trọng số $(0.670, 0, 0.330)$ và đầu ra $(1.660, 0.330)$. (Đầu ra ở vị trí đệm vốn bị bỏ qua khi pooling, nên không quan trọng.)

**T3 (3đ)** *(mỗi ý khoảng 0.5đ)*
- **(A) TF-IDF + LR:** dưới 1 ms mỗi tin trên CPU, vài MB bộ nhớ, dễ giải thích; khái quát kém sang cách diễn đạt mới (F1 theo template khoảng 0.71–0.73 ở project).
- **(B) Encoder nhỏ tự train:** khoảng 1 ms; nhưng không có tri thức ngôn ngữ từ pretrain, nên trên dữ liệu ít nó khái quát **kém nhất** (khoảng 0.52 ở project).
- **(C) LLM 7B:** khái quát tốt nhất (zero/few-shot), nhưng prefill 300 token trên CPU mất hàng giây, không đạt 50 ms. 200 tin/giây cần GPU và tốn khoảng 14 GB trọng số fp16 cộng KV-cache. Chi phí cao nhất.
- **Đề xuất:** (A), hoặc một encoder **pretrained** nhỏ đã fine-tune (bài 18), làm đường chính, có ngưỡng từ chối. Tin nhắn có độ tin cậy thấp thì chuyển cho tổng đài viên, hoặc cho LLM xử lý bất đồng bộ. Dùng nhãn từ tổng đài viên / LLM để mở rộng dữ liệu train theo thời gian.
- Đo độ trễ **p95** chứ không chỉ trung bình; đánh giá bằng chia theo template hoặc theo thời gian.

**T4 (4đ)**

(a) *(2đ)*
```python
def attend_step(q, k_new, v_new, K_cache, V_cache):
    K = torch.cat([K_cache, k_new[None]], 0)            # (t+1, d)
    V = torch.cat([V_cache, v_new[None]], 0)
    w = torch.softmax(K @ q / math.sqrt(q.shape[0]), dim=0)   # (t+1,)
    return w @ V, K, V
```

(b) *(1.5đ)* Ở bước $t$, cache chứa đúng $k_1..k_t$, $v_1..v_t$ (cộng với token mới), tức là **đúng tập vị trí mà causal mask cho phép** hàng $t$ nhìn. Hàng $t$ của causal attention trên cả chuỗi là $\mathrm{softmax}(q_t K_{1..t}^\top/\sqrt d)\,V_{1..t}$, đúng phép tính trong `attend_step`. Không cần mask vì cache **chưa có** tương lai. Đầu ra của các vị trí cũ không đổi khi thêm token mới, cũng chính nhờ causal mask, nên không cần tính lại.

(c) *(0.5đ)* $q$ của một vị trí chỉ được dùng **đúng một lần**, ở bước của chính vị trí đó. Còn $k$, $v$ của nó được **mọi** bước sau dùng lại.

</details>
