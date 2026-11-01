# Lời giải · Final Tuần 5

> Mọi con số ở đây đã chạy kiểm chứng trên máy bạn. Phần C có code đầy đủ ở cuối; bản đã kiểm chứng (11/11 test pass) **không** nằm trong repo.

---

## Phần A (mỗi câu 2đ)

**A1.** (i) Một tin ở dạng **NFD** (chữ và dấu tách rời), tin kia ở **NFC**. (ii) Hai **kiểu đặt dấu**: "hủy" và "huỷ". Ngoài ra có thể do chữ hoa/thường. Sửa: `unicodedata.normalize("NFC", ...)` và bảng thống nhất kiểu dấu, đặt trong hàm tiền xử lý chạy ở **cả** train lẫn lúc phục vụ (bài 11).

**A2.** Chia ngẫu nhiên đặt các câu **gần trùng nhau** (cùng template, chỉ khác slot và nhiễu) vào cả train lẫn test, nên model chỉ cần nhận lại chúng: một dạng rò rỉ. Chia theo template kiểm tra khả năng khái quát sang **cách diễn đạt mới**. Với log thật: chia **theo thời gian** (train quá khứ, test tương lai), và/hoặc **theo khách hàng / theo cuộc hội thoại** (`GroupKFold`); loại các câu trùng hoặc gần trùng giữa hai tập.

**A3.** $\text{idf}(t) = \ln\frac{1 + N}{1 + \text{df}(t)} + 1$. Có mặt trong mọi văn bản thì $\text{idf} = \ln 1 + 1 = 1$. Không đặt bằng 0, để từ đó không bị **xóa hẳn**: nó vẫn có thể mang thông tin khi kết hợp với các từ khác, và một văn bản chỉ gồm các từ phổ biến vẫn có vector khác 0.

**A4.** Lũy thừa 0.75 **làm phẳng** phân phối: giảm tỉ lệ của từ rất phổ biến, tăng tỉ lệ của từ hiếm (với số đếm 81, 16, 1: từ 0.83 / 0.16 / 0.01 thành 0.75 / 0.22 / 0.03). Nhờ vậy từ hiếm cũng được dùng làm nhiễu đủ nhiều để vector của chúng được "đẩy" đúng chỗ, còn từ phổ biến không chiếm hết số mẫu nhiễu. Số 0.75 được chọn bằng thực nghiệm (Mikolov et al. 2013).

**A5.** BPE chọn cặp có **tần suất** lớn nhất. WordPiece chọn cặp có $\frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$ lớn nhất (tăng likelihood nhiều nhất). Ví dụ: $\text{count}(g, i) = 50$ với $\text{count}(g) = 100$, $\text{count}(i) = 200$; và $\text{count}(q, u) = 10$ với $\text{count}(q) = 10$, $\text{count}(u) = 40$. BPE chọn `g i`; WordPiece chọn `q u` ($0.025 > 0.0025$), vì "q" luôn đi kèm "u".

**A6.** Chia $\sqrt{d_\text{head}} = \sqrt{64} = 8$. Chia cho $\sqrt{512} \approx 22.6$ làm điểm số nhỏ đi $\sqrt 8 \approx 2.8$ lần, nên softmax **phẳng** hơn (gần đều): model khó tập trung vào token quan trọng, học chậm hơn.

**A7.** $Q$ sau khi tách head: **(8, 8, 32, 32)**, tức $(B, h, T, d_\text{head})$ với $d_\text{head} = 32$. Điểm số: **(8, 8, 32, 32)**, tức $(B, h, T, T)$ (trùng shape chỉ vì $T = d_\text{head} = 32$). Đầu ra trước `out_proj`: **(8, 32, 256)**.

**A8.** Sau RoPE, $q_m^\top k_n$ chỉ phụ thuộc vào **vị trí tương đối** $m - n$ (cùng với nội dung của $q$, $k$). Hữu ích vì quan hệ giữa hai từ ("không" đứng ngay trước "muốn") thường phụ thuộc vào khoảng cách, không phụ thuộc vị trí tuyệt đối trong câu; ngoài ra RoPE ngoại suy sang chuỗi dài hơn tốt hơn learned PE.

**A9.** Post-LN: $x \leftarrow \mathrm{LN}(x + f(x))$. Pre-LN: $x \leftarrow x + f(\mathrm{LN}(x))$, thêm một LN cuối. Với pre-LN, đường residual là **đường thẳng** từ đầu vào tới đầu ra, không đi qua LayerNorm nào, nên gradient ổn định ngay từ đầu. Nhờ đó train được model sâu mà không cần warmup dài, và ít bị phân kỳ (Xiong et al. 2020).

**A10.** (a) **Encoder-only**: gán nhãn từng token cần ngữ cảnh hai phía. (b) **Encoder–decoder** (hoặc decoder-only lớn): đầu vào dài, đầu ra là văn bản mới, ngắn hơn. (c) **Decoder-only**: sinh văn bản tự do, dùng prompt cho mọi loại câu hỏi.

**A11.** $\mathrm{PPL} = \exp\big(-\frac1N\sum_t \log P(x_t \mid x_{<t})\big)$. **Không so được trực tiếp**, vì đơn vị (ký tự / từ) khác nhau. Quy về cùng đơn vị: với khoảng 4.2 ký tự mỗi từ, $1.4^{4.2} \approx 4.1$ mỗi từ, tốt hơn 30. Nhưng chỉ chắc chắn khi đo trên **cùng** tập test, cùng cách đếm.

**A12.** Rất chắc chắn (một token chiếm 0.95): top-p 0.9 giữ **1** token; top-k ($k = 50$) vẫn giữ 50 token, kể cả 49 token gần như vô nghĩa. Phân vân (phân phối phẳng): top-p giữ **nhiều** token; top-k chỉ giữ $k$, có thể cắt mất các lựa chọn hợp lý. Top-p tự co giãn theo độ chắc chắn của model.

---

## Phần B

**B1 (5đ).**

(a) *(3đ)* Ban đầu: `c u o c </w>` (3), `c u o n </w>` (2), `c u a </w>` (4).

| Cặp | Đếm |
|---|---|
| `c u` | 3 + 2 + 4 = **9** |
| `u o` | 5 |
| `a </w>` | 4 |
| `u a` | 4 |
| `c </w>` | 3 |
| `o c` | 3 |
| `n </w>` | 2 |
| `o n` | 2 |

1. **`c u` → `cu`** (9).
2. Còn `cu o c </w>`, `cu o n </w>`, `cu a </w>`. Cặp `cu o` = 5 lớn nhất, nên **`cu o` → `cuo`**.
3. Còn `cuo c </w>` (3), `cuo n </w>` (2), `cu a </w>` (4). Cặp `a </w>` = 4 và `cu a` = 4 **hòa**; so phần tử đầu, `"a" < "cu"`, nên **`a </w>` → `a</w>`**.

(b) *(1đ)* "cuoi": `c u o i </w>` → `cu o i </w>` → **`cuo` `i` `</w>`**. "cu": `c u </w>` → **`cu` `</w>`**.

(c) *(1đ)* Ký tự **`i`** không có trong corpus (chỉ có c, u, o, n, a), nên không có trong vocab và thành `<unk>` khi encode. Byte-level BPE dùng 256 byte làm vocab gốc, nên mọi ký tự (kể cả chưa từng thấy) đều biểu diễn được bằng các byte của nó.

**B2 (5đ).**

(a) *(2đ)* $d = 256$, $d_\text{ff} = 1\,024$:
- Attention: $4(256^2 + 256) = 263\,168$.
- FFN: $256 \cdot 1024 + 1024 + 1024 \cdot 256 + 256 = 525\,568$.
- 2 LayerNorm: $4 \cdot 256 = 1\,024$.
- Tổng một lớp: **789 760**.

(b) *(1.5đ)*
- Embedding: $8000 \cdot 256 + 128 \cdot 256 = 2\,048\,000 + 32\,768 = 2\,080\,768$.
- 4 lớp: $3\,159\,040$. LN cuối: $512$. Head: $256 \cdot 9 + 9 = 2\,313$.
- Tổng: **5 242 633**. Embedding chiếm **39.7%**.

(c) *(1.5đ)* $2 \times L \times T \times d \times 2\ \text{byte} \times B = 2 \times 4 \times 512 \times 256 \times 2 \times 32 = 67\,108\,864$ byte $=$ **64 MiB**.

**B3 (5đ).**

(a) *(2.5đ)* $P(\text{hủy} \mid \langle s\rangle) = 6/10$; $P(\text{gói} \mid \text{hủy}) = 5/(5 + 1) = 5/6$; $P(\langle/s\rangle \mid \text{gói}) = 5/5 = 1$. Tích: $0.6 \times 0.833 \times 1 =$ **0.5**. $\mathrm{PPL} = 0.5^{-1/3} =$ **1.26**.

(b) *(1đ)* $\text{count}(\text{nạp}, \text{gói}) = 0$, nên $P = 0$ và perplexity là **vô cực** ($\log 0$). Add-k chia một ít xác suất cho **mọi** bigram chưa thấy, nên perplexity hữu hạn và model không "chết" vì một câu lạ.

(c) *(1.5đ)* Tổng tích lũy trước mỗi token: 0, 0.45, 0.70, 0.85, 0.95.
- $p = 0.5$: bỏ token có tổng trước ≥ 0.5, nên giữ **2 token đầu** (0.45, 0.25).
- $p = 0.8$: giữ **3 token đầu** (0.45, 0.25, 0.15).

**B4 (5đ).**

(a) *(2đ)*

| Lớp | Dự đoán thành lớp này | TP | Precision | Recall | F1 |
|---|---|---|---|---|---|
| hủy gói | 7 + 2 + 1 = 10 | 7 | 0.700 | 7/10 = 0.700 | 0.700 |
| đăng ký | 3 + 6 = 9 | 6 | 0.667 | 6/8 = 0.750 | 0.706 |
| khác | 1 | 1 | 1.000 | 1/2 = 0.500 | 0.667 |

(b) *(1.5đ)* Macro-F1 $= (0.700 + 0.706 + 0.667)/3 =$ **0.691**. Micro-F1 $=$ accuracy $= 14/20 =$ **0.70**. Weighted-F1 $= (10 \cdot 0.700 + 8 \cdot 0.706 + 2 \cdot 0.667)/20 =$ **0.699**. Báo **macro-F1**, kèm bảng từng lớp: lớp "khác" hiếm nhưng quan trọng (recall chỉ 0.5); micro và weighted bị lớp đông chi phối.

(c) *(1.5đ)* $c = 5$, $r = 6$. $p_1 = 5/5$; bigram "nạp thẻ" ✓, "thẻ cho" ✗, "cho số" ✓, "số này" ✓, nên $p_2 = 3/4$. BP $= e^{1 - 6/5} = 0.819$. BLEU-2 $= 0.819 \times \sqrt{0.75} =$ **0.709**. BLEU-4: không có 4-gram nào trùng ("thẻ cho" phá vỡ mọi 4-gram), nên $p_4 = 0$ và **BLEU-4 = 0**.

---

## Phần C · Code (32đ)

```python
import math
from collections import Counter

import numpy as np


# C1 · WordPiece
def wordpiece_tokenize(word, vocab, unk="[UNK]", max_chars=100):
    if len(word) > max_chars:
        return [unk]
    out, start = [], 0
    while start < len(word):
        end, piece = len(word), None
        while start < end:                                       # thử từ dài nhất tới ngắn nhất
            sub = word[start:end] if start == 0 else "##" + word[start:end]
            if sub in vocab:
                piece = sub
                break
            end -= 1
        if piece is None:
            return [unk]                                         # cả từ thành UNK
        out.append(piece)
        start = end
    return out


# C2 · Causal MHA + KV-cache
def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mha(X, Wq, Wk, Wv, Wo, n_heads):
    T, d = X.shape
    dh = d // n_heads
    split = lambda M: M.reshape(T, n_heads, dh).transpose(1, 0, 2)          # (h, T, dh)
    q, k, v = split(X @ Wq), split(X @ Wk), split(X @ Wv)
    s = q @ k.transpose(0, 2, 1) / math.sqrt(dh)
    s = np.where(np.tril(np.ones((T, T), dtype=bool)), s, -np.inf)
    out = (_softmax(s) @ v).transpose(1, 0, 2).reshape(T, d)
    return out @ Wo


def init_cache(n_heads, d_head):
    return {"k": np.zeros((n_heads, 0, d_head)), "v": np.zeros((n_heads, 0, d_head))}


def decode_step(x_t, cache, Wq, Wk, Wv, Wo, n_heads):
    d = x_t.shape[0]
    dh = d // n_heads
    q, k, v = ((x_t @ W).reshape(n_heads, 1, dh) for W in (Wq, Wk, Wv))
    K = np.concatenate([cache["k"], k], axis=1)
    V = np.concatenate([cache["v"], v], axis=1)
    w = _softmax(q @ K.transpose(0, 2, 1) / math.sqrt(dh))                   # (h, 1, t): không cần mask
    return (w @ V).reshape(d) @ Wo, {"k": K, "v": V}


# C3 · Bigram LM
class BigramLM:
    BOS, EOS = "<s>", "</s>"

    def __init__(self, k=1.0):
        self.k = k

    def fit(self, sentences):
        self.vocab_ = sorted({w for s in sentences for w in s} | {self.EOS})
        self.counts_, self.ctx_ = Counter(), Counter()
        for s in sentences:
            seq = [self.BOS] + list(s) + [self.EOS]
            for a, b in zip(seq, seq[1:]):
                self.counts_[a, b] += 1
                self.ctx_[a] += 1
        return self

    def prob(self, prev, w):
        return (self.counts_[prev, w] + self.k) / (self.ctx_[prev] + self.k * len(self.vocab_))

    def perplexity(self, sentences):
        nll, n = 0.0, 0
        for s in sentences:
            seq = [self.BOS] + list(s) + [self.EOS]
            for a, b in zip(seq, seq[1:]):
                nll -= math.log(self.prob(a, b))
                n += 1
        return math.exp(nll / n)


# C4 · MLM masking
def mlm_mask(ids, mask_id, vocab_size, special_ids=(), p=0.15, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    ids = np.asarray(ids)
    inputs, labels = ids.copy(), np.full_like(ids, -100)
    chosen = ~np.isin(ids, list(special_ids)) & (rng.random(ids.shape) < p)
    labels[chosen] = ids[chosen]
    r = rng.random(ids.shape)
    inputs[chosen & (r < 0.8)] = mask_id
    rand = chosen & (r >= 0.8) & (r < 0.9)
    inputs[rand] = rng.integers(0, vocab_size, size=int(rand.sum()))
    return inputs, labels
```

**Điểm hay mất:**
- **C1:** khi không khớp, chỉ thay **một phần** bằng `[UNK]` (phải là cả từ); quên tiền tố `##` cho phần nối tiếp; vòng `end` chạy từ ngắn tới dài (phải là dài tới ngắn).
- **C2:** tách head sai thứ tự (`reshape(n_heads, T, dh)` thay vì `reshape(T, n_heads, dh).transpose(...)` cho kết quả khác); chia cho $\sqrt d$; `decode_step` thêm mask (thừa) hoặc dùng `X` thay vì `x_t`.
- **C3:** đưa `<s>` vào vocab (sai mẫu số $|V|$); quên bước dự đoán `</s>` khi tính perplexity.
- **C4:** sửa trực tiếp `ids`; dùng **cùng** một số ngẫu nhiên cho bước "chọn" và bước "80/10/10" (làm lệch tỉ lệ).

---

## Phần D · Thiết kế (24đ)

Chấm theo ý; mỗi phần 6đ. Không có đáp án duy nhất; dưới đây là một bài làm tốt.

**D1 · Dữ liệu và nhãn (6đ)**
- **Làm sạch nhãn** *(1.5đ)*: nhãn từ việc chuyển tiếp thủ công có nhiễu. Lấy mẫu khoảng 1 000 tin, cho 2 người gán lại theo một **hướng dẫn gán nhãn** viết rõ; đo **kappa** (bài 19); sửa hướng dẫn tới khi kappa ≥ khoảng 0.8. Dùng model ban đầu để tìm các tin có nhãn "đáng ngờ" (model rất chắc chắn nhưng khác nhãn) và cho người xem lại (*confident learning*).
- **Ngoài phạm vi** *(1đ)*: thu thập từ log (tin bị tổng đài viên bỏ qua hoặc trả lời "không hỗ trợ"), cộng thêm một ít câu tổng hợp; gán nhãn `khac`. Giữ riêng một tập ngoài phạm vi **mới**, không trùng chủ đề với train, để đo khả năng từ chối.
- **Đa intent** *(1đ)*: gán nhãn dạng **đa nhãn**, hoặc tách câu rồi phân loại từng câu. Ít nhất cần đo tần suất của chúng; nếu hiếm thì chuyển cho người.
- **Kênh** *(1đ)*: ghi trường `channel`; phân tầng train/test theo kênh; báo metric **theo kênh** (SMS ít dấu, Zalo nhiều emoji).
- **Chia** *(1.5đ)*: theo **thời gian** (train trên các tháng cũ, test trên tháng mới nhất), nhóm theo khách hàng (tránh một khách nằm ở cả hai tập), loại câu trùng hoặc gần trùng giữa hai tập (băm sau `normalize`). Đây là phiên bản đời thật của "chia theo template".

**D2 · Model và phục vụ (6đ)**
- **Tầng 1** *(2đ)*: `normalize` + TF-IDF (âm tiết và ký tự) + logistic regression. Dưới 1 ms mỗi tin, vài MB RAM. Đây là baseline và đường dự phòng.
- **Tầng 2** *(2đ)*: một encoder **pretrained** nhỏ, đa ngôn ngữ hoặc tiếng Việt, đã fine-tune và có thể chưng cất (distill) xuống vài chục triệu tham số; export ONNX, lượng tử hóa INT8 (Tuần 12). Chạy trên CPU, khoảng 10–30 ms mỗi tin (cần đo thật). Chỉ dùng tầng này nếu nó thắng tầng 1 **theo cặp** trên tập test theo thời gian.
- **Ước lượng** *(1đ)*: 80 tin/giây × 20 ms CPU mỗi tin = 1.6 core-giây mỗi giây, nên khoảng 4 core có dư; nhân 2–3 lần để có dự phòng và cho p95. Batch nhỏ (ví dụ gom trong 5 ms) để tăng throughput mà không vượt 100 ms.
- **GPU / LLM** *(1đ)*: không ở đường đồng bộ. Dùng LLM **bất đồng bộ** để gán nhãn nháp cho tin bị từ chối, sinh câu diễn đạt mới cho intent mới, hoặc làm "người chấm" khi đánh giá. Mọi nhãn do LLM tạo đều cần người kiểm tra mẫu.

**D3 · Đánh giá và ngưỡng (6đ)**
- **Metric** *(1.5đ)*: macro-F1 chính; bảng từng intent và confusion matrix; recall và precision riêng của **khiếu nại**; recall của lớp ngoài phạm vi; coverage (tỉ lệ tự động xử lý) và precision trên phần được tự động xử lý; độ trễ p95.
- **Ngưỡng có điều kiện theo lớp** *(2đ)*: yêu cầu "không xử lý sai khiếu nại" là một ràng buộc **bất đối xứng**. Mọi tin có xác suất khiếu nại ≥ một ngưỡng **thấp** (ví dụ 0.2) đều được chuyển thẳng tới hàng đợi khiếu nại có người xem, kể cả khi đó không phải lớp có xác suất cao nhất. Các lớp khác dùng ngưỡng max-probability để đạt precision mục tiêu (`choose_threshold` của project); dưới ngưỡng thì chuyển cho người.
- **Ngưỡng có "chuyển" được không** *(1.5đ)*: project cho thấy ngưỡng đạt precision 0.95 trên val chỉ cho khoảng 0.85–0.93 trên test theo template (tùy model). Phân phối mới làm calibration lệch. Vì vậy: chọn ngưỡng trên val **theo thời gian**; hiệu chỉnh xác suất (temperature scaling); để biên an toàn; **shadow mode** một thời gian (model chạy song song, chưa quyết định), so với tổng đài viên rồi mới bật.
- **Kiểm định** *(1đ)*: so ghép cặp nhiều split hoặc bootstrap theo ngày; A/B trên một phần lưu lượng, theo dõi tỉ lệ chuyển nhầm và thời gian xử lý.

**D4 · Vận hành và cải tiến (6đ)**
- **Theo dõi** *(2đ)*: phân phối intent dự đoán theo giờ và theo kênh (đột biến "lỗi mạng" có thể là sự cố thật, cũng có thể là drift); phân phối max-probability (tỉ lệ dưới ngưỡng tăng là dấu hiệu drift); tỉ lệ tin chứa từ ngoài vocabulary; **tín hiệu phản hồi**: tổng đài viên chuyển lại tin sang hàng đợi khác (nhãn sửa sai miễn phí); độ trễ p95; lỗi.
- **Gói cước mới** *(1.5đ)*: tên gói như "GX30" mang ít thông tin cho intent (đã được chuẩn hóa thành chính nó; có thể thay bằng placeholder `<goi>` nhờ danh mục gói). Cú pháp mới ("DK GX30 gửi 1xxx"): thêm vài câu mẫu, hoặc luật đơn giản đi trước model; theo dõi tỉ lệ từ chối ngay sau khi ra mắt gói.
- **Học lại** *(1.5đ)*: định kỳ (ví dụ hằng tháng) và khi có cảnh báo drift. Dữ liệu = nhãn đã xác nhận từ tổng đài viên cộng mẫu chủ động lấy từ các tin có độ tin cậy thấp (*active learning*). Trước khi thay model: chạy bộ test cố định (gồm các ca khó đã biết, khiếu nại, ngoài phạm vi) và so ghép cặp với model hiện tại; shadow mode; rollout dần; giữ được khả năng quay lại model cũ.
- **Quản lý phiên bản** *(1đ)*: version cho dữ liệu, cho bước tiền xử lý (`normalize`), cho tokenizer và model; log đủ thông tin để tái lập (Tuần 13).
