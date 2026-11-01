# Ngày 1 · Văn bản cổ điển: tiền xử lý tiếng Việt, BoW và TF-IDF, word embeddings

> **Loại tài liệu: Lý thuyết (bản nhẹ).** Công thức đầy đủ và ví dụ tính tay. Chứng minh chỉ phác thảo vài dòng.
> **Hôm nay gồm 3 bài:**
> - **11** Tiền xử lý văn bản tiếng Việt: Unicode, chuẩn hóa, teencode, stopword, stemming
> - **12** Bag-of-words, n-gram, TF-IDF và phân loại văn bản
> - **13** Word embeddings: word2vec, GloVe, FastText
>
> **Thời gian:** đọc ~2h45 · quiz 45' · bài tập ~2h
> **Dữ liệu xuyên suốt tuần:** tin nhắn khách hàng gửi tổng đài một nhà mạng (tổng hợp, `intent/data.py`), 9 nhãn: 8 intent (tra cước, nạp tiền, đăng ký gói, hủy gói, lỗi mạng, đổi SIM, khiếu nại, gặp tổng đài viên) và `khac` (ngoài phạm vi). 6 000 tin nhắn sinh từ **141 câu mẫu (template)**, có nhiễu như thật: 30% bỏ dấu, 34% có teencode, 10% ở dạng Unicode NFD, có gõ sai, emoji. Có 6 template **nhập nhằng** mà chính người gán nhãn cũng không thống nhất, nên accuracy tối đa đạt được là **0.986**.
> **Hai cách chia dữ liệu** dùng suốt tuần:
> - **Ngẫu nhiên:** mọi template đều có mặt ở cả train lẫn test.
> - **Theo template:** khoảng 20% template của mỗi intent bị giữ lại hoàn toàn cho test, tức là test toàn **cách diễn đạt chưa từng thấy**. Cách chia này gần với thực tế hơn nhiều.
>
> Mọi con số "đo trên máy bạn" chạy bằng scikit-learn 1.9, PyTorch 2.14 (CPU). Các con số trên chia theo template là trung bình của **5 split**, vì chúng dao động rất mạnh giữa các split.

---

## 11 · Tiền xử lý văn bản tiếng Việt: Unicode, chuẩn hóa, teencode, stopword, stemming

> ⏱ Tầng 1: 10' · Tầng 2: 45'
> **Cần biết trước:** chuỗi Python và Unicode NFC (Tuần 1, phụ lục P1); rò rỉ dữ liệu và chia train/test (Tuần 3, bài 01); feature engineering dạng hàm thuần (Tuần 3, phụ lục P1).

> **📖 Từ mới**
> - **Code point:** số hiệu của một ký tự trong bảng Unicode, viết dạng `U+1EE7` (là chữ "ủ").
> - **Dấu kết hợp (combining mark):** code point không đứng một mình mà "đè" lên ký tự trước nó, ví dụ `U+0309` là dấu hỏi.
> - **NFC / NFD:** hai dạng chuẩn của Unicode. NFC **gộp** chữ và dấu thành một code point khi có thể; NFD **tách** ra thành chữ gốc cộng các dấu kết hợp.
> - **Chuẩn hóa văn bản (text normalization):** đưa các cách viết khác nhau của **cùng một nội dung** về một dạng duy nhất.
> - **Teencode:** cách viết tắt kiểu chat: "ko" = không, "dc" = được, "mk" = mình.
> - **Placeholder:** token đại diện thay cho một giá trị cụ thể, ví dụ mọi số điện thoại thành `<phone>`.
> - **Stopword:** từ rất phổ biến, mang ít nội dung ("của", "là", "thì").
> - **Stemming / lemmatization:** đưa các biến thể hình thái của từ về một gốc chung ("cancelled", "cancelling" → "cancel").

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Một bưu cục nhận thư ghi địa chỉ theo nhiều kiểu: "Q.7", "quận 7", "Quan 7", "QUẬN BẢY". Với người đưa thư, đó là **một** nơi. Với máy tính, đó là **bốn chuỗi khác nhau**. Trước khi phân loại thư, phải có người viết lại địa chỉ theo một mẫu chung. Tiền xử lý văn bản chính là bước đó: gộp các cách viết khác nhau của cùng một ý, để model không phải học riêng từng cách viết.

**Ví dụ có số: hai chữ "hủy" trông giống hệt nhau nhưng khác nhau.**

| Dạng | Các code point | Số code point | Số byte UTF-8 |
|---|---|---|---|
| NFC | `h` `ủ` (U+1EE7) `y` | 3 | 5 |
| NFD | `h` `u` `◌̉` (U+0309, dấu hỏi) `y` | 4 | 5 |

Hai chuỗi hiển thị như nhau, nhưng `"hủy" == unicodedata.normalize("NFD", "hủy")` cho `False`. Bàn phím của một số hệ điều hành gửi đi dạng NFD. Không chuẩn hóa thì model coi đó là hai từ khác nhau.

Trên 6 000 tin nhắn, mỗi bước chuẩn hóa làm số từ khác nhau (vocabulary) giảm đi:

| Bước (cộng dồn) | Số từ khác nhau | Số từ chỉ xuất hiện 1 lần |
|---|---|---|
| Tách theo khoảng trắng, không làm gì | 1 872 | 560 |
| + NFC | 1 533 | 440 |
| + chữ thường | 1 059 | 228 |
| + bỏ dấu câu, emoji | 774 | 137 |
| + placeholder (tiền, SĐT, URL, email), thống nhất kiểu đặt dấu | 752 | 137 |
| + mở rộng teencode | 728 | 137 |
| + bỏ dấu tiếng Việt | 434 | 96 |

**Ba ý cốt lõi:**
1. Chuẩn hóa gộp các cách viết của **cùng một ý**. Đây là bước rẻ nhất và gần như luôn có lợi: NFC, chữ thường, placeholder.
2. Bước nào **xóa thông tin** (bỏ dấu, bỏ stopword) thì có thể gộp nhầm **những ý khác nhau**. Phải đo, không làm theo thói quen.
3. Cùng **một** hàm tiền xử lý phải chạy ở cả lúc train lẫn lúc phục vụ.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 11.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Unicode normalization (NFC, NFD, NFKC, NFKD)** | NFD: phân rã chính tắc; NFC: phân rã rồi tổ hợp lại. NFKC/NFKD còn gộp các ký tự "tương thích" (chữ toàn độ rộng, ký tự ①) | Unicode Standard Annex #15 |
| **Tokenization** | Cắt văn bản thành các đơn vị (token): từ, âm tiết, subword (bài 14), ký tự | Jurafsky & Martin, *Speech and Language Processing* (SLP3), ch. 2 |
| **Tách từ tiếng Việt (word segmentation)** | Tiếng Việt viết cách theo **âm tiết**; một từ có thể gồm nhiều âm tiết ("đăng ký", "tổng đài viên"). Công cụ tách từ nối chúng lại: `đăng_ký` | Nguyen et al., *VnCoreNLP*, NAACL 2018 (demo); thư viện `underthesea` |
| **Stopword removal** | Bỏ các từ phổ biến khỏi văn bản trước khi đếm | Manning, Raghavan, Schütze, *Introduction to Information Retrieval* (IIR) 2008, §2.2.2 |
| **Stemming** | Cắt hậu tố theo luật: "cancelling" → "cancel" | Porter, *An algorithm for suffix stripping*, 1980 |
| **Lemmatization** | Đưa về dạng từ điển dựa trên hình thái học: "went" → "go" | IIR §2.2.4; WordNet (Miller 1995) |
| **Sentence segmentation** | Tách câu; khó ở chỗ dấu chấm trong "TP.HCM", "3.5" | Kiss & Strunk, *Punkt*, Computational Linguistics 2006 |
| **Text normalization** | Viết lại các biến thể không chuẩn (viết tắt, sai chính tả, teencode) về dạng chuẩn | Han & Baldwin, ACL 2011 (chuẩn hóa văn bản mạng xã hội) |

**Tiếng Việt khác tiếng Anh ở đâu:**
- **Không biến hình:** "hủy" không có dạng quá khứ hay số nhiều. Stemming và lemmatization gần như **không áp dụng**. Thay vào đó, bài toán khó là **tách từ** và **dấu thanh**.
- **Dấu thanh mang nghĩa:** "đăng" (đăng ký) khác "dang"; "gói" khác "gội". Bỏ dấu là xóa thông tin (mục 11.4).
- **Hai kiểu đặt dấu** cho cụm "oa", "oe", "uy": kiểu cũ "hòa", "hủy" (dấu trên âm đầu) và kiểu mới "hoà", "huỷ" (dấu trên âm sau). Cả hai đều phổ biến; chỉ cần chọn **một** kiểu làm chuẩn.

#### 11.2 🐢 Đi chậm từng bước: một tin nhắn đi qua hàm `normalize`

Tin nhắn gốc (ở dạng NFD, gõ trên một điện thoại): `"Huỷ gói GX30, nạp 200.000đ cho số 0868 123 456 ko dc à 😡"`

| Bước | Kết quả | Vì sao |
|---|---|---|
| 1. NFC | (hiển thị như cũ; số code point giảm) | Gộp chữ và dấu kết hợp thành một code point |
| 2. Chữ thường | `huỷ gói gx30, nạp 200.000đ cho số 0868 123 456 ko dc à 😡` | "Hủy" và "hủy" là một |
| 3. URL, email, SĐT → placeholder | `… cho số <phone> ko dc à 😡` | 6 số điện thoại khác nhau không mang thông tin gì về intent |
| 4. Tiền → `<money>` | `huỷ gói gx30, nạp <money> cho số <phone> …` | Phải làm **trước** teencode: "50k" có chữ "k" (nghìn), không phải "k" (không) |
| 5. Thống nhất kiểu dấu | `hủy gói gx30, …` | "huỷ" → "hủy" |
| 6. Tách token, bỏ dấu câu và emoji | `hủy` `gói` `gx30` `nạp` `<money>` `cho` `số` `<phone>` `ko` `dc` `à` | Regex `<\w+>\|\w+` |
| 7. Teencode | `… không được à` | Tra bảng `ko` → không, `dc` → được |
| 8. *(Tùy chọn)* bỏ dấu | `huy goi gx30 nap <money> cho so <phone> khong duoc a` | Chỉ làm nếu đo thấy có lợi |

Kết quả bước 7: `hủy gói gx30 nạp <money> cho số <phone> không được à`.

**Thứ tự các bước có ý nghĩa.** Bước 3–4 phải đứng trước bước 6, vì regex số điện thoại cần các khoảng trắng bên trong "0868 123 456". Bước 4 phải đứng trước bước 7 (lý do ở bảng trên).

#### 11.3 Ví dụ tính tay: bỏ dấu gộp những từ nào?

Hàm bỏ dấu: phân rã NFD, xóa mọi code point loại `Mn` (dấu kết hợp), đổi `đ` → `d`, rồi NFC lại. Trên 728 từ đã chuẩn hóa của dữ liệu, bỏ dấu tạo ra **246 nhóm va chạm** (nhiều từ khác nhau thành cùng một chuỗi). Một số nhóm:

| Sau khi bỏ dấu | Các từ gốc bị gộp |
|---|---|
| `tu` | tự, tư, từ, tử, tu |
| `dung` | đúng, dụng, dừng, dùng, dung |
| `noi` | nói, nối, nội, noi |
| `nhan` | nhắn, nhân, nhận, nhan |
| `cuoc` | cước, cuộc, cuoc |
| `dang` | đang, đăng, dang |
| `gap` | gấp, gặp, gap |

Va chạm nguy hiểm nhất với bài toán của ta: "hông" (teencode của "không") và "hỏng" (hư) đều thành `hong`. "Sim bị hỏng" và "sim hông bị gì" không còn phân biệt được bằng từ đó nữa.

#### 11.4 Đo trên máy bạn: chuẩn hóa giúp bao nhiêu?

TF-IDF (âm tiết 1–2 gram, bài 12) + logistic regression, macro-F1 trên 9 lớp, trung bình 5 split. Cột "NFD" là accuracy chỉ trên các tin nhắn ở dạng NFD (10% dữ liệu).

| Tiền xử lý | Ngẫu nhiên: F1 | Ngẫu nhiên: acc NFD | **Theo template: F1** |
|---|---|---|---|
| Không làm gì | 0.969 | 0.932 | 0.707 |
| NFC + chữ thường | 0.981 | 0.975 | 0.715 |
| `normalize` không teencode | 0.981 | 0.973 | 0.731 |
| **`normalize` đầy đủ** | **0.981** | 0.971 | **0.733** |
| `normalize` + bỏ dấu | 0.981 | 0.979 | 0.717 |
| `normalize` + bỏ stopword | 0.983 | | 0.714 |
| *TF-IDF ký tự 2–5 gram, không làm gì* | 0.973 | 0.968 | 0.673 |
| *TF-IDF ký tự 2–5 gram + `normalize`* | 0.983 | 0.980 | 0.705 |

**So ghép cặp trên chia theo template** (cùng 5 split):

| So sánh | Chênh F1 trung bình | Số split thắng |
|---|---|---|
| `normalize` so với không làm gì | **+0.026** | **5/5** |
| + bỏ dấu, so với `normalize` | −0.015 | 2/5 |
| + teencode, so với không teencode | +0.002 | — (nhiễu) |

**Đọc bảng:**
- **NFC + chữ thường** là bước có lợi chắc chắn nhất. Nó sửa gần hết lỗi trên tin nhắn NFD (accuracy từ 0.932 lên 0.975).
- **Bỏ dấu** giúp được phần tin nhắn vốn đã không dấu, nhưng làm mất thông tin ở phần có dấu (mục 11.3), nên tính chung là **thua**, 3 trên 5 split.
- **Teencode** gần như không giúp. Lý do: dữ liệu train đã có đủ các biến thể "ko", "k", "hông", nên model tự học được chúng. Bảng tra chỉ có ích khi một biến thể hiếm hoặc chưa từng thấy.
- **Bỏ stopword** không giúp (−0.019 trên template). Với câu ngắn như tin nhắn, mỗi từ đều quan trọng. Danh sách stopword tiếng Anh thông dụng còn có "not", "no": bỏ đi là mất **phủ định**.
- **Độ dao động giữa các split rất lớn:** F1 của cùng một cấu hình theo template chạy từ 0.63 đến 0.87 tùy split. Chỉ có so sánh **ghép cặp** mới cho kết luận tin được (Tuần 3, bài 03).

#### 11.5 Code chuẩn

```python
import re
import unicodedata

TEENCODE = {"ko": "không", "k": "không", "dc": "được", "mk": "mình", "nv": "nhân viên", ...}
TONE_FIX = {"uỷ": "ủy", "oà": "òa", "oá": "óa", ...}             # kiểu mới → kiểu cũ (chọn một)

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"(\+84|0)(\d[\s.]?){8,10}\d?")
_MONEY = re.compile(r"\b\d+([.,]\d{3})*\s*(k|nghìn|ngàn|đ|đồng|vnd)\b")
_TOKEN = re.compile(r"<\w+>|\w+")

def strip_accents(text):
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return unicodedata.normalize("NFC", "".join(c for c in text if unicodedata.category(c) != "Mn"))

def normalize(text, *, strip=False, teencode=True):
    text = unicodedata.normalize("NFC", text).lower()
    for pat, ph in ((_URL, " <url> "), (_EMAIL, " <email> "), (_PHONE, " <phone> "), (_MONEY, " <money> ")):
        text = pat.sub(ph, text)
    for old, new in TONE_FIX.items():
        text = text.replace(old, new)
    tokens = _TOKEN.findall(text)                       # bỏ dấu câu, emoji
    if teencode:
        tokens = [TEENCODE.get(t, t) for t in tokens]
    text = " ".join(tokens)
    return strip_accents(text) if strip else text
```

**Lý do thiết kế:**
- **Hàm thuần, không học gì từ dữ liệu**, nên gọi trước khi chia train/test vẫn an toàn (Tuần 3, phụ lục P1). Bảng teencode là tri thức có sẵn, không phải thống kê từ dữ liệu.
- **Tham số `strip`, `teencode`** để bật tắt từng bước khi đo như mục 11.4.
- **Placeholder thay vì xóa:** "có số điện thoại trong tin nhắn" vẫn là thông tin (thường là intent nạp tiền hoặc gọi lại).

#### 11.6 Cạm bẫy

1. **Quên NFC:** cùng một từ thành hai token; accuracy trên tin nhắn NFD giảm từ 0.975 xuống 0.932 (mục 11.4). Lỗi này **vô hình** khi nhìn bằng mắt.
2. **Bỏ dấu theo thói quen:** gộp "đúng/dừng/dùng", "hông/hỏng". Đo xem có lợi không đã.
3. **Bỏ stopword với câu ngắn hoặc khi cần phủ định:** "không muốn dùng gói nữa" mà mất "không" thì đảo nghĩa. Stopword removal có ích cho tìm kiếm kiểu IR cổ điển, ít khi có ích cho phân loại bằng model học.
4. **Sai thứ tự bước:** chạy teencode trước khi nhận diện tiền thì "50 k" thành "50 không".
5. **Training–serving skew:** lúc train chạy `normalize`, lúc phục vụ thì quên, hoặc dùng một bản copy khác. Đóng gói tiền xử lý **cùng** model (bài 12: `TfidfVectorizer(preprocessor=normalize)` nằm trong Pipeline).
6. **Lowercase xóa thông tin quan trọng** trong một số bài toán: tên riêng trong NER ("Hòa" là tên người hay "hòa" trong "hòa mạng"?). Với phân loại intent thì an toàn.
7. **Tách câu bằng dấu chấm:** "Gói GX30 giá 30.000đ. Tôi muốn hủy." có dấu chấm trong số tiền. Một tin nhắn cũng có thể chứa **hai intent** ("hủy gói A rồi đăng ký gói B"), xem phần D của final.

**Sắc thái phỏng vấn:**
- *"Tiền xử lý văn bản tiếng Việt cần làm gì?"* NFC; chữ thường; thống nhất kiểu dấu; placeholder cho số, tiền, link; xử lý teencode và không dấu (bảng tra, hoặc dùng n-gram ký tự / subword để model tự chịu được); tách từ nếu dùng đặc trưng mức từ. Kèm câu: "mỗi bước xóa thông tin thì phải đo trên tập test đúng kiểu".
- *"Có nên bỏ stopword không?"* Tùy model. Với TF-IDF cho tìm kiếm thì có thể; với phân loại câu ngắn hoặc model Transformer thì không. Nêu ví dụ phủ định.

#### 11.7 Đọc thêm

- Unicode Standard Annex #15, *Unicode Normalization Forms*.
- D. Jurafsky & J. Martin, *Speech and Language Processing* (bản nháp 3rd ed., online), chương 2 *Regular Expressions, Tokenization, Edit Distance*.
- C. Manning, P. Raghavan, H. Schütze, *Introduction to Information Retrieval* (2008), chương 2.
- B. Han & T. Baldwin, *Lexical Normalisation of Short Text Messages: Makn Sens a #twitter*, ACL 2011.
- T. Vu et al., *VnCoreNLP: A Vietnamese Natural Language Processing Toolkit*, NAACL 2018 (demo).

---

## 12 · Bag-of-words, n-gram, TF-IDF và phân loại văn bản

> ⏱ Tầng 1: 10' · Tầng 2: 40'
> **Cần biết trước:** logistic regression đa lớp, regularization (Tuần 3, bài 05); ma trận thưa và feature hashing (Tuần 3, phụ lục P1); chuẩn L2, cosine (Tuần 2, bài 01).

> **📖 Từ mới**
> - **Bag-of-words (BoW):** biểu diễn văn bản bằng **số lần xuất hiện** của từng từ, bỏ qua thứ tự, như đổ các từ vào một cái túi.
> - **n-gram:** dãy $n$ token liên tiếp. Với "hủy gói data": unigram là `hủy`, `gói`, `data`; bigram là `hủy gói`, `gói data`.
> - **Document frequency (df):** số **văn bản** có chứa một từ (không phải số lần xuất hiện).
> - **TF-IDF:** trọng số = (số lần từ xuất hiện trong văn bản) × (độ "hiếm" của từ trên toàn bộ dữ liệu).
> - **Ma trận thưa (sparse):** ma trận gần như toàn số 0; chỉ lưu các ô khác 0.
> - **One-hot:** mỗi từ là một vector toàn 0, trừ một số 1 ở vị trí của từ đó.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Nhân viên tổng đài lâu năm nghe một câu là biết ngay cần chuyển cho bộ phận nào, chỉ nhờ vài **từ khóa**: nghe "trừ tiền" thì chuyển khiếu nại, nghe "mất sóng" thì chuyển kỹ thuật. Họ không cần phân tích ngữ pháp. BoW + logistic regression làm đúng như vậy: mỗi từ có một "điểm" cho từng intent, cộng các điểm lại rồi chọn intent điểm cao nhất. TF-IDF thêm một ý: từ nào **xuất hiện ở mọi nơi** ("gói", "tôi") thì ít giá trị phân biệt, nên được trọng số thấp.

**Ví dụ có số.** Ba tin nhắn: (1) "hủy gói", (2) "đăng ký gói", (3) "hủy hủy". Vocabulary sắp theo thứ tự chuỗi: `gói`, `hủy`, `ký`, `đăng`.

| | `gói` | `hủy` | `ký` | `đăng` |
|---|---|---|---|---|
| df (số văn bản chứa từ) | 2 | 2 | 1 | 1 |
| idf $= \ln\frac{1 + 3}{1 + \text{df}} + 1$ | 1.288 | 1.288 | **1.693** | **1.693** |
| Văn bản (2): đếm × idf | 1.288 | 0 | 1.693 | 1.693 |
| Văn bản (2): chia chuẩn L2 (2.719) | **0.474** | 0 | **0.623** | **0.623** |

"gói" có trong 2/3 văn bản nên bị hạ trọng số so với "đăng", "ký".

**Ba ý cốt lõi:**
1. BoW biến văn bản thành vector **cố định chiều** (bằng kích thước vocabulary), rất thưa. Từ đó dùng được mọi model của Tuần 3.
2. TF-IDF = đếm × độ hiếm, rồi chuẩn hóa độ dài. n-gram giữ lại **một chút** thứ tự ("đăng ký" khác "ký đăng"); n-gram **ký tự** chịu được lỗi gõ và không dấu.
3. One-hot coi mọi cặp từ khác nhau là **xa như nhau**: "hủy" và "huỷ" vuông góc với nhau. Đó là động lực của embedding (bài 13).

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 12.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Vector space model** | Biểu diễn văn bản là vector trọng số theo từ; so khớp bằng cosine | Salton, Wong, Yang, *CACM* 1975 |
| **TF-IDF** | $w_{t,d} = \text{tf}_{t,d} \cdot \text{idf}_t$; idf đo độ hiếm | Spärck Jones, *J. Documentation* 1972 (idf); IIR §6.2 |
| **IDF trong scikit-learn** (`smooth_idf=True`) | $\text{idf}(t) = \ln\dfrac{1 + N}{1 + \text{df}(t)} + 1$; sau đó chuẩn hóa L2 mỗi hàng | scikit-learn docs, *TfidfTransformer* |
| **n-gram (từ / ký tự)** | Dãy $n$ token liên tiếp. `char_wb` của scikit-learn: n-gram ký tự **trong từng từ**, thêm dấu cách hai đầu | SLP3 ch. 3; scikit-learn docs |
| **Multinomial Naive Bayes** | $P(c \mid d) \propto P(c)\prod_t P(t \mid c)^{\text{tf}_{t,d}}$, làm mượt Laplace với $\alpha$ | McCallum & Nigam, AAAI Workshop 1998 |
| **Baseline mạnh cho phân loại văn bản** | BoW/n-gram + model tuyến tính (LR, SVM, NB-SVM) là baseline khó vượt với dữ liệu ít | Wang & Manning, ACL 2012; Joulin et al., *fastText classification*, EACL 2017 |
| **Feature hashing** | Băm n-gram vào $2^k$ cột, không cần lưu vocabulary | Weinberger et al., ICML 2009 (Tuần 3, phụ lục P1) |

#### 12.2 Công thức

Với văn bản $d$, từ $t$, $N$ văn bản trong tập train:

$$\text{tf}_{t,d} = \text{số lần } t \text{ có trong } d, \qquad \text{idf}_t = \ln\frac{1 + N}{1 + \text{df}_t} + 1, \qquad x_d = \frac{(\text{tf}_{t,d}\,\text{idf}_t)_t}{\big\lVert(\text{tf}_{t,d}\,\text{idf}_t)_t\big\rVert_2}.$$

- **"+1" ở tử và mẫu** (smooth): như thể có thêm một văn bản chứa mọi từ, nên không bao giờ chia cho 0.
- **"+1" ở cuối:** từ có mặt ở **mọi** văn bản vẫn có idf $= 1$ chứ không bằng 0, nên không bị xóa hẳn.
- **Chuẩn hóa L2:** câu dài không tự động có điểm lớn hơn câu ngắn. Tích vô hướng của hai vector đã chuẩn hóa chính là cosine.
- **Từ chưa thấy lúc train** bị bỏ qua lúc transform. Câu toàn từ lạ cho vector 0.

Kết hợp với logistic regression đa lớp: $P(c \mid d) = \mathrm{softmax}(W x_d + b)_c$. Mỗi cột của $W$ là "điểm" của một từ cho từng intent, nên **đọc được** (mục 12.4).

#### 12.3 Ví dụ tính tay

**(a) TF-IDF đủ ba văn bản** (tiếp ví dụ Tầng 1):
- (1) "hủy gói": $(1.288, 1.288, 0, 0)$, chuẩn bằng $1.821$, nên $x = (0.707,\ 0.707,\ 0,\ 0)$.
- (2) "đăng ký gói": $(0.474,\ 0,\ 0.623,\ 0.623)$.
- (3) "hủy hủy": $(0,\ 2.575,\ 0,\ 0)$, chuẩn hóa thành $(0, 1, 0, 0)$. Lặp từ không làm vector "to" hơn, chỉ làm nó **dồn về** từ đó.
- Cosine (1)–(2) $= 0.707 \times 0.474 = 0.335$; cosine (1)–(3) $= 0.707$; cosine (2)–(3) $= 0$.

**(b) Số chiều.** Vocabulary $V$ từ, câu dài $L$ từ: vector BoW có $V$ chiều nhưng tối đa $L$ ô khác 0. Với $V = 3\,988$ (unigram + bigram của dữ liệu) và trung bình 15.6 ô khác 0 mỗi câu, độ thưa là 99.6%.

**(c) One-hot không đo được độ giống.** Với one-hot, $\cos(\text{"hủy"}, \text{"huỷ"}) = 0$ và $\cos(\text{"hủy"}, \text{"nạp"}) = 0$: mọi cặp từ khác nhau đều "xa" như nhau. Chuẩn hóa (bài 11) gộp được "hủy"/"huỷ", nhưng không gộp được "hủy"/"cắt"/"ngừng".

#### 12.4 Đo trên máy bạn

**(a) Kích thước biểu diễn** (4 800 tin nhắn train, sau `normalize`):

| Biểu diễn | Vocabulary | Ô khác 0 / câu | Mật độ | Bộ nhớ CSR | Nếu lưu dense (float64) |
|---|---|---|---|---|---|
| Âm tiết, 1-gram | 701 | 8.2 | 1.18% | 0.5 MB | 27 MB |
| Âm tiết, 1–2 gram | 3 988 | 15.6 | 0.39% | 0.9 MB | 153 MB |
| Ký tự `char_wb`, 2–5 gram | 4 629 | 87.7 | 1.90% | 5.1 MB | 178 MB |

**(b) Chọn n-gram** (logistic regression, $C = 10$; macro-F1 theo template, trung bình 5 split):

| Đặc trưng | F1 |
|---|---|
| Âm tiết 1-gram | **0.735** |
| Âm tiết 1–2 gram | 0.733 |
| Âm tiết 1–3 gram | 0.725 |
| Ký tự 1–3 gram | 0.690 |
| Ký tự 2–5 gram | 0.705 |
| Ký tự 3–6 gram | 0.687 |
| *Multinomial NB, âm tiết 1–2 gram* | 0.60–0.62 |

Trên chia ngẫu nhiên, mọi cấu hình đều đạt 0.976–0.983, sát trần 0.986.

**(c) Bigram sửa được một câu, nhưng không giúp trung bình.** Hai model train trên cùng dữ liệu:

| Câu thử | Chỉ 1-gram | 1–2 gram |
|---|---|---|
| "hủy đăng ký gói GX30" | hủy gói 0.98 ✓ | hủy gói 0.97 ✓ |
| "đăng ký lại gói vừa hủy" | **hủy gói 0.91** ✗ | **đăng ký gói 0.79** ✓ |

Bigram `đăng ký`, `ký lại` lấn được từ "hủy". Nhưng tính trung bình, 1–2 gram **không hơn** 1-gram (−0.0025, thắng 3/5 split): với ít dữ liệu, mỗi bigram xuất hiện quá ít lần để model học được trọng số tin cậy.

**(d) Model học được gì?** Các đặc trưng có hệ số lớn nhất (1–2 gram):

| Intent | Đặc trưng mạnh nhất |
|---|---|
| hủy gói | `hủy`, `huy`, `gói`, `goi`, `nữa`, `hủy gói`, `hủy đăng` |
| khiếu nại | `trừ`, `sao`, `tru`, `sai`, `trừ tiền`, `bị trừ`, `nại` |
| khác (ngoài phạm vi) | **`15`**, `bạn`, `nay`, `trang <url>`, `<url>`, `buồn` |

Lớp "khác" dựa vào **`15`**, lấy từ đúng một template ("tính giúp tôi 15 nhân 7"). Model học thuộc **dấu vết của template**, không học được khái niệm "ngoài phạm vi". Đó là lý do F1 tụt mạnh khi test bằng template mới.

**Tóm tắt:** khoảng cách giữa chia ngẫu nhiên (0.98) và chia theo template (0.73) lớn hơn rất nhiều so với mọi khác biệt giữa các cách tiền xử lý hay chọn n-gram. **Cách đánh giá quan trọng hơn cách chọn đặc trưng.**

#### 12.5 Code chuẩn

```python
import functools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def build_tfidf(kind="char", strip=False, C=10.0):
    prep = functools.partial(normalize, strip=strip)                  # picklable, khác lambda
    vec = (TfidfVectorizer(preprocessor=prep, token_pattern=r"<\w+>|\w+", ngram_range=(1, 2)) if kind == "word"
           else TfidfVectorizer(preprocessor=prep, analyzer="char_wb", ngram_range=(2, 5)))
    return Pipeline([("tfidf", vec), ("clf", LogisticRegression(C=C, max_iter=3000))])

model = build_tfidf("word").fit(train.text, train.intent)
proba = model.predict_proba(test.text)                                # cột theo model.classes_
```

**Lý do thiết kế:**
- **Tiền xử lý nằm trong vectorizer, vectorizer nằm trong Pipeline:** `fit` học vocabulary và idf **chỉ trên train**; `joblib.dump(model)` lưu cả hai, nên lúc phục vụ không thể quên bước nào.
- **`token_pattern=r"<\w+>|\w+"`:** mặc định của scikit-learn bỏ token 1 ký tự và tách `<phone>` thành `phone`.
- **`functools.partial` thay cho lambda:** lambda không pickle được (Tuần 4, phụ lục P2).

#### 12.6 Cạm bẫy

1. **Fit vectorizer trên toàn bộ dữ liệu** (kể cả test): idf và vocabulary "nhìn thấy" test. Rò rỉ nhẹ, nhưng là lỗi quy trình.
2. **Chỉ đánh giá bằng chia ngẫu nhiên** khi dữ liệu có nhiều câu gần trùng nhau (template, câu được copy, hội thoại của cùng một người): 0.98 so với 0.73 ở đây.
3. **Đọc hệ số như nhân quả:** `15` là đặc trưng mạnh nhất của "khác", nhưng không có ý nghĩa nào. Hệ số cho biết model **dựa vào đâu**, giúp phát hiện dấu vết lạ trong dữ liệu.
4. **Dùng `CountVectorizer` mặc định với tiếng Việt:** `token_pattern` mặc định bỏ token 1 ký tự ("k", "j", "t" là teencode quan trọng).
5. **Thêm n-gram dài theo thói quen:** 1–3 gram tệ hơn 1-gram trên template (0.725 so với 0.735). Vocabulary tăng mà dữ liệu không tăng.
6. **Dense hóa ma trận TF-IDF** (`.toarray()`) với vocabulary lớn: 178 MB thay vì 5 MB ở đây; với triệu văn bản thì hết RAM.

**Sắc thái phỏng vấn:** *"Bạn có một tuần và 5 000 tin nhắn có nhãn. Model phân loại đầu tiên là gì?"* TF-IDF (âm tiết + ký tự) + logistic regression: train vài giây, giải thích được, là baseline để mọi model phức tạp hơn phải vượt. Nói thêm cách đánh giá (chia theo template hoặc theo thời gian), và nói rằng model pretrained (bài 18) đáng thử khi cần khái quát sang cách diễn đạt mới.

#### 12.7 Đọc thêm

- C. Manning, P. Raghavan, H. Schütze, *Introduction to Information Retrieval* (2008), chương 6 (TF-IDF, vector space) và chương 13 (Naive Bayes).
- S. Wang & C. Manning, *Baselines and Bigrams: Simple, Good Sentiment and Topic Classification*, ACL 2012.
- A. Joulin et al., *Bag of Tricks for Efficient Text Classification*, EACL 2017.
- scikit-learn User Guide: *Text feature extraction*; ví dụ *Classification of text documents using sparse features*.

---

## 13 · Word embeddings: word2vec, GloVe, FastText

> ⏱ Tầng 1: 15' · Tầng 2: 45'
> **Cần biết trước:** `nn.Embedding` là bảng tra, tương đương one-hot nhân ma trận (Tuần 4, bài 03); sigmoid, BCE (Tuần 2, bài 05); cosine (Tuần 2, bài 01).

> **📖 Từ mới**
> - **Embedding (vector từ):** vector số thực **dày** (vài chục đến vài trăm chiều) đại diện cho một từ; từ gần nghĩa thì vector gần nhau.
> - **Giả thuyết phân bố (distributional hypothesis):** từ xuất hiện trong **ngữ cảnh giống nhau** thì có nghĩa giống nhau.
> - **Cửa sổ ngữ cảnh (context window):** các từ cách từ đang xét tối đa $m$ vị trí.
> - **Từ trung tâm / từ ngữ cảnh (center / context word):** từ đang xét / một từ trong cửa sổ quanh nó.
> - **Negative sampling:** thay vì so với **cả** vocabulary, chỉ so với vài từ "nhiễu" được rút ngẫu nhiên.
> - **OOV (out-of-vocabulary):** từ chưa từng thấy lúc train.
> - **Embedding tĩnh / theo ngữ cảnh:** một vector cố định cho mỗi từ / vector thay đổi theo câu chứa từ đó.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Bạn gặp một từ lạ "bít" trong ba câu: "ko **bít** sao", "mình **bít** rồi", "chưa **bít** gì cả". Dù không ai giải thích, bạn đoán được nó là "biết", vì nó đứng **đúng chỗ** mà "biết" thường đứng. Word2vec làm đúng như thế ở quy mô lớn: từ nào hay có cùng "hàng xóm" thì được đặt gần nhau trong không gian vector.

**Ví dụ có số (đo trên máy bạn).** Train word2vec trên 40 000 tin nhắn **chưa** mở rộng teencode, rồi hỏi "từ nào gần nhất":

| Từ | 5 hàng xóm gần nhất (cosine) |
|---|---|
| không | hông 0.92 · kông 0.87 · khnôg 0.86 · khôg 0.84 · hkông 0.79 |
| được | đk 0.91 · đc 0.88 · ưđợc 0.85 · đợưc 0.84 · nhé 0.81 |
| tôi | t 0.80 · tui 0.80 · mnìh 0.60 · … · mình 0.55 |
| mạng | ạmng 0.91 · mạg 0.89 · mng 0.74 · … |

Model **không được cho bảng teencode nào**, nhưng tự tìm ra "đk" = "đc" = "được" và các lỗi gõ của "không".

**Ba ý cốt lõi:**
1. Embedding là một **bảng tra** $V \times d$ được học, sao cho vector của một từ **dự đoán tốt** các từ quanh nó.
2. "Gần nhau" nghĩa là **thay thế được cho nhau trong câu**, không nhất thiết là đồng nghĩa: "hủy" và "đăng" cũng gần nhau, vì cả hai đứng trước "gói".
3. FastText ghép từ các **n-gram ký tự**, nên có vector cho cả từ chưa thấy. Embedding tĩnh không phân biệt được "gói cước" với "gói quà"; embedding theo ngữ cảnh (bài 18) thì có.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 13.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **Distributional hypothesis** | "You shall know a word by the company it keeps" | Harris, *Word* 1954; Firth 1957 |
| **Skip-gram** | Từ trung tâm dự đoán từng từ ngữ cảnh | Mikolov et al., *Efficient Estimation of Word Representations*, ICLR Workshop 2013 |
| **CBOW** | Trung bình các vector ngữ cảnh dự đoán từ trung tâm; nhanh hơn, tốt cho từ phổ biến | như trên |
| **Negative sampling (SGNS)** | Phân biệt cặp (trung tâm, ngữ cảnh) thật với $k$ cặp nhiễu; nhiễu rút theo $P(w) \propto \text{count}(w)^{0.75}$ | Mikolov et al., *Distributed Representations of Words and Phrases*, NeurIPS 2013 |
| **Subsampling từ phổ biến** | Bỏ ngẫu nhiên từ có tần suất $f$ với xác suất $1 - \sqrt{t/f}$ | như trên |
| **GloVe** | Hồi quy có trọng số trên log của ma trận đồng xuất hiện: $\sum_{ij} f(X_{ij})(w_i^\top \tilde w_j + b_i + \tilde b_j - \log X_{ij})^2$ | Pennington, Socher, Manning, EMNLP 2014 |
| **FastText** | Vector từ = tổng vector của các n-gram ký tự (3–6) của `<từ>`; có vector cho từ OOV | Bojanowski et al., TACL 2017 |
| **Analogy (3CosAdd)** | Tìm $x$ gần nhất với $v_b - v_a + v_c$ ("king − man + woman ≈ queen") | Mikolov, Yih, Zweig, NAACL 2013 |
| **SGNS ≈ phân tích ma trận PMI** | SGNS ngầm phân tích ma trận PMI dịch đi $\log k$ | Levy & Goldberg, NeurIPS 2014 |
| **Contextual embeddings** | Vector của từ phụ thuộc cả câu (ELMo, BERT) | Peters et al., NAACL 2018; Devlin et al., NAACL 2019 |

#### 13.2 🐢 Đi chậm từng bước: skip-gram với negative sampling

*Bước 1 · Tạo cặp huấn luyện.* Câu "tôi muốn hủy gói data", cửa sổ $m = 2$. Với từ trung tâm "hủy" (vị trí 3), các từ ngữ cảnh là các từ ở vị trí 1, 2, 4, 5:

| Trung tâm | Ngữ cảnh | Cặp |
|---|---|---|
| hủy | tôi, muốn, gói, data | (hủy, tôi), (hủy, muốn), (hủy, gói), (hủy, data) |
| tôi (vị trí 1) | muốn, hủy | (tôi, muốn), (tôi, hủy) |

Một câu 5 từ với $m = 2$ cho $2 + 3 + 4 + 3 + 2 = 14$ cặp. 40 000 tin nhắn của ta cho **1 085 084** cặp.

*Bước 2 · Hai bảng vector.* Mỗi từ có **hai** vector: $v_w$ khi nó là trung tâm, $u_w$ khi nó là ngữ cảnh. Điểm "hợp nhau" của một cặp là tích vô hướng $u_o^\top v_c$. Sau khi train, người ta thường chỉ giữ bảng $v$.

*Bước 3 · Bài toán gốc quá đắt.* Skip-gram gốc dùng softmax trên **cả** vocabulary: $P(o \mid c) = \dfrac{e^{u_o^\top v_c}}{\sum_{w=1}^{V} e^{u_w^\top v_c}}$. Mẫu số cộng $V$ số hạng cho **mỗi** cặp; với $V = 10^6$ thì quá chậm.

*Bước 4 · Đổi thành phân loại nhị phân.* Thay vì "$o$ là từ nào trong $V$ từ", ta hỏi "cặp $(c, o)$ này là **thật** hay **giả**?". Với mỗi cặp thật, rút $k$ từ nhiễu $n_1..n_k$ rồi tối thiểu hóa
$$\ell = -\log\sigma(u_o^\top v_c) - \sum_{i=1}^{k}\log\sigma(-u_{n_i}^\top v_c).$$
Số hạng đầu kéo $v_c$ **lại gần** $u_o$; các số hạng sau đẩy $v_c$ **ra xa** các $u_{n_i}$. Chi phí mỗi cặp là $k + 1$ tích vô hướng thay vì $V$.

*Bước 5 · Một con số.* Giả sử $u_o^\top v_c = 2$ (cặp thật) và hai từ nhiễu có tích vô hướng $0$ và $-1$:
$$\ell = -\log\sigma(2) - \log\sigma(0) - \log\sigma(1) = 0.127 + 0.693 + 0.313 = 1.133.$$
Từ nhiễu thứ nhất ($u^\top v = 0$, model "phân vân") đóng góp nhiều nhất. Gradient sẽ đẩy tích vô hướng của nó xuống âm.

*Bước 6 · Rút nhiễu theo lũy thừa 0.75.* Ba từ có số lần xuất hiện 100, 10, 1:

| | Theo tần suất | Theo tần suất$^{0.75}$ |
|---|---|---|
| Từ 100 lần | 0.901 | 0.827 |
| Từ 10 lần | 0.090 | 0.147 |
| Từ 1 lần | 0.009 | 0.026 |

Lũy thừa 0.75 **nâng** xác suất được chọn làm nhiễu của từ hiếm, nên từ hiếm cũng được "đẩy ra" đủ nhiều. Con số 0.75 được chọn bằng thực nghiệm.

**Vì sao điều này làm từ đồng nghĩa gần nhau (phác thảo).** "đk" và "đc" có cùng các từ ngữ cảnh ("ko", "à", "không"). Cả $v_\text{đk}$ lẫn $v_\text{đc}$ đều bị kéo về gần **cùng** các vector $u$ đó, nên chúng gần nhau. Levy & Goldberg (2014) chứng minh SGNS tương đương phân tích ma trận PMI (pointwise mutual information) giữa từ và ngữ cảnh.

#### 13.3 Ví dụ tính tay

**(a) Cosine.** $a = (1, 2, 0)$, $b = (2, 3, 1)$: $a \cdot b = 8$, $\lVert a\rVert = \sqrt5$, $\lVert b\rVert = \sqrt{14}$, nên $\cos = 8/\sqrt{70} =$ **0.956**.

**(b) Analogy với vector đồ chơi** (trục 1 = "là gói cước", trục 2 = "là hành động hủy"):

| Từ | Vector |
|---|---|
| đăng_ký | (0, 0) |
| hủy | (0, 1) |
| gói | (1, 0) |
| hủy_gói | (1, 1) |
| gói_quà | (1, −0.2) |

$v_\text{gói} - v_\text{đăng\_ký} + v_\text{hủy} = (1, 1)$, trùng hẳn với `hủy_gói` (cosine 1.0). Với embedding thật thì chỉ gần đúng, và quy ước là **loại** ba từ trong câu hỏi khỏi tập ứng viên; nếu không, đáp án thường lại chính là $c$.

**(c) FastText cho từ OOV.** Từ "nạpp" (gõ thừa chữ p) chưa từng thấy. Các 3-gram của `<nạpp>`: `<nạ`, `nạp`, `ạpp`, `pp>`. Hai trong bốn 3-gram (`<nạ`, `nạp`) trùng với `<nạp>`. Vector của "nạpp" là trung bình các vector n-gram, nên nằm gần "nạp".

#### 13.4 Đo trên máy bạn

Corpus 40 000 tin nhắn (sau `normalize(teencode=False)`), 332 027 token; giữ các từ xuất hiện ≥ 5 lần, được 660 từ; cửa sổ 2, $d = 50$, $k = 5$, Adam, 2 epoch. Loss giảm từ 1.91 xuống 1.15, mất vài giây.

**(a) Hàng xóm gần nhất:** bảng ở Tầng 1. Thêm hai dòng:

| Từ | 5 hàng xóm gần nhất |
|---|---|
| hủy | đăg (lỗi gõ của "đăng") 0.69 · gúp 0.67 · các 0.67 · **bỏ** 0.67 · ăđng 0.67 |
| gói | ăđng 0.62 · táng 0.61 · **hủy** 0.60 · thế 0.60 |

- **Trái nghĩa nằm gần nhau:** "hủy" và "đăng ký" cùng đứng trước "gói", nên rất gần nhau. Với bài toán phân loại intent, đây là **điểm yếu chết người** nếu chỉ dùng trung bình embedding tĩnh làm đặc trưng.
- "bỏ" (bỏ gói) cũng gần "hủy": đúng nghĩa.

**(b) FastText cho từ chưa thấy** (cùng corpus, vector từ = trung bình vector n-gram ký tự 3–5 cộng chính từ đó, băm vào 20 000 ô):

| Từ OOV | Hàng xóm gần nhất |
|---|---|
| khôngg | không 1.00 · hông 0.98 · khôg 0.93 |
| mạngg | mạng 0.98 · mạg 0.90 · ạmng 0.88 |
| đăngg | đăng 0.99 · đăg 0.90 · đnăg 0.89 |
| hủyy | hủy 0.87 · ăđng 0.70 · đăng 0.69 |

Word2vec thường không có vector nào cho các từ này.

#### 13.5 Code chuẩn (SGNS trong PyTorch, rút gọn)

```python
import torch
from torch import nn
import torch.nn.functional as F

class SGNS(nn.Module):
    def __init__(self, V: int, d: int = 50):
        super().__init__()
        self.inp = nn.Embedding(V, d)          # v: vector khi là từ trung tâm (giữ lại sau khi train)
        self.out = nn.Embedding(V, d)          # u: vector khi là từ ngữ cảnh
        nn.init.uniform_(self.inp.weight, -0.5 / d, 0.5 / d)
        nn.init.zeros_(self.out.weight)

    def forward(self, center, context, negatives):            # (B,), (B,), (B, k)
        v = self.inp(center)                                   # (B, d)
        pos = F.logsigmoid((v * self.out(context)).sum(-1))    # log σ(u_o·v)
        neg = F.logsigmoid(-(self.out(negatives) @ v.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return -(pos + neg).mean()

noise = counts.float() ** 0.75
noise /= noise.sum()
negatives = torch.multinomial(noise, B * k, replacement=True).view(B, k)
```

Trong thực tế, dùng `gensim` (`Word2Vec`, `FastText`, chưa cài trên máy bạn), hoặc vector FastText đã train sẵn cho tiếng Việt (Grave et al., LREC 2018; cần tải).

#### 13.6 Cạm bẫy

1. **Coi "gần" là "đồng nghĩa":** trái nghĩa và cùng loại ("hủy"/"đăng ký", "tăng"/"giảm") nằm rất gần nhau.
2. **Embedding tĩnh cho từ nhiều nghĩa:** "gói" (gói cước / gói quà) chỉ có **một** vector, là trung bình của mọi nghĩa.
3. **Đánh giá embedding chỉ bằng analogy:** kết quả analogy phụ thuộc nhiều vào cách loại từ và vào vài phép tính "đẹp"; nó tương quan yếu với chất lượng ở bài toán thật (Faruqui et al., RepEval 2016). Hãy đánh giá ở bài toán bạn cần.
4. **Trung bình embedding để biểu diễn câu:** mất thứ tự và phủ định; từ phổ biến lấn át. Đây là lý do cần model theo ngữ cảnh.
5. **Từ hiếm có vector tệ** (vài lần xuất hiện là quá ít): đặt `min_count`, hoặc dùng FastText.
6. **Embedding train trên miền khác:** vector từ báo chí không biết "đk" hay "GX30". Train thêm trên dữ liệu miền của bạn, hoặc dùng subword.

**Sắc thái phỏng vấn:**
- *"Word2vec học như thế nào?"* Skip-gram với negative sampling: tạo cặp từ trong cửa sổ, phân biệt cặp thật với cặp nhiễu (lũy thừa 0.75), hai bảng vector. Nêu được "vì sao gần nhau thì giống nghĩa" bằng giả thuyết phân bố.
- *"Word2vec khác BERT thế nào?"* Tĩnh so với theo ngữ cảnh; một vector mỗi từ so với một vector mỗi **lần xuất hiện**; mức từ so với subword; mạng nông so với Transformer sâu.

#### 13.7 Đọc thêm

- T. Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality*, NeurIPS 2013.
- J. Pennington, R. Socher, C. Manning, *GloVe: Global Vectors for Word Representation*, EMNLP 2014.
- P. Bojanowski, E. Grave, A. Joulin, T. Mikolov, *Enriching Word Vectors with Subword Information*, TACL 2017.
- O. Levy & Y. Goldberg, *Neural Word Embedding as Implicit Matrix Factorization*, NeurIPS 2014.
- Y. Goldberg & O. Levy, *word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method*, arXiv 2014.
- SLP3, chương 6 *Vector Semantics and Embeddings*.

---

## Bài tập Ngày 1 · `intent/textprep.py` (~2h)

Mảnh đầu tiên của project ([project.md](project.md)): tiền xử lý và TF-IDF tự viết. Lệnh test: `python -m pytest -q tests/test_textprep.py` (9 test).

**Dữ liệu cho sẵn** (chép vào đầu file):

```python
TEENCODE = {"ko": "không", "k": "không", "hông": "không", "hok": "không", "dc": "được", "đc": "được", "đk": "được",
            "j": "gì", "r": "rồi", "rùi": "rồi", "v": "vậy", "mk": "mình", "mik": "mình", "t": "tôi", "tui": "tôi",
            "bn": "bao nhiêu", "bnh": "bao nhiêu", "vs": "với", "nha": "nhé", "nhe": "nhé", "ntn": "thế nào",
            "nv": "nhân viên", "tk": "tài khoản", "đt": "điện thoại", "bít": "biết", "júp": "giúp", "xiền": "tiền"}
TONE_FIX = {"uỷ": "ủy", "uý": "úy", "uỳ": "ùy", "uỹ": "ũy", "uỵ": "ụy", "oà": "òa", "oá": "óa", "oả": "ỏa",
            "oã": "õa", "oạ": "ọa", "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe"}
```

| Hàm / lớp | Yêu cầu |
|---|---|
| `strip_accents(text)` | Đổi `đ`/`Đ` → `d`/`D`; NFD; bỏ mọi ký tự có `unicodedata.category(c) == "Mn"`; NFC lại |
| `normalize(text, *, strip=False, teencode=True)` | Đúng thứ tự ở mục 11.2: NFC → chữ thường → URL, email, SĐT, tiền thành `<url>`, `<email>`, `<phone>`, `<money>` (có dấu cách hai bên) → `TONE_FIX` → lấy token bằng regex `<\w+>\|\w+` → (teencode) tra `TEENCODE` từng token → nối bằng một dấu cách → (strip) `strip_accents` |
| `syllables(text)` | Danh sách token theo regex `<\w+>\|\w+` |
| `char_ngrams(text, n_min=2, n_max=5)` | **Giống hệt** `analyzer="char_wb"` của scikit-learn: với mỗi từ `w`, đặt `w = " " + w + " "`; với từng $n$ từ `n_min` tới `n_max`: nếu `len(w) <= n` thì thêm cả `w` **một lần** rồi dừng (bỏ các $n$ lớn hơn); nếu không thì thêm mọi `w[i:i+n]` |
| `word_ngrams(tokens, n_min=1, n_max=2)` | Mọi n-gram từ, nối bằng một dấu cách, theo thứ tự: hết các 1-gram rồi tới các 2-gram |
| `Tfidf(analyzer, min_df=1)` | `.fit(docs)`: df = số văn bản chứa token; giữ token có df ≥ `min_df`; `vocabulary_` = dict token → chỉ số, **sắp theo thứ tự chuỗi**; `idf_` (mảng NumPy) theo công thức 12.2. `.transform(docs)`: `scipy.sparse.csr_matrix`, tf = số lần đếm (bỏ token lạ), nhân idf, chuẩn hóa L2 từng hàng (hàng toàn 0 giữ nguyên). `.fit_transform` |

**Regex gợi ý** (các test kiểm tra đúng những dạng này):

```python
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"(\+84|0)(\d[\s.]?){8,10}\d?")              # 0912345678, 0868 123 456, +84912000111
_MONEY = re.compile(r"\b\d+([.,]\d{3})*\s*(k|nghìn|ngàn|đ|đồng|vnd)\b")  # 50k, 200.000đ, 100 nghìn
```

**Gợi ý:**
- Hai test so với scikit-learn (`test_char_ngrams_match_sklearn_char_wb`, `test_tfidf_matches_sklearn`) là cách kiểm tra mạnh nhất. Nếu lệch, in ra một câu cụ thể để so.
- `Tfidf.transform`: dùng `collections.Counter` cho từng văn bản, rồi dựng `csr_matrix((vals, (rows, cols)), shape=...)`.
