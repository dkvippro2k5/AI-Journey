# Quiz · Ngày 3 · API các nhà cung cấp, streaming, độ tin cậy / chi phí / bảo mật (bài 07–09)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> Làm trên giấy, được dùng máy tính cầm tay. **Không chạy code.** Đáp án ở cuối file. Mọi mức giá là giả định.

## Phần 1 — Quick test (14 câu, câu 14 được 2đ, các câu khác 1đ)

1. Gửi cùng một hội thoại (có system prompt) cho Anthropic Messages API: system prompt đặt ở đâu? Tham số nào là bắt buộc với Anthropic mà không bắt buộc với OpenAI Chat Completions?
2. Đối số của một tool call trong response của OpenAI Chat Completions có kiểu gì? Của Anthropic thì sao? Điều này ảnh hưởng thế nào tới code của bạn?
3. Ảnh 2048 × 1024, gửi cho OpenAI với `detail="high"` (co cho vừa 2048 × 2048, rồi co cạnh ngắn về 768, ô 512 × 512, 170 token/ô + 85). Bao nhiêu token?
4. Một prompt gồm: (A) ngày giờ hiện tại, (B) system prompt cố định 800 token, (C) few-shot cố định 600 token, (D) lịch sử hội thoại, (E) câu hỏi mới. Xếp lại thứ tự để tận dụng prompt caching tốt nhất.
5. Kể hai việc trong một tổng đài nên dùng Batch API, và một việc không được dùng.
6. Luồng SSE `data: dòng 1\ndata:dòng 2\n\n`. Sự kiện nhận được có `data` bằng gì?
7. Chữ "ủ" chiếm bao nhiêu byte UTF-8? Vì sao gọi `chunk.decode("utf-8")` trên từng mảnh nhận được lại gây lỗi với tiếng Việt?
8. Vì sao giao diện chat thường không dùng `EventSource` của trình duyệt để nhận luồng từ backend?
9. Gateway đang stream câu trả lời cho khách thì kết nối tới nhà cung cấp bị đứt sau 40 token. Có nên tự động thử lại không? Vì sao?
10. Token bucket `rate = 5`/s, `capacity = 10`, xô **rỗng** lúc $t = 0$. Lúc $t = 1.5$ s có 8 request cùng tới. Bao nhiêu request được đi qua?
11. Giới hạn "100 request mỗi phút" cài bằng fixed window theo phút đồng hồ. Trong trường hợp xấu nhất, bao nhiêu request có thể đi qua trong khoảng 2 giây?
12. Trong các mã sau, mã nào nên thử lại: 400, 401, 404, 408, 422, 429, 503, 529?
13. Vì sao không nên cache câu trả lời khi `temperature = 0.8`? Vì sao khóa cache phải chứa phiên bản (fingerprint) của prompt?
14. *(2đ)* Key có giới hạn 1 000 RPM và 300 000 TPM. Mỗi request trung bình 600 token vào + 150 token ra. (a) Mỗi phút tối đa bao nhiêu request? Cổ chai là gì? (b) Lúc cao điểm cần 30 request/giây. Cần tối thiểu bao nhiêu key như vậy (nếu không giảm token)?

## Phần 2 — Tự luận (4 câu, 30 phút)

**T1 · Ước lượng chi phí (4đ).** 500 000 request/ngày. Mỗi request: system + few-shot 1 000 token (giống nhau ở mọi request), tin nhắn 100 token, câu trả lời 120 token. Giá (USD / 1M token):
- Model lớn: vào 2.5, ra 10; token trúng cache tính 0.25.
- Model nhỏ: vào 0.15, ra 0.6; không dùng cache.

Câu hỏi:
- (a) Chi phí một ngày nếu gửi hết cho model lớn, không cache.
- (b) Có prompt caching cho phần 1 000 token, tỉ lệ trúng 95% (5% request trả giá vào đầy đủ cho phần đó). Chi phí một ngày?
- (c) Thêm định tuyến: 70% request đi model nhỏ, 30% đi model lớn (có cache như (b)). Chi phí một ngày?
- (d) Đòn bẩy nào lớn nhất? Nêu một rủi ro về **chất lượng** của định tuyến và cách theo dõi rủi ro đó.

**T2 · Lần theo parser SSE (4đ).** Luồng byte (viết `\r`, `\n` cho rõ) là:
```
event: delta\r\ndata: {"t":"hủ"}\r\n\r\n: ping\n\ndata: [DONE]\n\n
```
Mạng giao thành 4 mảnh:
- m1: `event: delta\r`
- m2: `\ndata: {"t":"h` + **2 byte đầu** của "ủ"
- m3: **byte cuối** của "ủ" + `"}\r\n\r`
- m4: `\n: ping\n\ndata: [DONE]\n\n`

Câu hỏi:
- (a) Với parser đúng chuẩn (bài 08), mỗi lần `feed(m1)`, …, `feed(m4)` trả về những sự kiện nào? Giải thích vì sao `feed(m1)` và `feed(m3)` không trả sự kiện nào.
- (b) Đoạn code sau có **ba lỗi**. Chỉ ra và nêu triệu chứng của từng lỗi trên luồng trên.
  ```python
  async for chunk in resp.aiter_bytes():
      text = chunk.decode("utf-8", errors="replace")
      for part in text.split("\n\n"):
          if part.startswith("data: "):
              handle(json.loads(part[6:]))
  ```

**T3 · Chính sách retry và fallback (3đ).** Tổng đài yêu cầu khách nhận được câu trả lời (hoặc thông báo chuyển người) trong **4 giây**. Model chính có p50 0.7 s, p99 2.5 s; thỉnh thoảng trả 429 kèm `Retry-After: 2`, và có những đợt sập kéo dài vài phút. Có một model dự phòng (p50 1.5 s) và một kho câu trả lời mẫu theo intent.
- (a) Đặt timeout mỗi lần gọi, số lần thử, và quy tắc với Retry-After sao cho không vượt 4 giây. Trình bày phép tính trường hợp xấu nhất.
- (b) Vì sao retry không đủ cho đợt sập kéo dài? Cơ chế nào giải quyết (nêu tên và ý tưởng)?
- (c) Liệt kê chuỗi fallback và những gì cần ghi log để biết chất lượng thực tế.

**T4 · Ngân sách context (4đ).** Model có context 8 192 token. Mỗi lượt gửi: system 1 500 token, 3 tài liệu tham khảo mỗi tài liệu 1 200 token, lịch sử 24 lượt hỏi đáp (mỗi lượt khoảng 150 token), câu hỏi mới 50 token, `max_tokens = 800`.
- (a) Tổng số token nếu gửi tất cả. Vượt bao nhiêu?
- (b) Giữ đủ 3 tài liệu: `fit_history` giữ được tối đa bao nhiêu lượt gần nhất?
- (c) Thay phần lịch sử bị bỏ bằng một bản tóm tắt 200 token: còn giữ được bao nhiêu lượt gần nhất? Ưu và nhược so với (b)?
- (d) Khách đã nói số thuê bao ở lượt thứ 2. Làm sao để thông tin đó không bị mất dù lịch sử bị cắt?

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Phần 1**

1. System là trường **`system` ở cấp ngoài** của request, không nằm trong `messages`. **`max_tokens`** bắt buộc với Anthropic.
2. OpenAI: `function.arguments` là **chuỗi JSON** (phải `json.loads`, có thể sai cú pháp). Anthropic: `input` là **object** đã parse. Adapter phải chuẩn hóa về một kiểu, và luôn kiểm tra bằng schema.
3. Đã vừa 2048. Cạnh ngắn 1024 → 768, ảnh thành 1536 × 768. Số ô: $\lceil 1536/512 \rceil \times \lceil 768/512 \rceil = 3 \times 2 = 6$. Token $= 85 + 6 \times 170 =$ **1 105**.
4. **B → C → D → E → A**, hoặc đưa A vào cuối cùng với câu hỏi (B, C, D, A+E). Phần cố định ở đầu; ngày giờ thay đổi mỗi request nên phải nằm sau phần muốn cache. Lịch sử thay đổi theo hội thoại nên nằm sau phần cố định.
5. Nên dùng: gán nhãn intent lại cho log cũ; tóm tắt cuộc gọi cuối ngày; sinh dữ liệu đánh giá; dịch kho FAQ. Không dùng: trả lời chat thời gian thực (kết quả có thể tới sau nhiều giờ).
6. `"dòng 1\ndòng 2"`: hai dòng `data` nối bằng `\n`; `data:dòng 2` (không có dấu cách) vẫn cho "dòng 2".
7. **3 byte** (`E1 BB A7`). Mạng có thể cắt giữa ký tự; giải mã từng mảnh riêng biến các byte dở dang thành `�` (hoặc ném lỗi). Phải dùng incremental decoder giữ byte dở dang cho mảnh sau.
8. `EventSource` chỉ gửi **GET** và không đặt được header tùy ý (như `Authorization`). Chat cần POST thân JSON (hội thoại dài) và xác thực, nên dùng `fetch()` + đọc `ReadableStream` + parser SSE.
9. **Không** tự động thử lại: khách đã thấy 40 token, gọi lại sẽ sinh câu trả lời mới (khác) và chữ bị lặp hoặc lệch. Nên báo lỗi/ghi nhận, cho khách nút "thử lại", hoặc thay bằng thông báo chuyển người.
10. Lúc 1.5 s có $\min(10, 0 + 1.5 \times 5) = 7.5$ vé, nên **7** request đi qua (request thứ 8 cần thêm 0.5 vé).
11. **200**: 100 request ở giây cuối của phút trước, 100 request ở giây đầu của phút sau.
12. Thử lại: **408, 429, 503, 529** (429 thì chờ đúng Retry-After). Không thử lại: 400, 401, 404, 422 (do request, gửi lại vẫn lỗi).
13. Với `temperature = 0.8`, cùng một prompt được kỳ vọng cho các câu trả lời khác nhau; cache làm mọi người nhận cùng một câu, mất tính đa dạng mà người dùng muốn. Khóa phải chứa fingerprint của prompt vì đổi prompt (sửa chỉ dẫn, thêm few-shot) thì câu trả lời đúng cũng đổi; thiếu nó, cache trả câu trả lời của phiên bản cũ.
14. *(2đ)* (a) Theo TPM: $300\,000/750 = 400$ request/phút; theo RPM: 1 000. **Tối đa 400 request/phút, cổ chai là TPM** (1đ). (b) Cần $30 \times 60 = 1\,800$ request/phút, tức $1\,800/400 = 4.5$, nên cần **5 key** (1đ) (hoặc giảm token mỗi request, hoặc xin tier cao hơn).

**T1 (4đ)**

(a) *(1đ)* Mỗi request: $1\,100 \times 2.5 + 120 \times 10 = 2\,750 + 1\,200 = 3\,950$ micro-USD. Một ngày: $3\,950 \times 500\,000 / 10^6 =$ **1 975 USD**.

(b) *(1đ)* Phần 1 000 token: $0.95 \times 1\,000 \times 0.25 + 0.05 \times 1\,000 \times 2.5 = 237.5 + 125 = 362.5$. Tin nhắn: $100 \times 2.5 = 250$. Đầu ra: 1 200. Tổng **1 812.5** micro-USD/request, tức **906 USD/ngày** (−54%).

(c) *(1đ)* Model nhỏ: $1\,100 \times 0.15 + 120 \times 0.6 = 165 + 72 = 237$ micro-USD. Trung bình: $0.7 \times 237 + 0.3 \times 1\,812.5 = 165.9 + 543.75 = 709.65$. Một ngày: **355 USD** (−82% so với (a)).

(d) *(1đ)* Lớn nhất ở đây là **prompt caching** (−54%), tiếp đến định tuyến (thêm −61% từ mức đã cache); cả hai cộng lại −82%. Rủi ro chất lượng: model nhỏ trả lời sai ở những intent khó, hoặc router gửi nhầm tin khó sang model nhỏ. Theo dõi: chạy cùng bộ đánh giá cố định cho cả hai tier theo **từng intent**; lấy mẫu một phần lưu lượng của model nhỏ gửi song song lên model lớn (shadow) để so sánh; theo dõi tỉ lệ 👎 và tỉ lệ chuyển người theo tier.

**T2 (4đ)**

(a) *(2.5đ)*
- `feed(m1)` → **không sự kiện**: bộ đệm kết thúc bằng `\r`, chưa biết đó là dòng kết thúc bằng CR hay nửa đầu của CRLF, nên phải chờ.
- `feed(m2)` → **không sự kiện**: `\n` hoàn tất dòng `event: delta` (đặt loại sự kiện là "delta"); phần `data: {"t":"h` + 2 byte chưa có ký tự xuống dòng; incremental decoder giữ 2 byte của "ủ".
- `feed(m3)` → **không sự kiện**: byte cuối ghép với 2 byte trước thành "ủ"; dòng `data: {"t":"hủ"}` hoàn tất (thêm vào data); lại có `\r` ở cuối nên chờ.
- `feed(m4)` → **2 sự kiện**: `\n` hoàn tất CRLF thành **dòng trống**, phát `SSEEvent(event="delta", data='{"t":"hủ"}')`; `: ping` là chú thích, dòng trống sau nó không phát gì (không có data); `data: [DONE]` + dòng trống phát `SSEEvent(event="message", data="[DONE]")`.

(b) *(1.5đ, mỗi lỗi 0.5đ)*
1. **Giải mã từng mảnh:** "ủ" bị cắt giữa m2 và m3 thành `�`.
2. **Tách `\n\n` trên từng mảnh, không có bộ đệm:** sự kiện nằm vắt qua nhiều mảnh (ở đây mọi sự kiện đều vắt qua) bị mất hoặc cắt cụt, nên `json.loads` lỗi.
3. **Chỉ xử lý `\n\n`, không xử lý `\r\n`:** sự kiện đầu dùng CRLF nên không bao giờ khớp `\n\n` đúng chỗ; dòng `event:` bị bỏ qua (không nhận loại sự kiện); và `[DONE]` bị đưa vào `json.loads`, gây lỗi.

**T3 (3đ)**

(a) *(1.5đ)* Một phương án hợp lý:
- Model chính: timeout 1.5 s (trên p50 khá xa, nhưng dưới p99, để không chờ các request "kẹt"); thử tối đa 2 lần.
- Nếu 429 có `Retry-After: 2`: chờ 2 s nữa sẽ vượt ngân sách, nên **không** chờ mà chuyển ngay sang dự phòng.
- Model dự phòng: timeout bằng phần ngân sách còn lại.
- Trường hợp xấu nhất: chính lần 1 timeout 1.5 s + backoff nhỏ (≤ 0.2 s) + chính lần 2 timeout 1.5 s = 3.2 s; còn 0.8 s, không đủ cho dự phòng (p50 1.5 s), nên trả câu trả lời mẫu theo intent hoặc thông báo chuyển người, trong khoảng 3.3 s.
- Phương án khác: chỉ thử chính một lần (1.5 s) rồi sang dự phòng (2.3 s), xấu nhất khoảng 3.8 s.

Điểm cốt lõi: **tổng mọi lần thử ≤ deadline**, và quyết định theo ngân sách còn lại.

(b) *(0.5đ)* Mọi lần thử trong đợt sập đều thất bại; retry chỉ cộng thêm độ trễ (và tải lên nhà cung cấp đang có sự cố). **Circuit breaker:** sau N lỗi liên tiếp, "mở mạch" một khoảng thời gian, gửi thẳng sang dự phòng; sau đó cho một request thử ("nửa mở"), thành công thì đóng mạch lại (Tuần 7).

(c) *(1đ)* Chuỗi: model chính → model dự phòng → câu trả lời trong cache (kể cả hơi cũ) → câu trả lời mẫu theo intent → chuyển tổng đài viên. Log mỗi request: model thực sự đã trả lời, `fallback_used`, số lần thử, mã lỗi từng lần, độ trễ từng bước, route cuối cùng, phiên bản prompt. Đo tỉ lệ 👎 / chuyển người **theo model đã trả lời**, để biết dự phòng kém hơn bao nhiêu.

**T4 (4đ)**

(a) *(1đ)* $1\,500 + 3 \times 1\,200 + 24 \times 150 + 50 + 800 = 9\,550$ token, **vượt 1 358 token**.

(b) *(1đ)* Còn cho lịch sử: $8\,192 - 800 - 1\,500 - 3\,600 - 50 = 2\,242$ token, tức $\lfloor 2\,242/150 \rfloor =$ **14 lượt** gần nhất (bỏ 10 lượt cũ).

(c) *(1đ)* $(2\,242 - 200)/150 = 13.6$, tức **13 lượt** + bản tóm tắt. Ưu: vẫn giữ ý chính của 11 lượt cũ (khách đã thử gì, đã được hướng dẫn gì). Nhược: tốn thêm một lần gọi LLM để tóm tắt; bản tóm tắt có thể bỏ sót hoặc sai; mất 1 lượt gần nhất so với (b).

(d) *(1đ)* Trích thông tin quan trọng (số thuê bao, tên gói, mã khiếu nại) thành **trạng thái có cấu trúc** (slot) ngay khi xuất hiện, lưu ở phía server, và chèn lại vào system prompt (hoặc một khối "Thông tin khách") ở **mỗi** lượt, không phụ thuộc lịch sử còn hay mất. Có thể cân nhắc bớt một tài liệu ít liên quan (giữ được 22 lượt nếu chỉ còn 2 tài liệu).

</details>
