# Ôn phỏng vấn · Tuần 6 · LLM và AI API

> **Dùng khi nào:** sau final, hoặc ôn lại vào các tuần sau (khuyến nghị: lần đầu ngay sau Tuần 6, rồi ôn lại sau 1 tuần và sau 1 tháng).
> **Cách luyện (khoảng 45'):**
> 1. Đọc câu hỏi, **chưa mở đáp án**.
> 2. Hẹn giờ **1 phút**, **nói to** câu trả lời như đang ngồi phỏng vấn (ghi âm càng tốt).
> 3. Mở đáp án, đối chiếu với **"Ý phải có"**. Thiếu ý nào thì đánh dấu, và nói lại lần nữa.
> 4. Câu nào thiếu từ 2 ý trở lên thì đọc lại bài được dẫn ở cuối câu.
>
> **Mẹo chung:** vị trí AI Engineer hỏi rất nhiều về **tích hợp LLM vào sản phẩm**: ước lượng chi phí và tài nguyên, structured output, prompt injection, xử lý lỗi, giảm chi phí. Hãy luôn trả lời kèm **con số** và **đánh đổi**. Câu 2, 7, 9, 13, 14 nên luyện tới mức trả lời trơn tru.

---

## A. Nền tảng LLM

**1. Một model 8B cần bao nhiêu bộ nhớ GPU để phục vụ? Bạn ước lượng thế nào?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Mình tính hai phần: trọng số và KV-cache. Trọng số là số tham số nhân số byte: 8 tỉ tham số ở fp16 là khoảng 16 GB, int4 khoảng 4 GB. KV-cache là $2 \times$ số lớp $\times$ số head KV $\times$ $d_\text{head}$ $\times$ số byte cho mỗi token. Với Llama-3-8B (32 lớp, 8 head KV nhờ GQA, $d_\text{head}$ 128, fp16) là 128 KiB mỗi token, tức 1 GiB cho một request 8K token. 32 request đồng thời là 32 GiB, **lớn hơn cả trọng số**. Nếu không có GQA thì gấp 4 lần. Rồi cộng overhead của framework và activation. Train full fine-tune với Adam thì khoảng 16 byte mỗi tham số, tức hơn 120 GB, nên người ta dùng LoRA.

**Ý phải có:** tham số × byte · công thức KV có GQA · KV theo số request đồng thời có thể vượt trọng số · fp16/int4 · train cần nhiều hơn.
**Nếu bị hỏi tiếp:** *"MoE thì sao?"* → bộ nhớ theo tham số **tổng** (Mixtral 46.7B), compute theo tham số **kích hoạt** (12.9B).
**Xem lại:** bài 01, 02.
</details>

**2. Vì sao token đầu ra đắt hơn token đầu vào? TTFT phụ thuộc vào gì?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Prefill xử lý cả prompt trong một lần forward song song: một lần đọc trọng số được dùng cho hàng nghìn token, nên bị chặn bởi **compute**. Decode sinh từng token, mỗi token lại phải đọc toàn bộ trọng số và KV-cache, nên bị chặn bởi **băng thông bộ nhớ**; mỗi token đầu ra chiếm GPU lâu hơn nhiều. Mình đo trên CPU với model 0.5B: prefill khoảng 170 token/s, decode khoảng 12 token/s, chênh 14 lần; trên GPU còn chênh nhiều hơn. TTFT xấp xỉ overhead cộng thời gian prefill, nên tăng tuyến tính theo độ dài prompt: prompt 4 096 token mất 26 s trên máy mình. Giảm TTFT bằng prompt ngắn hơn, prompt caching, model nhỏ hơn; streaming không giảm TTFT nhưng làm người dùng thấy chữ sớm.

**Ý phải có:** prefill compute-bound, decode memory-bound · số đo · TTFT ∝ độ dài prompt · cách giảm.
**Nếu bị hỏi tiếp:** *"Làm sao tăng thông lượng decode?"* → batching (một lần đọc trọng số cho nhiều request), lượng tử hóa, speculative decoding (Tuần 12).
**Xem lại:** bài 02.
</details>

**3. Context window bị giới hạn bởi gì? Model "128K context" có dùng tốt cả 128K không?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Ba thứ: cách mã hóa vị trí có ngoại suy được không, bộ nhớ KV tăng tuyến tính theo độ dài, và chi phí attention tăng theo bình phương. Kỹ thuật tương ứng là RoPE scaling như position interpolation hay YaRN, hoặc ALiBi; GQA và sliding window để giảm KV; FlashAttention để giảm đọc ghi bộ nhớ. Với Llama-3-8B, ở 128K token phần attention tốn gấp khoảng 4 lần toàn bộ phần ma trận còn lại. Còn "chạy được" 128K không có nghĩa là "dùng tốt": nghiên cứu "lost in the middle" cho thấy thông tin ở giữa context bị dùng kém hơn ở đầu và cuối. Vì vậy mình đặt chỉ dẫn quan trọng ở đầu hoặc cuối, và với RAG thì chỉ đưa vào ít đoạn liên quan nhất thay vì nhồi đầy context.

**Ý phải có:** vị trí, KV, $T^2$ · kỹ thuật tương ứng · lost in the middle · thực hành đặt thông tin.
**Xem lại:** bài 02.
</details>

**4. RLHF gồm những bước nào? DPO khác gì?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** RLHF kiểu InstructGPT có bốn bước. SFT trên câu trả lời mẫu. Thu dữ liệu so sánh: người chấm chọn câu tốt hơn. Train reward model bằng loss Bradley–Terry, $-\log\sigma(r_w - r_l)$. Rồi dùng PPO tối đa hóa điểm của reward model trừ đi một khoản phạt KL so với model SFT, để model không đi xa và không khai thác điểm yếu của reward model. DPO dùng một kết quả là mục tiêu đó có nghiệm dạng đóng, với phần thưởng ngầm là $\beta\log\frac{\pi}{\pi_\text{ref}}$. Thay vào Bradley–Terry được một loss phân loại chỉ cần log-prob của model đang train và model tham chiếu trên cặp câu trả lời. Không cần reward model riêng, không phải sinh mẫu khi train, chỉ cần 2 model trong bộ nhớ thay vì 4. Constitutional AI thì thay phần lớn nhãn của người bằng nhãn do AI tạo theo một bộ nguyên tắc.

**Ý phải có:** 4 bước · Bradley–Terry · phạt KL và lý do · DPO: phần thưởng ngầm, không RM, không RL, 2 model.
**Nếu bị hỏi tiếp:** *"$\beta$ trong DPO là gì?"* → độ chặt khi bám model tham chiếu; $\beta$ lớn thì bám sát hơn.
**Xem lại:** bài 03.
</details>

## B. Prompting và structured output

**5. Bạn viết system prompt cho một chatbot production thế nào?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Mình viết như một bản giao việc. Vai trò và nhiệm vụ trong một hai câu. Quy tắc phải làm và không được làm, mỗi điều cấm đi kèm điều nên làm thay thế, ví dụ câu ngoài phạm vi thì gán intent `khac` và trả lời đúng một câu mẫu. Định dạng đầu ra bằng schema. Vài ví dụ few-shot đa dạng, cân bằng nhãn. Dữ liệu của khách bọc trong thẻ, ghi rõ đó là dữ liệu chứ không phải lệnh. Phần cố định để đầu, phần thay đổi để cuối, cho trúng prompt caching. Prompt có version, có changelog, và mỗi lần sửa phải chạy lại bộ hồi quy. Trên project của mình, prompt với hai ví dụ few-shot dài 452 token; viết gọn system prompt giảm từ 251 xuống 131 token mà vẫn giữ đủ quy tắc.

**Ý phải có:** vai trò/nhiệm vụ/quy tắc/định dạng/ví dụ/dữ liệu · điều cấm kèm điều thay thế · thứ tự cho caching · version + bộ hồi quy.
**Xem lại:** bài 04, 06.
</details>

**6. Chain-of-thought và self-consistency là gì? Khi nào dùng?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** CoT là yêu cầu model viết các bước suy luận trước khi đưa đáp án. Nó giúp ở bài nhiều bước như tính cước có nhiều thành phần, chủ yếu với model đủ lớn. Cái giá là nhiều token đầu ra hơn, chậm và đắt hơn, và phải tách đáp án ra khỏi phần suy luận; trong schema thì đặt `reasoning` **trước** `answer`. Self-consistency lấy mẫu vài chuỗi CoT ở nhiệt độ lớn hơn 0 rồi bỏ phiếu đáp án cuối. Nó giúp khi các lỗi phân tán. Nếu model sai có hệ thống thì bỏ phiếu còn làm tệ đi: với hai lựa chọn và mỗi mẫu đúng 40%, đa số của 5 mẫu chỉ đúng 32%. Chi phí gấp $k$ lần. Mình không dùng CoT cho phân loại hay trích xuất đơn giản, và không dùng khi cần độ trễ thấp.

**Ý phải có:** định nghĩa · lợi ích theo loại bài và cỡ model · chi phí token · điều kiện để bỏ phiếu có ích · thứ tự trường trong schema.
**Xem lại:** bài 04.
</details>

**7. Làm sao để LLM luôn trả JSON đúng định dạng?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Có ba mức bảo đảm. Chỉ dẫn trong prompt thì không bảo đảm gì. JSON mode bảo đảm đúng cú pháp nhưng không bảo đảm đúng khóa hay enum. Structured outputs với schema strict, hoặc sinh có ràng buộc khi tự host, bảo đảm đúng schema bằng cách che logit của mọi token làm đầu ra sai ở mỗi bước. Nhưng không mức nào bảo đảm **nội dung** đúng, nên mình luôn kiểm tra bằng Pydantic ở phía mình. Khi sai, sửa cục bộ những lỗi an toàn như khối ```json hay dấu phẩy thừa; rồi hỏi lại đúng một lần kèm thông báo lỗi; rồi fallback. Trong project, bật vòng hỏi lại đưa tỉ lệ hợp lệ từ khoảng 94–95% lên 99.8–99.9%, tốn thêm 4–5% lần gọi. Mình tự viết sinh có ràng buộc cho một model 0.5B: số token phải sinh giảm 5 lần, intent bị ép vào đúng 9 nhãn. Nhưng chỉ 52% đúng schema, vì grammar chỉ ép kiểu chứ không ép `tien ≥ 1000`. Còn 1/60 JSON hỏng vì grammar của mình quên luật số không được bắt đầu bằng 0. Ràng buộc chỉ tốt bằng grammar, và không sửa được năng lực model.

**Ý phải có:** ba mức · constrained decoding là gì · luôn validate · sửa cục bộ → hỏi lại 1 lần → fallback · đúng schema ≠ đúng nội dung.
**Xem lại:** bài 05.
</details>

**8. Function calling hoạt động thế nào? Rủi ro là gì?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Mình khai báo các tool bằng tên, mô tả và JSON Schema của đối số. Model không chạy gì cả, nó chỉ trả về một yêu cầu gọi có cấu trúc: tên tool và đối số. Với OpenAI, đối số là một chuỗi JSON phải parse; với Anthropic là một block `tool_use` chứa object. Ứng dụng kiểm tra đối số bằng schema, kiểm tra quyền, thực thi, rồi gửi kết quả lại bằng message vai trò `tool` kèm `tool_call_id`, và lặp tới khi model trả lời. Có thể có nhiều lời gọi song song; khi stream thì đối số đến từng mẩu, phải nối theo index rồi mới parse. Rủi ro lớn nhất là model chọn tool và đối số dựa trên văn bản, kể cả văn bản do kẻ tấn công đưa vào. Nên hành động có tác dụng phụ phải có lớp kiểm tra quyền phía server, chỉ áp dụng cho chính khách của phiên, và cần xác nhận.

**Ý phải có:** khai báo bằng JSON Schema · model không thực thi · vòng gọi/kết quả · khác biệt OpenAI/Anthropic · kiểm tra quyền và xác nhận.
**Xem lại:** bài 05, 07.
</details>

**9. Prompt injection là gì? Bạn chống thế nào?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Đó là khi văn bản do kẻ tấn công kiểm soát chứa chỉ dẫn làm model làm trái ý người xây hệ thống. Trực tiếp là người dùng gõ, gián tiếp là nó nằm trong tài liệu, email hay trang web mà hệ thống tự đưa vào prompt. Gốc rễ là LLM xử lý lệnh và dữ liệu trong cùng một chuỗi token; chưa có cách tách triệt để như prepared statement của SQL. Thậm chí khi tự host, tokenizer mặc định còn biến chuỗi `<|im_start|>` do người dùng gõ thành special token thật, nên người dùng giả được một lượt system. Vì vậy mình phòng thủ nhiều lớp: lọc đầu vào, bọc dữ liệu trong thẻ, nhắc lại quy tắc sau dữ liệu, kiểm tra đầu ra bằng schema và canary. Quan trọng nhất là giới hạn hậu quả: quyền tối thiểu cho tool, kiểm tra quyền phía server, xác nhận hành động không đảo ngược. Mình đo trên model 0.5B: bọc thẻ giảm tỉ lệ lộ bí mật nhưng không về 0; chỉ lớp chặn đầu ra mới làm khách không bao giờ thấy.

**Ý phải có:** trực tiếp/gián tiếp · vì sao xảy ra · không có giải pháp tuyệt đối · nhiều lớp · quyền tối thiểu + xác nhận là lớp quan trọng nhất.
**Nếu bị hỏi tiếp:** *"Đặt API key trong system prompt được không?"* → không; giả định system prompt sẽ bị trích ra.
**Xem lại:** bài 06.
</details>

**10. Bạn quản lý và thử nghiệm thay đổi prompt trong production thế nào?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Mỗi prompt có tên và version **bất biến**, có fingerprint là hash của nội dung và một changelog. Muốn sửa thì tăng version. Version được ghi vào mọi dòng log và vào khóa cache, để đổi prompt không trả về câu trả lời cũ. Quy trình gồm bốn bước. Chạy bộ hồi quy offline, so ghép cặp theo intent với phiên bản cũ. Rồi shadow trên lưu lượng thật. Rồi A/B, chia ổn định theo hash của user ID, bắt đầu 5–10%. Đo tỉ lệ 👎, chuyển người, lỗi schema, chi phí, độ trễ. Phải đủ mẫu: phát hiện mức tăng từ 80% lên 85% câu trả lời hữu ích cần khoảng 900 mẫu có đánh giá mỗi nhánh, nên đánh giá offline ghép cặp rất quan trọng. Công cụ như PromptLayer hay LangSmith làm sẵn registry và tracing.

**Ý phải có:** version bất biến + fingerprint + changelog · version trong log và khóa cache · offline ghép cặp → shadow → A/B chia ổn định · cỡ mẫu.
**Xem lại:** bài 06.
</details>

## C. API và production

**11. Làm sao để đổi nhà cung cấp LLM mà không phải sửa code nghiệp vụ?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Mình có một lớp trừu tượng, ví dụ `chat(messages, **params)` trả về văn bản, usage và lý do dừng. Mỗi nhà cung cấp có một adapter chuyển đổi hai chiều. Khác biệt chính: Anthropic để system prompt thành trường riêng và bắt buộc `max_tokens`; Gemini dùng `contents` và `parts` với vai trò `model`; đối số tool call của OpenAI là chuỗi JSON còn của Anthropic là object; định dạng stream cũng khác. Rất nhiều nhà cung cấp và server tự host như Groq, OpenRouter, vLLM, Ollama nói giao thức kiểu OpenAI, nên client của mình chỉ cần đổi `base_url` và key. Tên model nằm trong cấu hình theo tier, không hard-code. Và có một provider giả nói đúng giao thức để chạy bộ test hợp đồng offline: project của mình có 49 test chạy không cần key.

**Ý phải có:** lớp trừu tượng + adapter · vài khác biệt cụ thể · giao thức OpenAI-compatible · cấu hình theo tier · provider giả cho test.
**Xem lại:** bài 07; project.
</details>

**12. Streaming LLM hoạt động thế nào? Bạn từng gặp lỗi gì khi parse?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Server trả `text/event-stream`. Mỗi sự kiện là vài dòng `data:`, kết thúc bằng một dòng trống. Với OpenAI, mỗi sự kiện là một chunk chứa delta và kết thúc bằng `data: [DONE]`. Cái bẫy là mạng giao byte theo mảnh tùy ý, không theo ranh giới sự kiện hay ký tự. Mình đo trên 300 luồng bị cắt ngẫu nhiên: parser tách `\n\n` trên từng mảnh hỏng cả 300; có bộ đệm dòng nhưng vẫn giải mã UTF-8 từng mảnh thì chỉ đúng 95 luồng và sinh 771 ký tự `�`, vì chữ Việt 2–3 byte bị cắt đôi. Parser đúng có ba tầng: incremental decoder, tách dòng (chờ khi `\r` nằm cuối bộ đệm), và phát sự kiện ở dòng trống. Streaming giảm độ trễ cảm nhận: trong project, khách thấy chữ đầu tiên sau 0.28 s thay vì 0.68 s, tổng thời gian không đổi. Và không được tự thử lại khi đã gửi một phần cho người dùng.

**Ý phải có:** định dạng SSE · mảnh byte ≠ sự kiện ≠ ký tự · ba tầng của parser · độ trễ cảm nhận · không retry giữa luồng.
**Nếu bị hỏi tiếp:** *"Vì sao không dùng `EventSource`?"* → chỉ GET, không đặt được header xác thực; dùng `fetch` + `ReadableStream`.
**Xem lại:** bài 08.
</details>

**13. Thiết kế xử lý lỗi khi gọi LLM API.**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Mình phân loại lỗi trước. 408, 429, 5xx, 529, timeout là thoáng qua, thử lại với full jitter; nếu có `Retry-After` thì chờ đúng thời gian đó. 400, 401, 404, 422 là do request, gửi lại vẫn lỗi, nên dừng ngay. Toàn bộ các lần thử nằm trong một deadline, và chỉ retry ở một tầng để tránh retry storm. Retry chỉ chữa lỗi thoáng qua. Với sự cố kéo dài cần **fallback**: model khác, nhà cung cấp khác, cache, câu trả lời mẫu, chuyển người. Trong project, khi model nhỏ sập 30 giây: không retry thì 79% request được trả lời, có retry chỉ lên 82%, có fallback thì 99.8%. Đổi lại p95 tăng từ 1.9 lên 3.4 giây vì phải thử hết với model hỏng trước. Bước tiếp theo là circuit breaker, ngắt mạch sau vài lỗi liên tiếp để gửi thẳng sang dự phòng. Và ghi `fallback_used` để biết chất lượng thực tế.

**Ý phải có:** phân loại lỗi · jitter + Retry-After · deadline, một tầng retry · fallback chain · số đo retry so với fallback · circuit breaker.
**Xem lại:** bài 09; project E3.
</details>

**14. Làm sao giảm chi phí LLM của một hệ thống nhiều lần?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Đầu tiên là **đo**: log token và tiền theo khách, tính năng, model, phiên bản prompt. Đòn bẩy lớn nhất thường là **định tuyến**. Trong gateway của mình, router TF-IDF dưới 1 ms quyết định trước: tin "gặp nhân viên" và tin ngoài phạm vi, khoảng 15% lưu lượng, không cần gọi LLM; phần lớn còn lại đi model nhỏ; chỉ khiếu nại và tin router không chắc mới lên model lớn. Chi phí giảm từ 2.37 xuống 0.48 USD mỗi 1 000 request, khoảng 5 lần, với độ chính xác intent giảm từ 0.97 xuống 0.94. Sau đó là cache: khóa chuẩn hóa nhẹ trúng 14%. Khóa chuẩn hóa mạnh, thay số điện thoại bằng placeholder, trúng nhiều hơn chút nhưng trả nhầm câu trả lời của khách này cho khách khác. Rồi giới hạn `max_tokens`, nén prompt, prompt caching cho phần đầu cố định, Batch API cho việc không gấp. Lưu lượng lớn và ổn định thì tính phương án tự host.

**Ý phải có:** đo trước · định tuyến có số · cache và cái bẫy của khóa · prompt caching/batch/nén · tự host.
**Xem lại:** bài 07, 09; project E1, E2.
</details>

**15. Làm sao bảo vệ API key và chống lạm dụng khi app mobile gọi LLM?**

<details><summary>Đáp án</summary>

**Trả lời mẫu.** Không bao giờ để key trong app hay JavaScript, vì ai cũng trích ra được. App gọi **backend proxy** của mình bằng token phiên đăng nhập; backend giữ key của nhà cung cấp. Khi phát triển thì key nằm trong `.env` đã được `.gitignore`; production thì trong secret manager, có phân quyền và nhật ký truy cập. Xoay vòng bằng cách luôn có hai key hợp lệ: triển khai key mới, chờ key cũ hết lưu lượng rồi thu hồi; lỡ commit lên git thì coi như đã lộ. Backend giới hạn tốc độ theo khách bằng token bucket, trạng thái để trong Redis nếu có nhiều instance. Trong project, khách gửi nhiều nhất bị chặn 81% request còn khách khác chỉ 10%. Kiểm tra đầu vào ở biên: độ dài, số lượt lịch sử, không cho khách tự gửi message system. Log thì che PII và không bao giờ chứa key. Với dữ liệu khách hàng ở Việt Nam còn phải tuân thủ Nghị định 13 về dữ liệu cá nhân.

**Ý phải có:** backend proxy · `.env` / secret manager · xoay vòng · rate limit theo khách (Redis) · kiểm tra đầu vào · log che PII · Nghị định 13.
**Xem lại:** bài 09; project `server.py`.
</details>

## D. Kể về project (câu gần như chắc chắn bị hỏi)

**16. "Kể về một hệ thống tích hợp LLM bạn đã xây."** *(luyện bản 2 phút)*

<details><summary>Khung trả lời và bản mẫu</summary>

**Khung:** bối cảnh → mục tiêu đo được → cách làm → **kết quả có số** → điều học được hoặc điều bất ngờ → sẽ làm gì tiếp.

**Bản mẫu (project `gateway`).**
- **Bối cảnh:** gateway LLM cho tổng đài nhà mạng, đặt sau router intent mình làm trước đó. Mục tiêu: trả lời tự động có cấu trúc, chi phí thấp, chịu được lỗi nhà cung cấp, an toàn trước prompt injection.
- **Cách làm:**
  - Router TF-IDF chọn đường: model nhỏ, model lớn, tổng đài viên, hoặc từ chối.
  - Prompt có version, dữ liệu khách bọc thẻ; đầu ra là JSON kiểm tra bằng Pydantic, có sửa cục bộ và hỏi lại một lần.
  - Client async có retry theo Retry-After, streaming SSE với parser tự viết, fallback giữa hai tier.
  - Cache khóa SHA-256; rate limit theo khách; canary để chặn rò rỉ system prompt; log che PII; backend proxy FastAPI.
  - Mọi thứ test offline trên một provider giả nói đúng giao thức, 49 test.
- **Kết quả** (2 000 request mô phỏng):
  - Định tuyến giảm chi phí khoảng **5 lần** (2.37 → 0.48 USD / 1 000 request); cache thêm 14% lượt trúng.
  - Khi model nhỏ sập 30 giây, fallback giữ tỉ lệ trả lời **99.8%** so với 82% nếu chỉ retry.
  - Streaming giảm thời gian tới chữ đầu tiên từ 0.68 xuống 0.28 giây.
  - Lớp chặn đầu ra đưa tỉ lệ rò rỉ tới tay khách về 0.
- **Bất ngờ:**
  - Khóa cache chuẩn hóa quá mạnh trả nhầm số điện thoại của khách này cho khách khác.
  - Model 0.5B dùng prompt phân loại intent chỉ đúng 38%, trong khi router TF-IDF đúng 96%: LLM không phải công cụ cho mọi việc.
- **Nếu làm tiếp:** circuit breaker và nhiều nhà cung cấp, rate limit theo token, Redis cho trạng thái chung, và giữ intent của router khi nó tự tin thay vì tin intent của LLM.

**Câu hỏi đào sâu hay gặp** (chuẩn bị sẵn):
- *"Số liệu đó đo thế nào?"* Provider giả với tỉ lệ lỗi và tốc độ giả định, thời gian mô phỏng để lặp lại được. Nó đo **đóng góp của từng lớp** gateway, không đo chất lượng LLM. Chất lượng LLM thật mình đo riêng trên một model 0.5B chạy local.
- *"Vì sao không cache mọi thứ?"* Lịch sử, nhiệt độ > 0, dấu hiệu injection, dữ liệu cá nhân, thông tin hay đổi.
- *"Sao không tin intent của LLM?"* Trên các request đi qua LLM, intent của router đúng 0.97 còn của LLM 0.92. Router tự tin thì giữ, LLM chỉ viết câu trả lời và trích slot.

Thay các chi tiết bằng **số của chính bạn** trong `REPORT.md`.
</details>

---

**Tự chấm:** mỗi câu được 1 điểm nếu nói đủ "Ý phải có" trong khoảng 1 phút (câu 16 trong khoảng 2 phút). Đạt **≥ 12/16** là sẵn sàng cho phần LLM và tích hợp API của vòng phỏng vấn kỹ thuật.
