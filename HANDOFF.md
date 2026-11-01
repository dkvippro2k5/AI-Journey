# Bàn giao · Viết tài liệu học, Tuần 6 trở đi

> Cho session Claude mới. Cập nhật 2026-11-01 (đã xong Tuần 1–5). Đọc hết file này, rồi đọc lướt [week-05-nlp/](week-05-nlp/) để nắm giọng văn.

## 1. Bối cảnh

- **Người dùng:** sinh viên năm cuối, ứng tuyển **VDT 2027 · AI Engineer** (hạn khoảng giữa tháng 3/2027; Tết khoảng 06/02/2027). Muốn ôn lại từ nền tảng tới nâng cao, không bỏ phần nào. **Không có GPU** (chỉ Kaggle/Colab). Trả lời bằng tiếng Việt, giữ thuật ngữ tiếng Anh.
- **Kế hoạch 16 tuần** (Phase 0–13 trong [README.md](README.md) gốc; 5 ngày/tuần; DSA 30–45' mỗi ngày):
  - ✅ 1 Python/async · 2 Toán · 3 ML · 4 DL+PyTorch · 5 NLP+Transformer
  - ⏭ **6 LLM API**
  - 7 Orchestration/MCP · 8–9 RAG+SQL/pgvector · 10 Agents · 11 Fine-tuning · 12 GenAI+vLLM · 13–14 MLOps · 15 System design · 16 Ôn tập
- **Các project nối thành một hệ thống chung** (RAG + Agent tiếng Việt, tự host):
  - T1 `multi_caller`: retry, Semaphore → gateway ở T6.
  - T2 `optlab`: NumPy, các hàm stats → T8, T13.
  - T3 `quality`: dự đoán 👎 bằng HGB → service ở T13.
  - T4 `qnet`: MLP/TabAttn PyTorch, so ghép cặp với HGB; `fit`, LayerNorm, `dot_attention` → T5, T11.
  - T5 `intent`: router intent tổng đài (8 intent + `khac`), `normalize` tiếng Việt, BPE, `MultiHeadAttention`, `TransformerBlock`, ngưỡng chuyển người (`choose_threshold`/`route`), chia theo template → gateway T6, agent T10, service T13.
- **Quy trình:**
  - Duyệt **từng tuần**. Chỉ viết tuần mới khi người dùng nói "làm tiếp".
  - Viết trọn tuần, kiểm chứng hết, rồi gửi một tin tổng kết ngắn và xin góp ý.
  - Muốn đổi bố cục thì **hỏi trước**, kèm bản mẫu ngắn.
  - Thỉnh thoảng báo tiến độ ngắn.

## 2. Các quyết định đã chốt (không quay lại cách cũ)

- Mỗi bài có hai tầng:
  - **Tầng 1 · Trực giác:** 📖 Từ mới (định nghĩa trước khi dùng), ví dụ đời thường, 3 ý cốt lõi, ví dụ có số.
  - **Tầng 2 · Kiến thức chuẩn** (đề thi chấm theo tầng này): bảng thuật ngữ **có nguồn** (paper/sách/docs, có năm), công thức kèm ý nghĩa, **ví dụ tính tay**, số đo thật, code chuẩn, cạm bẫy, sắc thái phỏng vấn, đọc thêm.
- **Không có chứng minh hình thức** (từ Tuần 3): chỉ phác thảo "vì sao" 2–3 dòng. **Giữ đầy đủ** tính tay và số đo. Quiz/final không có câu chứng minh.
- **Giữ độ khó hiện tại** (người dùng thích, vì giúp củng cố nền tảng). Nhưng một số khái niệm ở Tuần 4 bị **giải thích vắn tắt, khó hình dung**. Từ nay, mỗi cơ chế không hiển nhiên phải có giải thích chậm: hình dung bằng lời hoặc sơ đồ, ví dụ số đi từng bước **trước** công thức, ghi rõ các bước trung gian. Mẫu: khối **"🐢 Đi chậm từng bước (đọc trước công thức)"** đã thêm vào Tuần 4 ở mục 4.2 (phương sai khi khởi tạo), 8.2 (LSTM), 9.2 (q/k/v, $\sqrt d$); sau khối là "**Công thức (tóm tắt các bước trên).**". Các chỗ khác của Tuần 4 (vector–Jacobian, residual, normalization, AdamW) chờ người dùng đánh dấu.
- Mỗi bài đọc khoảng 30'; 3–4 bài mỗi ngày. Nhịp tuần **cố định**:
  - Ngày 1–3: đọc + quiz 45' + bài tập khoảng 2h (mỗi bài tập là một mảnh của project).
  - Ngày 4: ghép project.
  - Ngày 5: `REPORT.md` + final 180'.
- Ghi loại tài liệu ở đầu mỗi file ngày:
  - **Lý thuyết (bản nhẹ):** ML, DL, NLP. Công thức đầy đủ, tính tay nhiều.
  - **Kết hợp:** RAG, fine-tuning, GenAI. Chỉ có công thức cho cơ chế lõi.
  - **Kỹ thuật:** Python, LLM API, orchestration/MCP, agents, MLOps, system design. Toán chỉ ở mức ước lượng nhanh; phần B của final là bài ước lượng.
- **Đối chiếu độ phủ với roadmap** (người dùng hỏi 2026-10-27 "tài liệu có đủ kiến thức không"): trước khi viết một tuần, liệt kê **mọi gạch đầu dòng** của Phase tương ứng trong [README.md](README.md) gốc và đánh dấu bài nào dạy mục đó. Mục nào dời sang tuần khác thì ghi rõ trong README của tuần. Tuần 1–3 bị thiếu, nên đã được bổ sung bằng `phu-luc.md` (xem mục 3).
- **Trang "Ôn phỏng vấn"** (`on-phong-van.md`): **đã duyệt** (2026-11-01), mọi tuần đều có. Khoảng 16 câu, trả lời mẫu khoảng 1' trong `<details>`, mục "Ý phải có", câu hỏi tiếp, câu "kể về project" dùng số thật của tuần. Mẫu: [week-05-nlp/on-phong-van.md](week-05-nlp/on-phong-van.md).
- Người dùng chấp nhận tuần **nặng** ("trông kiến thức nặng phết, nhưng không sao"). README của tuần có mục **"Đối chiếu với roadmap"** (bảng mục roadmap → bài, ghi rõ mục dời sang tuần nào), như Tuần 5.

## 3. Định dạng

```
week-NN-<slug>/
├── README.md                     # mục tiêu, lịch 5 ngày, cài đặt, bảng kiểm tra, nhận/chuyển giữa các tuần
├── ngay-{1,2,3}.md               # bài đánh số liên tục; cuối file là "Bài tập Ngày N" (chữ ký hàm khớp test, số test)
├── ngay-{1,2,3}.quiz.md
├── project.md                    # đặc tả Ngày 4, lệnh chạy, KẾT QUẢ THẬT, câu hỏi REPORT, checklist
├── final-exam.md / final-solutions.md (có đủ code phần C)
├── on-phong-van.md
├── phu-luc.md                    # chỉ khi cần vá chỗ thiếu so với roadmap; có ở Tuần 1–3
├── code/test_final.py            # KHÔNG để final_lib.py trong repo
└── project/{pyproject.toml, .gitignore, <pkg>/, tests/, _reference/}
```

- **Phụ lục** (`phu-luc.md`, viết 2026-10-27): bài đánh số P1, P2, …; cùng định dạng hai tầng; có mục "Tự làm" thay cho bài tập project; cuối file là quiz 30', 20 điểm (10 câu nhanh + 2 tự luận × 5đ), đạt khi ≥ 14; README của tuần có mục "## Phụ lục" ngay trước "## Kiểm tra". Dẫn chéo dạng "(Tuần 1, phụ lục P5)". Nội dung đã có:
  - Tuần 1 (P1–P6): cú pháp và Unicode NFC, `collections`/`functools`/`itertools`, traceback/`pdb`, `timeit`/`cProfile`/`tracemalloc`, `np.stack`/`np.where`/`concat`/`melt`/`to_datetime` và múi giờ, `tqdm`/YAML/Hydra (chỉ mô tả, chưa cài)/`aiohttp`/Jupyter–Colab.
  - Tuần 2 (P1–P3): Lagrange/KKT, step decay/exponential/ReduceLROnPlateau/sàn nhiễu SGD, plateau.
  - Tuần 3 (P1–P3): feature engineering, mất cân bằng (SMOTE, hiệu chỉnh prior), metric hồi quy (smearing, pinball).
  - Tuần 4 (P1–P4): custom loss (pos_weight + hiệu chỉnh odds, focal, label smoothing, ổn định số), `Dataset`/`collate_fn`/mask/pack/bucketing/BiLSTM, object detection (IoU, anchor, NMS, mAP), device/GPU (P4 là Kỹ thuật, chưa chạy trên GPU).
- **Quiz:** 45', 30 điểm, đạt khi ≥ 21.
  - Phần 1: 14 câu (câu 14 được 2đ).
  - Phần 2: 4 câu tự luận T1–T4, mẫu 4 + 4 + 3 + 4 = 15đ (tính tay / phân tích / thiết kế / code).
  - Đáp án nằm trong một khối `<details>`, có barem.
- **Final:** 180', 100 điểm, đạt khi ≥ 70.
  - A: 12 câu × 2đ.
  - B: tính tay, 20đ.
  - C: code, 32đ, khoảng 8–14 test; chạy `python -m pytest -q code/test_final.py` ở thư mục tuần.
  - D: thiết kế, 24đ, bối cảnh Việt Nam hoặc Viettel.
  - Có danh sách "bài → câu".
- **Quy ước:**
  - Dùng `MODEL_ID` thay cho tên model thật; mọi mức giá là giá giả định.
  - Số thập phân dùng dấu chấm, phân cách hàng nghìn dùng khoảng trắng (`10 000`).
  - Số đo ghi là "đo trên máy bạn" (máy của người dùng chính là máy chạy session).
  - Dẫn bài cũ theo dạng "(Tuần 2, bài 06)".
  - LaTeX dùng cho KaTeX: chỉ số dưới là toán tử thì phải bọc ngoặc, ví dụ `\eta_{\min}`.

## 4. Kiểm chứng (bắt buộc)

1. **Mọi con số, output và đáp án** (kể cả đáp án quiz) đều chạy thật bằng script trong scratchpad. Không đoán.
2. Thứ tự làm việc:
   1. dàn bài + `data.py` (chỉnh cho kết quả có ý nghĩa sư phạm);
   2. `_reference/` + tests;
   3. script đo số liệu;
   4. `ngay-N.md` → quiz;
   5. `project.md` với kết quả thật;
   6. final + lời giải để trong scratchpad;
   7. **bản sao sạch**: copy `_reference` vào package → `python -m pytest -q` → chạy test final;
   8. xóa `__pycache__` và `.pytest_cache`.
3. Bài tính tay phải có **đáp án duy nhất** (kiểm tra trường hợp hòa).
4. Test phải bền với nhiều seed và nhiều cách cài đặt; không assert sát giá trị của mạng neural.
5. Không dùng API đã deprecated. Với `np.errstate`, dùng `over="raise", invalid="raise"`.

## 5. Môi trường (máy của người dùng)

- **Hệ thống:** Windows 11, 22 core, không GPU. PowerShell 5.1 (không có `&&`), có Git Bash.
- **Python 3.14.5:** NumPy 2.5 · pandas 3.0 (Copy-on-Write) · scikit-learn 1.9 · torch 2.14 CPU · transformers 5.18 · tokenizers 0.23 · datasets 5.0.
- **Đã tải (2026-11-01, được phép):** `intfloat/multilingual-e5-small` (118M tham số, 470 MB) trong cache Hugging Face. Trong tài liệu gọi là `MODEL_ID`; fine-tune trên CPU mất khoảng 10 phút mỗi split.
- Windows console là cp1252: chạy script in tiếng Việt thì đặt `PYTHONIOENCODING=utf-8`. Output nền nên dùng `python -u`, đừng pipe qua `grep` (bị giữ trong buffer).
- **Chưa cài:** torchvision, sentence-transformers, lightgbm, xgboost, optuna, umap, shap.
- Luôn gọi `python -m pytest`. Không dùng `python -I` cho script của mình (sẽ mất torch/httpx).
- **Tải dữ liệu hoặc model từ mạng thì phải hỏi trước**, hoặc để thành phần tùy chọn trên Kaggle.
- Huấn luyện chạy trên CPU trong vài phút; test dưới khoảng 60 s.
- **Chạy pin thì chậm 3–15 lần** (xung nhịp 1.4 GHz; matmul còn 33 thay vì 484 GFLOP/s). Kiểm tra `Win32_Battery` trước khi đo thời gian.
- bf16 trên CPU này chậm khoảng 850 lần so với fp32.

## 6. Tuần 6 · LLM và AI API (đề xuất, chưa duyệt)

- **Loại tài liệu:** chủ yếu **Kỹ thuật** (phần B của final là ước lượng: token, chi phí, độ trễ, bộ nhớ KV). Phần 6.1 (nền tảng LLM) là Kết hợp.
- **Roadmap Phase 6** có 6.1 LLM fundamentals, 6.2 prompt engineering, 6.3 AI API (nhiều nhà cung cấp), 6.4 API integration patterns, 6.5 secure API integration. **Đối chiếu đủ mọi gạch đầu dòng** trước khi lên dàn bài.
- **Tuần 5 đã dạy, không dạy lại:** KV-cache, RoPE, GQA (khái niệm), FlashAttention (khái niệm), tokenization, temperature/top-k/top-p, perplexity, BERT/GPT/T5, metric và hạn chế của BLEU, kappa. Tuần 6 chỉ thêm: ALiBi, sliding window, scaling laws, pretraining → instruction tuning → RLHF/DPO (khái niệm; thực hành DPO ở Tuần 11), context window và chi phí.
- **Đã hẹn ở Tuần 6:** bảng các LLM cụ thể (roadmap 5.7, nhưng theo quy ước dùng `MODEL_ID` và giá giả định), `tiktoken`/đếm token, chi phí theo token; gateway dùng lại `multi_caller` (Tuần 1) và router `intent` (Tuần 5).
- **Phải hỏi người dùng trước khi viết:**
  - Có API key của nhà cung cấp LLM nào không? Nếu không: dùng **fake server** (giống `multi_caller/fake.py` Tuần 1) cho mọi test, cộng một LLM nhỏ chạy local qua `transformers` (cần tải, phải hỏi).
  - Có cho cài thêm thư viện (`openai`, `anthropic`, `tiktoken`, `pydantic`, `fastapi`) không?
- **Project gợi ý:** gateway LLM: router intent (T5) → chọn model/prompt theo intent → gọi qua client có retry/timeout/rate limit (T1) → structured output kiểm tra bằng schema → cache, đếm token và chi phí → log để đánh giá. Test chạy offline bằng fake provider.
- **Final gợi ý:** B: ước lượng token, chi phí, độ trễ (prefill/decode), bộ nhớ KV. C: SSE streaming parser, JSON schema validator / repair, token bucket rate limiter, prompt template có version. D: gateway LLM cho tổng đài, có chống prompt injection.

## 7. Tham chiếu

- **Memory** (`C:\Users\ZENBOOK\.claude\projects\d--AI-Engineer-Journey\memory\`):
  - `project_vdt_plan.md`: cập nhật tiến độ sau mỗi tuần.
  - `feedback_doc_format.md`: cập nhật khi có góp ý mới.
- **Transcript cũ** (chỉ Grep, đừng đọc hết), cùng thư mục trên: `f1fc0acf-….jsonl` (Tuần 1–3), `9cf44ff1-….jsonl` (Tuần 4–5).
- **GitHub:** https://github.com/dkvippro2k5/AI-Journey, nhánh `main`, **mỗi tuần một commit**, **không** có dòng `Co-Authored-By` (người dùng yêu cầu). Git identity: Duong Kien. `.gitignore` gốc loại cache, `.venv`, `artifacts/`, `final_lib.py`.
- **Mẫu:** [week-04-dl/](week-04-dl/) (mới nhất); [week-03-ml/](week-03-ml/) (quiz, final, tests); [week-01-python-async/ngay-3.md](week-01-python-async/ngay-3.md) (bài Kỹ thuật).

## 8. Checklist khi xong một tuần

- [ ] Đủ file theo mục 3; mỗi bài có đủ tầng 1 và tầng 2; không có chứng minh hình thức; cơ chế khó có giải thích từng bước.
- [ ] Mọi gạch đầu dòng của Phase tương ứng trong roadmap đều có bài dạy, hoặc được ghi rõ là dời sang tuần nào.
- [ ] Mọi số liệu và đáp án đã chạy; bài tính tay có đáp án duy nhất.
- [ ] Bản sao sạch: test project và test final đều pass; repo không có cache và không có `final_lib.py`.
- [ ] Đã cập nhật memory và file này (mục 1 và mục 6).
- [ ] Đã gửi tin tổng kết ngắn: bảng các ngày, vài số đo đáng chú ý, kết quả kiểm chứng, những gì điều chỉnh, và lời xin góp ý.
