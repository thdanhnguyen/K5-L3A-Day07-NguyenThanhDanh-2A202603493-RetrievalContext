# Báo cáo cá nhân Ngày 7 — Agentic RAG: Retrieval và Context Engineering

- **Họ và tên:** Nguyễn Thành Danh
- **MSSV:** `2A202603493`

---

## Dữ liệu và benchmark

### 1. Nguồn dữ liệu và cách kết nối từ Ngày 6
Trong bài thực hành này, em kết nối trực tiếp kho dữ liệu đã xây dựng ở Ngày 6 (`D:\K5-L3A-DAY06-2A202603493-DataIngestion`) sang Ngày 7 thông qua script `src/export_day6.py`:
- **Tài liệu gốc:** *NIST Artificial Intelligence Risk Management Framework (AI RMF 1.0)* (`NIST.AI.100-1.pdf`), gồm 48 trang vật lý (42 trang nội dung chính từ trang vật lý 6 đến 47, sau khi loại bỏ bìa và mục lục ở trang 1–5).
- **Chiến lược chunking kế thừa từ Ngày 6:** Em sử dụng chiến lược **Structure-aware Paragraph/Section Chunking (Strategy B / `par`)** với `chunk_size = 800` ký tự, `overlap = 100` ký tự, giữ nguyên ranh giới đoạn văn (`\n\n`) và gắn metadata cấu trúc (`section`, `document_page`, `year=2023`, `version="1.0"`, `chunk_strategy`, `chunk_index`, `char_length`).
- **Quy mô tập `data/chunks.json`:** Gồm **173 chunks** có ID ổn định theo mẫu `nist-ai-rmf-1.0-p{page:02d}-par-{idx:02d}` (ví dụ: `nist-ai-rmf-1.0-p06-par-01`), bao phủ 4 phân vùng lớn (`section`):
  1. `executive_summary` (trang vật lý 6–8): 14 chunks.
  2. `part_1_foundational` (trang vật lý 9–24): 68 chunks.
  3. `part_2_core_profiles` (trang vật lý 25–39): 58 chunks.
  4. `appendix` (trang vật lý 40–47, gồm Appendix A, B, C, D): 33 chunks.
- **Kết nối Vector Store Ngày 6:** Toàn bộ 173 chunks được nhúng bằng model thật **`text-embedding-3-small` (1536 chiều)** của OpenAI. Class `DensePgVectorRetriever` (`src/retrievers.py`) hỗ trợ truy vấn trực tiếp bảng `rag_chunks` trên PostgreSQL + `pgvector` (`postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab`) và đồng bộ xuống kho vector cục bộ `data/day6_vector_store.json` (173 bản ghi × 1536 float, tổng 30,277 embedding tokens, chi phí index `$0.000606`) để bài thực hành có thể chạy tái lập 100% ngay cả khi tắt Docker.

### 2. Quy trình xây dựng Benchmark (`data/questions.json` và `data/cases.json`)
Em tự thiết kế bộ benchmark theo đúng quy tắc của `BENCHMARK.md`, viết câu hỏi bằng **tiếng Việt** để kiểm tra năng lực truy hồi xuyên ngôn ngữ (Cross-Lingual Retrieval: câu hỏi tiếng Việt $\rightarrow$ tài liệu gốc tiếng Anh) và diễn đạt lại câu hỏi thay vì chép nguyên văn câu chữ trong PDF:
- **`data/questions.json` (20 câu hỏi có đáp án dùng cho Retrieval CLI `starter/baseline.py`):**
  - Chia tách nghiêm ngặt theo tỷ lệ 75% / 25%: **15 câu `dev` (`q01`–`q15`)** và **5 câu `test` (`q16`–`q20`)**.
  - Bao phủ toàn bộ 4 phần của NIST AI RMF 1.0 (Executive Summary, Part 1 Foundational Information, Part 2 Core GOVERN/MAP/MEASURE/MANAGE, và Appendices A–D).
  - Mỗi câu đều có trường `gold` chứa `source_id`, `page` thực tế và chuỗi `evidence` trích nguyên văn từ chunk nguồn. Em đã kiểm chứng bằng hàm `validate(chunks, questions)` và unit test `tests/test_fixture.py`: **20/20 câu đều khớp ít nhất một chunk trong `data/chunks.json`**.
  - Có **5 câu gắn bộ lọc `filter` theo `section`** (`q02`, `q09`, `q12`, `q15` ở tập `dev` và `q19` ở tập `test`) để đánh giá khả năng thu hẹp không gian tìm kiếm bằng metadata.
- **`data/cases.json` (24 tình huống dùng cho Agentic RAG):**
  - Gồm toàn bộ 20 câu có đáp án (`q01`–`q20`, `"answerable": true`) cộng thêm **4 câu không có đáp án trong tài liệu (`u01`–`u04`, `"answerable": false`)**, chia đều 2 câu ở `dev` (`u01`, `u02`) và 2 câu ở `test` (`u03`, `u04`):
    - `u01` (`dev`): Hỏi về *mức phạt tiền hành chính tối đa (USD) khi doanh nghiệp thương mại không tuân thủ GOVERN 1.1* — đã rà soát toàn bộ 42 trang nội dung: NIST AI RMF 1.0 là khung tự nguyện (*voluntary framework*), không có điều khoản xử phạt tiền.
    - `u02` (`dev`): Hỏi về *lệ phí đăng ký kiểm toán định kỳ và cấp chứng chỉ tuân thủ AI RMF hàng năm của NIST* — tài liệu không hề có chương trình cấp chứng chỉ thu phí của NIST.
    - `u03` (`test`): Hỏi về *ngưỡng số lượng tham số (parameters) hoặc số phép tính huấn luyện (FLOPs) tối thiểu trong chức năng MEASURE để xếp mô hình vào nhóm rủi ro nghiêm trọng* — đã kiểm tra toàn bộ mục MEASURE và toàn văn tài liệu: NIST AI RMF 1.0 không đặt ngưỡng định lượng về parameters hay FLOPs.
    - `u04` (`test`): Hỏi về *thời hạn bắt buộc (số ngày tối đa) trong Appendix A để nhà phát triển AI báo cáo sự cố nghiêm trọng lên Quốc hội Hoa Kỳ* — đã rà soát toàn bộ Appendix A (trang 40–42) và toàn văn tài liệu: không có quy định số ngày báo cáo lên Quốc hội.

---

## Retrieval và thử nghiệm

### 1. Các phương pháp Retrieval đã cài đặt (`src/retrievers.py`)
Ngoài baseline **TF-IDF Cosine** (`TfidfIndex` trong `starter/baseline.py`), em đã cài đặt và thử nghiệm 3 họ phương pháp truy hồi cùng 2 bước cải tiến có kiểm soát trên tập `dev`:
1. **Lexical Okapi BM25 (`BM25Retriever`):** Chuẩn hóa Unicode NFC, tách từ và tính điểm BM25 với bão hòa tần suất từ ($k_1 = 1.5$) và chuẩn hóa độ dài tài liệu ($b = 0.75$).
2. **Dense Vector Search từ Ngày 6 (`DensePgVectorRetriever`):** Nhúng câu hỏi bằng `text-embedding-3-small` (1536 chiều) và tính tương đồng Cosine với vector của 173 chunks Ngày 6. Nhờ biểu diễn ngữ nghĩa đa ngôn ngữ của `text-embedding-3-small`, phương pháp này vượt qua rào cản khác biệt ngôn ngữ Việt – Anh mà TF-IDF/BM25 gặp phải.
3. **Hybrid Retrieval với Reciprocal Rank Fusion (`HybridRRFRetriever`):** Kết hợp danh sách ứng viên từ Dense Vector ($w_{\text{dense}} = 1.0$) và Lexical BM25 ($w_{\text{lexical}} = 0.8$) theo công thức $\text{RRF}(d) = \sum_{m} \frac{w_m}{k_{\text{rrf}} + \text{rank}_m(d)}$ với $k_{\text{rrf}} = 20$.
4. **Hai cải tiến có kiểm soát trên tập `dev`:**
   - **Cải tiến 1 (Metadata Filter + Context Budget `800 -> 2400` ký tự):** Kích hoạt bộ lọc `filters` theo `metadata.section` và nâng ngân sách đóng gói ngữ cảnh `pack_context` từ `800` lên `2,400` ký tự. Với `budget = 800`, trung bình một chunk dài ~650–780 ký tự cộng thêm tiêu đề `[chunk_id | source | page]` (~74 ký tự) đã chiếm trọn ngân sách, khiến các chunk ở Rank 2 và Rank 3 bị loại bỏ hoàn toàn dù nằm trong `top_k = 3`.
   - **Cải tiến 2 (Cross-Lingual Query Rewrite bằng `gpt-6-luna` + Hybrid RRF + Filter + Budget 2400):** Dùng `gpt-6-luna` (`QueryRewriteHybridRetriever`) chuyển câu hỏi tiếng Việt sang truy vấn tiếng Anh chứa đúng thuật ngữ chuyên ngành và mã tiểu mục của NIST AI RMF 1.0 (ví dụ: `GOVERN 1.1`, `MEASURE 2.5`, `Appendix A`), đồng thời ghi log cả câu hỏi gốc lẫn `rewritten_query` vào `traces.json`.

### 2. Bảng kết quả thực nghiệm Retrieval trên tập `dev` (15 câu) và `test` (5 câu)

> **Quy tắc chống rò rỉ dữ liệu (`test` leakage):** Em chỉ thử nghiệm, so sánh và tinh chỉnh tham số trên 15 câu `dev`. Sau khi chốt phương pháp tốt nhất trên `dev` là **Phương án 6 (`QueryRewrite + Hybrid RRF + Filter + Budget 2400`)**, em mới chạy đánh giá đúng 1 lần trên 5 câu `test` (`q16`–`q20`).

| Cấu hình chạy (Split & Phương pháp) | `top_k` | `budget` | `filter` | Hit@1 | Hit@3 | ContextHit | Ký tự Context TB | Độ trễ trung vị (ms) | Chi phí API (USD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dev` 1. Baseline TF-IDF** (`dev_1_baseline_tfidf`) | 3 | 800 | Tắt | 7/15 (46.67%) | 9/15 (60.00%) | 5/15 (33.33%) | 603.4 | 0.24 ms | \$0.000000 |
| **`dev` 2. Lexical BM25** (`dev_2_bm25_lexical`) | 3 | 800 | Tắt | 8/15 (53.33%) | 8/15 (53.33%) | 6/15 (40.00%) | 687.9 | 0.80 ms | \$0.000000 |
| **`dev` 3. Dense Vector Day 6** (`dev_3_dense_day6`) | 3 | 800 | Tắt | **10/15 (66.67%)** | 12/15 (80.00%) | 8/15 (53.33%) | 667.2 | 459.24 ms | \$0.000017 |
| **`dev` 4. Hybrid RRF (Dense + BM25)** (`dev_4_hybrid_rrf`) | 3 | 800 | Tắt | 9/15 (60.00%) | 10/15 (66.67%) | 7/15 (46.67%) | 595.9 | 32.56 ms | \$0.000000* |
| **`dev` 5. Cải tiến 1: Hybrid RRF + Filter + Budget 2400** (`dev_5_improv1_...`) | 3 | 2400 | **Bật** | 9/15 (60.00%) | 10/15 (66.67%) | 10/15 (66.67%) | 1958.7 | 32.87 ms | \$0.000000* |
| **`dev` 6. Cải tiến 2 (CHỐT): Query Rewrite (`gpt-6-luna`) + Hybrid + Filter + Budget 2400** (`dev_6_improv2_...`) | 3 | 2400 | **Bật** | 9/15 (60.00%) | **13/15 (86.67%)** | **12/15 (80.00%)** | 2025.3 | 3504.25 ms | \$0.002107 |
| **`test` Baseline TF-IDF** (`test_baseline_tfidf`) | 3 | 800 | Tắt | 2/5 (40.00%) | 4/5 (80.00%) | 1/5 (20.00%) | 657.0 | 0.33 ms | \$0.000000 |
| **`test` Phương pháp đã chọn (Phương án 6)** (`test_selected_rewrite_...`) | 3 | 2400 | **Bật** | 2/5 (40.00%) | 4/5 (80.00%) | **4/5 (80.00%)** | 1911.4 | 4614.92 ms | \$0.000917 |

*(\*Ghi chú: Các lần chạy Hybrid tiếp theo trên cùng tập câu hỏi tái sử dụng bộ nhớ đệm query embedding trong phiên nên không tốn thêm phí embedding).*

### 3. Đánh giá thủ công chất lượng Context (thang điểm 0 / 1 / 2) trên tập `test` (`q16`–`q20`)
Em trực tiếp đọc từng chuỗi `context` sau bước đóng gói `pack_context` trong `starter/results/test_baseline_tfidf/traces.json` và `starter/results/test_selected_rewrite_hybrid_filter_budget2400/traces.json` để chấm điểm mức độ đủ bằng chứng theo thang `0` (không có bằng chứng), `1` (có liên quan nhưng thiếu ý), `2` (đủ bằng chứng trả lời trọn vẹn):

| Mã câu | Nội dung câu hỏi (`test`) | Điểm Context Baseline (`budget=800`) | Điểm Context Phương pháp chọn (`budget=2400`) | Phân tích chi tiết nguyên nhân thay đổi |
| :---: | :--- | :---: | :---: | :--- |
| `q16` | Tiểu mục `GOVERN 4.3` yêu cầu tổ chức thiết lập các thông lệ nào để xử lý kiểm thử và sự cố AI? | **0 / 2** | **0 / 2** | Ở Baseline TF-IDF, chunk đúng `nist-ai-rmf-1.0-p29-par-00` (794 ký tự) đứng **Rank 1** nhưng khi ghép thêm dòng header `[nist-ai-rmf-1.0-p29-par-00 \| ... \| page 29]\n` (74 ký tự) thì tổng độ dài là `868 > 800`, nên bị hàm `pack_context(budget=800)` loại bỏ và chỉ giữ lại Rank 2 (`p28-par-02`, trang 28). Ở phương pháp chọn, câu truy vấn viết lại kéo các chunk `p28-par-03`, `p27-par-03` và `p29-par-01` (chứa `GOVERN 5.2`–`6.2` cùng trang 29) lên top 3 nên trượt `p29-par-00`. |
| `q17` | Theo tiểu mục `MEASURE 3.3`, quy trình phản hồi dành cho người dùng cuối và cộng đồng bị ảnh hưởng cần phục vụ mục đích gì? | **2 / 2** | **2 / 2** | Cả hai phương pháp đều xếp chunk vàng `nist-ai-rmf-1.0-p36-par-00` (540 ký tự, trang 36) ở Rank 1 và đóng gói trọn vẹn câu định nghĩa `MEASURE 3.3` vào context. |
| `q18` | Kế hoạch giám sát hệ thống AI sau triển khai theo tiểu mục `MANAGE 4.1` phải bao gồm những cơ chế nào? | **0 / 2** | **2 / 2** | Ở Baseline TF-IDF, từ khóa `MANAGE` đẩy `p37-par-00` (661 ký tự) lên Rank 1 và đẩy chunk đúng `p38-par-00` (686 ký tự) xuống Rank 2; do `budget=800` đã bị `p37-par-00` (735 ký tự tính cả header) chiếm chỗ nên chunk đúng ở Rank 2 bị cắt bỏ! Nhờ Query Rewrite (`gpt-6-luna`) dịch *"giám sát sau triển khai"* thành `"Post-deployment AI system monitoring plans"` và nâng `budget=2400`, chunk vàng `p38-par-00` vươn lên **Rank 1** và nằm trọn trong context. |
| `q19` | Trong phần Phụ lục (`Appendix B`), những tính năng nào của hệ thống AI vừa tiềm ẩn rủi ro nhưng đồng thời cũng mang lại lợi ích giúp nâng cao độ chính xác và khả năng phục hồi? | **0 / 2** | **2 / 2** | Câu hỏi tiếng Việt dùng cụm *"mô hình tiền huấn luyện / học chuyển giao"* hoàn toàn không trùng từ vựng tiếng Anh (`pre-trained models and transfer learning`) nếu chỉ dùng TF-IDF. Ở Baseline, `p06-par-02` (695 ký tự) đứng Rank 1 chiếm hết `budget=800` nên chunk vàng `p43-par-00` (794 ký tự, ở Rank 2) bị loại khỏi context. Ở phương pháp chọn, nhờ `filter={"section": "appendix"}` và `budget=2400`, cả 3 chunk thuộc Appendix (`p47-par-00`, `p44-par-02`, `p43-par-00`) đều được giữ lại đầy đủ (`2,298` ký tự). |
| `q20` | Theo `Appendix C` về tương tác người-AI, hạn chế cốt lõi khi biểu diễn các hiện tượng phức tạp của con người bằng mô hình toán học là gì? | **0 / 2** | **2 / 2** | Baseline TF-IDF chỉ khớp chữ `"Appendix C"` nên lấy `p45-par-00` (đoạn mở đầu Appendix C) và hai chunk trang 16, trang 7; hoàn toàn bỏ sót đoạn thứ hai `p45-par-01` chứa đáp án. Nhờ `gpt-6-luna` dịch câu hỏi sang `"Appendix C human-AI interaction limitations mathematical models..."`, phương pháp chọn lấy trọn cả `p45-par-00` (Rank 1) và `p45-par-01` (Rank 2: *"Representing complex human phenomena with mathematical models can come at the cost of removing necessary context"*) vào context. |
| **Tổng kết** | **Điểm Context trung bình trên 5 câu `test`** | **0.40 / 2.00** (1/5 câu đạt mức 2) | **1.60 / 2.00** (4/5 câu đạt mức 2) | **ContextHit trên tập `test` tăng gấp 4 lần: từ `20.00%` (`1/5`) lên `80.00%` (`4/5`).** |

---

## Agent và đánh giá

### 1. Kiến trúc vòng lặp Agentic RAG (`starter/agentic.py` & `src/agent_runner.py`)
Em giữ nguyên vẹn các rào chắn an toàn trong `starter/agentic.py` và kết nối với bộ điều khiển thực tế `OpenAIController` (`src/agent_runner.py`, sử dụng model **`gpt-6-luna`**):
1. **Không trả lời từ trí nhớ (No Memory Answering):** Ở lượt đầu tiên khi `context` đang rỗng, nếu LLM cố tình trả về `action="answer"`, `run_agent` lập tức chặn lại và ghi sự kiện `"blocked_answer_without_search"`.
2. **Viết lại truy vấn đa bước & Giới hạn ngân sách (`max_searches = 3`, `budget = 2400`):** Ở mỗi lượt, `OpenAIController` kiểm tra danh sách `past_queries` và `context` hiện tại. Nếu bằng chứng chưa đủ trả lời trọn vẹn mọi vế của câu hỏi, Agent sinh truy vấn tiếng Anh mới (`action="search"`). Toàn bộ các chunk thu được qua các lượt được khử trùng lặp theo `id` và đóng gói nghiêm ngặt trong `budget = 2,400` ký tự bởi `pack_context`.
3. **Kiểm tra trích dẫn nghiêm ngặt (Citation Verification):** Khi Agent trả lời (`action="answer"`), mọi mã `[chunk_id]` trong `citations` bắt buộc phải thuộc tập `context_ids` thực sự đã được nạp vào `context`. Nếu trích dẫn rỗng hoặc chứa bất kỳ ID bịa đặt nào ngoài `context_ids`, hệ thống ép sang trạng thái từ chối an toàn (`status="abstain"`, sự kiện `"invalid_or_missing_citations"`).
4. **Hành vi từ chối có kiểm chứng (Search-then-Abstain):** Với các câu hỏi không có đáp án (`u01`–`u04`), Agent bắt buộc thực hiện ít nhất 2 lượt tìm kiếm với các góc tiếp cận từ khóa khác nhau; khi xác nhận tài liệu không đề cập, Agent trả về `answer=""` để kích hoạt trạng thái `status="abstain"` (`"Không đủ bằng chứng để trả lời."`).

### 2. Bảng tổng hợp kết quả Agentic RAG (Đã kiểm định bằng `starter/agent_metrics.py`)
Toàn bộ kết quả dưới đây được tổng hợp tự động bằng lệnh `starter/agent_metrics.py` từ `traces.json` và nhãn chấm thủ công từng câu trong `reviews.json` (`starter/results/agent_*/summary.json`):

| Thước đo đánh giá (Metric) | `agent_dev_selected` (`dev`: 15 câu có đáp án + 2 câu `u01`–`u02`) | `agent_test_baseline` (`test` TF-IDF, `budget=800`: 5 câu + 2 câu `u03`–`u04`) | `agent_test_selected` (`test` Hybrid RRF, `budget=2400`: 5 câu + 2 câu `u03`–`u04`) |
| :--- | :---: | :---: | :---: |
| **1. Tỷ lệ dùng tool đúng lúc (`tool_use`)** | **17 / 17 (100.0%)** | **7 / 7 (100.0%)** | **7 / 7 (100.0%)** |
| **2. Câu trả lời dựa trên bằng chứng (`grounded_answer`)** | Mức 2: **13/15** \| Mức 1: **0/15** \| Mức 0: **2/15** (**TB: 1.73 / 2.00**) | Mức 2: **2/5** \| Mức 1: **0/5** \| Mức 0: **3/5** (**TB: 0.80 / 2.00**) | Mức 2: **4/5** \| Mức 1: **0/5** \| Mức 0: **1/5** (**TB: 1.60 / 2.00**) |
| **3. Độ chính xác trích dẫn (`citation_validity`)** | **19 / 19 (100.0%)** | **2 / 2 (100.0%)** | **4 / 4 (100.0%)** |
| **4. Tỷ lệ câu trả lời có trích dẫn (`answer_citation_coverage`)** | **13 / 13 (100.0%)** | **2 / 2 (100.0%)** | **4 / 4 (100.0%)** |
| **5. Từ chối đúng câu không có đáp án (`abstention`)** | **2 / 2 (100.0%)** (`u01`, `u02`) | **2 / 2 (100.0%)** (`u03`, `u04`) | **2 / 2 (100.0%)** (`u03`, `u04`) |
| **6. Từ chối sai trên câu có đáp án (`false_abstention`)** | **2 / 15 (13.33%)** (`q01`, `q04`) | **3 / 5 (60.00%)** (`q16`, `q19`, `q20`) | **1 / 5 (20.00%)** (`q16`) |
| **7. Lỗi hạ tầng / API (`provider_errors`)** | **0** | **0** | **0** |
| **8. Số lượt gọi `search` trung bình / câu** | 1.41 lượt | 1.71 lượt | 1.43 lượt |
| **9. Độ trễ trung vị / p95 mỗi câu** | 4,923.7 ms / 9,197.2 ms | 4,207.7 ms / 4,353.3 ms | 5,007.2 ms / 5,349.8 ms |
| **10. Tổng chi phí LLM `gpt-6-luna` (USD)** | \$0.008057 (17 cases) | \$0.002969 (7 cases) | \$0.003143 (7 cases) |

### 3. Bảng đối chiếu chi tiết từng tình huống của Agent trên tập `test` (`q16`–`q20`, `u03`–`u04`)

| Case ID | Loại câu | Truy vấn tiếng Anh do Agent (`gpt-6-luna`) tự sinh | Số lượt `search` | Trạng thái (`status`) | Trích dẫn (`citations`) | Điểm Grounded / Abstention | Nhận xét kiểm định thủ công |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| `q16` | Có đáp án (`GOVERN 4.3`) | L1: `"NIST AI RMF GOVERN 4.3 testing practices AI incidents"`<br>L2: `"NIST AI RMF GOVERN 4.3 organizational practices testing AI systems incident response documentation"` | 2 | `abstain` | `[]` | **0 / 2** (*False Abstention*) | Cả 2 lượt tìm kiếm đều lấy các đoạn nói về đánh giá hiệu quả AI RMF (`p24-par-00`), `MEASURE 1` (`p34-par-00`) và `MANAGE 1` (`p37-par-00`), không kéo được `p29-par-00` vào context. Agent trung thực từ chối trả lời thay vì bịa nội dung `GOVERN 4.3`. |
| `q17` | Có đáp án (`MEASURE 3.3`) | `"NIST AI RMF MEASURE 3.3 feedback processes end users and impacted communities purpose"` | 1 | `answered` | `["nist-ai-rmf-1.0-p36-par-00"]` | **2 / 2** | Trả lời đầy đủ mục đích báo cáo vấn đề (*report problems*), khiếu nại kết quả hệ thống (*appeal system outcomes*) và tích hợp vào chỉ số đánh giá AI. |
| `q18` | Có đáp án (`MANAGE 4.1`) | `"NIST AI RMF MANAGE 4.1 post-deployment monitoring plan mechanisms"` | 1 | `answered` | `["nist-ai-rmf-1.0-p38-par-00"]` | **2 / 2** | Liệt kê chính xác 6 cơ chế trong `MANAGE 4.1`: thu thập/đánh giá ý kiến người dùng & AI actors, kháng nghị và ghi đè (*appeal and override*), ngừng vận hành, ứng phó sự cố, khôi phục và quản lý thay đổi. |
| `q19` | Có đáp án (`Appendix B`, có `filter`) | `"Appendix B AI system features pose risks while providing benefits improve accuracy and resilience"` | 1 | `answered` | `["nist-ai-rmf-1.0-p43-par-00"]` | **2 / 2** | Nhờ `filter={"section": "appendix"}` và Hybrid RRF, Agent tìm ra ngay `pre-trained models` và `transfer learning` ở trang 43 ngay lượt đầu tiên (trong khi Agent Baseline bị `abstain`). |
| `q20` | Có đáp án (`Appendix C`) | `"Appendix C human-AI interaction limitations mathematical models represent complex human phenomena"` | 1 | `answered` | `["nist-ai-rmf-1.0-p45-par-01"]` | **2 / 2** | Trích xuất chính xác ý cốt lõi ở trang 45: biểu diễn hiện tượng con người bằng mô hình toán học làm mất bối cảnh cần thiết (*removing necessary context*), gây khó khăn khi đánh giá tác động cá nhân và xã hội. |
| `u03` | Không có đáp án (Bẫy ngưỡng FLOPs / tham số) | L1: `"AI RMF 1.0 MEASURE function minimum model parameter count training compute FLOPs severe risk threshold"`<br>L2: `"NIST AI RMF 1.0 Appendix A model scale number of parameters training compute FLOPs threshold high impact risk"` | 2 | `abstain` | `[]` | **Đúng (`true`)** | Agent tìm kiếm 2 lượt ở cả phần `MEASURE` (trang 34–35) và `Appendix`, xác nhận tài liệu không quy định ngưỡng FLOPs/parameters và từ chối trả lời chính xác. |
| `u04` | Không có đáp án (Bẫy hạn ngày báo cáo Quốc hội) | L1: `"Appendix A congressional reporting deadline days serious AI incident developers report to Congress"`<br>L2: `"Appendix A AI RMF 1.0 congressional reporting deadline serious incident reporting Congress"` | 2 | `abstain` | `[]` | **Đúng (`true`)** | Agent tra cứu 2 lượt quanh chủ đề báo cáo sự cố (`MAP 5.1`, `Appendix C`, `Appendix D`), xác nhận không có quy định số ngày báo cáo Quốc hội và từ chối an toàn. |

### 4. Phân tích ít nhất 3 trường hợp lỗi cụ thể và bài học rút ra
Qua quá trình kiểm tra nhật ký `traces.json` trên cả tập `dev` và `test`, em bóc tách **3 trường hợp lỗi điển hình** để thấy rõ sự tương tác giữa Chunking, Retrieval, Context Packing và Agent:

1. **Lỗi 1 — Rớt bằng chứng ở biên ngân sách ký tự (`q04` trên `agent_dev_selected` và `q16` trên `test_baseline_tfidf`):**
   - **Hiện tượng:** Ở câu `q04` (*"Nếu không thể đo lường được rủi ro AI một cách phù hợp thì có thể kết luận hệ thống AI đó có rủi ro cao hoặc rủi ro thấp hay không?"*), ngay từ lượt tìm kiếm đầu tiên (`"AI risk cannot be measured appropriately determine high or low risk..."`), bộ truy hồi Hybrid RRF đã tìm ra đúng chunk vàng `nist-ai-rmf-1.0-p10-par-00` ở **Rank 3** (chứa nguyên văn câu *"The inability to appropriately measure AI risks does not imply that an AI system necessarily poses either a high or low risk"*). Tuy nhiên, 2 chunk đứng trước là `p11-par-00` (835 ký tự tính cả header) và `p30-par-00` (794 ký tự tính cả header) đã chiếm `1,631` ký tự; khi ghép thêm `p10-par-00` (771 ký tự tính cả header) thì tổng độ dài là **`2,404` ký tự — vượt quá ngân sách `budget = 2,400` đúng 4 ký tự!** Tương tự, ở `q16` của `test_baseline_tfidf` (`budget = 800`), chunk vàng `p29-par-00` đứng **Rank 1** nhưng dài `794` ký tự, cộng thêm header 74 ký tự thành `868 > 800` nên bị loại bỏ trắng.
   - **Nguyên nhân gốc rễ:** Hàm `pack_context` kiểm tra điều kiện cứng `len(candidate) <= budget` theo thứ tự xuất hiện. Khi `chunk_size = 800` ký tự được ghép thêm chuỗi header `[id | source | page]\n`, một chunk có thể dài tới ~870 ký tự (vượt mốc 800), và 3 chunks có thể dài tới ~2,450 ký tự (vượt mốc 2,400).
   - **Hướng khắc phục:** Đặt `budget >= 3 * (chunk_size + 100)` (tức khoảng `2,700` ký tự cho `top_k = 3`, `chunk_size = 800`) hoặc thêm bước Reranker để đẩy chunk chứa câu trả lời trực tiếp lên Rank 1 trước khi đóng gói.

2. **Lỗi 2 — Tích lũy ngữ cảnh theo thứ tự FIFO làm nghẽn bằng chứng ở các lượt tìm kiếm sau (`q01` trên `agent_dev_selected`):**
   - **Hiện tượng:** Ở câu `q01` (*định nghĩa AI system ở trang 6*), lượt tìm kiếm 1 (`"AI system definition NIST AI RMF 1.0"`) trả về 3 đoạn giới thiệu chung (`p08-par-02`, `p47-par-00`, `p07-par-05`) chiếm gần hết ngân sách `2,400` ký tự. Đến lượt tìm kiếm 3, Agent đã tự sửa truy vấn rất chuẩn thành `'"AI system" "machine-based system" NIST AI RMF definition'`, nhưng vì hàm `run_agent` nối kết quả mới vào cuối mảng `collected` (`collected.extend(results)`) rồi gọi `pack_context(collected, budget=2400)` từ đầu mảng `collected`, nên các chunk cũ từ lượt 1 vẫn chiếm hết chỗ trong `context`, khiến chunk mới tìm được ở lượt 3 không thể lọt vào `context_ids`!
   - **Nguyên nhân gốc rễ:** Cơ chế tích lũy `collected` theo thứ tự FIFO ưu tiên giữ lại các chunk của lượt tìm kiếm đầu tiên (vốn là lượt có truy vấn chưa tốt nên Agent mới phải tìm lại lần 2, lần 3).
   - **Hướng khắc phục:** Khi Agent chủ động gọi `search` lần thứ 2 hoặc thứ 3, cần ưu tiên đóng gói các kết quả mới nhất (LIFO hoặc chấm điểm lại theo truy vấn mới) trước khi lấp đầy phần dư bằng kết quả cũ.

3. **Lỗi 3 — Trượt tiểu mục trong bảng dài do nhiễu từ khóa chung (`q16` trên `agent_test_selected`):**
   - **Hiện tượng:** Câu `q16` hỏi về tiểu mục `GOVERN 4.3` (*"Organizational practices are in place to enable AI testing, identification of incidents, and information sharing"* — nằm trong `nist-ai-rmf-1.0-p29-par-00`). Khi Agent sinh truy vấn `"NIST AI RMF GOVERN 4.3 testing practices AI incidents"`, các từ khóa phổ biến như `AI RMF`, `practices`, `testing`, `incidents` xuất hiện dày đặc ở các đoạn tổng quan (`p24-par-00`, `p34-par-00`, `p37-par-00`), trong khi mã số `"4.3"` là chuỗi ngắn dễ bị pha loãng trong vector ngữ nghĩa 1536 chiều. Kết quả là cả 2 lượt tìm kiếm đều không đưa được `p29-par-00` vào top 3, dẫn đến trường hợp *False Abstention* duy nhất trên tập `test` (`1/5`).
   - **Nguyên nhân gốc rễ:** Với các truy vấn chứa mã định danh chính xác (`GOVERN 4.3`, `MEASURE 3.3`, `MANAGE 4.1`), trọng số BM25 hiện tại (`w_lexical = 0.8` so với `w_dense = 1.0`) và việc tách token `"4.3"` thành `"4"` và `"3"` làm giảm lợi thế khớp chính xác cụm mã tiểu mục.
   - **Hướng khắc phục:** Giữ nguyên token mã số có dấu chấm (regex `\d+\.\d+`) trong bộ tách từ BM25 và tự động tăng trọng số `w_lexical` khi phát hiện truy vấn có chứa mã tiểu mục định dạng `GOVERN/MAP/MEASURE/MANAGE X.Y`.

---

## Thử thách nâng cao (nếu thực hiện)

Em đã thực hiện trọn vẹn phần **Thử thách nâng cao (+15 điểm)** theo đúng yêu cầu của `ADVANCED.md`: xây dựng hệ thống **Multimodal Agentic RAG** trên một báo cáo thực tế quy mô lớn (**110 trang PDF**) có cấu trúc phân cấp phức tạp, nhiều bảng biểu số liệu, biểu đồ định lượng và sơ đồ quy trình.

### 1. Hồ sơ tài liệu 110 trang và cấu trúc Index đa phương thức (`data/advanced/`, `src/advanced_indexer.py`)
- **Tài liệu lựa chọn:** Báo cáo thường niên của Ngân hàng Thế giới — **World Bank *World Development Report 2022: Finance for an Equitable Recovery*** (Giấy phép mở CC BY 3.0 IGO, nguồn UN Digital Library).
- **Cắt gọn 110 trang nội dung cốt lõi (`data/advanced/WDR2022_110p.pdf`):** Từ bản gốc của World Bank, em loại bỏ 12 trang bìa/mục lục đầu (Front Matter) và trích xuất chính xác **110 trang nội dung liên tục** (tương ứng trang vật lý `13–122` của bản gốc, đánh số lại từ `page 1` đến `page 110` trong `WDR2022_110p.pdf`), bao phủ trọn vẹn:
  - *Foreword, Preface & Abbreviations* (trang 1–10)
  - *Overview* (trang 11–34)
  - *Introduction* (trang 35–58)
  - *Chapter 1: Emerging risks to the recovery* (trang 59–88)
  - *Chapter 2: Resolving bank asset distress & Spotlight 2.1* (trang 89–110)
- **Trích xuất đa phương thức (`src/advanced_indexer.py`):**
  1. **528 Chunks đa phương thức (`data/advanced/chunks.json` & `data/advanced/vector_store.json`):**
     - **497 Text Chunks (`chunk_type = "text"`):** Chia theo đoạn văn có nhận diện mục lục (`max_size = 900` ký tự, `overlap = 80` ký tự), bảo toàn số trang PDF (`1..110`), trang gốc (`original_phys_page 13..122`), nhãn chương (`section`) và đường dẫn ảnh chụp trang tương ứng.
     - **4 Table Chunks (`chunk_type = "table"`):** Phát hiện tự động bằng `page.find_tables()` của PyMuPDF tại các trang **42, 74, 98, 108**, chuyển đổi cấu trúc hàng/cột sang bảng Markdown chuẩn và cắt ảnh bảng độ phân giải cao lưu tại `data/advanced/crops/` (ví dụ: *Table 1.1: Classification of macroprudential policy tools* ở trang 74; *Table B2.2.1: Examples of forbearance measures* ở trang 98; bảng phân loại công cụ xử lý ngân hàng yếu kém ở trang 108).
     - **27 Visual Chunks (`chunk_type = "visual"`):** Toàn bộ **27 trang có biểu đồ, đồ thị hoặc sơ đồ quy trình** (ví dụ: *Figure O.1*, *Figure O.2*, *Figure O.3*, *Figure 1.1*, *Figure B1.1.1*, *Figure 1.5*, *Figure 2.1*, *Figure B2.1.1*, *Figure 2.2*, *Figure 2.3*, *Figure 2.4*...) được kết xuất thành ảnh PNG 150 DPI tại `data/advanced/pages/page_XXX.png` và đưa qua **`gpt-6-luna` Vision** trong lúc lập chỉ mục để trích xuất bản mô tả chi tiết từng trục, nhãn, con số, xu hướng và dòng chảy mũi tên.
  2. **Kho Vector Đa phương thức (`data/advanced/vector_store.json`):** Toàn bộ 528 chunks (text + bảng Markdown + mô tả visual) được nhúng bằng **`text-embedding-3-small` (1536 chiều)** với tổng cộng **95,110 embedding tokens**.
  3. **Chi phí và thời gian lập chỉ mục (`data/advanced/manifest.json`):**
     - Phí Vision Indexing (`gpt-6-luna` trên 27 trang biểu đồ/sơ đồ): **`$0.017107`** (`90,564` prompt tokens, `5,871` completion tokens).
     - Phí Embedding Indexing (`text-embedding-3-small` cho 528 chunks): **`$0.001902`** (`95,110` tokens).
     - **Tổng chi phí lập chỉ mục toàn bộ 110 trang đa phương thức:** **`$0.019009`**.

### 2. Kiến trúc Multimodal Agentic RAG 2 công cụ (`src/advanced_agent.py`)
Để xử lý các câu hỏi đòi hỏi đọc chính xác con số trên biểu đồ hoặc chiều mũi tên trong sơ đồ mà bản tóm tắt văn bản có thể bỏ sót, em trang bị cho Agent (`gpt-6-luna`) **2 công cụ chuyên biệt**:
1. `search_index(query, section=None, page=None, chunk_type=None)`: Tìm kiếm Hybrid RRF (Dense `text-embedding-3-small` + Okapi BM25) trên toàn bộ 528 chunks với bộ lọc metadata (`section`, `page`, `chunk_type` gồm `text` / `table` / `visual`).
2. `inspect_visual_page(page, focus_question)`: Cho phép Agent mở trực tiếp file ảnh chụp trang gốc `data/advanced/pages/page_XXX.png` (150 DPI) và gọi **`gpt-6-luna` Vision** tại thời điểm suy luận (inference-time visual verification) để đọc các chi tiết nhỏ trên đồ thị, chú thích hoặc bảng biểu trước khi kết luận.

### 3. Thiết kế bộ Benchmark nâng cao 48 tình huống (`data/advanced/benchmark.json`)
Em thiết kế **48 tình huống kiểm thử** (`18` câu `dev` và `30` câu `test`), chia theo **6 nhóm năng lực bắt buộc của `ADVANCED.md`**:
1. **`text` — Văn bản phân cấp (7 câu: `adv_dev_txt_01`–`03`, `adv_test_txt_01`–`04`):** Truy xuất thông tin văn bản có cấu trúc mục/tiểu mục sâu (ví dụ: cơ chế đồng thuận của Paris Club; định nghĩa zombies và tác động lên tăng trưởng; quy trình rà soát chất lượng tài sản AQR).
2. **`table` — Bảng số liệu (9 câu: `adv_dev_tab_01`–`03`, `adv_test_tab_01`–`06`):** Đọc, đối chiếu hàng/cột và tính toán từ bảng số liệu (ví dụ: *Table 1.1* trang 74 về công cụ chính sách an toàn vĩ mô; *Table B1.2.1* trang 85 về tỷ lệ trích lập dự phòng ở Ấn Độ; *Table B2.2.1* trang 98 về các biện pháp khoan hồng nợ; bảng công cụ xử lý ngân hàng trang 108).
3. **`chart` — Biểu đồ định lượng (8 câu: `adv_dev_cht_01`–`03`, `adv_test_cht_01`–`05`):** Đọc số liệu định lượng, đỉnh/đáy và xu hướng từ biểu đồ (ví dụ: *Figure O.1*, *Figure 1.1*, *Figure B1.1.1*, *Figure 2.2*, *Figure 2.5*).
4. **`diagram` — Sơ đồ luồng / quy trình (8 câu: `adv_dev_dia_01`–`03`, `adv_test_dia_01`–`05`):** Truy vết luồng nhân quả và sơ đồ khối (ví dụ: *Figure O.2* trang 25 về 4 khu vực kinh tế kết nối rủi ro bảng cân đối kế toán; *Figure O.3* trang 26 về vòng tuần hoàn tích cực *Virtuous cycle*; *Figure 1.5* trang 72 về kênh truyền dẫn rủi ro chủ quyền – ngân hàng; *Figure 2.4* trang 103).
5. **`multihop` — Đa trang / Đa phương thức (7 câu: `adv_dev_mhp_01`–`03`, `adv_test_mhp_01`–`04`):** Tổng hợp bằng chứng nằm rải rác trên nhiều trang và kết hợp giữa hộp văn bản (*Box*) với biểu đồ/bảng biểu (ví dụ: kết hợp *Box 1.2* trang 85 và *Figure B1.2.1* trang 86 về ngân hàng nhà nước tại Ấn Độ).
6. **`unanswerable` — Không có đáp án (9 câu: `adv_dev_una_01`–`03`, `adv_test_una_01`–`06`):** Câu hỏi bẫy hợp lý cùng chủ đề tài chính – ngân hàng nhưng hoàn toàn không có trong báo cáo WDR 2022 (ví dụ: tỷ lệ trích lập dự phòng cho khoản vay thế chấp tiền mã hóa trong *Table B1.2.1* trang 85; dự báo tỷ lệ nợ xấu NPL của Serbia cho năm 2025 trong *Figure 2.5* trang 121; hay số tiền USD chính xác Fed mua trái phiếu doanh nghiệp vào tháng 6/2020 trong *Box I.1* trang 65).

### 4. Kết quả thực nghiệm trên tập `dev` (18 cases) và `test` (30 cases)

Bảng tổng hợp từ `starter/results/advanced_dev/summary.json`, `starter/results/advanced_test/metrics.json` và `starter/results/advanced_test/summary.json`:

| Nhóm câu hỏi (`Category`) | Số câu (`dev` / `test`) | Kết quả đạt trên `dev` (`advanced_dev`) | Kết quả đạt trên `test` (`advanced_test`) | Công cụ Agent đã kích hoạt trên `test` |
| :--- | :---: | :---: | :---: | :--- |
| **1. Văn bản phân cấp (`text`)** | 3 / 4 | **3 / 3 (100.0%)** | **4 / 4 (100.0%)** | `search` (kết hợp lọc `section`/`page`) |
| **2. Bảng số liệu (`table`)** | 3 / 6 | **3 / 3 (100.0%)** | **6 / 6 (100.0%)** | `search(chunk_type="table")` + `inspect_page` |
| **3. Biểu đồ định lượng (`chart`)** | 3 / 5 | **2 / 3 (66.7%)** | **5 / 5 (100.0%)** | `search(chunk_type="visual")` + `inspect_page` |
| **4. Sơ đồ luồng / quy trình (`diagram`)** | 3 / 5 | **3 / 3 (100.0%)** | **5 / 5 (100.0%)** | `search(chunk_type="visual")` + `inspect_page` |
| **5. Đa trang / Đa phương thức (`multihop`)** | 3 / 4 | **3 / 3 (100.0%)** | **3 / 4 (75.0%)** | Đa lượt `search` + `inspect_page` liên trang |
| **6. Không có đáp án (`unanswerable`)** | 3 / 6 | **3 / 3 (100.0%)** | **6 / 6 (100.0%)** | Tra cứu kiểm chứng rồi `abstain` |
| **TỔNG KẾT TOÀN BỘ** | **18 / 30** | **17 / 18 (94.44%)** | **29 / 30 (96.67%)** | **47 lượt gọi tool trên 30 câu `test` (0 lỗi API)** |

**Các chỉ số vận hành và chi phí thực tế trên tập `test` (30 cases — `starter/results/advanced_test/metrics.json`):**
- **Điểm Grounded Answer trung bình (24 câu có đáp án):** **1.92 / 2.00** (23/24 câu đạt điểm tối đa `2/2`, chỉ 1 câu `adv_test_mhp_03` bị `false_abstention`).
- **Độ chính xác trích dẫn (`citation_validity`) & Độ phủ trích dẫn (`answer_citation_coverage`):** **100.0%** (`32/32` trích dẫn hợp lệ; `23/23` câu trả lời đều trích dẫn đúng mã chunk `[wdr2022-pXXX-...]` hoặc `[page:XX]` trong context).
- **Tỷ lệ từ chối đúng câu không có đáp án (`abstention`):** **6 / 6 (100.0%)** (`adv_test_una_01`–`adv_test_una_06`).
- **Độ trễ (`latency`):** Trung bình **`9,551.75 ms`** (~9.55 giây/câu), p95 **`22,150.76 ms`** (~22.15 giây/câu do có bước gửi ảnh PNG qua `gpt-6-luna` Vision).
- **Chi phí thực thi 30 câu `test`:** Tổng cộng **`$0.034699`** (gồm `$0.000013` phí query embedding, `$0.010271` phí `inspect_page` Vision và `$0.024415` phí LLM điều phối `gpt-6-luna`).

### 5. Phân tích lỗi trên phần Nâng cao (`adv_test_mhp_03`)
- **Trường hợp `adv_test_mhp_03` (Câu hỏi đa bước kết hợp *Box 1.2* trang 85 và *Figure B1.2.1* trang 86 — *False Abstention*):** Câu hỏi yêu cầu kết hợp lý giải văn bản ở *Box 1.2* (vì sao các ngân hàng quốc doanh Ấn Độ tích tụ nhiều nợ xấu hơn ngân hàng tư nhân sau chính sách khoan hồng năm 2008) với con số tỷ trọng ngân hàng quốc doanh trong hệ thống ngân hàng Ấn Độ ở trang 86. Do câu hỏi đòi hỏi cả tìm kiếm văn bản trên 2 trang liên tiếp lẫn gọi `inspect_page` đọc biểu đồ *Figure B1.2.1*, sau khi sử dụng hết các bước tìm kiếm và kiểm tra hình ảnh mà chưa gom đủ đồng thời cả hai vế định tính và định lượng, Agent đã chọn `abstain` an toàn thay vì trả lời thiếu vế.
- **Bài học thiết kế cho tài liệu quy mô lớn 100+ trang:** Việc kết hợp **bản mô tả biểu đồ bằng Vision ngay lúc lập chỉ mục (Index-time Vision Captioning)** với **công cụ soi ảnh trang gốc lúc suy luận (`inspect_page`)** là chìa khóa giúp Agent vừa tìm đúng trang chứa biểu đồ bằng Hybrid RRF (`search`), vừa đọc chính xác từng con số và nhãn trục trên ảnh PNG gốc mà không cần nạp hàng trăm ảnh vào ngữ cảnh cùng lúc.
