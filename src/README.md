# Source Modules (`src/`) — Day 7 Agentic RAG

Thư mục này chứa mã nguồn hiện thực của sinh viên **Nguyễn Thành Danh (`2A202603493`)**, kết nối dữ liệu từ Day 6 và mở rộng trên khung [starter](../starter/README.md) (bao gồm cả Phần cơ bản 85 điểm và Thử thách nâng cao +15 điểm).

## Cấu trúc các module

### A. Phần cơ bản (85 điểm — NIST AI RMF 1.0)
1. **[src/export_day6.py](export_day6.py)**:
   - Kết nối PostgreSQL + pgvector từ Day 6 (hoặc tái tạo tất định từ PDF gốc + `by_paragraph(max_size=800)` khi container dừng) để xuất `173` chunks (Chiến lược B/C) ra `data/chunks.json`.
   - Chuyển đổi `20` câu hỏi có đáp án (`15 dev` + `5 test`) sang `data/questions.json` và kiểm chứng bằng `validate()`.
   - Tổng hợp `24` câu hỏi (`20` câu có đáp án + `4` câu thiếu bằng chứng `u01`..`u04`) vào `data/cases.json`.
2. **[src/retrievers.py](retrievers.py)**:
   - `DensePgVectorRetriever`: Truy xuất ngữ nghĩa dùng model `text-embedding-3-small` (1536 chiều) trên bảng `rag_chunks` (pgvector) hoặc kho vector `data/day6_vector_store.json`.
   - `BM25Retriever`: Truy xuất từ khóa theo thuật toán Okapi BM25 ($k_1=1.5, b=0.75$) có chuẩn hóa Unicode NFC.
   - `HybridRRFRetriever`: Kết hợp hạng giữa Dense và Lexical (BM25) bằng Reciprocal Rank Fusion ($k=60$).
   - `QueryRewriteHybridRetriever`: Dùng LLM dịch/mở rộng truy vấn tiếng Việt sang thuật ngữ chuẩn tiếng Anh của NIST AI RMF 1.0 trước khi truy xuất Hybrid RRF.
3. **[src/agent_runner.py](agent_runner.py)**:
   - `OpenAIController`: Hiện thực giao thức `Controller` trong `starter/agentic.py`, điều phối vòng lặp `search` (tối đa 3 bước) và `answer` (kèm `[chunk_id]` hoặc `abstain=true`).

### B. Thử thách nâng cao (+15 điểm — WDR 2022 110 trang đa phương thức)
4. **[src/advanced_indexer.py](advanced_indexer.py)**:
   - Index toàn bộ 110 trang của `data/advanced/WDR2022_110p.pdf` (gồm `497` text chunks, `4` bảng layout kèm ảnh crop `data/advanced/crops/`, `27` trang hình/biểu đồ/sơ đồ qua `gpt-6-luna` Vision kèm ảnh PNG `data/advanced/pages/`) và tạo vector 1536 chiều (`text-embedding-3-small`) lưu tại `data/advanced/vector_store.json`.
5. **[src/advanced_agent.py](advanced_agent.py)**:
   - `MultimodalAgent` (`gpt-6-luna`): Điều phối 2 công cụ `search` (Hybrid RRF trên toàn bộ 528 chunks) và `inspect_page` (xem trực tiếp ảnh PNG trang gốc bằng `gpt-6-luna` Vision) để trả lời bộ benchmark 18 câu `dev` và 30 câu `test` (`data/advanced/benchmark.json`).

## Lệnh chạy tái lập từ thư mục gốc

```bash
# 1. Phần cơ bản (85 điểm)
python src/export_day6.py
python src/agent_runner.py --mode all

# 2. Phần nâng cao (+15 điểm)
python src/advanced_indexer.py
python src/advanced_agent.py --split dev --out starter/results/advanced_dev
python src/advanced_agent.py --split test --out starter/results/advanced_test

# 3. Chạy bộ kiểm thử đơn vị (Unit Tests)
python -m unittest discover -s tests -v
```
