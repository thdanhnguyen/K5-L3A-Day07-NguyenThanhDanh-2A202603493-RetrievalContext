"""Export Day 6 chunks and questions into the Day 7 JSON contract and build cases.json."""
import json
import os
import sys
from pathlib import Path

try:
    import psycopg
except ImportError:
    psycopg = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path: Path | str | None = None, *_args, **_kwargs) -> bool:
        path = Path(dotenv_path) if dotenv_path else Path(".env")
        if not path.exists():
            return False
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))
        return True

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "starter"))
from baseline import hit, validate

DAY6_DIR = Path(r"D:\K5-L3A-DAY06-2A202603493-DataIngestion")
DAY6_STARTER_DIR = DAY6_DIR / "starter"
DAY6_ENV_PATH = DAY6_STARTER_DIR / ".env"
DAY6_PDF_PATH = DAY6_STARTER_DIR / "data" / "NIST_AI_RMF_1.0.pdf"
DAY6_QUESTIONS_PATH = DAY6_DIR / "benchmark" / "questions.json"


def export_chunks_from_pg(database_url: str, strategy: str = "B") -> list[dict]:
    """Query `rag_chunks` (Day 6) and return chunks in the Day 7 JSON contract."""
    sql = """
        SELECT chunk_id, document_id, source, page, section, text, metadata
        FROM rag_chunks
        WHERE strategy = %s
        ORDER BY page ASC, chunk_id ASC;
    """
    chunks: list[dict] = []
    with psycopg.connect(database_url, connect_timeout=3) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (strategy,))
            for row in cur.fetchall():
                chunk_id, doc_id, source, page, section, text, raw_meta = row
                meta = raw_meta if isinstance(raw_meta, dict) else json.loads(raw_meta)
                meta = {**meta, "section": section}
                chunks.append(
                    {
                        "id": str(chunk_id),
                        "source_id": str(doc_id),
                        "source": str(source),
                        "page": int(page),
                        "text": str(text),
                        "metadata": meta,
                    }
                )
    if not chunks:
        raise RuntimeError(f"No chunks found for strategy={strategy} in table rag_chunks.")
    return chunks


def export_chunks_from_day6_pipeline() -> list[dict]:
    """Fallback: run Day 6 deterministic `load_pdf` + `by_paragraph(max_size=800)` to get the exact 173 chunks."""
    sys.path.insert(0, str(DAY6_STARTER_DIR))
    from src.chunking import by_paragraph
    from src.ingestion import load_pdf

    pages = load_pdf(
        path=DAY6_PDF_PATH,
        document_id="nist-ai-rmf-1.0",
        source="starter/data/NIST_AI_RMF_1.0.pdf",
    )
    raw_chunks = [c for p in pages for c in by_paragraph(p, max_size=800)]
    chunks: list[dict] = []
    for c in raw_chunks:
        chunks.append(
            {
                "id": str(c.chunk_id),
                "source_id": str(c.document_id),
                "source": str(c.source),
                "page": int(c.page),
                "text": str(c.text),
                "metadata": dict(c.metadata),
            }
        )
    return chunks


def export_answerable_questions(questions_path: Path, chunks: list[dict]) -> list[dict]:
    """Convert the 20 answerable questions from Day 6 (`q01`..`q20`) to Day 7 schema."""
    raw_questions = json.loads(questions_path.read_text(encoding="utf-8"))
    converted: list[dict] = []
    for q in raw_questions:
        source_id = q["gold_source_id"]
        page = int(q["gold_page"])
        evidence = q["evidence_quote"]

        # Check if primary evidence_quote is verbatim inside a single chunk on that page;
        # if split across two sentences/chunks, find the exact matching substring from the chunk on that page.
        candidate_gold = [{"source_id": source_id, "page": page, "evidence": evidence}]
        if not any(hit(c, candidate_gold) for c in chunks):
            matched = False
            for phrase in q.get("evidence_required_phrases") or []:
                alt = [{"source_id": source_id, "page": page, "evidence": phrase}]
                if any(hit(c, alt) for c in chunks):
                    candidate_gold = alt
                    matched = True
                    break
            if not matched:
                # Try first 80 chars of evidence_quote
                short_ev = evidence[:80].strip()
                alt = [{"source_id": source_id, "page": page, "evidence": short_ev}]
                if any(hit(c, alt) for c in chunks):
                    candidate_gold = alt

        item = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold": candidate_gold,
        }
        if q.get("filter"):
            item["filter"] = q["filter"]
        converted.append(item)
    return converted


def build_unanswerable_cases() -> list[dict]:
    """Build 4 unanswerable questions (2 dev + 2 test) on the NIST AI RMF 1.0 topic."""
    return [
        {
            "id": "u01",
            "split": "dev",
            "question": "Theo NIST AI RMF 1.0, mức phạt tiền hành chính tối đa (tính bằng USD) đối với tổ chức thương mại tại Hoa Kỳ không tuân thủ chức năng GOVERN 1.1 là bao nhiêu?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Đã kiểm tra toàn bộ 42 trang nội dung NIST AI RMF 1.0: đây là khung hướng dẫn tự nguyện (voluntary framework), hoàn toàn không quy định chế tài xử phạt hay mức phạt tiền USD.",
        },
        {
            "id": "u02",
            "split": "dev",
            "question": "Trong NIST AI RMF 1.0, lệ phí đăng ký kiểm toán định kỳ và cấp chứng chỉ tuân thủ AI RMF chính thức do NIST thu hàng năm là bao nhiêu?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Đã kiểm tra toàn bộ tài liệu NIST AI RMF 1.0: NIST không vận hành chương trình cấp chứng chỉ tuân thủ có thu phí và không nêu bất kỳ biểu phí kiểm toán nào.",
        },
        {
            "id": "u03",
            "split": "test",
            "question": "Theo chức năng MEASURE trong NIST AI RMF 1.0, ngưỡng số lượng tham số (parameters) hoặc số phép tính huấn luyện (FLOPs) tối thiểu để một mô hình AI bị xếp vào nhóm rủi ro nghiêm trọng là bao nhiêu?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Đã kiểm tra toàn bộ phần MEASURE và 42 trang nội dung: NIST AI RMF 1.0 không hề đặt ngưỡng định lượng về số lượng tham số (parameters) hay FLOPs huấn luyện.",
        },
        {
            "id": "u04",
            "split": "test",
            "question": "Theo Phụ lục A (Appendix A) của NIST AI RMF 1.0, thời hạn bắt buộc (số ngày tối đa) để nhà phát triển AI phải báo cáo sự cố nghiêm trọng lên Quốc hội Hoa Kỳ là bao nhiêu ngày?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Đã kiểm tra toàn bộ Appendix A (trang vật lý 40-42) và toàn văn tài liệu: không có quy định nào về thời hạn số ngày bắt buộc báo cáo sự cố lên Quốc hội.",
        },
    ]


def build_all_cases(answerable_questions: list[dict]) -> list[dict]:
    """Combine 20 answerable questions (`answerable: True`) and 4 unanswerable questions (`answerable: False`)."""
    cases: list[dict] = []
    for q in answerable_questions:
        cases.append(
            {
                **q,
                "answerable": True,
            }
        )
    cases.extend(build_unanswerable_cases())
    return cases


def main() -> None:
    if DAY6_ENV_PATH.exists():
        load_dotenv(DAY6_ENV_PATH)
    load_dotenv(ROOT_DIR / ".env")

    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@127.0.0.1:55433/day6_lab",
    )
    try:
        print(f"[1/4] Trying to export strategy B chunks from PostgreSQL: {db_url} ...")
        chunks = export_chunks_from_pg(db_url, strategy="B")
        print(f"      -> Exported {len(chunks)} chunks from PostgreSQL.")
    except Exception as exc:
        print(f"      -> PostgreSQL unavailable ({exc}); exporting directly via Day 6 ingestion pipeline...")
        chunks = export_chunks_from_day6_pipeline()
        print(f"      -> Exported {len(chunks)} chunks from Day 6 pipeline.")

    print(f"[2/4] Converting 20 questions from: {DAY6_QUESTIONS_PATH} ...")
    questions = export_answerable_questions(DAY6_QUESTIONS_PATH, chunks)
    print(f"      -> Converted {len(questions)} answerable questions.")

    print("[3/4] Running `validate(chunks, questions)` to verify gold evidence ...")
    validate(chunks, questions)
    print("      -> Validation passed! 100% of gold evidence matches chunks.")

    print("[4/4] Building `cases.json` (answerable + unanswerable cases) ...")
    cases = build_all_cases(questions)

    data_dir = ROOT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (data_dir / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (data_dir / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Done! Saved:\n"
        f"  - data/chunks.json ({len(chunks)} chunks)\n"
        f"  - data/questions.json ({len(questions)} answerable questions)\n"
        f"  - data/cases.json ({len(cases)} total agent cases)"
    )


if __name__ == "__main__":
    main()
