"""Multimodal PDF indexer for Day 7 Bonus (100+ pages: WDR 2022)."""
import argparse
import base64
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pymupdf as fitz  # PyMuPDF

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path: Path | str | None = None, *_args: Any, **_kwargs: Any) -> bool:
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
ADV_DIR = ROOT_DIR / "data" / "advanced"
PDF_PATH = ADV_DIR / "WDR2022_110p.pdf"
PAGES_DIR = ADV_DIR / "pages"
CROPS_DIR = ADV_DIR / "crops"
CHUNKS_PATH = ADV_DIR / "chunks.json"
VECTOR_STORE_PATH = ADV_DIR / "vector_store.json"
MANIFEST_PATH = ADV_DIR / "manifest.json"

SOURCE_ID = "world-bank-wdr-2022-110p"
SOURCE_REL = "data/advanced/WDR2022_110p.pdf"
OFFICIAL_URL = "https://digitallibrary.un.org/record/3963937/files/9781464817304.pdf"


def load_api_key() -> str:
    load_dotenv(ROOT_DIR / ".env", override=True)
    day6_env = Path(r"D:\K5-L3A-DAY06-2A202603493-DataIngestion\starter\.env")
    if not os.getenv("OPENAI_API_KEY") and day6_env.exists():
        load_dotenv(day6_env)
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return key


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad\n", "").replace("-\n", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def map_section(phys_page: int) -> str:
    """Deterministic section mapping for WDR 2022 (282 physical pages)."""
    if phys_page <= 12:
        return "Front Matter (Excluded)"
    if phys_page <= 22:
        return "Foreword, Preface & Abbreviations"
    if phys_page <= 46:
        return "Overview"
    if phys_page <= 70:
        return "Introduction"
    if phys_page <= 100:
        return "Chapter 1: Emerging risks to the recovery"
    if phys_page <= 144:
        return "Chapter 2: Resolving bank asset distress & Spotlight 2.1"
    if phys_page <= 178:
        return "Chapter 3: Restructuring firm and household debt"
    if phys_page <= 224:
        return "Chapter 4: Lending during the recovery"
    if phys_page <= 270:
        return "Chapter 5: Managing sovereign debt & Spotlight 5.1"
    if phys_page <= 279:
        return "Chapter 6: Policy priorities for the recovery"
    return "Back Matter (Excluded)"


def chunk_paragraphs(text: str, max_size: int = 900, overlap: int = 80) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paras:
        if len(para) > max_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(para):
                end = min(len(para), start + max_size)
                chunks.append(para[start:end].strip())
                if end >= len(para):
                    break
                start = max(start + 1, end - overlap)
            continue
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks


def format_table_markdown(rows: list[list[Any]]) -> str:
    cleaned_rows = []
    for row in rows:
        cells = [re.sub(r"\s+", " ", str(c or "")).strip() for c in row]
        if any(cells):
            cleaned_rows.append(cells)
    if len(cleaned_rows) < 2:
        return ""
    lines = ["| " + " | ".join(cleaned_rows[0]) + " |", "| " + " | ".join(["---"] * len(cleaned_rows[0])) + " |"]
    for r in cleaned_rows[1:]:
        padded = r + [""] * max(0, len(cleaned_rows[0]) - len(r))
        lines.append("| " + " | ".join(padded[: len(cleaned_rows[0])]) + " |")
    return "\n".join(lines)


def describe_visual_page(api_key: str, image_path: Path, page_num: int, page_text_hint: str) -> tuple[str, int, int]:
    """Use gpt-6-luna Vision to extract structured descriptions of charts, diagrams, maps, and tables."""
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    captions = re.findall(r"(?:Figure|Map|Table|Box)\s+[A-Z]?\d+(?:\.\d+)*[^\n]*", page_text_hint)
    hint = "; ".join(captions[:5]) if captions else "Visual page"

    prompt = (
        f"You are indexing page {page_num} of World Development Report 2022 (World Bank).\n"
        f"Detected captions on this page: {hint}\n"
        "Carefully inspect ALL charts, graphs, diagrams, flowcharts, maps, or tables visible in this page image.\n"
        "For EACH visual element (especially Figures/Maps/Diagrams/Tables):\n"
        "1. State its exact title and type (bar chart, line chart, scatter plot, flowchart, diagram, map, or table).\n"
        "2. List axes, units (%, $ billions, index, years), and all legend categories.\n"
        "3. Transcribe key numeric data points, country/income-group comparisons, trends, or flowchart steps/arrows accurately.\n"
        "Be concise, factual, and preserve exact numbers/labels visible in the image. Do not invent any numbers."
    )
    payload = json.dumps(
        {
            "model": "gpt-6-luna",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                        },
                    ],
                }
            ],
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            usage = body.get("usage") or {}
            pt = int(usage.get("prompt_tokens") or 0)
            ct = int(usage.get("completion_tokens") or 0)
            content = body["choices"][0]["message"]["content"].strip()
            return content, pt, ct
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    return "", 0, 0


def embed_texts(api_key: str, texts: list[str]) -> tuple[list[list[float]], int]:
    all_vecs: list[list[float]] = []
    total_tokens = 0
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = [t.replace("\n", " ").strip()[:8000] or "empty" for t in texts[i : i + batch_size]]
        payload = json.dumps({"model": "text-embedding-3-small", "input": batch}).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        usage = body.get("usage") or {}
        total_tokens += int(usage.get("total_tokens") or 0)
        data = sorted(body["data"], key=lambda x: x["index"])
        all_vecs.extend([d["embedding"] for d in data])
    return all_vecs, total_tokens


def run_index(skip_vision: bool = False) -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Missing {PDF_PATH}")
    api_key = load_api_key()
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    excluded_pages: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    visual_pages_indexed: list[int] = []
    table_count = 0
    text_chunk_count = 0
    vision_prompt_tokens = 0
    vision_completion_tokens = 0

    existing_visual_cache: dict[int, str] = {}
    if CHUNKS_PATH.exists():
        for old in json.loads(CHUNKS_PATH.read_text(encoding="utf-8")):
            if old.get("metadata", {}).get("chunk_type") == "visual":
                existing_visual_cache[int(old["page"])] = old["text"]

    t0 = time.perf_counter()
    for idx in range(total_pages):
        split_page = idx + 1
        phys_page = idx + 13
        page = doc[idx]

        section = map_section(phys_page)
        raw_text = page.get_text("text") or ""
        clean_text = normalize_text(raw_text)
        if not clean_text:
            excluded_pages.append({"page": phys_page, "split_page": split_page, "reason": "Blank/divider page with no extractable text"})
            continue

        # 1. Text chunks
        for p_idx, part in enumerate(chunk_paragraphs(clean_text, max_size=900, overlap=80)):
            chunks.append(
                {
                    "id": f"wdr2022-p{phys_page:03d}-txt-{p_idx:02d}",
                    "source_id": SOURCE_ID,
                    "source": SOURCE_REL,
                    "page": phys_page,
                    "text": part,
                    "metadata": {
                        "section": section,
                        "chunk_type": "text",
                        "split_page": split_page,
                    },
                }
            )
            text_chunk_count += 1

        # 2. Layout table extraction + crop image
        try:
            tabs = page.find_tables()
            for t_idx, tab in enumerate(tabs.tables):
                md_table = format_table_markdown(tab.extract())
                if not md_table or len(md_table) < 40:
                    continue
                bbox = [round(v, 1) for v in tab.bbox]
                clip = fitz.Rect(tab.bbox) + (-10, -24, 10, 10)
                clip = clip & page.rect
                crop_path = CROPS_DIR / f"p{phys_page:03d}_tab_{t_idx:02d}.png"
                if not crop_path.exists():
                    pix = page.get_pixmap(dpi=160, clip=clip)
                    pix.save(str(crop_path))
                chunks.append(
                    {
                        "id": f"wdr2022-p{phys_page:03d}-tab-{t_idx:02d}",
                        "source_id": SOURCE_ID,
                        "source": SOURCE_REL,
                        "page": phys_page,
                        "text": f"[Table extracted from page {phys_page} (split page {split_page})]\n{md_table}",
                        "metadata": {
                            "section": section,
                            "chunk_type": "table",
                            "split_page": split_page,
                            "bbox": bbox,
                            "image_path": str(crop_path.relative_to(ROOT_DIR)).replace("\\", "/"),
                        },
                    }
                )
                table_count += 1
        except Exception:
            pass

        # 3. Visual elements (Figure / Map / Table caption pages)
        has_visual_caption = bool(re.search(r"\b(?:Figure|Map|Table)\s+[A-Z0-9]+(?:\.\d+)+", raw_text))
        if has_visual_caption:
            page_img = PAGES_DIR / f"p{phys_page:03d}.png"
            if not page_img.exists():
                pix = page.get_pixmap(dpi=150)
                pix.save(str(page_img))

            visual_desc = existing_visual_cache.get(phys_page, "")
            if not visual_desc and not skip_vision:
                visual_desc, pt, ct = describe_visual_page(api_key, page_img, phys_page, raw_text)
                vision_prompt_tokens += pt
                vision_completion_tokens += ct

            if visual_desc:
                chunks.append(
                    {
                        "id": f"wdr2022-p{phys_page:03d}-vis-00",
                        "source_id": SOURCE_ID,
                        "source": SOURCE_REL,
                        "page": phys_page,
                        "text": f"[Visual description of page {phys_page} (split page {split_page})]\n{visual_desc}",
                        "metadata": {
                            "section": section,
                            "chunk_type": "visual",
                            "split_page": split_page,
                            "image_path": str(page_img.relative_to(ROOT_DIR)).replace("\\", "/"),
                        },
                    }
                )
                visual_pages_indexed.append(phys_page)

    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4. Embed all chunks with text-embedding-3-small (1536 dims)
    vectors, emb_tokens = embed_texts(api_key, [c["text"] for c in chunks])
    records = [{**c, "embedding": v} for c, v in zip(chunks, vectors)]
    VECTOR_STORE_PATH.write_text(
        json.dumps(
            {
                "model": "text-embedding-3-small",
                "dimensions": 1536,
                "count": len(records),
                "index_tokens": emb_tokens,
                "records": records,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    elapsed = round(time.perf_counter() - t0, 2)
    emb_cost = (emb_tokens / 1_000_000.0) * 0.02
    vision_cost = (vision_prompt_tokens * 0.15 + vision_completion_tokens * 0.60) / 1_000_000.0
    manifest = {
        "source_id": SOURCE_ID,
        "title": "World Development Report 2022: Finance for an Equitable Recovery",
        "publisher": "World Bank Group",
        "isbn": "978-1-4648-1730-4",
        "official_url": OFFICIAL_URL,
        "license": "Creative Commons Attribution 3.0 IGO (CC BY 3.0 IGO)",
        "total_physical_pages": total_pages,
        "indexed_page_range": "1-110 (original WDR 2022 physical pages 13-122)",
        "excluded_pages": excluded_pages,
        "chunk_counts": {
            "total": len(chunks),
            "text": text_chunk_count,
            "table": table_count,
            "visual": len(visual_pages_indexed),
        },
        "visual_pages": visual_pages_indexed,
        "embedding_model": "text-embedding-3-small",
        "dimensions": 1536,
        "vision_model": "gpt-6-luna",
        "tokens": {
            "embedding_tokens": emb_tokens,
            "vision_prompt_tokens": vision_prompt_tokens,
            "vision_completion_tokens": vision_completion_tokens,
        },
        "cost_usd": {
            "embedding": round(emb_cost, 6),
            "vision": round(vision_cost, 6),
            "total": round(emb_cost + vision_cost, 6),
        },
        "elapsed_seconds": elapsed,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "excluded_pages"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-vision", action="store_true")
    args = parser.parse_args()
    run_index(skip_vision=args.skip_vision)
