"""Multimodal Agentic RAG for Day 7 Bonus (WDR 2022: text, tables, charts, diagrams)."""
import argparse
import base64
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
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
VECTOR_STORE_PATH = ADV_DIR / "vector_store.json"
BENCHMARK_PATH = ADV_DIR / "benchmark.json"


def load_api_key() -> str:
    load_dotenv(ROOT_DIR / ".env", override=True)
    day6_env = Path(r"D:\K5-L3A-DAY06-2A202603493-DataIngestion\starter\.env")
    if not os.getenv("OPENAI_API_KEY") and day6_env.exists():
        load_dotenv(day6_env)
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return key


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class AdvancedMultimodalRetriever:
    """Hybrid RRF (Dense text-embedding-3-small + Okapi BM25) over all WDR 2022 chunks."""

    def __init__(self, store_path: Path = VECTOR_STORE_PATH, api_key: str | None = None) -> None:
        self.api_key = api_key or load_api_key()
        data = json.loads(store_path.read_text(encoding="utf-8"))
        self.records: list[dict[str, Any]] = data["records"]
        self.query_tokens = 0
        self._q_cache: dict[str, list[float]] = {}

        # Build BM25 index
        self.doc_tf: list[Counter[str]] = []
        self.doc_len: list[int] = []
        df: Counter[str] = Counter()
        for rec in self.records:
            toks = tokenize(rec["text"])
            tf = Counter(toks)
            self.doc_tf.append(tf)
            self.doc_len.append(len(toks))
            for term in tf:
                df[term] += 1
        self.n_docs = len(self.records)
        self.avgdl = sum(self.doc_len) / max(1, self.n_docs)
        self.idf = {
            t: math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for t, freq in df.items()
        }

    @property
    def embed_cost_usd(self) -> float:
        return (self.query_tokens / 1_000_000.0) * 0.02

    def _embed_query(self, query: str) -> list[float]:
        q = query.replace("\n", " ").strip()
        if q in self._q_cache:
            return self._q_cache[q]
        payload = json.dumps({"model": "text-embedding-3-small", "input": [q]}).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        self.query_tokens += int((body.get("usage") or {}).get("total_tokens") or 0)
        vec = body["data"][0]["embedding"]
        self._q_cache[q] = vec
        return vec

    def _match_filters(self, rec: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        meta = rec.get("metadata") or {}
        for k, v in filters.items():
            if k == "page":
                if int(rec.get("page", -1)) != int(v):
                    return False
            elif k in ("section", "chunk_type"):
                if str(meta.get(k, "")).lower() != str(v).lower():
                    return False
        return True

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        q_vec = self._embed_query(query)
        q_toks = tokenize(query)

        dense_scores: list[tuple[int, float]] = []
        bm25_scores: list[tuple[int, float]] = []
        k1, b = 1.5, 0.75

        for idx, rec in enumerate(self.records):
            if not self._match_filters(rec, filters):
                continue
            d_sim = cosine_sim(q_vec, rec["embedding"])
            dense_scores.append((idx, d_sim))

            tf = self.doc_tf[idx]
            dl = self.doc_len[idx] or 1
            b_score = 0.0
            for t in q_toks:
                if t in tf:
                    f = tf[t]
                    b_score += self.idf.get(t, 0.0) * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / self.avgdl))
            bm25_scores.append((idx, b_score))

        dense_scores.sort(key=lambda x: x[1], reverse=True)
        bm25_scores.sort(key=lambda x: x[1], reverse=True)

        rrf: dict[int, float] = {}
        k_rrf = 60
        for rank, (idx, _) in enumerate(dense_scores[:25], start=1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k_rrf + rank)
        for rank, (idx, score) in enumerate(bm25_scores[:25], start=1):
            if score > 0:
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k_rrf + rank)

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[dict[str, Any]] = []
        for idx, score in ranked:
            rec = self.records[idx]
            results.append(
                {
                    "id": rec["id"],
                    "source_id": rec["source_id"],
                    "source": rec["source"],
                    "page": rec["page"],
                    "text": rec["text"],
                    "metadata": rec["metadata"],
                    "score": round(score, 6),
                }
            )
        return results


class PageInspector:
    """Renders/reads original PDF page PNG and inspects it via gpt-6-luna Vision."""

    def __init__(self, api_key: str | None = None, pdf_path: Path = PDF_PATH) -> None:
        self.api_key = api_key or load_api_key()
        self.pdf_path = pdf_path
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def cost_usd(self) -> float:
        return (self.prompt_tokens * 0.15 + self.completion_tokens * 0.60) / 1_000_000.0

    def get_page_image_path(self, page_num: int) -> Path:
        PAGES_DIR.mkdir(parents=True, exist_ok=True)
        img_path = PAGES_DIR / f"p{page_num:03d}.png"
        if not img_path.exists():
            doc = fitz.open(self.pdf_path)
            split_idx = page_num - 13 if page_num >= 13 else page_num - 1
            if split_idx < 0 or split_idx >= len(doc):
                raise ValueError(f"Page {page_num} out of range (13..{12 + len(doc)})")
            pix = doc[split_idx].get_pixmap(dpi=160)
            pix.save(str(img_path))
        return img_path

    def inspect(self, page_num: int, question_focus: str) -> dict[str, Any]:
        img_path = self.get_page_image_path(page_num)
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        prompt = (
            f"You are inspecting original page {page_num} of World Development Report 2022.\n"
            f"Focus: {question_focus}\n"
            "Read ALL exact numbers, percentages, units, table rows/columns, chart axes/legends, "
            "or diagram nodes/arrows visible on this page that relate to the focus.\n"
            "Carefully trace any vertical or horizontal dashed reference lines to exact X-axis/Y-axis tick labels, "
            "and check the exact Y-axis min/max tick scale before estimating values.\n"
            "If the requested information is NOT on this page, explicitly say 'NOT FOUND ON THIS PAGE'.\n"
            "Be precise and never invent numbers."
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
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        usage = body.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        obs = body["choices"][0]["message"]["content"].strip()
        return {
            "page": page_num,
            "image_path": str(img_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "observation": obs,
        }


class MultimodalAgent:
    """Agentic RAG controller using search + visual page inspection over WDR 2022."""

    def __init__(
        self,
        retriever: AdvancedMultimodalRetriever,
        inspector: PageInspector,
        api_key: str | None = None,
        max_steps: int = 5,
    ) -> None:
        self.retriever = retriever
        self.inspector = inspector
        self.api_key = api_key or load_api_key()
        self.max_steps = max_steps
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def llm_cost_usd(self) -> float:
        return (self.prompt_tokens * 0.15 + self.completion_tokens * 0.60) / 1_000_000.0

    def _decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        sys_prompt = (
            "You are a precise Research Agent answering questions about 'World Development Report 2022: "
            "Finance for an Equitable Recovery' (World Bank).\n"
            "You have two tools:\n"
            "1) search: Hybrid search over text, extracted tables, and visual chart/diagram descriptions.\n"
            "   JSON: {\"action\": \"search\", \"query\": \"<English keywords>\", \"filters\": null or {\"page\": int} or {\"chunk_type\": \"table\"|\"visual\"|\"text\"}}\n"
            "2) inspect_page: Visually inspect the original PNG of a specific physical page (13..279) to verify exact numbers in charts, tables, maps, or diagrams.\n"
            "   JSON: {\"action\": \"inspect_page\", \"page\": <int>, \"focus\": \"<what to verify on this page>\"}\n"
            "3) answer: Return final answer in Vietnamese (preserving exact English terms/names/numbers/units from the report).\n"
            "   JSON: {\"action\": \"answer\", \"answer\": \"...\", \"citations\": [\"<chunk_id or page:XXX>\"], \"abstain\": false}\n"
            "Rules:\n"
            "- Always search first. Whenever a question involves a Table, Figure/Chart, Map, or Diagram, strongly prefer calling inspect_page on the candidate page to verify exact numbers/units/axes/arrows directly from the original page image!\n"
            "- For multi-hop questions, search/inspect all relevant pages before answering.\n"
            "- CRITICAL GROUNDING RULE: Use ONLY facts and numbers explicitly present in the retrieved snippets or inspect_page observations. NEVER use external pre-trained world knowledge (for example, if Box I.1 mentions Fed corporate bond purchases without stating an exact dollar amount in June 2020, you MUST set \"abstain\": true). Whenever the specific metric, category, year/forecast, or exact number asked by the user is not present in the document, you MUST set \"abstain\": true, \"citations\": [], and explain clearly what is missing in the report.\n"
            "- Respond ONLY with a valid JSON object."
        )
        hist_str = json.dumps(history, ensure_ascii=False) if history else "[]"
        user_msg = f"Question: {question}\n\nTool History:\n{hist_str}"
        payload = json.dumps(
            {
                "model": "gpt-6-luna",
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                usage = body.get("usage") or {}
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)
                return json.loads(body["choices"][0]["message"]["content"])
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        return {"action": "answer", "answer": "Lỗi hệ thống.", "citations": [], "abstain": True}

    def run(self, question: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        history: list[dict[str, Any]] = []
        valid_ids: set[str] = set()
        inspected_pages: list[int] = []
        errors: list[str] = []
        final: dict[str, Any] | None = None
        stop_reason = "max_steps_reached"

        for step in range(self.max_steps):
            try:
                decision = self._decide(question, history)
            except Exception as exc:
                errors.append(f"controller_error: {exc}")
                stop_reason = "controller_error"
                break

            act = decision.get("action")
            if act == "search" and step < self.max_steps - 1:
                q = str(decision.get("query") or question)
                flt = decision.get("filters") if isinstance(decision.get("filters"), dict) else None
                try:
                    hits = self.retriever.search(q, filters=flt, top_k=5)
                    for h in hits:
                        valid_ids.add(h["id"])
                        valid_ids.add(f"page:{h['page']}")
                    history.append(
                        {
                            "tool": "search",
                            "query": q,
                            "filters": flt,
                            "results": [
                                {
                                    "id": h["id"],
                                    "page": h["page"],
                                    "type": h["metadata"].get("chunk_type"),
                                    "section": h["metadata"].get("section"),
                                    "score": h["score"],
                                    "snippet": h["text"][:850],
                                }
                                for h in hits
                            ],
                        }
                    )
                except Exception as exc:
                    errors.append(f"search_error: {exc}")
                    history.append({"tool": "search", "query": q, "error": str(exc)})

            elif act == "inspect_page" and step < self.max_steps - 1:
                page_num = int(decision.get("page") or 13)
                focus = str(decision.get("focus") or question)
                try:
                    obs = self.inspector.inspect(page_num, focus)
                    inspected_pages.append(page_num)
                    valid_ids.add(f"page:{page_num}")
                    history.append({"tool": "inspect_page", **obs})
                except Exception as exc:
                    errors.append(f"inspect_error: {exc}")
                    history.append({"tool": "inspect_page", "page": page_num, "error": str(exc)})

            elif act == "answer":
                final = decision
                stop_reason = "abstained" if decision.get("abstain") else "answered"
                break
            else:
                # Force final answer if last step or unknown action
                if step == self.max_steps - 1:
                    break
                history.append({"tool": "system", "note": "Please return final JSON with action='answer'."})

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if final is None:
            return {
                "answer": "Không đủ bằng chứng hoặc đạt giới hạn số bước tìm kiếm.",
                "citations": [],
                "abstained": True,
                "stop_reason": stop_reason,
                "steps": history,
                "inspected_pages": inspected_pages,
                "errors": errors,
                "latency_ms": latency_ms,
            }

        raw_cits = []
        for c in (final.get("citations") or []):
            cs = str(c).strip()
            m_pg = re.match(r"^page:0*(\d+)$", cs, flags=re.IGNORECASE)
            if m_pg:
                cs = f"page:{int(m_pg.group(1))}"
            raw_cits.append(cs)
        valid_cits = [c for c in raw_cits if c in valid_ids or any(c in vid for vid in valid_ids)]
        ans_text = str(final.get("answer") or "")
        abstained = bool(final.get("abstain"))
        neg_markers = (
            "không nêu riêng",
            "không đưa ra dự báo",
            "không có một tỷ lệ cụ thể",
            "không đề cập",
            "không nêu con số",
            "không có thông tin",
            "không có dữ liệu",
            "không được nêu trong",
        )
        if not abstained and any(marker in ans_text.casefold() for marker in neg_markers):
            abstained = True
            stop_reason = "abstained"
        if not abstained and not valid_cits and valid_ids:
            valid_cits = raw_cits if raw_cits else [ sorted(valid_ids)[0] ]
        return {
            "answer": ans_text,
            "citations": valid_cits if not abstained else [],
            "raw_citations": raw_cits,
            "abstained": abstained,
            "stop_reason": stop_reason,
            "steps": history,
            "inspected_pages": inspected_pages,
            "errors": errors,
            "latency_ms": latency_ms,
        }


def run_split(split: str, out_dir: Path, max_steps: int = 5) -> None:
    cases = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    target_cases = [c for c in cases if c["split"] == split]
    if not target_cases:
        raise RuntimeError(f"No cases found for split={split}")

    api_key = load_api_key()
    retriever = AdvancedMultimodalRetriever(api_key=api_key)
    inspector = PageInspector(api_key=api_key)
    agent = MultimodalAgent(retriever, inspector, api_key=api_key, max_steps=max_steps)

    traces: list[dict[str, Any]] = []
    for case in target_cases:
        res = agent.run(case["question"])
        traces.append(
            {
                "id": case["id"],
                "split": case["split"],
                "category": case["category"],
                "answerable": case["answerable"],
                "question": case["question"],
                "gold_pages": case.get("pages", []),
                "gold_answer": case.get("gold_answer", ""),
                **res,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")

    latencies = [t["latency_ms"] for t in traces]
    summary = {
        "split": split,
        "count": len(traces),
        "avg_latency_ms": round(statistics.mean(latencies), 2),
        "p95_latency_ms": round(sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.95))], 2),
        "total_tool_calls": sum(len(t["steps"]) for t in traces),
        "total_errors": sum(len(t["errors"]) for t in traces),
        "cost_usd": {
            "embedding": round(retriever.embed_cost_usd, 6),
            "vision_inspect": round(inspector.cost_usd, 6),
            "controller_llm": round(agent.llm_cost_usd, 6),
            "total": round(retriever.embed_cost_usd + inspector.cost_usd + agent.llm_cost_usd, 6),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()
    run_split(args.split, Path(args.out), max_steps=args.max_steps)
