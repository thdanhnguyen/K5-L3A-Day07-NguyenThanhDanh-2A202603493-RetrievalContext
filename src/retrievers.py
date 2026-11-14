"""Retrieval methods for Day 7: Dense (pgvector / persistent Day 6 vector store), BM25 Lexical, Hybrid RRF, and Query Rewrite Hybrid."""
import json
import math
import os
import re
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import psycopg
except ImportError:
    psycopg = None  # type: ignore[assignment]

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
ALLOWED_FILTER_KEYS = {"section", "page", "source_id", "source"}


def tokenize_text(text: str) -> list[str]:
    """Normalize Unicode NFC, casefold, and split words for BM25."""
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def _vector_literal(vec: list[float]) -> str:
    """Format a float list as a pgvector literal string."""
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute exact cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class DensePgVectorRetriever:
    """Semantic dense retrieval over Day 6 chunks using `text-embedding-3-small` (1536 dims).

    Queries PostgreSQL + pgvector (`rag_chunks` table, `strategy='B'`) when the Day 6
    Docker container is reachable, and automatically syncs/falls back to a persistent local
    vector store (`data/day6_vector_store.json`) built with the exact same real OpenAI
    `text-embedding-3-small` embeddings when Docker is stopped.
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]] | None = None,
        database_url: str | None = None,
        api_key: str | None = None,
        model_name: str = "text-embedding-3-small",
        dimensions: int = 1536,
        chunk_strategy: str = "B",
        local_store_path: Path | None = None,
    ) -> None:
        load_dotenv(ROOT_DIR / ".env", override=True)
        day6_env = Path(r"D:\K5-L3A-DAY06-2A202603493-DataIngestion\starter\.env")
        if not os.getenv("OPENAI_API_KEY") and day6_env.exists():
            load_dotenv(day6_env)

        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"
        )
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")

        self.model_name = model_name
        self.dimensions = dimensions
        self.chunk_strategy = chunk_strategy
        self.local_store_path = local_store_path or (ROOT_DIR / "data" / "day6_vector_store.json")
        self.chunks = chunks or json.loads((ROOT_DIR / "data" / "chunks.json").read_text(encoding="utf-8"))
        self.total_tokens = 0
        self.index_tokens = 0
        self._query_cache: dict[str, list[float]] = {}
        self._pg_available: bool | None = None
        self._local_index: list[dict[str, Any]] | None = None

    @property
    def cost_usd(self) -> float:
        """Query embedding cost for `text-embedding-3-small` ($0.02 / 1M tokens)."""
        return (self.total_tokens / 1_000_000.0) * 0.02

    @property
    def index_cost_usd(self) -> float:
        """Document indexing cost if local vector store was built ($0.02 / 1M tokens)."""
        return (self.index_tokens / 1_000_000.0) * 0.02

    def _call_embeddings_api(self, inputs: list[str], is_index: bool = False) -> list[list[float]]:
        clean_inputs = [t.replace("\n", " ").strip() for t in inputs]
        payload = json.dumps({"model": self.model_name, "input": clean_inputs}).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        usage = body.get("usage") or {}
        tokens = int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
        if is_index:
            self.index_tokens += tokens
        else:
            self.total_tokens += tokens

        data = sorted(body["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in data]

    def embed_query(self, query: str) -> list[float]:
        """Call OpenAI Embeddings API to generate a 1536-dim vector compatible with Day 6."""
        clean_q = query.replace("\n", " ").strip()
        if not clean_q:
            raise ValueError("query must not be empty")
        if clean_q in self._query_cache:
            return self._query_cache[clean_q]

        vec = self._call_embeddings_api([clean_q], is_index=False)[0]
        self._query_cache[clean_q] = vec
        return vec

    def _check_pg(self) -> bool:
        if self._pg_available is not None:
            return self._pg_available
        try:
            with psycopg.connect(self.database_url, connect_timeout=2) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM rag_chunks WHERE strategy = %s;",
                        (self.chunk_strategy,),
                    )
                    cnt = cur.fetchone()[0]
                    self._pg_available = cnt > 0
        except Exception:
            self._pg_available = False
        return self._pg_available

    def _ensure_local_index(self) -> list[dict[str, Any]]:
        if self._local_index is not None:
            return self._local_index

        if self.local_store_path.exists():
            stored = json.loads(self.local_store_path.read_text(encoding="utf-8"))
            if (
                isinstance(stored, dict)
                and stored.get("model") == self.model_name
                and stored.get("dimensions") == self.dimensions
                and len(stored.get("records", [])) == len(self.chunks)
            ):
                self._local_index = stored["records"]
                return self._local_index

        # Build persistent vector index for all 173 Day 6 strategy B chunks using real OpenAI API
        batch_size = 64
        all_vectors: list[list[float]] = []
        texts = [c["text"] for c in self.chunks]
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_vectors.extend(self._call_embeddings_api(batch, is_index=True))

        records: list[dict[str, Any]] = []
        for c, vec in zip(self.chunks, all_vectors):
            records.append({**c, "embedding": vec})

        self.local_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_store_path.write_text(
            json.dumps(
                {
                    "model": self.model_name,
                    "dimensions": self.dimensions,
                    "strategy": self.chunk_strategy,
                    "count": len(records),
                    "index_tokens": self.index_tokens,
                    "records": records,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._local_index = records
        return self._local_index

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Query top_k nearest chunks by exact cosine similarity (via pgvector or persistent vector store)."""
        if filters:
            for k in filters:
                if k not in ALLOWED_FILTER_KEYS:
                    raise ValueError(f"Unsupported filter field: {k}")

        q_vec = self.embed_query(query)

        if self._check_pg():
            q_lit = _vector_literal(q_vec)
            where_clauses = ["strategy = %s"]
            where_params: list[Any] = [self.chunk_strategy]
            if filters:
                for k, v in filters.items():
                    col = "document_id" if k == "source_id" else k
                    where_clauses.append(f"{col} = %s")
                    where_params.append(v)

            where_sql = " AND ".join(where_clauses)
            sql = f"""
                SELECT
                    chunk_id,
                    document_id,
                    source,
                    page,
                    section,
                    text,
                    metadata,
                    1 - (embedding <=> %s::vector) AS cosine_similarity
                FROM rag_chunks
                WHERE {where_sql}
                ORDER BY embedding <=> %s::vector ASC, chunk_id ASC
                LIMIT %s;
            """
            params = [q_lit] + where_params + [q_lit, top_k]
            results: list[dict[str, Any]] = []
            with psycopg.connect(self.database_url, connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    for row in cur.fetchall():
                        meta = row[6] if isinstance(row[6], dict) else json.loads(row[6])
                        meta = {**meta, "section": row[4]}
                        results.append(
                            {
                                "id": str(row[0]),
                                "source_id": str(row[1]),
                                "source": str(row[2]),
                                "page": int(row[3]),
                                "text": str(row[5]),
                                "metadata": meta,
                                "score": round(float(row[7]), 6),
                                "strategy": "dense_pgvector",
                            }
                        )
            return results

        # Exact cosine scan over the persistent 1536-dim Day 6 vector store
        records = self._ensure_local_index()
        scored: list[dict[str, Any]] = []
        for rec in records:
            if filters:
                matched = True
                for k, v in filters.items():
                    if k in rec and rec[k] == v:
                        continue
                    if rec.get("metadata", {}).get(k) == v:
                        continue
                    matched = False
                    break
                if not matched:
                    continue
            sim = _cosine_similarity(q_vec, rec["embedding"])
            item = {k: v for k, v in rec.items() if k != "embedding"}
            scored.append({**item, "score": round(float(sim), 6), "strategy": "dense_pgvector"})

        scored.sort(key=lambda c: (-c["score"], c["id"]))
        return scored[:top_k]


class BM25Retriever:
    """Lexical retrieval using Okapi BM25 with metadata filtering support."""

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize_text(c["text"]) for c in chunks]
        self.doc_freqs = [Counter(toks) for toks in self.doc_tokens]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0

        df = Counter(term for freq in self.doc_freqs for term in freq)
        n_docs = len(chunks)
        self.idf = {
            term: math.log(1.0 + (n_docs - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def _match_filter(self, chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        for k, v in filters.items():
            if k in chunk and chunk[k] == v:
                continue
            if chunk.get("metadata", {}).get(k) == v:
                continue
            return False
        return True

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        q_terms = tokenize_text(query)
        scored: list[dict[str, Any]] = []
        for chunk, tf_map, dl in zip(self.chunks, self.doc_freqs, self.doc_lens):
            if not self._match_filter(chunk, filters):
                continue
            score = 0.0
            for term in q_terms:
                if term not in tf_map:
                    continue
                tf = tf_map[term]
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                score += idf * (tf * (self.k1 + 1.0)) / denom
            if score > 0:
                scored.append({**chunk, "score": round(score, 6), "strategy": "bm25"})
        scored.sort(key=lambda c: (-c["score"], c["id"]))
        return scored[:top_k]


class HybridRRFRetriever:
    """Combine Dense pgvector and Lexical BM25 via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense_retriever: DensePgVectorRetriever,
        lexical_retriever: BM25Retriever,
        k_rrf: int = 60,
        w_dense: float = 1.0,
        w_lexical: float = 0.5,
        candidate_multiplier: int = 4,
    ) -> None:
        self.dense = dense_retriever
        self.lexical = lexical_retriever
        self.k_rrf = k_rrf
        self.w_dense = w_dense
        self.w_lexical = w_lexical
        self.candidate_multiplier = candidate_multiplier

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        cand_k = max(top_k * self.candidate_multiplier, 10)
        dense_hits = self.dense.search(query, filters=filters, top_k=cand_k)
        lex_hits = self.lexical.search(query, filters=filters, top_k=cand_k)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(dense_hits, start=1):
            cid = item["id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.w_dense / (self.k_rrf + rank)

        for rank, item in enumerate(lex_hits, start=1):
            cid = item["id"]
            if cid not in chunk_map:
                chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.w_lexical / (self.k_rrf + rank)

        merged: list[dict[str, Any]] = []
        for cid, score in rrf_scores.items():
            merged.append(
                {
                    **chunk_map[cid],
                    "score": round(score, 6),
                    "strategy": "hybrid_rrf",
                }
            )
        merged.sort(key=lambda c: (-c["score"], c["id"]))
        return merged[:top_k]


class QueryRewriteHybridRetriever:
    """Translate/expand Vietnamese questions into English NIST AI RMF 1.0 queries
    before searching with Hybrid RRF. Logs both original and rewritten queries."""

    def __init__(
        self,
        base_retriever: HybridRRFRetriever,
        api_key: str | None = None,
        model_name: str = "gpt-6-luna",
    ) -> None:
        load_dotenv(ROOT_DIR / ".env", override=True)
        day6_env = Path(r"D:\K5-L3A-DAY06-2A202603493-DataIngestion\starter\.env")
        if not os.getenv("OPENAI_API_KEY") and day6_env.exists():
            load_dotenv(day6_env)

        self.base = base_retriever
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self.model_name = model_name
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.rewrite_log: dict[str, str] = {}

    @property
    def cost_usd(self) -> float:
        """gpt-6-luna cost: $0.15 / 1M input tokens, $0.60 / 1M output tokens."""
        return (self.prompt_tokens * 0.15 + self.completion_tokens * 0.60) / 1_000_000.0

    def rewrite_query(self, question: str) -> str:
        """Convert Vietnamese question into an English search query matching NIST AI RMF 1.0 terminology."""
        if question in self.rewrite_log:
            return self.rewrite_log[question]

        sys_prompt = (
            "You are a search query optimizer for the NIST AI Risk Management Framework (AI RMF 1.0) document. "
            "Convert the user's Vietnamese question into a concise English search query containing exact "
            "technical keywords, subcategory codes (e.g., GOVERN 1.1, MAP, MEASURE 2.5, MANAGE, Appendix A/B/C/D), "
            "and phrases likely to appear verbatim in NIST AI RMF 1.0. Return ONLY the English search query text."
        )
        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": question},
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        usage = body.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        rewritten = body["choices"][0]["message"]["content"].strip()
        self.rewrite_log[question] = rewritten
        return rewritten

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        en_query = self.rewrite_query(query)
        hits = self.base.search(en_query, filters=filters, top_k=top_k)
        return [
            {
                **h,
                "strategy": "hybrid_rrf_query_rewrite",
                "rewritten_query": en_query,
            }
            for h in hits
        ]
