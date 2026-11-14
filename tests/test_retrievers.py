import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from retrievers import BM25Retriever, HybridRRFRetriever, _cosine_similarity


class DummyDenseRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.cost_usd = 0.0

    def search(self, query, *, filters=None, top_k=3):
        out = []
        for idx, c in enumerate(self.chunks):
            if filters and c.get("metadata", {}).get("section") != filters.get("section"):
                continue
            out.append({**c, "score": round(1.0 / (idx + 1), 6), "strategy": "dense_pgvector"})
        return out[:top_k]


class RetrieverUnitTests(unittest.TestCase):
    def setUp(self):
        self.sample_chunks = [
            {
                "id": "c1",
                "source_id": "nist-ai-rmf-1.0",
                "source": "NIST.pdf",
                "page": 27,
                "text": "GOVERN 1.1: Legal and regulatory requirements involving AI are understood.",
                "metadata": {"section": "part_2_core_profiles"},
            },
            {
                "id": "c2",
                "source_id": "nist-ai-rmf-1.0",
                "source": "NIST.pdf",
                "page": 40,
                "text": "Appendix A: AI Design tasks are performed during the Application Context phase.",
                "metadata": {"section": "appendix"},
            },
            {
                "id": "c3",
                "source_id": "nist-ai-rmf-1.0",
                "source": "NIST.pdf",
                "page": 38,
                "text": "MANAGE 4.1: Post-deployment AI system monitoring plans are implemented.",
                "metadata": {"section": "part_2_core_profiles"},
            },
        ]

    def test_bm25_search_and_metadata_filter(self):
        bm25 = BM25Retriever(self.sample_chunks)
        hits = bm25.search("GOVERN 1.1 legal requirements", top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["id"], "c1")

        filtered = bm25.search("AI tasks phase", filters={"section": "appendix"}, top_k=3)
        self.assertEqual([h["id"] for h in filtered], ["c2"])

    def test_hybrid_rrf_combines_dense_and_lexical(self):
        dense = DummyDenseRetriever(self.sample_chunks)
        bm25 = BM25Retriever(self.sample_chunks)
        hybrid = HybridRRFRetriever(dense, bm25, k_rrf=60, w_dense=1.0, w_lexical=0.5)
        hits = hybrid.search("MANAGE 4.1 post-deployment monitoring", top_k=2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["strategy"], "hybrid_rrf")
        self.assertIn("c3", [h["id"] for h in hits])

    def test_cosine_similarity(self):
        self.assertAlmostEqual(_cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(_cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
