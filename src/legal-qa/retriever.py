"""Hybrid retrieval: dense (Qdrant) + BM25, fused with Reciprocal Rank Fusion.

Dense side embeds the article body only. BM25 side indexes body + breadcrumb/section titles
(metadata), so a query like "الجمعيات" can hit articles under that heading even when the
word is not in the article text. Repealed articles are filtered out unless the query asks.
"""
import json
import os
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from chunk import normalize_ar, normalize_en

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
_AR = re.compile(r"[؀-ۿ]")
_REPEAL_WORDS = re.compile(r"ملغ|الغ|ألغ|repeal", re.I)


def norm_query(q: str) -> str:
    return normalize_ar(q) if _AR.search(q) else normalize_en(q)


def tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


class Embedder:
    """e5 models need 'query: ' / 'passage: ' prefixes; other models don't."""
    def __init__(self, name: str = DEFAULT_MODEL, fake: bool = False):
        self.name, self.fake = name, fake
        self.is_e5 = "e5" in name.lower()
        if not fake:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(name)
            self.dim = self.model.get_sentence_embedding_dimension()
        else:
            self.dim = 64

    def _enc(self, texts: list[str]):
        if self.fake:  # test-only: deterministic hashed bag-of-words vectors
            import numpy as np
            out = np.zeros((len(texts), self.dim), dtype="float32")
            for i, t in enumerate(texts):
                for w in tok(t):
                    out[i, hash(w) % self.dim] += 1
            return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)

    def passages(self, texts):
        return self._enc([f"passage: {t}" for t in texts] if self.is_e5 else texts)

    def queries(self, texts):
        return self._enc([f"query: {t}" for t in texts] if self.is_e5 else texts)


def bm25_text(c: dict) -> str:
    meta = " ".join(filter(None, [c.get(f"breadcrumb_{c['lang']}"), c.get("citation")]))
    return f"{c['text']} {normalize_ar(meta) if c['lang'] == 'ar' else normalize_en(meta)}"


class Retriever:
    def __init__(self, index_dir: Path, embedder: Embedder | None = None):
        from qdrant_client import QdrantClient
        self.dir = Path(index_dir)
        meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.chunks = [json.loads(l) for l in (self.dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        self.embedder = embedder or Embedder(meta["embedding_model"], fake=meta.get("fake", False))
        self.collection = meta.get("collection", "civil_code")
        self.client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                                   api_key=os.getenv("QDRANT_API_KEY") or None)
        self.bm25 = BM25Okapi([tok(bm25_text(c)) for c in self.chunks])

    def search(self, query: str, k: int = 5, pool: int = 30, include_repealed: bool | None = None) -> list[dict]:
        if include_repealed is None:
            include_repealed = bool(_REPEAL_WORDS.search(query))
        q = norm_query(query)
        ok = lambda i: include_repealed or not self.chunks[i]["is_repealed"]

        qv = self.embedder.queries([q])[0].tolist()
        dense = [p.id for p in self.client.query_points(self.collection, query=qv, limit=pool * 2).points if ok(p.id)][:pool]
        scores = self.bm25.get_scores(tok(q))
        sparse = [i for i in sorted(range(len(scores)), key=scores.__getitem__, reverse=True) if scores[i] > 0 and ok(i)][:pool]

        fused: dict[int, float] = {}
        for ranking in (dense, sparse):  # Reciprocal Rank Fusion, k=60
            for r, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + r + 1)

        out, seen = [], set()
        for i in sorted(fused, key=fused.get, reverse=True):
            c = self.chunks[i]
            if c["article_number"] in seen:  # AR + EN chunk of one article -> keep the better one
                continue
            seen.add(c["article_number"])
            out.append({**c, "score": fused[i]})
            if len(out) == k:
                break
        return out