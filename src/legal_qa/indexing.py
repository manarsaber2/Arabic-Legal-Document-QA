"""Core index builder (no MLflow): articles -> chunks -> embeddings -> Qdrant + chunks.jsonl + meta.json.

Used by index.py (sweep + MLflow) and reindex.py (batch re-index of new documents).
"""
import json
import os
import shutil
from pathlib import Path

from legal_qa.chunk import build_chunks
from legal_qa.retriever import Embedder, make_client


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def build_index(articles: list[dict], model: str, chunk_size: int, overlap: int, out: Path,
                tag: str, fake: bool = False, embedder: Embedder | None = None) -> dict:
    """Writes <out>/{chunks.jsonl, meta.json} and the Qdrant collection `tag`
    (embedded store at <out>/qdrant, or the server in QDRANT_URL)."""
    from qdrant_client.models import Distance, PointStruct, VectorParams

    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    chunks = build_chunks(articles, chunk_size, overlap)
    emb = embedder or Embedder(model, fake=fake)
    vecs = emb.passages([c["text"] for c in chunks])

    client = make_client(out)
    if client.collection_exists(tag):
        client.delete_collection(tag)
    client.create_collection(tag, vectors_config=VectorParams(size=emb.dim, distance=Distance.COSINE))
    points = [PointStruct(id=i, vector=v.tolist(),
                          payload={k: c[k] for k in ("article_number", "lang", "is_repealed")})
              for i, (c, v) in enumerate(zip(chunks, vecs))]
    for b in range(0, len(points), 256):
        client.upsert(tag, points[b:b + 256])
    client.close()  # embedded mode: release the folder lock before a Retriever opens it

    (out / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
    meta = {"embedding_model": model, "chunk_size": chunk_size, "overlap": overlap, "fake": fake,
            "collection": tag, "n_chunks": len(chunks), "n_articles": len(articles),
            "qdrant_mode": "server" if os.getenv("QDRANT_URL") else "embedded"}
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta