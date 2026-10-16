"""Build one Qdrant index per config and log an MLflow run with retrieval quality.

  python src/legal_qa/index.py --model intfloat/multilingual-e5-small --chunk-size 600 --overlap 80
  python src/legal_qa/index.py --sweep          # runs the 6 configs below -> 6 MLflow runs

Retrieval quality = hit@5 / MRR on data/eval/eval.jsonl ({"question", "article_number"}).
The best run (by MRR) is copied to data/index/best/ and registered in MLflow.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import mlflow

from chunk import build_chunks
from retriever import Embedder, Retriever

ROOT = Path(__file__).resolve().parents[2]
ARTICLES = ROOT / "data" / "articles.jsonl"
EVAL = ROOT / "data" / "eval" / "eval.jsonl"
INDEX_ROOT = ROOT / "data" / "index"

E5S, E5B = "intfloat/multilingual-e5-small", "intfloat/multilingual-e5-base"
SWEEP = [  # (model, chunk_size, overlap) -- small sizes force real splits on long articles
    (E5S, 600, 80), (E5S, 150, 30), (E5S, 80, 20),
    (E5B, 600, 80), (E5B, 150, 30), ("BAAI/bge-m3", 600, 80),
]


def evaluate(r: Retriever, k: int = 5) -> dict:
    if not EVAL.exists():
        return {}
    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    hits, rr = 0, 0.0
    for row in rows:
        got = [h["article_number"] for h in r.search(row["question"], k=k)]
        if row["article_number"] in got:
            hits += 1
            rr += 1.0 / (got.index(row["article_number"]) + 1)
    return {f"hit_at_{k}": hits / len(rows), "mrr": rr / len(rows), "n_eval": len(rows)}


def run(model: str, chunk_size: int, overlap: int, fake: bool = False) -> dict:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    tag = f"{model.split('/')[-1]}_cs{chunk_size}_ov{overlap}" + ("_fake" if fake else "")
    out = INDEX_ROOT / tag
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    articles = [json.loads(l) for l in ARTICLES.read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks = build_chunks(articles, chunk_size, overlap)
    emb = Embedder(model, fake=fake)

    with mlflow.start_run(run_name=tag):
        mlflow.log_params({"embedding_model": model, "chunk_size": chunk_size, "chunk_overlap": overlap,
                           "retrieval": "hybrid_bm25_dense_rrf", "n_articles": len(articles)})
        t0 = time.time()
        vecs = emb.passages([c["text"] for c in chunks])
        client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                              api_key=os.getenv("QDRANT_API_KEY") or None)
        if client.collection_exists(tag):  # one collection per config; rebuild from scratch
            client.delete_collection(tag)
        client.create_collection(tag, vectors_config=VectorParams(size=emb.dim, distance=Distance.COSINE))
        points = [PointStruct(id=i, vector=v.tolist(),
                              payload={k: c[k] for k in ("article_number", "lang", "is_repealed")})
                  for i, (c, v) in enumerate(zip(chunks, vecs))]
        for b in range(0, len(points), 256):
            client.upsert(tag, points[b:b + 256])
        client.close()
        (out / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
        (out / "meta.json").write_text(json.dumps({"embedding_model": model, "chunk_size": chunk_size,
                                                    "overlap": overlap, "fake": fake, "collection": tag}), encoding="utf-8")
        metrics = {"n_chunks": len(chunks), "multi_part_chunks": sum(c["num_parts"] > 1 for c in chunks),
                   "embed_seconds": round(time.time() - t0, 1)}
        metrics.update(evaluate(Retriever(out, emb)))
        mlflow.log_metrics(metrics)
        print(tag, metrics)
    return {"tag": tag, "dir": out, **metrics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=E5S)
    ap.add_argument("--chunk-size", type=int, default=600)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--fake", action="store_true", help="test-only hashed embeddings, no model download")
    a = ap.parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")  # file store is deprecated in new MLflow
    mlflow.set_experiment("arabic-legal-qa-retrieval")
    configs = SWEEP if a.sweep else [(a.model, a.chunk_size, a.overlap)]
    results = [run(m, cs, ov, a.fake) for m, cs, ov in configs]

    scored = [r for r in results if "mrr" in r]
    if scored:
        best = max(scored, key=lambda r: r["mrr"])
        shutil.rmtree(INDEX_ROOT / "best", ignore_errors=True)
        shutil.copytree(best["dir"], INDEX_ROOT / "best")
        print(f"BEST: {best['tag']}  mrr={best['mrr']:.3f}  hit@5={best['hit_at_5']:.3f}  -> data/index/best")


if __name__ == "__main__":
    main()