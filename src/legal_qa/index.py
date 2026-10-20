"""Build one Qdrant index per config and log an MLflow run with retrieval quality.

  python src/legal_qa/index.py --model intfloat/multilingual-e5-small --chunk-size 600 --overlap 80
  python src/legal_qa/index.py --sweep          # runs the 6 configs below -> 6 MLflow runs

Retrieval quality = hit@5 / MRR on data/eval/eval.jsonl ({"question", "article_number", "lang"}).
The best run (by MRR) is copied to data/index/best/ and registered in MLflow (alias "champion").
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import mlflow
import mlflow.pyfunc

from legal_qa.indexing import build_index, read_jsonl
from legal_qa.retriever import Embedder, Retriever

ROOT = Path(__file__).resolve().parents[2]
ARTICLES = ROOT / "data" / "articles.jsonl"
EVAL = ROOT / "data" / "eval" / "eval.jsonl"
INDEX_ROOT = ROOT / "data" / "index"

E5S, E5B = "intfloat/multilingual-e5-small", "intfloat/multilingual-e5-base"
SWEEP = [  # (model, chunk_size, overlap) -- small sizes force real splits on long articles
    (E5S, 600, 80), (E5S, 150, 30), (E5S, 80, 20),
    (E5B, 600, 80), (E5B, 150, 30), ("BAAI/bge-m3", 600, 80),
]


class RetrieverConfig(mlflow.pyfunc.PythonModel):
    """Registry entry for the winning retrieval config (embedding model, chunking, Qdrant collection)."""

    def load_context(self, context):
        self.config = json.load(open(context.artifacts["config"], encoding="utf-8"))

    def predict(self, context, model_input, params=None):
        return [self.config] * len(model_input)


def register_best(best: dict) -> None:
    cfg = {k: best[k] for k in ("tag", "model", "chunk_size", "overlap", "hit_at_5", "mrr")}
    cfg["collection"] = best["tag"]
    cfg_path = INDEX_ROOT / "best_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    with mlflow.start_run(run_name=f"BEST-{best['tag']}"):
        mlflow.log_params({k: cfg[k] for k in ("model", "chunk_size", "overlap")})
        mlflow.log_metrics({k: cfg[k] for k in ("hit_at_5", "mrr")})
        info = mlflow.pyfunc.log_model(name="retriever_config", python_model=RetrieverConfig(),
                                       artifacts={"config": str(cfg_path)}, pip_requirements=["mlflow"])
        mv = mlflow.register_model(info.model_uri, "arabic-legal-qa-retriever")
        mlflow.MlflowClient().set_registered_model_alias("arabic-legal-qa-retriever", "champion", mv.version)
        print(f"registered arabic-legal-qa-retriever v{mv.version} (alias: champion) -> {best['tag']}")


def export_report() -> None:
    """reports/mlflow_runs.csv: all runs side by side (README evidence)."""
    df = mlflow.search_runs(experiment_names=["arabic-legal-qa-retrieval"])
    keep = [c for c in df.columns if c.startswith(("params.", "metrics.")) or c == "tags.mlflow.runName"]
    (ROOT / "reports").mkdir(exist_ok=True)
    df[keep].to_csv(ROOT / "reports" / "mlflow_runs.csv", index=False)
    print(f"{len(df)} runs -> reports/mlflow_runs.csv")


def evaluate(r: Retriever, k: int = 5, eval_path: Path = EVAL) -> dict:
    if not Path(eval_path).exists():
        return {}
    rows = read_jsonl(eval_path)
    hits, rr, by_lang = 0, 0.0, {}
    for row in rows:
        got = [h["article_number"] for h in r.search(row["question"], k=k)]
        ok = row["article_number"] in got
        hits += ok
        rr += 1.0 / (got.index(row["article_number"]) + 1) if ok else 0.0
        h_, n_ = by_lang.get(row.get("lang", "all"), (0, 0))
        by_lang[row.get("lang", "all")] = (h_ + ok, n_ + 1)
    out = {f"hit_at_{k}": hits / len(rows), "mrr": rr / len(rows), "n_eval": len(rows)}
    if len(by_lang) > 1:
        out.update({f"hit_at_{k}_{lg}": h_ / n_ for lg, (h_, n_) in by_lang.items()})
    return out


def run(model: str, chunk_size: int, overlap: int, fake: bool = False) -> dict:
    tag = f"{model.split('/')[-1]}_cs{chunk_size}_ov{overlap}" + ("_fake" if fake else "")
    out = INDEX_ROOT / tag
    articles = read_jsonl(ARTICLES)
    emb = Embedder(model, fake=fake)
    with mlflow.start_run(run_name=tag):
        mlflow.log_params({"embedding_model": model, "chunk_size": chunk_size, "chunk_overlap": overlap,
                           "retrieval": "hybrid_bm25_dense_rrf", "n_articles": len(articles)})
        t0 = time.time()
        meta = build_index(articles, model, chunk_size, overlap, out, tag, fake, embedder=emb)
        metrics = {"n_chunks": meta["n_chunks"], "embed_seconds": round(time.time() - t0, 1)}
        r = Retriever(out, emb)
        metrics["multi_part_chunks"] = sum(c["num_parts"] > 1 for c in r.chunks)
        metrics.update(evaluate(r))
        r.close()
        mlflow.log_metrics(metrics)
        print(tag, metrics)
    return {"tag": tag, "dir": out, "model": model, "chunk_size": chunk_size, "overlap": overlap, **metrics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=E5S)
    ap.add_argument("--chunk-size", type=int, default=600)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--fake", action="store_true", help="test-only hashed embeddings, no model download")
    a = ap.parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("arabic-legal-qa-retrieval")
    configs = SWEEP if a.sweep else [(a.model, a.chunk_size, a.overlap)]
    results = [run(m, cs, ov, a.fake) for m, cs, ov in configs]

    scored = [r for r in results if "mrr" in r]
    best = max(scored, key=lambda r: r["mrr"]) if scored else results[-1]
    shutil.rmtree(INDEX_ROOT / "best", ignore_errors=True)
    shutil.copytree(best["dir"], INDEX_ROOT / "best")
    print(f"BEST: {best['tag']}  mrr={best.get('mrr', float('nan')):.3f}  -> data/index/best")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "retrieval_metrics.json").write_text(
        json.dumps({k: v for k, v in best.items() if k != "dir"}, indent=2), encoding="utf-8")
    if scored and not a.fake:
        register_best(best)
    export_report()


if __name__ == "__main__":
    main()
