"""RAGAS evaluation through the live pipeline (retriever + Gemini).

  python src/legal_qa/ragas_eval.py                # full run: needs >= 50 rows in data/eval/eval.jsonl
  python src/legal_qa/ragas_eval.py --limit 20     # CI gate: first 20 questions

Metrics: faithfulness, answer_relevancy, context_precision, context_recall  -> MLflow + data/eval/ragas_scores.json
Gate:   exit 1 if faithfulness < FAITHFULNESS_GATE (default 0.75, the rubric's CI gate)
Alert:  POST to ALERT_WEBHOOK_URL (Slack-compatible {"text": ...}) if faithfulness < ALERT_THRESHOLD (default 0.80)
Langfuse: if enabled, each question's faithfulness is attached to its trace as a score.
Env: GEMINI_API_KEY, MODEL, INDEX_DIR.  NOTE: ragas' API moves between versions; this targets 0.2/0.3.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE = float(os.getenv("FAITHFULNESS_GATE", 0.75))
ALERT = float(os.getenv("ALERT_THRESHOLD", 0.80))


def send_alert(faithfulness: float) -> None:
    url = os.getenv("ALERT_WEBHOOK_URL")
    msg = f"Arabic Legal QA: RAGAS faithfulness {faithfulness:.3f} < {ALERT} (gate {GATE})"
    print("ALERT:", msg)
    if not url:
        return
    import httpx
    try:
        httpx.post(url, json={"text": msg}, timeout=10).raise_for_status()
    except Exception as e:  # noqa: BLE001
        print("alert webhook failed:", e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval", type=Path, default=ROOT / "data" / "eval" / "eval.jsonl")
    a = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from fastapi.testclient import TestClient  # noqa: F401  (lifespan loads retriever + Gemini client)
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    from legal_qa import tracing
    from legal_qa.api import answer_question, app

    rows = [json.loads(l) for l in a.eval.read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    elif len(rows) < 50:
        sys.exit(f"need >= 50 eval questions for a full run, found {len(rows)} (use --limit for the CI subset)")

    data, trace_ids = [], []
    with TestClient(app):  # runs lifespan
        for r in rows:
            out = answer_question(r["question"])
            trace_ids.append(out["trace_id"])
            data.append({"user_input": r["question"], "response": out["answer"],
                         "retrieved_contexts": out["contexts"], "reference": r["reference"]})

    judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=os.getenv("MODEL", "gemini-3.5-flash-lite")))
    emb = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"))
    res = evaluate(EvaluationDataset.from_list(data),
                   metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
                   llm=judge, embeddings=emb)
    scores = {k: float(v) for k, v in res._repr_dict.items()}
    scores["n_questions"] = len(rows)
    print(json.dumps(scores, indent=2))
    (ROOT / "data" / "eval").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "eval" / "ragas_scores.json").write_text(json.dumps(scores), encoding="utf-8")

    for tid, f in zip(trace_ids, res.to_pandas()["faithfulness"]):
        if f == f:  # skip NaN
            tracing.score(tid, "faithfulness", f)

    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("arabic-legal-qa-ragas")
        with mlflow.start_run():
            mlflow.log_params({"n_questions": len(rows), "judge_model": os.getenv("MODEL", "gemini-3.5-flash-lite")})
            mlflow.log_metrics({k: v for k, v in scores.items() if k != "n_questions"})
    except Exception as e:  # noqa: BLE001
        print("mlflow logging skipped:", e)

    if scores["faithfulness"] < ALERT:
        send_alert(scores["faithfulness"])
    if scores["faithfulness"] < GATE:
        sys.exit(f"faithfulness {scores['faithfulness']:.3f} < gate {GATE}")


if __name__ == "__main__":
    main()
