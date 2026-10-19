"""RAGAS evaluation through the live pipeline (retriever + Gemini).

  python -m legal_qa.ragas_eval                # full run: needs >= 50 rows in data/eval/eval.jsonl
  python -m legal_qa.ragas_eval --limit 20     # CI gate: first 20 questions

Metrics: faithfulness, answer_relevancy, context_precision, context_recall  -> MLflow + data/eval/ragas_scores.json
Gate:   exit 1 if faithfulness over answered (non-refusal) questions < FAITHFULNESS_GATE (default 0.75)
Alert:  POST to ALERT_WEBHOOK_URL (Slack-compatible {"text": ...}) if faithfulness < ALERT_THRESHOLD (default 0.80)
Langfuse: if enabled, each question's faithfulness is attached to its trace as a score.
Env: GEMINI_API_KEY, MODEL, INDEX_DIR.  NOTE: ragas' API moves between versions; this targets 0.2/0.3.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE = float(os.getenv("FAITHFULNESS_GATE", 0.75))
ALERT = float(os.getenv("ALERT_THRESHOLD", 0.80))
JUDGE_MODEL = os.getenv("JUDGE_MODEL", os.getenv("MODEL", "gemini-3.5-flash-lite"))


def _patch_ragas_imports() -> None:
    """ragas 0.4.x imports VertexAI classes that langchain-community 0.4 removed; stub them (we use Gemini via AI Studio)."""
    import sys
    import types
    name = "langchain_community.chat_models.vertexai"
    try:
        __import__(name)
    except ImportError:
        mod = types.ModuleType(name)
        mod.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[name] = mod
    import langchain_community.llms as llms
    try:
        llms.VertexAI
    except (ImportError, AttributeError):
        llms.VertexAI = type("VertexAI", (), {})


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
    if sys.platform == "win32":  # avoids "Event loop is closed" noise from aiohttp/httpx at exit
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval", type=Path, default=ROOT / "data" / "eval" / "eval.jsonl")
    a = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from fastapi import HTTPException
    from fastapi.testclient import TestClient  # noqa: F401  (lifespan loads retriever + Gemini client)
    _patch_ragas_imports()
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    from ragas.run_config import RunConfig

    from legal_qa import tracing
    from legal_qa.api import answer_question, app

    rows = [json.loads(l) for l in a.eval.read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    elif len(rows) < 50:
        sys.exit(f"need >= 50 eval questions for a full run, found {len(rows)} (use --limit for the CI subset)")

    data, trace_ids, skipped, diag = [], [], [], []
    with TestClient(app):  # runs lifespan
        for i, r in enumerate(rows, 1):
            out = None
            for attempt in range(5):  # Gemini 503/429 are transient: wait and retry, then skip the question
                try:
                    out = answer_question(r["question"])
                    break
                except HTTPException as e:
                    wait = 15 * (attempt + 1)
                    print(f"[{i}/{len(rows)}] LLM error {e.status_code}, retry in {wait}s")
                    time.sleep(wait)
            if out is None:
                skipped.append(i)
                continue
            trace_ids.append(out["trace_id"])
            diag.append({"i": i, "lang": r.get("lang"), "expected_article": r["article_number"],
                         "retrieved": out["articles"], "hit": r["article_number"] in out["articles"],
                         "answerable": out["answerable"]})
            data.append({"user_input": r["question"], "response": out["answer"],
                         "retrieved_contexts": out["contexts"], "reference": r["reference"]})
            time.sleep(float(os.getenv("ASK_PAUSE", 1)))
    if skipped:
        print(f"WARNING: {len(skipped)} question(s) skipped after retries: {skipped}")
    if len(data) < 0.8 * len(rows):
        sys.exit(f"only {len(data)}/{len(rows)} questions answered; Gemini is overloaded, try again later")

    judge_llm = ChatGoogleGenerativeAI(model=JUDGE_MODEL)
    judge_emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # preflight: one tiny call each, so a quota / model-name / key problem shows its real error now
    # instead of silently turning every RAGAS job into NaN after an hour
    for name, call in (("judge model " + JUDGE_MODEL, lambda: judge_llm.invoke("Reply with: ok")),
                       ("embedding model", lambda: judge_emb.embed_query("test"))):
        try:
            call()
        except Exception as e:  # noqa: BLE001
            sys.exit(f"preflight failed for {name}: {type(e).__name__}: {str(e)[:600]}")
    judge = LangchainLLMWrapper(judge_llm)
    emb = LangchainEmbeddingsWrapper(judge_emb)
    res = evaluate(EvaluationDataset.from_list(data),
                   metrics=[Faithfulness(), AnswerRelevancy(strictness=1), ContextPrecision(), ContextRecall()],
                   llm=judge, embeddings=emb,
                   # few workers + long timeout: Gemini rate limits otherwise turn into TimeoutError (-> NaN, dropped from the mean)
                   run_config=RunConfig(max_workers=int(os.getenv("RAGAS_WORKERS", 3)), timeout=300, max_retries=6))
    df = res.to_pandas()
    if df[[c for c in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
           if c in df]].isna().all().all():
        sys.exit("every RAGAS job failed (all scores NaN): judge quota/rate limit or timeouts. "
                 "Wait for the quota to reset or set JUDGE_MODEL to another model, then re-run.")
    scores = {k: float(v) for k, v in df.select_dtypes("number").mean().items()}  # mean per metric
    # per-question diagnostics: is the problem retrieval (hit), refusals, or the judge?
    import pandas as pd
    report = pd.concat([pd.DataFrame(diag), df[["user_input", "response"] + list(scores)]], axis=1)
    (ROOT / "reports").mkdir(exist_ok=True)
    report.to_csv(ROOT / "reports" / "ragas_per_question.csv", index=False, encoding="utf-8-sig")
    hit = float(report["hit"].mean())
    print(f"\nretrieval hit on these {len(report)} questions: {hit:.2f} | refusals: {int((~report['answerable']).sum())}"
          f" | judge: {JUDGE_MODEL}")
    print("  faithfulness when answered (non-refusal):",
          round(float(report[report["answerable"]]["faithfulness"].mean()), 3))
    print("  per-question table -> reports/ragas_per_question.csv")
    failed = {f"failed_{k}": int(df[k].isna().sum()) for k in scores}  # jobs that errored/timed out are NaN, excluded from the mean
    if any(failed.values()):
        print(f"WARNING: {failed} -> the means exclude these. Re-run with RAGAS_WORKERS=2 for a cleaner number.")
    # Gate metric: faithfulness over questions the system actually answered. A refusal ("not in the articles")
    # is the intended behaviour when retrieval misses, but it scores low on faithfulness, so the plain mean mixes
    # hallucination with refusal rate. Both are reported; the refusal count is printed above.
    answered_f = report[report["answerable"]]["faithfulness"].dropna()
    scores["faithfulness_answered"] = float(answered_f.mean()) if len(answered_f) else scores["faithfulness"]
    scores["n_refusals"] = int((~report["answerable"]).sum())
    scores["n_questions"] = len(data)
    scores.update(failed)
    print(json.dumps(scores, indent=2))
    (ROOT / "data" / "eval").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "eval" / "ragas_scores.json").write_text(json.dumps(scores), encoding="utf-8")

    for tid, f in zip(trace_ids, df["faithfulness"]):
        if f == f:  # skip NaN
            tracing.score(tid, "faithfulness", f)

    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("arabic-legal-qa-ragas")
        with mlflow.start_run():
            mlflow.log_params({"n_questions": len(data), "n_skipped": len(skipped), "judge_model": JUDGE_MODEL})
            mlflow.log_metrics({k: v for k, v in scores.items() if k != "n_questions"})
    except Exception as e:  # noqa: BLE001
        print("mlflow logging skipped:", e)

    if scores["faithfulness"] < ALERT:
        send_alert(scores["faithfulness"])
    if scores["faithfulness_answered"] < GATE:
        sys.exit(f"faithfulness (answered questions) {scores['faithfulness_answered']:.3f} < gate {GATE}")


if __name__ == "__main__":
    main()