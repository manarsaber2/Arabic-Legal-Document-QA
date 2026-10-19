# Arabic Legal Document Q&A

Question answering over the bilingual (Arabic/English) Egyptian Civil Code. Hybrid retrieval (Qdrant dense + BM25, fused with RRF) feeding a Gemini model that answers with article citations and refuses when the articles don't contain the answer.

## Quick start (3 commands)

```bash
cp .env.example .env && pip install -e ".[dev]"        # add GEMINI_API_KEY to .env
python -m legal_qa.index                            # build the index (add --sweep for the 6-config comparison)
uvicorn legal_qa.api:app --port 8000                    # or: docker compose up --build
```

```bash
curl -X POST localhost:8000/ask -H "content-type: application/json" \
     -d '{"question": "متى تبدأ الشخصية القانونية للإنسان؟"}'
# {"answer": "...", "sources": ["المادة 29", ...]}
curl localhost:8000/health       # {"status": "healthy", "documents_indexed": N}
```
Also: `POST /ask/stream` (SSE: `sources`, `token`…, `done`) and `POST /search` (retrieval only).

## API keys

Only answering a question (`/ask`, `/ask/stream`) needs `GEMINI_API_KEY`. Building the index, `/search`, `/health` and the test suite need no key. The offline eval scripts (`gen_eval.py`, `ragas_eval.py`) also call Gemini.

## Pipeline

| Step | Command | Notes |
|---|---|---|
| Extraction (PDF → `articles.jsonl`) | `extraction.py`, `merge.py`, `clean_and_reprocess.py` | Gemini OCR: paid and non-deterministic, so it is **not** a DVC stage; its output is tracked as data |
| Chunk → index | `dvc repro` | stages `chunk`, `index`, parameters in `params.yaml` |
| Retrieval sweep | `python -m legal_qa.index --sweep` | 6 configs → 6 MLflow runs, best registered as `arabic-legal-qa-retriever@champion` |
| Eval set | `python -m legal_qa.gen_eval --n 60` | synthetic AR/EN questions + reference answers, **reviewed by hand** |
| RAGAS | `python -m legal_qa.ragas_eval` | faithfulness, answer relevancy, context precision/recall → MLflow; CI gate: faithfulness over answered questions ≥ 0.75; alert if overall faithfulness < 0.80 |
| Add a new document | `python -m legal_qa.reindex --add new.jsonl` | rebuild + atomic swap; tested with a new article (`tests/test_reindex.py`) |

Design choices: one chunk per (article, language); only the normalized article body is embedded (breadcrumbs, chapter titles and citations are metadata used by BM25 and for display, because shared headings would pull sibling articles together in vector space); repealed articles are filtered unless the query mentions repeal.

## Results

Fill from your own runs (do not publish numbers you have not measured):

| | value |
|---|---|
| Retrieval, best config `multilingual-e5-small`, chunk 600 / overlap 80 (60 questions) | hit@5 **0.40**, MRR **0.28** (Arabic 0.33, English 0.50); all configs in `reports/mlflow_runs.csv` |
| RAGAS, first 30 of the 60 eval questions, judge `gemini-3.5-flash-lite` | faithfulness **0.76**, answer relevancy **0.56** (1 of 30 judge calls failed and is excluded), context precision **0.60**, context recall **0.80** |
| Same run, diagnostics | gold article in the retrieved set for 80% of questions (top 8, with query rewriting); 8 of 30 answers were refusals; faithfulness on the 22 answered questions: **0.94** |

## Deviations from the project brief

- **Gemini API instead of self-hosted vLLM**: therefore no BentoML serving and no AWQ quantization step. FastAPI serves the model-agnostic RAG logic.
- **Langfuse Cloud** instead of self-hosted (set `LANGFUSE_*` in `.env`; `LANGFUSE_HOST` switches to a self-hosted instance with no code change).
- **MLflow Registry holds the winning retrieval config** (embedding model, chunking, Qdrant collection) as a pyfunc model, not a trained model, since nothing is trained.
- **DVC covers chunk → index, not OCR** (see Pipeline).
- **Qdrant**: embedded local store shipped inside the Docker image by default (the image carries its vector store); set `QDRANT_URL` for a Qdrant server. Embedded mode allows one API process, so scale with replicas.
- **Query rewriting** (`QUERY_REWRITE=1`, default): `/ask` makes one extra Gemini call that turns the (mostly colloquial Egyptian) question into formal legal Arabic before searching, because the corpus is formal Arabic and hit@5 on Arabic is only 0.33. It falls back to the raw question on any error. The retrieval numbers above are measured without it; RAGAS numbers include it.
- **CI gate definition**: the gate uses faithfulness over the questions the system answered, not over all questions. A refusal ("not in the articles") is the intended behaviour when retrieval misses but scores low on faithfulness, so the plain mean mixes hallucination with refusal rate. Both numbers are reported (overall 0.76, answered 0.94, 8 of 30 refusals) and refusals are tracked separately.
- `/ask` passes the top `DEFAULT_K=8` articles to the LLM (`k` is adjustable per request, max 10).
- Eval questions are synthetic (LLM-written, manually reviewed), so RAGAS numbers measure this pipeline on that set, not general legal QA quality. The faithfulness average is pulled down by refusals (an honest "not in the articles" answer scores low), not by invented content. Known limits: the run covers 30 questions, so one question moves a score by about 3 points; most Arabic questions are colloquial while the articles are formal Arabic with some OCR errors; each question has one gold article although others may also be valid, which lowers context precision/recall (with several articles retrieved, precision cannot approach 1).

## Tests

`pytest -q` runs offline (hashed fake embeddings, stub LLM, embedded Qdrant in a temp dir): API contract incl. 422 on empty question, refusal handling, repeal filtering, SSE, chunking/normalization, re-index of a new document, corpus sanity (skipped if `data/articles.jsonl` is absent).

## CI (`.github/workflows/ci.yml`)

lint (ruff) → pytest → `dvc pull` + `dvc repro` + RAGAS gate on the same 30 questions (gate: faithfulness over answered questions ≥ 0.75, see Deviations) → build and push the image to GHCR (main only). Needs secrets: `GEMINI_API_KEY`, `DAGSHUB_USER`, `DAGSHUB_TOKEN`, optional `ALERT_WEBHOOK_URL`.