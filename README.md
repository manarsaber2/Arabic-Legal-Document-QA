# Arabic Legal Document Q&A

Question answering over the bilingual (Arabic/English) Egyptian Civil Code. Hybrid retrieval (Qdrant dense + BM25, fused with RRF) feeding a Gemini model that answers with article citations and refuses when the articles don't contain the answer.

## Quick start (3 commands)

```bash
cp .env.example .env && pip install -e ".[dev]"        # add GEMINI_API_KEY to .env
python src/legal_qa/index.py                            # build the index (add --sweep for the 6-config comparison)
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
| Retrieval sweep | `python src/legal_qa/index.py --sweep` | 6 configs → 6 MLflow runs, best registered as `arabic-legal-qa-retriever@champion` |
| Eval set | `python src/legal_qa/gen_eval.py --n 60` | synthetic AR/EN questions + reference answers, **reviewed by hand** |
| RAGAS | `python src/legal_qa/ragas_eval.py` | faithfulness, answer relevancy, context precision/recall → MLflow; CI gate faithfulness ≥ 0.75, alert < 0.80 |
| OCR spot-check | `python src/legal_qa/spotcheck.py sample` / `score` | 20 articles, CER against hand-typed gold |
| Add a new document | `python src/legal_qa/reindex.py --add new.jsonl` | rebuild + atomic swap; tested with a new article (`tests/test_reindex.py`) |

Design choices: one chunk per (article, language); only the normalized article body is embedded (breadcrumbs, chapter titles and citations are metadata used by BM25 and for display, because shared headings would pull sibling articles together in vector space); repealed articles are filtered unless the query mentions repeal.

## Results

Fill from your own runs (do not publish numbers you have not measured):

| | value |
|---|---|
| Retrieval sweep (hit@5 / MRR, best config) | see `reports/mlflow_runs.csv`, screenshots in `reports/` |
| RAGAS (faithfulness / relevancy / ctx precision / ctx recall, n=?) | |
| OCR CER on 20-article gold sample | `reports/spotcheck_report.json` |

## Deviations from the project brief

- **Gemini API instead of self-hosted vLLM**: therefore no BentoML serving and no AWQ quantization step. FastAPI serves the model-agnostic RAG logic.
- **Langfuse Cloud** instead of self-hosted (set `LANGFUSE_*` in `.env`; `LANGFUSE_HOST` switches to a self-hosted instance with no code change).
- **MLflow Registry holds the winning retrieval config** (embedding model, chunking, Qdrant collection) as a pyfunc model, not a trained model, since nothing is trained.
- **DVC covers chunk → index, not OCR** (see Pipeline).
- **Qdrant**: embedded local store shipped inside the Docker image by default (the image carries its vector store); set `QDRANT_URL` for a Qdrant server. Embedded mode allows one API process, so scale with replicas.
- Eval questions are synthetic (LLM-written, manually reviewed), so RAGAS numbers measure this pipeline on that set, not general legal QA quality.

## Tests

`pytest -q` runs offline (hashed fake embeddings, stub LLM, embedded Qdrant in a temp dir): API contract incl. 422 on empty question, refusal handling, repeal filtering, SSE, chunking/normalization, re-index of a new document, CER, corpus sanity (skipped if `data/articles.jsonl` is absent).

## CI (`.github/workflows/ci.yml`)

lint (ruff) → pytest → `dvc pull` + `dvc repro` + RAGAS gate on 20 questions → build and push the image to GHCR (main only). Needs secrets: `GEMINI_API_KEY`, `DAGSHUB_USER`, `DAGSHUB_TOKEN`, optional `ALERT_WEBHOOK_URL`.