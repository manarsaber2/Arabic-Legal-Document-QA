"""FastAPI service.   uvicorn legal_qa.api:app --port 8000

  GET  /health      -> {"status": "healthy", "documents_indexed": N}
  POST /ask         -> {"answer": str, "sources": [str], "passages": [{article, citation, text}]}
  POST /ask/stream  -> Server-Sent Events: sources, token..., done
  POST /search      -> retrieval only (debugging / evaluation)
"""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from legal_qa import tracing
from legal_qa.retriever import Retriever

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index" / "best"))
MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-3.1-flash-lite")

DEFAULT_K = int(os.getenv("DEFAULT_K", "8"))
QUERY_REWRITE = os.getenv("QUERY_REWRITE", "1") == "1"

REWRITE_PROMPT = """Rewrite this question about the Egyptian Civil Code as a search query.
Output ONE line: the question in formal legal Arabic (Modern Standard Arabic, no dialect) followed by the key legal terms in English.
If the question is in English, output it with the matching formal Arabic legal terms. No explanations.

Question: {q}"""

REFUSAL_AR = "المواد المتاحة لا تحتوي على إجابة لهذا السؤال."
REFUSAL_EN = "The available articles do not contain an answer to this question."

SYSTEM = f"""You answer questions about the Egyptian Civil Code using ONLY the articles provided.
Answer in the language of the question. Cite article numbers like [المادة 53] / [Article 53].
If the articles do not contain the answer, reply with exactly this sentence and nothing else:
Arabic question: {REFUSAL_AR}
English question: {REFUSAL_EN}
Never use outside knowledge and never invent law."""

state: dict = {}


def _retryable(e) -> bool:
    return getattr(e, "code", None) in (429, 500, 503)


def call_llm(prompt: str) -> str:
    """Gemini 503/429 are transient: retry with backoff, then the fallback model, then fail cleanly."""
    from google.genai import errors
    last = None
    for model in (MODEL, FALLBACK_MODEL):
        for attempt in range(3):
            try:
                resp = state["llm"].models.generate_content(model=model, contents=prompt, config={"temperature": 0})
                return resp.text.strip()
            except errors.APIError as e:
                last = e
                if not _retryable(e):
                    raise HTTPException(502, f"LLM error {e.code}")
                time.sleep(2 ** attempt)
    raise HTTPException(503, f"LLM temporarily unavailable ({getattr(last, 'code', '?')}); try again shortly")


def stream_llm(prompt: str):
    """Yields text pieces. Falls back to the second model only if nothing was emitted yet."""
    from google.genai import errors
    last = None
    for model in (MODEL, FALLBACK_MODEL):
        emitted = False
        try:
            for part in state["llm"].models.generate_content_stream(model=model, contents=prompt,
                                                                    config={"temperature": 0}):
                if part.text:
                    emitted = True
                    yield part.text
            return
        except errors.APIError as e:
            last = e
            if emitted or not _retryable(e):
                break
    raise RuntimeError(f"LLM unavailable ({getattr(last, 'code', '?')})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = Retriever(INDEX_DIR)
    if os.getenv("GEMINI_API_KEY"):
        from google import genai
        state["llm"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    yield
    state["retriever"].close()
    state.clear()


app = FastAPI(title="Arabic Legal QA", lifespan=lifespan)


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(DEFAULT_K, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


def search_query(question: str) -> str:
    """Colloquial Egyptian Arabic vs formal-Arabic corpus is the main retrieval gap (hit@5 0.33 on Arabic).
    One cheap LLM call rewrites the question into formal legal wording; the original stays in the query.
    Any failure falls back to the plain question, so /ask never breaks because of this step."""
    if not QUERY_REWRITE or "llm" not in state:
        return question
    try:
        rewritten = call_llm(REWRITE_PROMPT.format(q=question)).splitlines()[0].strip()
        return f"{question}\n{rewritten}" if rewritten else question
    except Exception:  # noqa: BLE001
        return question


def _citation(h: dict) -> str:
    return h.get("citation") or (f"المادة {h['article_number']}" if h["lang"] == "ar" else f"Article {h['article_number']}")


def _prompt(hits: list[dict], question: str) -> str:
    context = "\n\n".join(f"[Article {h['article_number']}]\n{h['display_text']}" for h in hits)
    return f"{SYSTEM}\n\nArticles:\n{context}\n\nQuestion: {question}"


def _is_refusal(answer: str) -> bool:
    return REFUSAL_AR in answer or REFUSAL_EN in answer


def answer_question(question: str, k: int = DEFAULT_K) -> dict:
    """Full pipeline. Returns answer, sources (citation strings), contexts (for RAGAS), trace_id."""
    if "llm" not in state:
        raise HTTPException(503, "GEMINI_API_KEY not set: /ask needs it; /search and /health work without one")
    with tracing.trace("ask", question=question, k=k) as t:
        t0 = time.time()
        hits = state["retriever"].search(search_query(question), k=k)
        answer = call_llm(_prompt(hits, question))
        answerable = not _is_refusal(answer)
        out = {"answer": answer, "answerable": answerable,
               "sources": [_citation(h) for h in hits] if answerable else [],
               "passages": [{"article": h["article_number"], "citation": _citation(h), "text": h["display_text"]}
                            for h in hits] if answerable else [],
               "contexts": [f"[Article {h['article_number']}]\n{h['display_text']}" for h in hits],  # same as the prompt
               "articles": [h["article_number"] for h in hits], "trace_id": t["trace_id"]}
        t["output"] = answer
        t["metadata"] = {"answerable": answerable, "retrieved": [h["article_number"] for h in hits],
                         "latency_s": round(time.time() - t0, 3)}
    return out


@app.get("/health")
def health():
    return {"status": "healthy", "documents_indexed": len(state["retriever"].chunks)}


@app.post("/ask")
def ask(req: Ask):
    out = answer_question(req.question, req.k)
    # contract: answer + sources (list[str]); passages = the article texts behind those sources
    return {"answer": out["answer"], "sources": out["sources"], "passages": out["passages"]}


@app.post("/ask/stream")
def ask_stream(req: Ask):
    if "llm" not in state:
        raise HTTPException(503, "GEMINI_API_KEY not set: /ask needs it; /search and /health work without one")
    hits = state["retriever"].search(search_query(req.question), k=req.k)
    prompt = _prompt(hits, req.question)

    def sse(event: str, data) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def gen():
        # sources go first so the UI can show them while the answer streams
        yield sse("sources", [_citation(h) for h in hits])
        full = []
        try:
            for piece in stream_llm(prompt):
                full.append(piece)
                yield sse("token", piece)
        except RuntimeError as e:
            yield sse("error", str(e))
            return
        yield sse("done", {"answerable": not _is_refusal("".join(full))})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/search")
def search(req: Ask):
    hits = state["retriever"].search(search_query(req.question), k=req.k)
    return [{"article_number": h["article_number"], "lang": h["lang"], "citation": _citation(h),
             "score": round(h["score"], 4), "text": h["display_text"]} for h in hits]