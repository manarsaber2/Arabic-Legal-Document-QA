"""FastAPI service: GET /health, POST /ask.   uvicorn legal_qa.api:app --port 8000"""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from retriever import Retriever

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index" / "best"))
MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-3.1-flash-lite")

REFUSAL_AR = "المواد المتاحة لا تحتوي على إجابة لهذا السؤال."
REFUSAL_EN = "The available articles do not contain an answer to this question."

SYSTEM = f"""You answer questions about the Egyptian Civil Code using ONLY the articles provided.
Answer in the language of the question. Cite article numbers like [المادة 53] / [Article 53].
If the articles do not contain the answer, reply with exactly this sentence and nothing else:
Arabic question: {REFUSAL_AR}
English question: {REFUSAL_EN}
Never use outside knowledge and never invent law."""

state: dict = {}


def call_llm(prompt: str) -> str:
    """Gemini 503/429 are transient: retry with backoff, then try the fallback model, then fail cleanly."""
    from google.genai import errors
    last = None
    for model in (MODEL, FALLBACK_MODEL):
        for attempt in range(3):
            try:
                resp = state["llm"].models.generate_content(model=model, contents=prompt, config={"temperature": 0})
                return resp.text.strip()
            except errors.APIError as e:
                last = e
                if e.code not in (429, 500, 503):
                    raise HTTPException(502, f"LLM error {e.code}")
                time.sleep(2 ** attempt)
    raise HTTPException(503, f"LLM temporarily unavailable ({getattr(last, 'code', '?')}); try again shortly")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = Retriever(INDEX_DIR)
    if os.getenv("GEMINI_API_KEY"):
        from google import genai
        state["llm"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    yield
    state.clear()


app = FastAPI(title="Arabic Legal QA", lifespan=lifespan)


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(5, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok", "index": INDEX_DIR.name, "chunks": len(state["retriever"].chunks),
            "llm": "llm" in state}


@app.post("/ask")
def ask(req: Ask):
    hits = state["retriever"].search(req.question, k=req.k)
    sources = [{"article_number": h["article_number"], "citation": h["citation"], "lang": h["lang"],
                "breadcrumb": h.get("breadcrumb_ar") or h.get("breadcrumb_en"),
                "text": h["display_text"], "score": round(h["score"], 4)} for h in hits]
    if "llm" not in state:
        raise HTTPException(503, "GEMINI_API_KEY not set")
    context = "\n\n".join(f"[Article {s['article_number']}]\n{s['text']}" for s in sources)
    answer = call_llm(f"{SYSTEM}\n\nArticles:\n{context}\n\nQuestion: {req.question}")
    answerable = REFUSAL_AR not in answer and REFUSAL_EN not in answer
    return {"answer": answer, "answerable": answerable, "sources": sources if answerable else []}