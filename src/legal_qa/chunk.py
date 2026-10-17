"""Chunking stage: data/articles.jsonl -> data/chunks.jsonl   (self-contained, no extra imports)

One chunk per (article, language). A legal article is the smallest unit that still means
something alone, so we never split inside one unless it exceeds CHUNK_SIZE tokens (rare).

  - embedded text = article body only, normalized. Breadcrumb is NOT inside it: dozens of
    articles share the same breadcrumb and it would pull them artificially close together.
  - breadcrumb / chapter / section / citation / is_repealed live in metadata (payload),
    usable for BM25, filtering and citations.
  - Arabic and English are separate chunks (lang field), same article metadata.

Usage:
  python src/legal_qa/chunk.py
  CHUNK_SIZE=600 CHUNK_OVERLAP=80 python src/legal_qa/chunk.py
"""
import json
import os
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "articles.jsonl"
OUT_PATH = ROOT / "data" / "chunks.jsonl"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 600))      # word cap, only hit by rare long articles
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 80))

# ---------- normalization (retrieval only; display text stays untouched) ----------
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DIACRITICS = re.compile("[ً-ٰٟۖ-ۭ]")
_BIDI = re.compile("[‎‏‪-‮⁦-⁩]")
# leaked row label at the start of the body, e.g. "مادة ( ٧ )\n" -> must not be embedded
_AR_LABEL = re.compile(r"^\s*مادة\s*\(?\s*[\d٠-٩]+\s*\)?\s*[:\-–]?\s*")
_EN_LABEL = re.compile(r"^\s*Article\s+\d+\s*[:\-–.]?\s*", re.I)


def normalize_ar(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("ی", "ي").replace("ک", "ك")  # Persian yeh/kaf
    t = _BIDI.sub("", t)
    t = _AR_LABEL.sub("", t)
    t = _DIACRITICS.sub("", t).replace("ـ", "")                # diacritics + tatweel
    t = re.sub("[أإآٱ]", "ا", t)
    t = t.replace("ؤ", "و").replace("ئ", "ي").replace("ى", "ي").replace("ة", "ه")
    t = t.translate(_DIGITS)
    return re.sub(r"\s+", " ", t).strip()


def normalize_en(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = _EN_LABEL.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------- chunking ----------
def split_if_long(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    parts, start, step = [], 0, chunk_size - overlap
    while start < len(words):
        parts.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += step
    return parts


def strip_label(text: str, lang: str) -> str:
    """Display text: same label strip as normalize_*, but letters/diacritics stay as printed."""
    t = _AR_LABEL.sub("", text or "") if lang == "ar" else _EN_LABEL.sub("", text or "")
    return t.strip()


def build_chunks(articles: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    chunks, skipped = [], 0
    for rec in articles:
        for lang, norm in (("ar", normalize_ar), ("en", normalize_en)):
            text = norm(rec.get(f"text_{lang}") or "")
            if not text:
                skipped += 1
                continue
            parts = split_if_long(text, chunk_size, overlap)
            for i, part in enumerate(parts):
                n = rec["article_number"]
                chunks.append({
                    "chunk_id": f"{n}_{lang}" if len(parts) == 1 else f"{n}_{lang}_{i}",
                    "article_number": n,
                    "article_number_end": rec.get("article_number_end"),
                    "lang": lang,
                    "text": part,                      # normalized: what gets embedded / BM25-indexed
                    "display_text": strip_label(rec.get(f"text_{lang}") or "", lang) if len(parts) == 1 else part,
                    "part_index": i,
                    "num_parts": len(parts),
                    # metadata: filtering / BM25 / citation, never embedded
                    "is_repealed": rec.get("is_repealed", False),
                    "citation": rec.get("citation"),
                    "citation_ar": rec.get("citation_ar"),
                    "book": rec.get("book"), "book_ar": rec.get("book_ar"),
                    "chapter": rec.get("chapter"), "chapter_ar": rec.get("chapter_ar"),
                    "section": rec.get("section"), "section_ar": rec.get("section_ar"),
                    "topic": rec.get("topic"), "topic_ar": rec.get("topic_ar"),
                    "subtopic": rec.get("subtopic"), "subtopic_ar": rec.get("subtopic_ar"),
                    "breadcrumb_en": rec.get("breadcrumb_en"),
                    "breadcrumb_ar": rec.get("breadcrumb_ar"),
                    "source_page": rec.get("source_page"),
                    "qa_flags": rec.get("qa_flags", []),
                })
    if skipped:
        print(f"skipped {skipped} empty (article, lang) pairs")
    return chunks


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"{IN_PATH} not found")
    articles = [json.loads(l) for l in IN_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks = build_chunks(articles)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    split = sum(1 for c in chunks if c["num_parts"] > 1)
    print(f"{len(chunks)} chunks from {len(articles)} articles ({split} multi-part) -> {OUT_PATH}")


if __name__ == "__main__":
    main()