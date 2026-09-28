"""
extraction.py  (stage 1)
------------------------
PDF page -> image -> Gemini -> data/processed/page_XXX.json

Each page file contains an ORDERED list of blocks (headers and articles mixed,
in the same order as on the page). Hierarchy inheritance across pages, merging
of articles split across pages, repealed ranges, etc. are done in stage 2
(assemble.py), deterministically, not by the model.

"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Literal, Optional

import pymupdf  # PyMuPDF
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel

# ============================================================
# 1. Paths and configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

PDF_PATH = BASE_DIR / "data" / "legal_data_raw.pdf"
OUT_DIR = BASE_DIR / "data" / "processed"

MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL")  
MAX_OUTPUT_TOKENS = 16384


ATTEMPTS = [
    {"model": MODEL, "temperature": 0.0, "dpi": 150},
    {"model": MODEL, "temperature": 0.3, "dpi": 200},
    {"model": FALLBACK_MODEL or MODEL, "temperature": 0.3, "dpi": 200},
]

TRANSIENT_CODES = {429, 500, 502, 503, 504}


# ============================================================
# 2. Structured output schema (pydantic)
# ============================================================

class Block(BaseModel):
    kind: Literal["header", "article"]
    front_matter: bool

    # --- header fields (null for articles) ---
    label_en: Optional[str] 
    label_ar: Optional[str]  
    title_en: Optional[str]   
    title_ar: Optional[str]   

    
    number_en: Optional[str]
    number_ar: Optional[str]
    number_end: Optional[str]  
    text_en: Optional[str]
    text_ar: Optional[str]
    is_repealed: bool
    continues_from_previous_page: bool
    continues_on_next_page: bool


class PageExtraction(BaseModel):
    blocks: list[Block]


# ============================================================
# 3. Prompt
# ============================================================

PROMPT = """
You are transcribing ONE page of a bilingual (English | Arabic) Egyptian Civil Code.

Layout: a two-column table. English is in the LEFT column, Arabic is in the RIGHT
column. Each row is either one article or one heading.

Return "blocks": an ORDERED list of every heading and article on the page, in the
exact top-to-bottom order in which they appear. Each block has kind = "header"
or "article".

GENERAL RULES
- Copy text VERBATIM. Never translate, summarize, correct, or invent text.
- Extract only what is visibly present on this page. If something is absent, use null.
- Inside an article, keep paragraph breaks as "\\n" and keep Arabic clause markers
  such as (١) (٢) exactly as printed.
- For a header block: all article fields are null / false.
  For an article block: all header fields are null.

HEADERS (kind = "header")
- Heading rows that are NOT article text: BOOK / PART / TITLE / CHAPTER / SECTION
  headings, numbered topic headings such as "1. Laws and Rights" / "١- القانون والحق",
  and unnumbered sub-headings such as "Conflicts of law as to time:".
- label_* = the numbering part ("SECTION I", "الفصل الأول", "1.", "٢-").
  title_*  = the heading text ("Laws and their Applications", "القانون وتطبيقه").
  A heading with its numbering and its title (even on two lines) is ONE block.
  If there is no numbering, label is null and the text goes in title.
- A heading can exist in one language only: leave the other language null.
- Do not treat article text as a header.

ARTICLES (kind = "article")
- number_en / number_ar: only the printed number ("12" / "١٢").
- text_en / text_ar: the article body only. Do NOT include the words "Article 12"
  or "مادة ( ١٢ )" in the text.
- If a row continues an article that started on the previous page (no article number
  printed), set number_en = number_ar = null and continues_from_previous_page = true.
- If the last row is cut off at the bottom of the page (including a row that has only
  the "Article N" label and no text yet), set continues_on_next_page = true.
- REPEALED: if the row only states that articles are repealed, copy the notice exactly
  and set is_repealed = true. If it names a range (e.g. "Articles 54-80 have been
  repealed"), put the LAST number of the range in number_end (e.g. "80"), and in
  number_ar use only the first number. Otherwise number_end = null.
  Do NOT set is_repealed just because the word "repealed" appears inside normal legal
  text (for example Article 2 talks about how a law can be repealed).
- Never invent an article number that is not printed.

FRONT MATTER
- The boxed preamble at the top of the first page (the issuing law: "قانون الإصدار",
  its own numbered "مادة" items, dates) is NOT part of the Code. Set front_matter = true
  on every block inside that box, EXCEPT the heading "باب تمهيدي / أحكام عامة", which is
  a normal header (front_matter = false). Everywhere else front_matter = false.
"""


# ============================================================
# 4. Helpers
# ============================================================

AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def to_int(value):
    """First number in the string: '٥٢' -> 52, '( ١٠٠ )' -> 100, '٥٤ إلى ٨٠' -> 54."""
    if not value:
        return None
    m = re.search(r"\d+", str(value).translate(AR_DIGITS))
    return int(m.group()) if m else None


class EmptyResponse(Exception):
    pass


def diagnose(response):
    """Explain WHY response.text is empty (finish reason, block reason, thinking tokens)."""
    parts = []
    fb = getattr(response, "prompt_feedback", None)
    if fb is not None and getattr(fb, "block_reason", None):
        parts.append(f"prompt_block_reason={fb.block_reason}")
    cands = getattr(response, "candidates", None) or []
    if not cands:
        parts.append("no candidates")
    for c in cands:
        parts.append(f"finish_reason={getattr(c, 'finish_reason', None)}")
    um = getattr(response, "usage_metadata", None)
    if um is not None:
        parts.append(
            f"output_tokens={getattr(um, 'candidates_token_count', None)} "
            f"thoughts_tokens={getattr(um, 'thoughts_token_count', None)}"
        )
    return "; ".join(parts) or "unknown"


def call_once(client, model, temperature, image_bytes):
    """One model call with backoff for transient API errors only."""
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=PageExtraction,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    for i in range(1, 5):
        try:
            response = client.models.generate_content(
                model=model, contents=[image_part, PROMPT], config=config
            )
            break
        except APIError as e:
            code = getattr(e, "code", None)
            if code not in TRANSIENT_CODES or i == 4:
                raise
            wait = i * 5
            print(f"  API error {code}; retry {i}/3 in {wait}s ...")
            time.sleep(wait)

    text = response.text  
    if not text or not text.strip():
        raise EmptyResponse(diagnose(response))
    return text


# ============================================================
# 5. One page
# ============================================================

def page_summary(blocks):
    headers = [b for b in blocks if b.kind == "header"]
    articles = [b for b in blocks if b.kind == "article"]
    print(f"  Headers: {len(headers)} | Articles: {len(articles)}")
    for h in headers:
        en = f"{h.label_en or ''} {h.title_en or ''}".strip()
        ar = f"{h.label_ar or ''} {h.title_ar or ''}".strip()
        print(f"   H  {en} | {ar}" + ("  [front matter]" if h.front_matter else ""))
    for a in articles:
        n_en, n_ar = to_int(a.number_en), to_int(a.number_ar)
        flags = []
        if a.front_matter:
            flags.append("front matter")
        elif n_en != n_ar and not (n_en is None and n_ar is None):
            flags.append("EN/AR number mismatch")
        if (not a.text_en or not a.text_ar) and not a.continues_on_next_page and not a.front_matter:
            flags.append("missing text")
        if a.continues_from_previous_page:
            flags.append("continues from previous page")
        if a.continues_on_next_page:
            flags.append("continues on next page")
        if a.number_end:
            flags.append(f"range to {a.number_end}")
        print(f"   A  EN={n_en} AR={n_ar} | {', '.join(flags) or 'OK'}")


def process_page(doc, client, page_num, force):
    json_path = OUT_DIR / f"page_{page_num:03d}.json"
    err_path = OUT_DIR / f"page_{page_num:03d}.error.txt"

    print("\n" + "=" * 60)
    print(f"Page {page_num}/{len(doc)}")
    print("=" * 60)

    if json_path.exists() and not force:
        print(f"  cached: {json_path.name} (use --force to redo)")
        return True

    errors = []
    for n, att in enumerate(ATTEMPTS, start=1):
        try:
            pix = doc[page_num - 1].get_pixmap(dpi=att["dpi"])
            img = pix.tobytes("png")
            print(f"  attempt {n}: model={att['model']} temp={att['temperature']} "
                  f"dpi={att['dpi']} img={len(img)/1024:.0f}KB")

            raw = call_once(client, att["model"], att["temperature"], img)
            parsed = PageExtraction.model_validate_json(raw) 

            payload = {
                "page": page_num,  
                "model": att["model"],
                "blocks": [b.model_dump() for b in parsed.blocks],
            }
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if err_path.exists():
                err_path.unlink()
            page_summary(parsed.blocks)
            print(f"  saved {json_path.name}")
            return True

        except Exception as e:  
            msg = f"attempt {n} ({att['model']}, temp={att['temperature']}, dpi={att['dpi']}): " \
                  f"{type(e).__name__}: {e}"
            print(f"  FAILED {msg}")
            errors.append(msg)

    err_path.write_text("\n".join(errors), encoding="utf-8")
    print(f"  !! page {page_num} failed after {len(ATTEMPTS)} attempts -> {err_path.name}")
    return False


# ============================================================
# 6. Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pages", help="comma-separated page numbers, e.g. 4,8")
    args = ap.parse_args()

    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found: {PDF_PATH}")

    if OUT_DIR.is_file():
        OUT_DIR.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    client = genai.Client()
    doc = pymupdf.open(str(PDF_PATH))
    try:
        total = len(doc)
        pages = (
            [int(p) for p in args.pages.split(",")] if args.pages else range(1, total + 1)
        )
        failed = [p for p in pages if not process_page(doc, client, p, args.force)]
    finally:
        doc.close()

    print("\n" + "=" * 60)
    if failed:
        print(f"Extraction finished. FAILED pages: {failed}")
        print("Retry with: python src/legal-qa/extraction.py --pages "
              + ",".join(map(str, failed)))
    else:
        print("Extraction finished. All pages OK. Next: python src/legal-qa/assemble.py")
    print("=" * 60)


if __name__ == "__main__":
    main()