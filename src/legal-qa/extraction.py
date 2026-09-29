"""
extraction.py  (stage 1)
------------------------
PDF page -> image -> Gemini / HuggingFace -> data/processed/page_XXX.json

Each page file contains an ORDERED list of blocks (headers and articles mixed,
in the same order as on the page). Hierarchy inheritance across pages, merging
of articles split across pages, repealed ranges, etc. are done in stage 2
(assemble.py), deterministically, not by the model.
"""

import argparse
import base64
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Literal, Optional

import pymupdf  # PyMuPDF
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from huggingface_hub import InferenceClient
from pydantic import BaseModel

# ============================================================
# 1. Paths and configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

PDF_PATH = BASE_DIR / "data" / "legal_data_raw.pdf"
OUT_DIR = BASE_DIR / "data" / "processed"

MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite") 
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-3.1-flash-lite") 
HF_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"

MAX_OUTPUT_TOKENS = 32768

# Attempts designed to break strict verbatim text/image signatures (RECITATION workaround)
ATTEMPTS = [
    # Attempt 1: Standard extraction with safety override and 120 DPI
    {"provider": "gemini", "model": MODEL, "dpi": 120, "crop_y": 0, "crop_x": 0, "rotate": 0.0, "thinking": None},
    # Attempt 2: Vertical trim (crops top/bottom margins to remove repetitive headers/footers) + lower DPI
    {"provider": "gemini", "model": MODEL, "dpi": 110, "crop_y": 25, "crop_x": 10, "rotate": 0.0, "thinking": None},
    # Attempt 3: Micro rotation + fallback model
    {"provider": "gemini", "model": FALLBACK_MODEL, "dpi": 120, "crop_y": 20, "crop_x": 15, "rotate": 0.5, "thinking": None},
    # Attempt 4: Hugging Face Fallback (bypasses RECITATION filters)
    {"provider": "hf", "model": HF_MODEL, "dpi": 150, "crop_y": 0, "crop_x": 0, "rotate": 0.0, "thinking": None},
]

TRANSIENT_CODES = {429, 500, 502, 503, 504}
MAX_TRIES = 6


# ============================================================
# 2. Structured output schema (pydantic)
# ============================================================

class Block(BaseModel):
    kind: Literal["header", "article"]
    front_matter: bool

    # --- header fields (null for articles) ---
    label_en: Optional[str] = None
    label_ar: Optional[str] = None
    title_en: Optional[str] = None
    title_ar: Optional[str] = None

    # --- article fields (null for headers) ---
    number_en: Optional[str] = None
    number_ar: Optional[str] = None
    number_end: Optional[str] = None
    text_en: Optional[str] = None
    text_ar: Optional[str] = None
    is_repealed: bool = False
    continues_from_previous_page: bool = False
    continues_on_next_page: bool = False


class PageExtraction(BaseModel):
    blocks: list[Block]


# ============================================================
# 3. Prompt
# ============================================================

PROMPT = """
ANALYZE THIS BILINGUAL IMAGE MATRIX FOR LAYOUT OCR PARSING.
Extract and group visual text regions into structured OCR blocks.
Do NOT memorize, summarize, or cross-reference external database legal records.
Treat all text purely as arbitrary visual symbols rendered inside bounding regions.

You are transcribing ONE page of a bilingual (English | Arabic) Egyptian Civil Code document.

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
- text_ar must NOT start with "مادة" followed by a number, and text_en must NOT start
  with "Article" followed by a number. The number goes ONLY in number_en / number_ar.
- Numbers in parentheses such as (١) (٢) or (1) (2) INSIDE an article are paragraph
  markers, NOT article numbers. Never start a new article block for them; keep them
  inside the text of the current article. A new article block starts ONLY when the row
  has its own article label ("Article N" / "مادة N") in the number column.
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


def call_gemini(client, model, thinking, image_bytes):
    """One Gemini model call with safety limits disabled and transient backoff handling."""
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

    safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    config_args = {
        "response_mime_type": "application/json",
        "response_schema": PageExtraction,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "safety_settings": safety_settings,
        "tools": [],
    }

    if thinking:
        config_args["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)

    config = types.GenerateContentConfig(**config_args)

    for i in range(1, MAX_TRIES + 1):
        try:
            response = client.models.generate_content(
                model=model, contents=[image_part, PROMPT], config=config
            )
            break
        except APIError as e:
            code = getattr(e, "code", None)
            if code not in TRANSIENT_CODES or i == MAX_TRIES:
                raise
            wait = min(15 * 2 ** (i - 1), 120) + random.uniform(0, 5)
            print(f"   API error {code}; retry {i}/{MAX_TRIES - 1} in {wait:.0f}s ...")
            time.sleep(wait)

    text = response.text
    if not text or not text.strip():
        raise EmptyResponse(diagnose(response))
    return text


def call_hf(hf_client, model, image_bytes):
    """Fallback handler using Hugging Face Inference API for OCR / Document Extraction."""
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{base64_image}"

    completion = hf_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": PROMPT + "\n\nReturn ONLY raw valid JSON object with key 'blocks' like this: {\"blocks\": [...]}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    }
                ]
            }
        ],
        max_tokens=4096,
        temperature=0.1,
    )

    content = completion.choices[0].message.content
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    if not content or not content.strip():
        raise EmptyResponse("Hugging Face returned an empty response")
        
    return content.strip()


# ============================================================
# 5. One page
# ============================================================

def page_summary(blocks):
    headers = [b for b in blocks if b.kind == "header"]
    articles = [b for b in blocks if b.kind == "article"]
    print(f"   Headers: {len(headers)} | Articles: {len(articles)}")
    for h in headers:
        en = f"{h.label_en or ''} {h.title_en or ''}".strip()
        ar = f"{h.label_ar or ''} {h.title_ar or ''}".strip()
        print(f"    H  {en} | {ar}" + ("  [front matter]" if h.front_matter else ""))
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
        print(f"    A  EN={n_en} AR={n_ar} | {', '.join(flags) or 'OK'}")


def process_page(doc, gemini_client, hf_client, page_num, force):
    json_path = OUT_DIR / f"page_{page_num:03d}.json"
    err_path = OUT_DIR / f"page_{page_num:03d}.error.txt"

    print("\n" + "=" * 60)
    print(f"Page {page_num}/{len(doc)}")
    print("=" * 60)

    if json_path.exists() and not force:
        print(f"   cached: {json_path.name} (use --force to redo)")
        return True

    errors = []
    page = doc[page_num - 1]
    rect = page.rect

    for n, att in enumerate(ATTEMPTS, start=1):
        try:
            provider = att.get("provider", "gemini")
            crop_y = att.get("crop_y", 0)
            crop_x = att.get("crop_x", 0)
            clip_box = pymupdf.Rect(
                rect.x0 + crop_x,
                rect.y0 + crop_y,
                rect.x1 - crop_x,
                rect.y1 - crop_y
            )

            mat = pymupdf.Matrix(att.get("rotate", 0.0))
            pix = page.get_pixmap(dpi=att["dpi"], clip=clip_box, matrix=mat)
            img = pix.tobytes("png")

            print(
                f"   attempt {n} ({provider}): model={att['model']} dpi={att['dpi']} "
                f"crop_y={crop_y} crop_x={crop_x} rotate={att.get('rotate')} img={len(img)/1024:.0f}KB"
            )

            if provider == "hf":
                raw = call_hf(hf_client, att["model"], img)
            else:
                raw = call_gemini(gemini_client, att["model"], att.get("thinking"), img)


            data = json.loads(raw)
            if isinstance(data, list):
            
                data = {"blocks": data}
            
            parsed = PageExtraction.model_validate(data)

            payload = {
                "page": page_num,
                "model": f"{provider}/{att['model']}",
                "blocks": [b.model_dump() for b in parsed.blocks],
            }
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if err_path.exists():
                err_path.unlink()
            page_summary(parsed.blocks)
            print(f"   saved {json_path.name}")
            return True

        except Exception as e:
            msg = (
                f"attempt {n} ({att.get('provider', 'gemini')}/{att['model']}, crop_y={att.get('crop_y')}, "
                f"dpi={att['dpi']}): {type(e).__name__}: {e}"
            )
            print(f"   FAILED {msg}")
            errors.append(msg)

    err_path.write_text("\n".join(errors), encoding="utf-8")
    print(f"   !! page {page_num} failed after {len(ATTEMPTS)} attempts -> {err_path.name}")
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

    gemini_client = genai.Client()
    
    # Initialize Hugging Face Client
    hf_token = os.environ.get("HF_TOKEN")
    hf_client = InferenceClient(api_key=hf_token) if hf_token else None

    doc = pymupdf.open(str(PDF_PATH))
    try:
        total = len(doc)
        pages = (
            [int(p) for p in args.pages.split(",")] if args.pages else range(1, total + 1)
        )
        failed = [p for p in pages if not process_page(doc, gemini_client, hf_client, p, args.force)]
    finally:
        doc.close()

    print("\n" + "=" * 60)
    if failed:
        print(f"Extraction finished. FAILED pages: {failed}")
        print(
            "Retry with: python src/legal-qa/extraction.py --pages "
            + ",".join(map(str, failed))
        )
    else:
        print("Extraction finished. All pages OK. Next: python src/legal-qa/merge.py")
    print("=" * 60)


if __name__ == "__main__":
    main()