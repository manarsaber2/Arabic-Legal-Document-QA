"""Make a DRAFT eval set (mixed Arabic/English) with reference answers, from sampled articles.

For each sampled article Gemini writes one question + a short reference answer grounded ONLY in
that article, in the requested language. The label is that article's number. REVIEW the output by
hand before trusting any metric computed from it, and call it "synthetic, manually reviewed" in reports.

  python src/legal-qa/gen_eval.py --n 60              # ~60% Arabic, ~40% English
Row: {"question", "article_number", "lang", "reference"}
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "eval" / "eval.jsonl"
MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")
FALLBACK = os.getenv("FALLBACK_MODEL", "gemini-3.1-flash-lite")

PROMPT = """Below is one article of the Egyptian Civil Code, in Arabic and in English.
Write, in {lang_name}:
1. "question": ONE natural question a layperson might ask that this article answers. Do NOT mention
   the article number and do not copy long phrases from the text.
2. "reference": a short, correct answer (1-3 sentences) based ONLY on this article. No outside knowledge.
Return JSON only: {{"question": "...", "reference": "..."}}

ARABIC:
{ar}

ENGLISH:
{en}"""
LANG_NAME = {"ar": "simple Egyptian-flavoured Arabic", "en": "plain English"}
LABEL = re.compile(r"^\s*(مادة\s*\(?\s*[\d٠-٩]+\s*\)?|Article\s+\d+)\s*[:\-–.]?\s*", re.I)


def ask_llm(client, prompt: str) -> dict:
    """Gemini 503/429 are transient: retry with backoff, then the fallback model."""
    for model in (MODEL, FALLBACK):
        for attempt in range(4):
            try:
                r = client.models.generate_content(
                    model=model, contents=prompt,
                    config={"temperature": 0.4, "response_mime_type": "application/json"})
                d = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", r.text.strip()))
                if d.get("question") and d.get("reference"):
                    return d
            except Exception as e:  # noqa: BLE001
                print(f"  {model} attempt {attempt + 1} failed: {str(e)[:80]}")
            time.sleep(2 ** attempt)
    raise SystemExit("LLM unavailable, try again later")


def main() -> None:
    from dotenv import load_dotenv
    from google import genai
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    arts = [json.loads(l) for l in (ROOT / "data" / "articles.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    pool = [x for x in arts if not x["is_repealed"] and not x["qa_flags"]
            and len(x.get("text_ar") or "") > 80 and len(x.get("text_en") or "") > 80]
    random.seed(7)
    sample = random.sample(pool, a.n)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, art in enumerate(sample):
            lang = "ar" if i % 5 < 3 else "en"  # 3 of every 5 Arabic
            d = ask_llm(client, PROMPT.format(lang_name=LANG_NAME[lang], ar=LABEL.sub("", art["text_ar"]),
                                              en=LABEL.sub("", art["text_en"])))
            f.write(json.dumps({"question": d["question"], "article_number": art["article_number"],
                                "lang": lang, "reference": d["reference"]}, ensure_ascii=False) + "\n")
            print(f"{i + 1}/{a.n} [{lang}] art {art['article_number']}")
    print(f"wrote {a.n} rows -> {OUT}  (now REVIEW them by hand)")


if __name__ == "__main__":
    main()
