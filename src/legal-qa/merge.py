import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
PAGES_DIR = BASE_DIR / "data" / "processed"
OUT_JSONL = BASE_DIR / "data" / "articles.jsonl"
FRONT_JSON = BASE_DIR / "data" / "promulgation_law.json"
REPORT = BASE_DIR / "data" / "assembly_report.txt"

CODE_NAME_EN = "Egyptian Civil Code"
CODE_NAME_AR = "القانون المدني المصري"
EXPAND_REPEALED_RANGES = True

LEVELS = ["book", "chapter", "section", "topic", "subtopic"]

AR_LEVEL_PATTERNS = [
    (r"^(?:ال)?كتاب", "book"),
    (r"^(?:ال)?باب", "chapter"),
    (r"^(?:ال)?فصل", "section"),
    (r"^(?:ال)?فرع", "topic"),
]
EN_LEVEL_PATTERNS = [
    (r"^BOOK\b", "book"),
    (r"^(?:PART|TITLE)\b", "chapter"),
    (r"^(?:CHAPTER|SECTION)\b", "section"),
    (r"^(?:BRANCH|SUB-?SECTION)\b", "topic"),
]
NUMBERED = re.compile(r"^\s*[\d٠-٩]+\s*[-–.)]")
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

REPEAL_EN = re.compile(r"\b(?:has|have|was|were)\s+been\s+repealed\b", re.I)
REPEAL_AR = re.compile(r"(?m)^\W*(?:ألغي|ألغيت|أُلغي|أُلغيت)")
RANGE_EN = re.compile(
    r"Articles?\s+(\d+)\s*(?:-|–|to)\s*(\d+)\s+(?:has|have)\s+been\s+repealed", re.I
)
REF_EN = re.compile(r"\bArticles?\s+(\d+)")

LEVEL_WORD = re.compile(
    r"^\s*(?:ال)?(?:كتاب|باب|فصل|فرع)|^\s*(?:BOOK|PART|TITLE|CHAPTER|SECTION|PRELIMINARY)\b", re.I
)
NUM_SPLIT = re.compile(r"^\s*([\d٠-٩]+\s*[-–.)])\s*(.+)$", re.S)

AR_ONLY_HEADERS = ("باب تمهيدي",)


def to_int(v):
    if v is None:
        return None
    m = re.search(r"\d+", str(v).translate(AR_DIGITS))
    return int(m.group()) if m else None


def join(a, b):
    return "\n".join(x for x in (a, b) if x and x.strip())


def normalize_header(h):
    h = dict(h)
    for lang in ("en", "ar"):
        lab, tit = h.get(f"label_{lang}"), h.get(f"title_{lang}")
        if not lab and tit and "\n" in tit:
            first, rest = tit.split("\n", 1)
            if LEVEL_WORD.match(first) or NUMBERED.match(first):
                h[f"label_{lang}"], h[f"title_{lang}"] = first.strip(), rest.strip()

    if any(k in (h.get("label_ar") or "") for k in AR_ONLY_HEADERS):
        h["label_en"] = h["title_en"] = None

    m = {lang: NUM_SPLIT.match(h.get(f"label_{lang}") or "") for lang in ("en", "ar")}
    if (m["en"] or m["ar"]) and (h.get("title_en") or h.get("title_ar")):
        first, second = dict(h), dict(h)
        for lang in ("en", "ar"):
            if m[lang]:
                first[f"label_{lang}"], first[f"title_{lang}"] = m[lang].group(1), m[lang].group(2)
                second[f"label_{lang}"] = None
            else:
                second[f"label_{lang}"], second[f"title_{lang}"] = None, h.get(f"title_{lang}")
                first[f"title_{lang}"] = h.get(f"title_{lang}") if not h.get(f"label_{lang}") else first.get(f"title_{lang}")
        return [first, second]
    return [h]


def header_text(h, lang):
    return " ".join(str(x) for x in (h.get(f"label_{lang}"), h.get(f"title_{lang}")) if x)


def classify_header(h):
    ar, en = header_text(h, "ar"), header_text(h, "en")
    if NUMBERED.match(ar) or NUMBERED.match(en):
        return "topic"
    for pat, lvl in AR_LEVEL_PATTERNS:
        if re.match(pat, ar):
            return lvl
    for pat, lvl in EN_LEVEL_PATTERNS:
        if re.match(pat, en, re.I):
            return lvl
    return "subtopic"


def clean_title(s):
    return s.strip(" :\n") if s else None


def apply_header(state, level, h):
    i = LEVELS.index(level)
    for l in LEVELS[i:]:
        state[l] = None
    state[level] = {
        "label_en": h.get("label_en"),
        "label_ar": h.get("label_ar"),
        "en": clean_title(h.get("title_en") or h.get("label_en")),
        "ar": clean_title(h.get("title_ar") or h.get("label_ar")),
    }


def breadcrumb(state, lang):
    other = "ar" if lang == "en" else "en"
    parts = []
    for l in LEVELS:
        s = state[l]
        if s and (s[lang] or s[other]):
            parts.append(s[lang] or s[other])
    return " > ".join(parts)


def make_record(b, page, state):
    n_en, n_ar = to_int(b.get("number_en")), to_int(b.get("number_ar"))
    flags = []
    if n_en is not None and n_ar is not None and n_en != n_ar:
        flags.append("EN_AR_number_mismatch")
    num = n_en if n_en is not None else n_ar

    rec = {
        "id": f"civil_code_art_{num}",
        "article_number": num,
        "article_number_end": to_int(b.get("number_end")),
    }
    for l in LEVELS:
        s = state[l]
        rec[l] = s["en"] if s else None
        rec[f"{l}_ar"] = s["ar"] if s else None
    rec.update({
        "breadcrumb_en": breadcrumb(state, "en"),
        "breadcrumb_ar": breadcrumb(state, "ar"),
        "text_ar": b.get("text_ar") or "",
        "text_en": b.get("text_en") or "",
        "is_repealed": bool(b.get("is_repealed")),
        "source_page": page,
        "source_page_end": page,
        "citation": f"{CODE_NAME_EN}, Article {num}",
        "citation_ar": f"{CODE_NAME_AR}، المادة {num}",
        "qa_flags": flags,
        "_open": bool(b.get("continues_on_next_page")),
    })
    return rec


def load_pages():
    pages = {}
    for p in sorted(PAGES_DIR.glob("page_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        pages[d["page"]] = d["blocks"]
    return pages


def main():
    pages = load_pages()
    if not pages:
        raise SystemExit(f"No page_*.json in {PAGES_DIR}")

    total = int(sys.argv[1]) if len(sys.argv) > 1 else max(pages)
    warnings, outline, front = [], [], []
    missing = [p for p in range(1, total + 1) if p not in pages]
    if missing:
        warnings.append(f"MISSING PAGES (not extracted): {missing}")

    state = {l: None for l in LEVELS}
    records, current, prev_page = [], None, None

    for page in sorted(pages):
        if prev_page is not None and page != prev_page + 1:
            warnings.append(
                f"page gap: {prev_page} -> {page}. Articles/headings between them are lost"
                + (" (article %s was cut at the bottom of page %s)"
                   % (current["article_number"], prev_page) if current and current["_open"] else "")
            )
        prev_page = page

        for b in pages[page]:
            # ---------------- headers ----------------
            if b["kind"] == "header":
                if b.get("front_matter"):
                    continue
                headers = normalize_header(b)
                for idx, hb in enumerate(headers):
                    level = classify_header(hb)
                    # إذا تم تقسيم الهيدر، أجبر الثاني أن يكون subtopic لمنع مسح الـ topic الأول
                    if idx > 0 and level == "topic":
                        level = "subtopic"
                    apply_header(state, level, hb)
                    outline.append((page, level, header_text(hb, "en"), header_text(hb, "ar")))
                continue

            # ---------------- articles ----------------
            if b.get("front_matter"):
                front.append({
                    "number_ar": b.get("number_ar"), 
                    "text_ar": b.get("text_ar"),
                    "source_page": page,
                })
                continue

            no_number = to_int(b.get("number_en")) is None and to_int(b.get("number_ar")) is None

            if no_number:
                if current is None:
                    warnings.append(f"page {page}: continuation with no previous article (skipped)")
                    continue
                if not current["_open"]:
                    warnings.append(
                        f"page {page}: article {current['article_number']} was not marked as cut "
                        f"but a continuation follows (merged anyway)")
                current["text_en"] = join(current["text_en"], b.get("text_en"))
                current["text_ar"] = join(current["text_ar"], b.get("text_ar"))
                current["source_page_end"] = page
                current["_open"] = bool(b.get("continues_on_next_page"))
                continue

            if current is not None and current["_open"]:
                warnings.append(
                    f"page {page}: article {current['article_number']} was cut at the end of the "
                    f"previous page but the next row is a new article "
                    f"({b.get('number_en') or b.get('number_ar')}). Something is missing.")
            if b.get("continues_from_previous_page"):
                warnings.append(
                    f"page {page}: article {b.get('number_en') or b.get('number_ar')} is marked "
                    f"'continues from previous page' but has its own number")

            current = make_record(b, page, state)
            records.append(current)

    # ---------------- finalize / validate ----------------
    final, expected = [], None
    for r in records:
        r.pop("_open", None)
        flags = r["qa_flags"]
        en, ar = r["text_en"], r["text_ar"]

        notice = (len(en) < 300 and REPEAL_EN.search(en)) or (len(ar) < 500 and REPEAL_AR.search(ar))
        if notice and not r["is_repealed"]:
            r["is_repealed"] = True
            flags.append("repealed_flag_set_by_code")
        elif r["is_repealed"] and not notice:
            flags.append("repealed_by_model_only_check_manually")

        if r["article_number_end"] is None and en:
            m = RANGE_EN.search(en)
            if m and int(m.group(1)) == r["article_number"]:
                r["article_number_end"] = int(m.group(2))
        if r["article_number_end"] and r["article_number_end"] <= r["article_number"]:
            r["article_number_end"] = None

        if not en:
            flags.append("missing_text_en")
        if not ar:
            flags.append("missing_text_ar")

        n = r["article_number"]
        if n is not None:
            if expected is not None and n != expected:
                flags.append(f"sequence_break_expected_{expected}")
                warnings.append(
                    f"sequence break before article {n} (page {r['source_page']}): expected {expected}")
            expected = (r["article_number_end"] or n) + 1

        refs = sorted({int(x) for x in REF_EN.findall(en)} - {n}) if (en and not r["is_repealed"] and n is not None) else []
        r["references"] = refs

        final.append(r)

        # Repealed range handling
        if EXPAND_REPEALED_RANGES and r["is_repealed"] and r["article_number_end"] and n is not None:
            for k in range(n + 1, r["article_number_end"] + 1):
                stub = dict(r)
                stub.update({
                    "id": f"civil_code_art_{k}",
                    "article_number": k,
                    "article_number_end": None,  # تصفير المدى للـ Stub
                    "citation": f"{CODE_NAME_EN}, Article {k}",
                    "citation_ar": f"{CODE_NAME_AR}، المادة {k}",
                    "qa_flags": ["range_stub_of_%d" % n],
                    "references": [],
                })
                final.append(stub)

    # Duplicates Check
    seen = {}
    for r in final:
        num = r["article_number"]
        if num in seen:
            warnings.append(
                f"DUPLICATE article {num} (pages {seen[num]} and {r['source_page']})")
        seen[num] = r["source_page"]

    # ---------------- write ----------------
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    FRONT_JSON.write_text(json.dumps(front, ensure_ascii=False, indent=2), encoding="utf-8")

    flagged = [r for r in final if r["qa_flags"] and not r["qa_flags"][0].startswith("range_stub")]
    lines = [
        f"pages loaded: {len(pages)} / {total}",
        f"article records: {len(final)}  (repealed: {sum(r['is_repealed'] for r in final)})",
        f"promulgation-law items (excluded): {len(front)}",
        "",
        "=== WARNINGS ===", *(warnings or ["none"]),
        "",
        "=== FLAGGED RECORDS ===",
        *([f"art {r['article_number']} (p{r['source_page']}): {', '.join(r['qa_flags'])}"
           for r in flagged] or ["none"]),
        "",
        "=== HEADER OUTLINE (check the levels!) ===",
        *[f"p{p:>3} [{lvl:8}] {en} | {ar}" for p, lvl, en, ar in outline],
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4]))
    print(f"warnings: {len(warnings)} | flagged: {len(flagged)} -> see {REPORT}")


if __name__ == "__main__":
    main()