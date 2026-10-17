import json
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

JSONL_PATH = BASE_DIR / "articles.jsonl"
CLEAN_JSONL_PATH = BASE_DIR / "articles.jsonl"
BASE_DIR = Path(__file__).resolve().parent
ARTICLES_PATH = BASE_DIR / "articles.jsonl"

def fix_duplicates_and_empty_texts():
    if not JSONL_PATH.exists():
        print(f"File not found: {JSONL_PATH}")
        return


    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Total records before cleaning: {len(records)}")

    seen_articles = set()
    cleaned_records = []
    duplicates_count = 0

    for rec in records:
        art_num = rec.get("article_number")

        if art_num is not None and art_num in seen_articles:
            duplicates_count += 1
            continue

        if art_num is not None:
            seen_articles.add(art_num)

        cleaned_records.append(rec)

    print(f" Removed {duplicates_count} duplicate records.")


    empty_ar_records = [
        r for r in cleaned_records
        if not r.get("is_repealed") and not (r.get("text_ar") or "").strip()
    ]

    if empty_ar_records:
        print(f"Found {len(empty_ar_records)} active records with empty Arabic text.")

    else:
        print(" All active articles have valid Arabic text!")


    with open(CLEAN_JSONL_PATH, "w", encoding="utf-8") as f:
        for rec in cleaned_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f" Success! Cleaned dataset saved with {len(cleaned_records)} records.")


def find_missing_text_pages():
    if not ARTICLES_PATH.exists():
        print(f"File not found: {ARTICLES_PATH}")
        return []

    empty_pages = set()
    missing_count = 0

    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            # Check non-repealed articles with empty or whitespace-only Arabic text
            is_repealed = item.get("is_repealed", False)
            text_ar = item.get("text_ar", "") or ""

            if not is_repealed and not text_ar.strip():
                page = item.get("source_page")
                art_num = item.get("article_number")
                if page:
                    empty_pages.add(page)
                missing_count += 1
                print(f"Missing Arabic text: Article {art_num} on Page {page}")

    sorted_pages = sorted(list(empty_pages))
    pages_str = ",".join(map(str, sorted_pages))

    print(f"\nFound {missing_count} active articles missing Arabic text across {len(sorted_pages)} pages.")
    print(f"Target Pages: {pages_str}")
    return pages_str


if __name__ == "__main__":
    fix_duplicates_and_empty_texts()
    find_missing_text_pages()


