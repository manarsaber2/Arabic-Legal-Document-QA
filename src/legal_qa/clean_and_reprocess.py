import json
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

JSONL_PATH = BASE_DIR / "data" / "articles.jsonl"
CLEAN_JSONL_PATH = BASE_DIR / "data" / "articles.jsonl"

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

if __name__ == "__main__":
    fix_duplicates_and_empty_texts()
