"""Batch re-index: add/replace articles from a new document and rebuild the serving index.

  python src/legal_qa/reindex.py --add data/new_docs/amendment.jsonl

Input = JSONL in the articles.jsonl schema (article_number, text_ar, text_en, citation, ...).
Articles with an existing article_number are replaced, new ones are added. The index is rebuilt
with the config currently in data/index/best/meta.json (same model/chunking), built in a temp
folder and swapped in only on success. Restart the API afterwards (embedded Qdrant is loaded at startup).
"""
import argparse
import json
import shutil
from pathlib import Path

from legal_qa.indexing import build_index, read_jsonl

ROOT = Path(__file__).resolve().parents[2]


def merge_articles(existing: list[dict], new: list[dict]) -> tuple[list[dict], int, int]:
    by_no = {a["article_number"]: a for a in existing}
    added = sum(1 for a in new if a["article_number"] not in by_no)
    replaced = len(new) - added
    for a in new:
        by_no[a["article_number"]] = a
    return [by_no[k] for k in sorted(by_no)], added, replaced


def reindex(add_path: Path, articles_path: Path, index_dir: Path) -> dict:
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    merged, added, replaced = merge_articles(read_jsonl(articles_path), read_jsonl(add_path))
    tmp = index_dir.parent / f"{index_dir.name}_rebuild"
    new_meta = build_index(merged, meta["embedding_model"], meta["chunk_size"], meta["overlap"], tmp,
                           meta["collection"], fake=meta.get("fake", False))
    shutil.rmtree(index_dir)           # only reached if the rebuild succeeded
    tmp.rename(index_dir)
    articles_path.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in merged) + "\n", encoding="utf-8")
    print(f"+{added} new, {replaced} replaced -> {new_meta['n_articles']} articles, {new_meta['n_chunks']} chunks")
    return {"added": added, "replaced": replaced, **new_meta}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", required=True, type=Path)
    ap.add_argument("--articles", type=Path, default=ROOT / "data" / "articles.jsonl")
    ap.add_argument("--index", type=Path, default=ROOT / "data" / "index" / "best")
    a = ap.parse_args()
    reindex(a.add, a.articles, a.index)


if __name__ == "__main__":
    main()
