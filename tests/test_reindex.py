import json

from legal_qa.indexing import build_index
from legal_qa.reindex import merge_articles, reindex
from legal_qa.retriever import Retriever
from tests.conftest import ARTICLES

NEW_DOC = [
    {"article_number": 99, "text_ar": "مادة (99)\nتتقادم دعوى التعويض بمرور ثلاث سنوات.",
     "text_en": "Article 99\nA claim for compensation is time-barred after three years.", "citation": "Article 99", "is_repealed": False},
    {"article_number": 4, "text_ar": "مادة (4)\nنص معدل.", "text_en": "Article 4\nAmended text.", "citation": "Article 4", "is_repealed": False},
]


def test_merge_articles():
    merged, added, replaced = merge_articles(ARTICLES, NEW_DOC)
    assert (added, replaced) == (1, 1) and [a["article_number"] for a in merged][-1] == 99


def test_reindex_new_document_is_searchable(tmp_path, monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    art_path, add_path, idx = tmp_path / "articles.jsonl", tmp_path / "new.jsonl", tmp_path / "idx" / "best"
    art_path.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in ARTICLES), encoding="utf-8")
    add_path.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in NEW_DOC), encoding="utf-8")
    build_index(ARTICLES, "fake", 600, 80, idx, "col", fake=True)

    stats = reindex(add_path, art_path, idx)
    assert stats["added"] == 1 and stats["n_articles"] == 7

    r = Retriever(idx)
    top = r.search("A claim for compensation time-barred after three years", k=3)
    r.close()
    assert top[0]["article_number"] == 99