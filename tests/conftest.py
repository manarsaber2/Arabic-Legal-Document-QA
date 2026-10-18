"""Offline fixtures: tiny bilingual corpus, fake hashed embeddings, embedded Qdrant in tmp, stub LLM."""

import os

import pytest

os.environ["QUERY_REWRITE"] = "0"  # tests use a fixed-reply fake LLM; rewriting has its own test

from legal_qa.indexing import build_index

ARTICLES = [
    {"article_number": 1, "text_ar": "مادة (1)\nتسري النصوص التشريعية على المسائل التي تتناولها في لفظها أو في فحواها.",
     "text_en": "Article 1\nLegislative provisions apply to all matters they address in wording or spirit.",
     "citation": "Article 1", "is_repealed": False, "breadcrumb_en": "Preliminary Title", "breadcrumb_ar": "الباب التمهيدي"},
    {"article_number": 2, "text_ar": "مادة (2)\nلا يجوز إلغاء نص تشريعي إلا بتشريع لاحق ينص على هذا الإلغاء صراحة.",
     "text_en": "Article 2\nA legislative provision may only be repealed by a later enactment expressly providing so.",
     "citation": "Article 2", "is_repealed": False},
    {"article_number": 3, "text_ar": "مادة (3)\nتحدد القوانين الخاصة الاختصاص القضائي.",
     "text_en": "Article 3\nSpecial laws define judicial jurisdiction.", "citation": "Article 3", "is_repealed": True},
    {"article_number": 4, "text_ar": "مادة (4)\nمن استعمل حقه استعمالا مشروعا لا يكون مسؤولا عما ينشأ عن ذلك من ضرر.",
     "text_en": "Article 4\nA person who lawfully exercises a right is not liable for resulting damage.",
     "citation": "Article 4", "is_repealed": False},
    {"article_number": 5, "text_ar": "مادة (5)\nيكون استعمال الحق غير مشروع إذا لم يقصد به سوى الإضرار بالغير.",
     "text_en": "Article 5\nExercising a right is unlawful if its only aim is to harm another.",
     "citation": "Article 5", "is_repealed": False},
    {"article_number": 6, "text_ar": "مادة (6)\nالشخص الطبيعي تبدأ شخصيته بتمام ولادته حيا.",
     "text_en": "Article 6\nA natural person's legal personality begins at live birth.", "citation": "Article 6", "is_repealed": False},
]


@pytest.fixture(scope="session")
def index_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("idx") / "best"
    build_index(ARTICLES, "fake", 600, 80, d, "test_col", fake=True)
    return d


class FakeLLM:
    """Mimics google-genai client: client.models.generate_content(...).text"""
    def __init__(self, reply="الجواب حسب [المادة 4]."):
        self.reply = reply
        self.models = self

    def generate_content(self, model, contents, config=None):
        class R:  # noqa: D106
            text = self.reply
        return R()

    def generate_content_stream(self, model, contents, config=None):
        for w in self.reply.split(" "):
            yield type("P", (), {"text": w + " "})()


@pytest.fixture()
def client(index_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from legal_qa import api
    monkeypatch.setattr(api, "INDEX_DIR", index_dir)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    with TestClient(api.app) as c:
        api.state["llm"] = FakeLLM()
        yield c