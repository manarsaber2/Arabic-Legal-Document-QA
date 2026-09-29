import json
import re
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "articles.jsonl"


@pytest.fixture(scope="module")
def articles():
    items = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def test_article_numbers_are_contiguous(articles):
    assert len(articles) > 0
    numbers = [a["article_number"] for a in articles]
    expected = list(range(1, len(articles) + 1))
    assert numbers == expected


def test_every_article_has_arabic_text(articles):
    arabic_pattern = re.compile(r"[\u0600-\u06FF]")
    for article in articles:
        text_ar = article.get("text_ar", "")
        assert text_ar and isinstance(text_ar, str)
        assert arabic_pattern.search(text_ar)


def test_no_article_exceeds_max_length(articles):
    max_length = 10000
    for article in articles:
        text_ar = article.get("text_ar", "")
        text_en = article.get("text_en", "")
        assert len(text_ar) <= max_length
        assert len(text_en) <= max_length


def test_repealed_articles_are_flagged(articles):
    for article in articles:
        assert "is_repealed" in article
        assert isinstance(article["is_repealed"], bool)