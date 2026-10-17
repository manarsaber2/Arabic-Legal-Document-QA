"""Corpus sanity checks on data/articles.jsonl (skipped if the file is absent, e.g. in a fresh CI checkout)."""
import json
from pathlib import Path

import pytest

from legal_qa.chunk import build_chunks

PATH = Path(__file__).resolve().parents[1] / "data" / "articles.jsonl"
MAX_MISSING_FRAC = 0.02  # tolerated share of article numbers with no record inside [min, max]

pytestmark = pytest.mark.skipif(not PATH.exists(), reason="data/articles.jsonl not present")


@pytest.fixture(scope="module")
def arts():
    return [json.loads(l) for l in PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_required_fields(arts):
    for a in arts:
        assert isinstance(a["article_number"], int)
        assert (a.get("text_ar") or "").strip() or (a.get("text_en") or "").strip(), a["article_number"]


def test_unique_article_numbers(arts):
    nums = [a["article_number"] for a in arts]
    assert len(nums) == len(set(nums))


def test_numbers_cover_range(arts):
    covered = set()
    for a in arts:
        covered.update(range(a["article_number"], (a.get("article_number_end") or a["article_number"]) + 1))
    lo, hi = min(covered), max(covered)
    missing = [n for n in range(lo, hi + 1) if n not in covered]
    assert len(missing) <= MAX_MISSING_FRAC * (hi - lo + 1), f"{len(missing)} missing, e.g. {missing[:15]}"


def test_no_label_leak_in_embedded_text(arts):
    bad = [c["chunk_id"] for c in build_chunks(arts) if c["text"].startswith(("ماده", "Article "))]
    assert not bad, bad[:10]
