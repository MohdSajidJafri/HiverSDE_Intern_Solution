"""
Unit tests for TextNormalizer.
"""

import pytest
from src.hiver_agent.nlp.normalizer import TextNormalizer


def test_normalizer_handles_and_urls():
    normalizer = TextNormalizer()
    raw = "@SpotifyCares my app crashed! Check screenshot: https://t.co/xyz123 &amp; help me"
    norm = normalizer.normalize(raw)
    assert "@SpotifyCares" not in norm
    assert "https://t.co/xyz123" not in norm
    assert "&amp;" not in norm
    assert "&" in norm
    assert "my app crashed! Check screenshot: http://link & help me" == norm


def test_normalizer_empty_and_none():
    normalizer = TextNormalizer()
    assert normalizer.normalize("") == ""
    assert normalizer.normalize(None) == ""


def test_extract_features():
    normalizer = TextNormalizer()
    features = normalizer.extract_features("@SpotifyCares why is shuffle broken??? https://t.co/abc")
    assert features["has_question"] is True
    assert features["has_url"] is True
    assert features["has_mention"] is True
    assert features["word_count"] > 0
