"""
Unit tests for text processing utilities.
"""

import pytest
from src.utils.text_processing import (
    token_count,
    truncate_to_tokens,
    strip_html,
    split_by_headings,
    split_with_overlap,
    preprocess_query,
    compute_sentiment,
    compute_readability,
    count_citations,
    count_hedge_words,
    detect_refusal,
    classify_topic,
)


class TestTokenisation:

    def test_token_count_non_negative(self):
        assert token_count("Hello world") >= 0

    def test_empty_string(self):
        assert token_count("") == 0

    def test_truncate_preserves_content(self):
        text = "Hello world this is a test"
        truncated = truncate_to_tokens(text, max_tokens=3)
        assert token_count(truncated) <= 3


class TestHTMLProcessing:

    def test_strip_html_removes_tags(self):
        html = "<p>Hello <b>world</b></p>"
        assert strip_html(html) == "Hello world"

    def test_strip_html_handles_none(self):
        assert strip_html("") == ""

    def test_split_by_headings(self):
        html = "<h2>Section 1</h2><p>Content 1</p><h2>Section 2</h2><p>Content 2</p>"
        sections = split_by_headings(html)
        assert len(sections) >= 1


class TestQueryPreprocessing:

    def test_preserves_content(self):
        result = preprocess_query("HELLO WORLD")
        assert "HELLO" in result and "WORLD" in result

    def test_strips_whitespace(self):
        result = preprocess_query("  hello   world  ")
        assert "  " not in result


class TestDescriptors:

    def test_sentiment_in_range(self):
        score = compute_sentiment("This is a great test")
        assert -1.0 <= score <= 1.0

    def test_readability_positive(self):
        text = "Universal Credit is a monthly payment to help with your living costs. You may be able to get it if you are on a low income, out of work, or you cannot work."
        score = compute_readability(text)
        assert score > 0

    def test_count_citations_finds_links(self):
        text = "See https://www.gov.uk/example for details. Also check http://example.com"
        assert count_citations(text) >= 1

    def test_count_citations_finds_govuk(self):
        text = "According to GOV.UK guidance on this matter"
        count = count_citations(text)
        assert count >= 1

    def test_hedge_words_detected(self):
        text = "It might possibly be the case that perhaps this could work"
        count = count_hedge_words(text)
        assert count >= 2

    def test_no_hedge_words(self):
        text = "Universal Credit is a benefit"
        count = count_hedge_words(text)
        assert count == 0

    def test_refusal_detected(self):
        assert detect_refusal("I cannot provide personal advice on this matter")

    def test_no_refusal(self):
        assert not detect_refusal("Universal Credit is a benefit that helps people")


class TestTopicClassification:

    def test_universal_credit(self):
        assert classify_topic("How do I apply for Universal Credit?") == "universal_credit"

    def test_housing(self):
        assert classify_topic("Can I get housing benefit for my rent?") == "housing_benefit"

    def test_disability(self):
        assert classify_topic("What is PIP disability payment?") == "disability_benefits"

    def test_unknown(self):
        topic = classify_topic("What time is it?")
        # Should return some topic or "unknown"
        assert isinstance(topic, str)
