"""Tests for trace parsing and evaluation metrics."""

from app.evaluation import (
    RecommendationRecord,
    _binary_relevance_metrics,
    _catalog_lookup,
    _evaluate_groundedness,
    parse_trace_markdown,
)


SAMPLE_TRACE = """## Conversation

### Turn 1

**User**

> We need a solution for senior leadership.

**Agent**

Happy to help narrow that down. Who is this meant for?

_No recommendations this turn (`recommendations: null`)._

_`end_of_conversation`: **false**_

### Turn 2

**User**

> Selection - comparing candidates against a leadership benchmark.

**Agent**

For selection with a leadership benchmark, the instrument plus two relevant report formats:

| # | Name | Test Type | Keys | Duration | Languages | URL |
|---|------|-----------|------|----------|-----------|-----|
| 1 | Occupational Personality Questionnaire OPQ32r | P | Personality & Behavior | 25 minutes | English (USA) | <https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/> |
| 2 | OPQ Leadership Report | P | Personality & Behavior | - | - | <https://www.shl.com/products/product-catalog/view/opq-leadership-report/> |

_`end_of_conversation`: **true**_
"""


def test_parse_trace_markdown_extracts_turns_and_recommendations():
    trace = parse_trace_markdown(SAMPLE_TRACE, "sample")

    assert trace.trace_id == "sample"
    assert len(trace.turns) == 2
    assert trace.turns[0].expected_recommendations == []
    assert trace.turns[1].expected_end_of_conversation is True
    assert trace.turns[1].expected_recommendations[0].name == "Occupational Personality Questionnaire OPQ32r"
    assert trace.turns[1].expected_recommendations[1].url.endswith("/opq-leadership-report/")


def test_binary_relevance_metrics_capture_partial_overlap():
    ranked = [
        {"name": "A", "url": "https://example.com/a"},
        {"name": "B", "url": "https://example.com/b"},
        {"name": "C", "url": "https://example.com/c"},
    ]
    relevant = [
        RecommendationRecord(name="B", url="https://example.com/b"),
        RecommendationRecord(name="D", url="https://example.com/d"),
    ]

    metrics = _binary_relevance_metrics(ranked, relevant)

    assert metrics.precision_at_10 == 0.3333
    assert metrics.recall_at_10 == 0.5
    assert metrics.hit_rate == 1.0
    assert metrics.mrr_at_10 == 0.5


def test_groundedness_rewards_catalog_and_reply_alignment():
    assessments = [
        {
            "name": "Docker (New)",
            "url": "https://www.shl.com/products/product-catalog/view/docker-new/",
        },
        {
            "name": "Spring (New)",
            "url": "https://www.shl.com/products/product-catalog/view/spring-new/",
        },
    ]
    catalog_names, catalog_urls = _catalog_lookup(assessments)
    response = {
        "reply": "Docker (New) and Spring (New) are the closest matches here.",
        "recommendations": [
            {
                "name": "Docker (New)",
                "url": "https://www.shl.com/products/product-catalog/view/docker-new/",
            },
            {
                "name": "Spring (New)",
                "url": "https://www.shl.com/products/product-catalog/view/spring-new/",
            },
        ],
        "end_of_conversation": False,
    }
    retrieved = [
        {"name": "Docker (New)", "url": "https://www.shl.com/products/product-catalog/view/docker-new/"},
        {"name": "Spring (New)", "url": "https://www.shl.com/products/product-catalog/view/spring-new/"},
    ]

    groundedness = _evaluate_groundedness(response, catalog_names, catalog_urls, retrieved)

    assert groundedness.catalog_alignment == 1.0
    assert groundedness.retrieval_support == 1.0
    assert groundedness.reply_alignment == 1.0
    assert groundedness.groundedness_score == 1.0
