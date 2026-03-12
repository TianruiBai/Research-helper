"""Score calculator — computes Interest, Motivation, Confidence, Market,
Public Sentiment, and Comprehensive scores (0–100).

Works with both LLM-derived and heuristic input.
Redesigned to combine academic research signals with public/news sentiment
for a more accurate picture of a field's real-world status.
"""

from __future__ import annotations

import math


def compute_interest_score(
    total_papers: int,
    growth_rate_pct: float,
    cumulative_citations: int,
    avg_citation_velocity: float,
    news_article_count: int = 0,
) -> float:
    """Interest score: statistical + public attention.

    Academic research is slow — news coverage reflects real-time interest.
    """
    academic_raw = (
        0.25 * math.log(total_papers + 1)
        + 0.35 * _sigmoid(growth_rate_pct / 100.0)
        + 0.20 * math.log(cumulative_citations + 1)
        + 0.20 * _sigmoid(avg_citation_velocity / 10.0)
    )
    academic_norm = _normalise(academic_raw, ceiling=10.0)

    # News boost: each news article adds signal, diminishing returns
    news_boost = _sigmoid(news_article_count / 20.0 - 1.0) if news_article_count > 0 else 0.0

    # Blend: 70% academic, 30% public attention
    blended = 0.70 * academic_norm + 0.30 * news_boost
    return min(round(blended * 100, 1), 100.0)


def compute_motivation_score(
    problem_sentence_count: int,
    total_abstract_sentences: int,
) -> float:
    """Motivation score: ratio of problem/motivation sentences to total.

    Works the same whether input comes from LLM or heuristic regex.
    """
    if total_abstract_sentences <= 0:
        return 0.0
    ratio = problem_sentence_count / total_abstract_sentences
    return min(round(ratio * 100, 1), 100.0)


def compute_confidence_score(
    strong_count: int,
    moderate_count: int,
    hedged_count: int,
    negative_count: int,
    total_result_sentences: int,
    public_sentiment_score: float = 0.0,
) -> float:
    """Confidence score: blends academic claim strength with public sentiment.

    Academic confidence: weighted sum of claim-strength labels.
    Public sentiment adjusts: positive public reception boosts confidence,
    negative reception dampens it.
    """
    if total_result_sentences <= 0:
        academic_conf = 0.0
    else:
        weighted = (
            1.0 * strong_count
            + 0.5 * moderate_count
            + 0.1 * hedged_count
            + 0.0 * negative_count
        )
        academic_conf = min((weighted / total_result_sentences) * 100, 100.0)

    # Public sentiment is -100 to +100; normalise to 0–100 scale
    public_norm = (public_sentiment_score + 100) / 2.0

    # Blend: 65% academic, 35% public sentiment
    if public_sentiment_score == 0.0 and total_result_sentences > 0:
        # No public data available — use academic only
        blended = academic_conf
    else:
        blended = 0.65 * academic_conf + 0.35 * public_norm

    return min(round(blended, 1), 100.0)


def compute_market_score(
    industry_ratio: float,
    funding_ratio: float,
    patent_ratio: float,
    news_positive_ratio: float = 0.0,
    news_article_count: int = 0,
) -> float:
    """Market interest score — combines academic industry signals with news coverage.

    News positive ratio reflects how the public/industry views the technology.
    More news coverage + positive tone = higher market interest.
    """
    academic_raw = 0.45 * industry_ratio + 0.35 * funding_ratio + 0.20 * patent_ratio
    academic_score = min(academic_raw * 100, 100.0)

    if news_article_count == 0:
        return min(round(academic_score, 1), 100.0)

    # News market signal: coverage volume * positive tone
    news_volume = _sigmoid(news_article_count / 15.0 - 1.0)
    news_signal = news_volume * news_positive_ratio * 100

    # Blend: 55% academic, 45% news/public
    blended = 0.55 * academic_score + 0.45 * news_signal
    return min(round(blended, 1), 100.0)


def compute_public_sentiment_score(
    positive_ratio: float,
    negative_ratio: float,
    total_articles: int,
) -> float:
    """Public sentiment score (0–100): how positive is public discourse.

    50 = neutral, >50 = positive-leaning, <50 = negative-leaning.
    Adjusted for coverage volume (more articles = higher confidence in score).
    """
    if total_articles == 0:
        return 50.0  # no data = neutral
    raw = (positive_ratio - negative_ratio + 1.0) / 2.0  # map -1..+1 to 0..1
    # Volume confidence: dampens extreme scores when few articles
    volume_factor = min(total_articles / 10.0, 1.0)
    adjusted = 50.0 + (raw - 0.5) * 100 * volume_factor
    return min(max(round(adjusted, 1), 0.0), 100.0)


def compute_comprehensive_score(
    interest: float,
    motivation: float,
    confidence: float,
    market: float,
    public_sentiment: float = 50.0,
    field_weights: dict[str, float] | None = None,
) -> float:
    """Comprehensive score — weighted blend of ALL dimension scores.

    This is the top-level "field health" score that answers:
    "How promising is this research area overall?"

    When field_weights is provided (from field_awareness.detect_field),
    uses domain-specific weighting. Otherwise uses balanced defaults.

    Default weights:
        Interest     25%  — Is the field active and growing?
        Confidence   20%  — Are results reliable? Public agrees?
        Market       25%  — Is there real-world / commercial interest?
        Motivation   15%  — Are there clear unsolved problems?
        Public       15%  — What does the general discourse say?
    """
    if field_weights:
        w_i = field_weights.get("interest", 0.25)
        w_m = field_weights.get("motivation", 0.15)
        w_c = field_weights.get("confidence", 0.20)
        w_k = field_weights.get("market", 0.25)
        w_s = field_weights.get("sentiment", 0.15)
    else:
        w_i, w_m, w_c, w_k, w_s = 0.25, 0.15, 0.20, 0.25, 0.15

    raw = (
        w_i * interest
        + w_m * motivation
        + w_c * confidence
        + w_k * market
        + w_s * public_sentiment
    )
    return min(round(raw, 1), 100.0)


# -- helpers ---------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Map unbounded value to (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


def _normalise(value: float, ceiling: float = 10.0) -> float:
    """Map a positive value to 0–1 using the ceiling as reference."""
    if value <= 0:
        return 0.0
    return min(value / ceiling, 1.0)
