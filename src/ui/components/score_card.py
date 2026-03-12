"""Score card component — renders all dimension scores + sentiment."""

from __future__ import annotations

import streamlit as st

from src.reports.charts import sentiment_donut_chart


def render_score_cards(stats: dict) -> None:
    """Render the comprehensive score prominently, then dimension scores."""

    # --- Comprehensive score banner ---
    comprehensive = stats.get("comprehensive_score", 0)
    colour = _score_colour(comprehensive)
    st.markdown(
        f"### 🏆 Comprehensive Score: **{comprehensive:.0f}/100** {colour}"
    )
    st.progress(comprehensive / 100)

    # Field-aware weighting info
    field_cat = stats.get("field_category")
    field_pace = stats.get("field_pace")
    if field_cat:
        pace_icon = {"fast": "⚡", "medium": "🔄", "slow": "🐢"}.get(field_pace or "", "")
        display = stats.get("field_display_name", field_cat)
        st.caption(f"{pace_icon} Weighted for **{display}** ({field_pace} pace)")
    st.divider()

    # --- Individual dimension scores (3+3 layout) ---
    row1 = st.columns(3)
    row2 = st.columns(3)

    scores_row1 = [
        ("Interest", stats.get("interest_score", 0), "Publication volume & growth"),
        ("Motivation", stats.get("motivation_score", 0), "Problem-statement prevalence"),
        ("Confidence", stats.get("confidence_score", 0), "Claim strength + public trust"),
    ]
    scores_row2 = [
        ("Market", stats.get("market_score", 0), "Industry, funding & news buzz"),
        ("Public Sentiment", stats.get("public_sentiment_score", 0), "Attitude from news & web (50=neutral)"),
        ("News Articles", stats.get("news_article_count", 0), "Number of news/web articles found"),
    ]

    for col, (label, value, help_text) in zip(row1, scores_row1):
        with col:
            st.metric(label=label, value=f"{value:.0f}/100", help=help_text)
            st.progress(value / 100)
            st.caption(_score_colour(value))

    for col, (label, value, help_text) in zip(row2, scores_row2):
        with col:
            if label == "News Articles":
                st.metric(label=label, value=str(int(value)), help=help_text)
            else:
                st.metric(label=label, value=f"{value:.0f}/100", help=help_text)
                st.progress(value / 100)
                st.caption(_score_colour(value))


def render_sentiment_details(stats: dict) -> None:
    """Render positive / negative / neutral sentiment breakdown with donut chart."""
    pos_ratio  = stats.get("sentiment_positive_ratio", 0)
    neg_ratio  = stats.get("sentiment_negative_ratio", 0)
    neu_ratio  = stats.get("sentiment_neutral_ratio",
                           max(0.0, 1.0 - pos_ratio - neg_ratio))

    # Overall counts (derive from ratios if raw counts unavailable)
    total_sents = stats.get("total_papers", 100)
    pos_count = int(pos_ratio * total_sents)
    neg_count = int(neg_ratio * total_sents)
    neu_count = max(0, total_sents - pos_count - neg_count)

    st.subheader("💬 Sentiment Breakdown")

    # Metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric("👍 Positive", f"{pos_ratio:.0%}")
    c2.metric("😐 Neutral",  f"{neu_ratio:.0%}")
    c3.metric("👎 Negative", f"{neg_ratio:.0%}")

    # Donut chart
    fig = sentiment_donut_chart(pos_count, neg_count, neu_count,
                                title="Sentiment Distribution (all sources)")
    if fig.data:
        from src.ui.components.trend_chart import export_chart_buttons
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Export this chart", expanded=False):
            export_chart_buttons(fig, "sentiment_distribution")

    # Sample sentences
    pos_samples = stats.get("sentiment_positive_samples", [])
    neg_samples = stats.get("sentiment_negative_samples", [])

    if pos_samples:
        with st.expander("🟢 Positive sentiment samples", expanded=False):
            for s in pos_samples[:8]:
                if isinstance(s, dict):
                    st.markdown(f"- *{s.get('sentence', '')}*  \n  **{s.get('title', '')}**")
                else:
                    st.markdown(f"- {s}")
    if neg_samples:
        with st.expander("🔴 Negative sentiment samples", expanded=False):
            for s in neg_samples[:8]:
                if isinstance(s, dict):
                    st.markdown(f"- *{s.get('sentence', '')}*  \n  **{s.get('title', '')}**")
                else:
                    st.markdown(f"- {s}")


def _score_colour(value: float) -> str:
    if value >= 75:
        return "🟢 High"
    elif value >= 50:
        return "🟡 Moderate"
    elif value >= 25:
        return "🟠 Low"
    else:
        return "🔴 Very Low"
