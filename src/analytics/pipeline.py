"""Unified analytics pipeline — orchestrates statistical + LLM analytics.

Design: every analysis function has signature run(papers, llm_client=None).
When llm_client is provided → uses LLM. When None → heuristic fallback.
Both tiers return the same output types.

The pipeline separates academic papers from news/web articles and combines
signals from both for comprehensive scoring.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Callable

from src.analytics.trend import compute_trend_stats
from src.analytics.citations import compute_citation_stats
from src.analytics.venues import compute_venue_stats
from src.analytics.nlp_fast import compute_nlp_stats
from src.analytics.field_awareness import detect_field
from src.analytics.paper_selector import select_papers_for_llm
from src.analytics.sentiment import (
    analyze_sentiment_heuristic,
    analyze_sentiment_by_source_type,
    compute_sentiment_by_year,
)
from src.analytics.scores import (
    compute_comprehensive_score,
    compute_confidence_score,
    compute_interest_score,
    compute_market_score,
    compute_motivation_score,
    compute_public_sentiment_score,
)
from src.analytics.heuristics import (
    heuristic_confidence,
    heuristic_market,
    heuristic_motivation,
)
from src.llm.client import LLMClient
from src.storage.models import FieldStats, Paper

logger = logging.getLogger(__name__)

NEWS_VENUE_TYPES = {"news", "web", "blog"}


class AnalyticsPipeline:
    """Runs the full analytics pipeline on a set of papers."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        field_context_client: LLMClient | None = None,
        use_parallel: bool = False,
    ):
        self.llm_client = llm_client
        self.field_context_client = field_context_client
        self._llm_available = False
        self.use_parallel = use_parallel

    async def check_llm(self) -> bool:
        """Check if the LLM is available."""
        if self.llm_client is None:
            self._llm_available = False
            return False
        self._llm_available = await self.llm_client.health_check()
        if self._llm_available:
            self._llm_available = await self.llm_client.is_model_available()
        return self._llm_available

    async def run(
        self,
        papers: list[Paper],
        query: str = "",
        year_start: int | None = None,
        year_end: int | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
        token_callback: Callable[[str], None] | None = None,
    ) -> FieldStats:
        """Run the full analytics pipeline.

        Splits input into academic papers and news/web articles.
        Runs statistical analytics on academic, sentiment on news,
        and combines both for comprehensive scoring.
        """
        if not papers:
            return FieldStats(query=query)

        def _emit(msg: str, count: int = 0) -> None:
            if progress_callback:
                progress_callback(msg, count)

        # Split into academic vs news/web
        academic_papers = [
            p for p in papers if (p.venue_type or "") not in NEWS_VENUE_TYPES
        ]
        news_articles = [
            p for p in papers if (p.venue_type or "") in NEWS_VENUE_TYPES
        ]

        logger.info(
            "Pipeline input: %d total (%d academic, %d news/web)",
            len(papers), len(academic_papers), len(news_articles),
        )

        # Determine year range from academic papers
        years = [p.year for p in academic_papers if p.year]
        if not years:
            years = [p.year for p in papers if p.year]
        if not year_start:
            year_start = min(years) if years else datetime.now().year
        if not year_end:
            year_end = max(years) if years else datetime.now().year

        logger.info(
            "Running analytics (%d–%d), LLM=%s",
            year_start, year_end, self._llm_available,
        )

        # ── Part A: Statistical analytics on ACADEMIC papers ──
        _emit(f"Statistical analysis ({len(academic_papers or papers)} papers)...", len(papers))
        analysis_papers = academic_papers if academic_papers else papers
        trend = compute_trend_stats(analysis_papers)
        citation = compute_citation_stats(analysis_papers)
        venue = compute_venue_stats(analysis_papers)
        nlp = compute_nlp_stats(analysis_papers)

        # ── Part A2: Field awareness — detect research domain ──
        _emit("Detecting research field...")
        field_sample = select_papers_for_llm(analysis_papers, max_papers=50)
        abstracts_sample = [
            p.abstract for p in field_sample if p.abstract
        ]
        field_profile = detect_field(query, abstracts_sample)
        field_weights = {
            "interest": field_profile.weight_interest,
            "motivation": field_profile.weight_motivation,
            "confidence": field_profile.weight_confidence,
            "market": field_profile.weight_market,
            "sentiment": field_profile.weight_sentiment,
        }
        logger.info(
            "Field detected: %s (pace=%s)",
            field_profile.field_category, field_profile.pace,
        )
        _emit(f"Field: {field_profile.field_category.replace('_', ' ').title()} (pace={field_profile.pace})")

        # ── Part B: Sentiment analysis on ALL articles ──
        _emit(f"Sentiment analysis ({len(papers)} articles)...")
        sentiment_by_type = analyze_sentiment_by_source_type(papers)
        news_sentiment = sentiment_by_type["news"]
        combined_sentiment = sentiment_by_type["combined"]
        sentiment_by_year = compute_sentiment_by_year(papers)

        # ── Part C: LLM or heuristic for motivation, confidence, market ──
        if self._llm_available and self.llm_client:
            motivation, confidence, market, themes, narrative = await self._run_llm(
                analysis_papers, query, year_start, year_end, trend, venue,
                progress_callback=progress_callback,
                token_callback=token_callback,
            )
            # Supplement: LLM-classified paper sentiment gives richer sample sentences
            _emit("LLM: analyzing paper sentiment...")
            from src.llm.tasks.sentiment_analyzer import analyze_sentiment_llm
            if token_callback:
                self.llm_client._stream_callback = token_callback
            try:
                llm_sent = await analyze_sentiment_llm(analysis_papers, self.llm_client)
            finally:
                self.llm_client._stream_callback = None
            if llm_sent["positive_samples"] or llm_sent["negative_samples"]:
                combined_sentiment = {
                    **combined_sentiment,
                    "positive_samples": llm_sent["positive_samples"],
                    "negative_samples": llm_sent["negative_samples"],
                }
            # Deep field-context analysis — deferred until after FieldStats assembly
            from src.llm.tasks.field_context import analyze_field_context
            _run_field_context = True
        else:
            _emit("Running heuristic analysis (LLM not available)...")
            motivation, confidence, market, themes, narrative = self._run_heuristic(
                analysis_papers
            )
            field_context = {}
            _run_field_context = False

        # ── Part D: Compute scores (academic + public blended) ──
        news_count = len(news_articles)

        # Public sentiment score (0–100, 50=neutral)
        public_sentiment = compute_public_sentiment_score(
            positive_ratio=news_sentiment["positive_ratio"],
            negative_ratio=news_sentiment["negative_ratio"],
            total_articles=news_count,
        )

        # Sentiment score from combined text (-100 to +100)
        combined_sentiment_score = combined_sentiment["sentiment_score"]

        interest = compute_interest_score(
            total_papers=trend["total_papers"],
            growth_rate_pct=trend["growth_rate_pct"],
            cumulative_citations=citation["cumulative_citations"],
            avg_citation_velocity=citation["avg_citation_velocity"],
            news_article_count=news_count,
        )

        motivation_score = compute_motivation_score(
            problem_sentence_count=motivation["problem_sentence_count"],
            total_abstract_sentences=motivation["total_abstract_sentences"],
        )

        confidence_score = compute_confidence_score(
            strong_count=confidence["strong_count"],
            moderate_count=confidence["moderate_count"],
            hedged_count=confidence["hedged_count"],
            negative_count=confidence["negative_count"],
            total_result_sentences=confidence["total_result_sentences"],
            public_sentiment_score=combined_sentiment_score,
        )

        # Market: compute ratios
        total_p = len(analysis_papers) or 1
        funding_ratio = len(market.get("funders", [])) / max(total_p, 1)
        patent_ratio = market.get("patent_paper_count", 0) / max(total_p, 1)
        market_sc = compute_market_score(
            industry_ratio=venue["industry_ratio"],
            funding_ratio=min(funding_ratio, 1.0),
            patent_ratio=min(patent_ratio, 1.0),
            news_positive_ratio=news_sentiment["positive_ratio"],
            news_article_count=news_count,
        )

        # Comprehensive score (the new top-level metric)
        comprehensive = compute_comprehensive_score(
            interest=interest,
            motivation=motivation_score,
            confidence=confidence_score,
            market=market_sc,
            public_sentiment=public_sentiment,
            field_weights=field_weights,
        )

        # Assemble FieldStats
        stats = FieldStats(
            query=query,
            year_range=(year_start, year_end),
            total_papers=trend["total_papers"],
            review_papers=trend["review_papers"],
            papers_per_year=trend["papers_per_year"],
            growth_rate_pct=round(trend["growth_rate_pct"], 2),
            cagr_pct=round(trend["cagr_pct"], 2),
            cumulative_citations=citation["cumulative_citations"],
            avg_citation_velocity=citation["avg_citation_velocity"],
            median_citations=citation["median_citations"],
            h_index_estimate=citation["h_index_estimate"],
            top_cited_papers=citation["top_cited_papers"],
            most_cited_authors=citation.get("most_cited_authors", []),
            top_cited_details=citation.get("top_cited_details", []),
            venue_impact=citation.get("venue_impact", []),
            top_venues=venue["top_venues"],
            top_authors=venue["top_authors"],
            country_distribution=venue["country_distribution"],
            industry_ratio=venue["industry_ratio"],
            interest_score=interest,
            motivation_score=motivation_score,
            confidence_score=confidence_score,
            market_score=market_sc,
            public_sentiment_score=public_sentiment,
            comprehensive_score=comprehensive,
            news_article_count=news_count,
            sentiment_positive_ratio=round(combined_sentiment["positive_ratio"], 3),
            sentiment_negative_ratio=round(combined_sentiment["negative_ratio"], 3),
            sentiment_positive_samples=combined_sentiment.get("positive_samples"),
            sentiment_negative_samples=combined_sentiment.get("negative_samples"),
            sentiment_neutral_ratio=round(combined_sentiment.get("neutral_ratio", 0.0), 3),
            sentiment_by_year=sentiment_by_year,
            sentiment_by_source={
                "academic": {
                    "positive_ratio": round(sentiment_by_type["academic"]["positive_ratio"], 3),
                    "negative_ratio": round(sentiment_by_type["academic"]["negative_ratio"], 3),
                    "neutral_ratio": round(sentiment_by_type["academic"].get("neutral_ratio", 0.0), 3),
                    "positive_count": sentiment_by_type["academic"]["positive_count"],
                    "negative_count": sentiment_by_type["academic"]["negative_count"],
                    "neutral_count": sentiment_by_type["academic"]["neutral_count"],
                },
                "news": {
                    "positive_ratio": round(sentiment_by_type["news"]["positive_ratio"], 3),
                    "negative_ratio": round(sentiment_by_type["news"]["negative_ratio"], 3),
                    "neutral_ratio": round(sentiment_by_type["news"].get("neutral_ratio", 0.0), 3),
                    "positive_count": sentiment_by_type["news"]["positive_count"],
                    "negative_count": sentiment_by_type["news"]["negative_count"],
                    "neutral_count": sentiment_by_type["news"]["neutral_count"],
                },
            },
            top_themes=themes,
            top_funders=market.get("funder_counts"),
            field_narrative=narrative.get("narrative") if narrative else None,
            maturity_label=narrative.get("maturity_label") if narrative else None,
            # Field awareness
            field_category=field_profile.field_category,
            field_display_name=field_profile.field_category.replace("_", " ").title(),
            field_pace=field_profile.pace,
        )

        # ── Part E: Deep field-context analysis (LLM, needs assembled stats) ──
        if _run_field_context:
            # Prefer the dedicated smaller model; fall back to primary
            fc_client = self.field_context_client or self.llm_client
            if fc_client:
                _emit("LLM: deep field-context analysis...")
                if token_callback:
                    fc_client._stream_callback = token_callback
                try:
                    field_context = await analyze_field_context(
                        papers=analysis_papers,
                        stats=stats,
                        field_profile=field_profile,
                        llm_client=fc_client,
                        most_cited_authors=citation.get("most_cited_authors", []),
                        top_cited_details=citation.get("top_cited_details", []),
                    )
                finally:
                    fc_client._stream_callback = None
                stats.motivation_depth = field_context.get("motivation_depth")
                stats.confidence_assessment = field_context.get("confidence_assessment")
                stats.market_reality = field_context.get("market_reality")
                stats.velocity_context = field_context.get("velocity_context")
                stats.gaps_and_opportunities = field_context.get("gaps_and_opportunities")
                stats.field_specific_risks = field_context.get("field_specific_risks")
                stats.recommended_focus_areas = field_context.get("recommended_focus_areas")

        _emit("Pipeline complete.", len(papers))
        return stats

    async def _run_llm(
        self,
        papers: list[Paper],
        query: str,
        year_start: int,
        year_end: int,
        trend: dict,
        venue: dict,
        progress_callback: Callable[[str, int], None] | None = None,
        token_callback: Callable[[str], None] | None = None,
    ) -> tuple[dict, dict, dict, list[str], dict]:
        """Run LLM-backed analytics tasks.

        When self.use_parallel is True (max_concurrent_llm_calls > 1),
        the four independent tasks run concurrently via asyncio.gather.
        Token streaming is disabled for parallel tasks to avoid interleaved
        output; narrative always runs sequentially with streaming enabled.
        """
        from src.llm.tasks.theme_extractor import extract_themes
        from src.llm.tasks.motivation_classifier import classify_motivation
        from src.llm.tasks.confidence_detector import detect_confidence
        from src.llm.tasks.market_extractor import extract_market_signals
        from src.llm.tasks.narrative import generate_narrative

        assert self.llm_client is not None

        def _emit(msg: str, count: int = 0) -> None:
            if progress_callback:
                progress_callback(msg, count)

        n = len(papers)
        logger.info("Running LLM analytics with model: %s (parallel=%s)", self.llm_client.model, self.use_parallel)

        if self.use_parallel:
            # ── Parallel mode: run independent tasks concurrently ──
            # Token streaming is suppressed during gather to avoid interleaved output.
            _emit(f"LLM: parallel analysis — themes + motivation + confidence + market ({n} papers)...", n)
            raw = await asyncio.gather(
                extract_themes(papers, self.llm_client),
                classify_motivation(papers, self.llm_client),
                detect_confidence(papers, self.llm_client),
                extract_market_signals(papers, self.llm_client),
                return_exceptions=True,
            )
            themes: list[str] = raw[0] if not isinstance(raw[0], Exception) else []
            motivation: dict = raw[1] if not isinstance(raw[1], Exception) else {
                "motivation_sentences": [], "total_abstract_sentences": 1, "problem_sentence_count": 0
            }
            confidence: dict = raw[2] if not isinstance(raw[2], Exception) else {
                "strong_count": 0, "moderate_count": 0, "hedged_count": 0,
                "negative_count": 0, "total_result_sentences": 0, "claims": [],
            }
            market: dict = raw[3] if not isinstance(raw[3], Exception) else {
                "companies": [], "funders": [], "funder_counts": [],
                "patent_paper_count": 0, "total_papers_analysed": 0,
            }
            for i, exc in enumerate(raw):
                if isinstance(exc, Exception):
                    logger.warning("Parallel LLM task %d failed: %s", i, exc)

            # Narrative is sequential (depends on themes) + token streaming re-enabled
            partial_stats = FieldStats(
                query=query,
                year_range=(year_start, year_end),
                total_papers=trend["total_papers"],
                growth_rate_pct=trend["growth_rate_pct"],
                top_venues=venue["top_venues"],
                top_themes=themes,
            )
            _emit("LLM: generating narrative summary...")
            self.llm_client._stream_callback = token_callback
            try:
                narrative = await generate_narrative(papers, partial_stats, self.llm_client)
            finally:
                self.llm_client._stream_callback = None

        else:
            # ── Sequential mode: stream tokens per task ──
            self.llm_client._stream_callback = token_callback
            try:
                _emit(f"LLM: extracting themes ({n} papers)...", n)
                themes = await extract_themes(papers, self.llm_client)

                _emit(f"LLM: classifying motivation ({n} papers)...", n)
                motivation = await classify_motivation(papers, self.llm_client)

                _emit(f"LLM: detecting confidence ({n} papers)...", n)
                confidence = await detect_confidence(papers, self.llm_client)

                _emit(f"LLM: extracting market signals ({n} papers)...", n)
                market = await extract_market_signals(papers, self.llm_client)

                # Build partial FieldStats for narrative generation
                partial_stats = FieldStats(
                    query=query,
                    year_range=(year_start, year_end),
                    total_papers=trend["total_papers"],
                    growth_rate_pct=trend["growth_rate_pct"],
                    top_venues=venue["top_venues"],
                    top_themes=themes,
                )
                _emit("LLM: generating narrative summary...")
                narrative = await generate_narrative(papers, partial_stats, self.llm_client)
            finally:
                self.llm_client._stream_callback = None

        return motivation, confidence, market, themes, narrative

    def _run_heuristic(
        self, papers: list[Paper]
    ) -> tuple[dict, dict, dict, list[str], dict | None]:
        """Run heuristic (non-LLM) fallback analytics."""
        logger.info("Running heuristic analytics (no LLM)")
        motivation = heuristic_motivation(papers)
        confidence = heuristic_confidence(papers)
        market = heuristic_market(papers)
        themes: list[str] = []  # no theme extraction without LLM
        narrative = None
        return motivation, confidence, market, themes, narrative
