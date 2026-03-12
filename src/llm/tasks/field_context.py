"""LLM task: field-context-aware deep analysis.

Uses a smaller/lighter model when available for efficiency,
providing domain-specific interpretation of motivation, confidence,
and market signals based on the detected field pace.
"""

from __future__ import annotations

import logging

from src.analytics.field_awareness import FieldProfile
from src.analytics.paper_selector import select_papers_for_llm
from src.llm.client import LLMClient
from src.llm.prompts import FIELD_CONTEXT_ANALYSIS_PROMPT
from src.storage.models import FieldStats, Paper

logger = logging.getLogger(__name__)

# Evidence standard descriptions per field pace
_EVIDENCE_STANDARDS = {
    "fast": "conference peer review, benchmark comparisons, ablation studies",
    "moderate": "journal peer review, experimental validation, reproducibility",
    "slow": "randomized controlled trials, systematic reviews, regulatory approval",
}

# Velocity interpretation templates
_VELOCITY_INTERPRETATIONS = {
    "fast": "rapid acceleration typical of tech hype cycles — validate staying power",
    "moderate": "steady growth suggesting sustained interest — check for plateau signals",
    "slow": "significant movement in a traditionally slow domain — may signal a breakthrough",
}


async def analyze_field_context(
    papers: list[Paper],
    stats: FieldStats,
    field_profile: FieldProfile,
    llm_client: LLMClient,
    most_cited_authors: list[dict] | None = None,
    top_cited_details: list[dict] | None = None,
    sample_size: int = 20,
) -> dict:
    """Generate field-context-aware deep analysis.

    Uses year-stratified, citation-weighted selection so every year is
    represented and the LLM sees the most influential papers.
    Returns:
        {
            "motivation_depth": str,
            "confidence_assessment": str,
            "market_reality": str,
            "velocity_context": str,
            "gaps_and_opportunities": list[str],
            "field_specific_risks": list[str],
            "recommended_focus_areas": list[str],
        }
    """
    sample = select_papers_for_llm(papers, max_papers=sample_size)
    if not sample:
        return _empty()

    sample_text = "\n\n".join(
        f"[{i}] ({p.year or '?'}) {p.title}\n{p.abstract[:400]}"
        for i, p in enumerate(sample)
    )

    # Format most-cited author info
    if most_cited_authors and len(most_cited_authors) > 0:
        top_authors_str = ", ".join(
            f"{a['author']} ({a['total_citations']} cites, {a['paper_count']} papers)"
            for a in most_cited_authors[:5]
        )
    else:
        top_authors_str = "N/A"

    # Format most-cited paper info
    if top_cited_details and len(top_cited_details) > 0:
        top_paper = top_cited_details[0]
        top_paper_str = top_paper.get("title", "N/A")
        top_paper_cites = top_paper.get("citations", 0)
    else:
        top_paper_str = "N/A"
        top_paper_cites = 0

    top_venues_str = ", ".join(v for v, _ in stats.top_venues[:5]) if stats.top_venues else "N/A"
    top_themes_str = ", ".join(stats.top_themes[:8]) if stats.top_themes else "N/A"

    prompt = FIELD_CONTEXT_ANALYSIS_PROMPT.format(
        query=stats.query,
        field_category=field_profile.field_category,
        field_display_name=field_profile.display_name,
        field_pace=field_profile.pace,
        cycle_years=field_profile.typical_cycle_years,
        total_papers=stats.total_papers,
        growth_rate=stats.growth_rate_pct,
        top_themes=top_themes_str,
        top_venues=top_venues_str,
        most_cited_authors=top_authors_str,
        most_cited_paper=top_paper_str,
        most_cited_paper_citations=top_paper_cites,
        evidence_standard=_EVIDENCE_STANDARDS.get(field_profile.pace, _EVIDENCE_STANDARDS["moderate"]),
        velocity_interpretation=_VELOCITY_INTERPRETATIONS.get(field_profile.pace, _VELOCITY_INTERPRETATIONS["moderate"]),
        sample_abstracts=sample_text,
    )

    # When web search is enabled, prepend an instruction for the model to
    # ground its analysis in current online sources
    if llm_client.web_search:
        prompt = (
            "IMPORTANT: You have web search enabled. Search the internet for the "
            "latest developments, funding announcements, industry adoption, and "
            "recent breakthroughs related to the query below. Cite specific recent "
            "findings to ground your analysis in up-to-date information.\n\n"
            + prompt
        )
        logger.info("Field-context analysis using web-search-enabled model: %s", llm_client.model)
    else:
        logger.info("Field-context analysis using model: %s", llm_client.model)

    try:
        result = await llm_client.complete_json(prompt, temperature=0.5, max_tokens=8192)
        return {
            "motivation_depth": result.get("motivation_depth", ""),
            "confidence_assessment": result.get("confidence_assessment", ""),
            "market_reality": result.get("market_reality", ""),
            "velocity_context": result.get("velocity_context", ""),
            "gaps_and_opportunities": result.get("gaps_and_opportunities", []),
            "field_specific_risks": result.get("field_specific_risks", []),
            "recommended_focus_areas": result.get("recommended_focus_areas", []),
        }
    except Exception as e:
        logger.error("Field context analysis failed: %s", e)
        return _empty()


def _empty() -> dict:
    return {
        "motivation_depth": "",
        "confidence_assessment": "",
        "market_reality": "",
        "velocity_context": "",
        "gaps_and_opportunities": [],
        "field_specific_risks": [],
        "recommended_focus_areas": [],
    }
