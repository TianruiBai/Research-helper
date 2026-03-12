"""LLM task: generate narrative field summary."""

from __future__ import annotations

import logging

from src.analytics.paper_selector import select_papers_for_llm
from src.llm.client import LLMClient
from src.llm.prompts import NARRATIVE_SUMMARY_PROMPT
from src.storage.models import FieldStats, Paper

logger = logging.getLogger(__name__)


async def generate_narrative(
    papers: list[Paper],
    stats: FieldStats,
    llm_client: LLMClient,
    sample_size: int = 20,
) -> dict:
    """Generate a 3-5 paragraph field overview.

    Uses year-stratified, citation-weighted selection so every year is
    represented and the LLM sees the most influential papers.
    Returns:
        {
            "narrative": str,
            "maturity_label": str,
            "open_questions": list[str],
        }
    """
    sample = select_papers_for_llm(papers, max_papers=sample_size)
    if not sample:
        return _empty()

    sample_text = "\n\n".join(
        f"[{i}] ({p.year or '?'}) {p.title}\n{p.abstract[:400]}"
        for i, p in enumerate(sample)
    )

    top_venues_str = ", ".join(v for v, _ in stats.top_venues[:5]) if stats.top_venues else "N/A"
    top_themes_str = ", ".join(stats.top_themes[:8]) if stats.top_themes else "N/A"

    prompt = NARRATIVE_SUMMARY_PROMPT.format(
        query=stats.query,
        total_papers=stats.total_papers,
        year_start=stats.year_range[0],
        year_end=stats.year_range[1],
        growth_rate=stats.growth_rate_pct,
        top_venues=top_venues_str,
        top_themes=top_themes_str,
        sample_abstracts=sample_text,
    )

    try:
        result = await llm_client.complete_json(prompt, temperature=0.7, max_tokens=8192)
        return {
            "narrative": result.get("narrative", ""),
            "maturity_label": result.get("maturity_label", "Unknown"),
            "open_questions": result.get("open_questions", []),
        }
    except Exception as e:
        logger.error("Narrative generation failed: %s", e)
        return _empty()


def _empty() -> dict:
    return {
        "narrative": "",
        "maturity_label": "Unknown",
        "open_questions": [],
    }
