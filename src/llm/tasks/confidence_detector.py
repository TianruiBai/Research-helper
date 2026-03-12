"""LLM task: detect claim strength / confidence in result sentences."""

from __future__ import annotations

import logging

from src.analytics.paper_selector import select_papers_for_llm
from src.llm.client import LLMClient
from src.llm.prompts import CONFIDENCE_DETECTION_PROMPT, format_abstracts_batch
from src.storage.models import Paper

logger = logging.getLogger(__name__)


async def detect_confidence(
    papers: list[Paper],
    llm_client: LLMClient,
    batch_size: int = 8,
    max_papers: int = 80,
) -> dict:
    """Detect claim strength in abstract result sentences.

    Uses year-stratified, citation-weighted selection so every year is
    represented and the LLM sees the most influential papers.
    Returns:
        {
            "strong_count": int,
            "moderate_count": int,
            "hedged_count": int,
            "negative_count": int,
            "total_result_sentences": int,
            "claims": list[dict],
        }
    """
    selected = select_papers_for_llm(papers, max_papers=max_papers)
    abstracts = [
        (i, p.abstract)
        for i, p in enumerate(selected)
        if p.abstract
    ]
    if not abstracts:
        return _empty_result()

    counts = {"strong": 0, "moderate": 0, "hedged": 0, "negative": 0}
    all_claims: list[dict] = []

    for batch_start in range(0, len(abstracts), batch_size):
        batch = abstracts[batch_start : batch_start + batch_size]
        abstracts_text = format_abstracts_batch(batch)
        prompt = CONFIDENCE_DETECTION_PROMPT.format(abstracts_text=abstracts_text)

        try:
            result = await llm_client.complete_json(prompt)
            claims = result.get("claims", [])
            if isinstance(claims, list):
                for c in claims:
                    if isinstance(c, dict):
                        label = c.get("label", "").lower()
                        if label in counts:
                            counts[label] += 1
                            all_claims.append(c)
        except Exception as e:
            logger.warning("Confidence detection batch failed: %s", e)
            continue

    total = sum(counts.values())
    return {
        "strong_count": counts["strong"],
        "moderate_count": counts["moderate"],
        "hedged_count": counts["hedged"],
        "negative_count": counts["negative"],
        "total_result_sentences": max(total, 1),
        "claims": all_claims,
    }


def _empty_result() -> dict:
    return {
        "strong_count": 0,
        "moderate_count": 0,
        "hedged_count": 0,
        "negative_count": 0,
        "total_result_sentences": 1,
        "claims": [],
    }
