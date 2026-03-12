"""LLM task: classify motivation/problem sentences in abstracts."""

from __future__ import annotations

import logging

from src.analytics.paper_selector import select_papers_for_llm
from src.llm.client import LLMClient
from src.llm.prompts import MOTIVATION_CLASSIFICATION_PROMPT, format_abstracts_batch
from src.storage.models import Paper

logger = logging.getLogger(__name__)


async def classify_motivation(
    papers: list[Paper],
    llm_client: LLMClient,
    batch_size: int = 8,
    max_papers: int = 80,
) -> dict:
    """Classify problem/motivation sentences in abstracts.

    Uses year-stratified, citation-weighted selection so every year is
    represented and the LLM sees the most influential papers.
    Returns:
        {
            "motivation_sentences": list[str],
            "total_abstract_sentences": int,
            "problem_sentence_count": int,
        }
    """
    selected = select_papers_for_llm(papers, max_papers=max_papers)
    abstracts = [
        (i, p.abstract)
        for i, p in enumerate(selected)
        if p.abstract
    ]
    if not abstracts:
        return {"motivation_sentences": [], "total_abstract_sentences": 0, "problem_sentence_count": 0}

    all_motivation: list[str] = []
    total_sentences = 0

    for batch_start in range(0, len(abstracts), batch_size):
        batch = abstracts[batch_start : batch_start + batch_size]
        abstracts_text = format_abstracts_batch(batch)
        prompt = MOTIVATION_CLASSIFICATION_PROMPT.format(abstracts_text=abstracts_text)

        # Estimate total sentences in this batch
        for _, abstract in batch:
            total_sentences += len([s for s in abstract.split(".") if len(s.strip()) > 10])

        try:
            result = await llm_client.complete_json(prompt)
            sentences = result.get("sentences", [])
            if isinstance(sentences, list):
                for s in sentences:
                    if isinstance(s, dict) and s.get("label") in ("problem", "motivation"):
                        all_motivation.append(s.get("sentence", ""))
        except Exception as e:
            logger.warning("Motivation classification batch failed: %s", e)
            continue

    return {
        "motivation_sentences": all_motivation,
        "total_abstract_sentences": max(total_sentences, 1),
        "problem_sentence_count": len(all_motivation),
    }
