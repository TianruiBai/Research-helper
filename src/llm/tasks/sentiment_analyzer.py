"""LLM task: paper-level sentiment classification.

Classifies each abstract as positive / negative / neutral toward its research
area's prospects.  Returns the same dict structure as
analytics.sentiment.analyze_sentiment_heuristic so the pipeline can use either
interchangeably.
"""

from __future__ import annotations

import logging

from src.analytics.paper_selector import select_papers_for_llm
from src.llm.client import LLMClient
from src.llm.prompts import LLM_SENTIMENT_ANALYSIS_PROMPT, format_abstracts_batch
from src.storage.models import Paper

logger = logging.getLogger(__name__)


async def analyze_sentiment_llm(
    papers: list[Paper],
    llm_client: LLMClient,
    batch_size: int = 8,
    max_papers: int = 80,
) -> dict:
    """Classify each abstract as positive/negative/neutral using the LLM.

    Uses year-stratified, citation-weighted selection so every year is
    represented and the LLM sees the most influential papers.
    Returns a dict compatible with analyze_sentiment_heuristic:
        {positive_count, negative_count, neutral_count, total_sentences,
         positive_ratio, negative_ratio, neutral_ratio, sentiment_score,
         positive_samples, negative_samples}
    """
    selected = select_papers_for_llm(papers, max_papers=max_papers)
    abstracts = [
        (i, p.abstract)
        for i, p in enumerate(selected)
        if p.abstract
    ]
    if not abstracts:
        return _empty()

    counts: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
    positive_samples: list[dict] = []
    negative_samples: list[dict] = []
    paper_map = {i: p for i, p in enumerate(selected) if p.abstract}

    for batch_start in range(0, len(abstracts), batch_size):
        batch = abstracts[batch_start: batch_start + batch_size]
        abstracts_text = format_abstracts_batch(batch)
        prompt = LLM_SENTIMENT_ANALYSIS_PROMPT.format(abstracts_text=abstracts_text)

        try:
            result = await llm_client.complete_json(prompt)
            for item in result.get("classifications", []):
                if not isinstance(item, dict):
                    continue
                label = item.get("label", "neutral").lower()
                if label not in counts:
                    label = "neutral"
                counts[label] += 1
                idx = item.get("paper_index", -1)
                paper = paper_map.get(idx)
                if paper:
                    entry = {
                        "sentence": item.get("reason", ""),
                        "title": paper.title,
                        "source_type": paper.venue_type or "unknown",
                    }
                    if label == "positive" and len(positive_samples) < 20:
                        positive_samples.append(entry)
                    elif label == "negative" and len(negative_samples) < 20:
                        negative_samples.append(entry)
        except Exception as e:
            logger.warning("LLM sentiment batch %d failed: %s", batch_start, e)
            continue

    total = sum(counts.values()) or 1
    return {
        "positive_count": counts["positive"],
        "negative_count": counts["negative"],
        "neutral_count": counts["neutral"],
        "total_sentences": total,
        "positive_ratio": round(counts["positive"] / total, 3),
        "negative_ratio": round(counts["negative"] / total, 3),
        "neutral_ratio": round(counts["neutral"] / total, 3),
        "sentiment_score": (counts["positive"] - counts["negative"]) / total * 100,
        "positive_samples": positive_samples,
        "negative_samples": negative_samples,
    }


def _empty() -> dict:
    return {
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
        "total_sentences": 1,
        "positive_ratio": 0.0,
        "negative_ratio": 0.0,
        "neutral_ratio": 1.0,
        "sentiment_score": 0.0,
        "positive_samples": [],
        "negative_samples": [],
    }
