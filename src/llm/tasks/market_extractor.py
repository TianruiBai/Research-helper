"""LLM task: extract market / industry signals from abstracts."""

from __future__ import annotations

import logging

from src.analytics.paper_selector import select_papers_for_llm
from src.llm.client import LLMClient
from src.llm.prompts import MARKET_EXTRACTION_PROMPT, format_abstracts_batch
from src.storage.models import Paper

logger = logging.getLogger(__name__)


async def extract_market_signals(
    papers: list[Paper],
    llm_client: LLMClient,
    batch_size: int = 8,
    max_papers: int = 80,
) -> dict:
    """Extract industry/market signals from abstracts.

    Uses year-stratified, citation-weighted selection so every year is
    represented and the LLM sees the most influential papers.
    Returns:
        {
            "companies": list[str],
            "funders": list[str],
            "patent_paper_count": int,
            "total_papers_analysed": int,
        }
    """
    selected = select_papers_for_llm(papers, max_papers=max_papers)
    abstracts = [
        (i, p.abstract)
        for i, p in enumerate(selected)
        if p.abstract
    ]
    if not abstracts:
        return {"companies": [], "funders": [], "patent_paper_count": 0, "total_papers_analysed": 0}

    all_companies: list[str] = []
    all_funders: list[str] = []
    patent_count = 0

    for batch_start in range(0, len(abstracts), batch_size):
        batch = abstracts[batch_start : batch_start + batch_size]
        abstracts_text = format_abstracts_batch(batch)
        prompt = MARKET_EXTRACTION_PROMPT.format(abstracts_text=abstracts_text)

        try:
            result = await llm_client.complete_json(prompt)
            signals = result.get("signals", [])
            if isinstance(signals, list):
                for s in signals:
                    if isinstance(s, dict):
                        all_companies.extend(s.get("companies", []))
                        all_funders.extend(s.get("funders", []))
                        if s.get("has_patent_ref"):
                            patent_count += 1
        except Exception as e:
            logger.warning("Market extraction batch failed: %s", e)
            continue

    # Deduplicate
    from collections import Counter
    company_counts = Counter(c.strip() for c in all_companies if c.strip())
    funder_counts = Counter(f.strip() for f in all_funders if f.strip())

    return {
        "companies": [c for c, _ in company_counts.most_common(20)],
        "funders": [f for f, _ in funder_counts.most_common(20)],
        "funder_counts": funder_counts.most_common(20),
        "patent_paper_count": patent_count,
        "total_papers_analysed": len(abstracts),
    }
