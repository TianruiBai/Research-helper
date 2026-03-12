"""Year-stratified, multi-signal paper selection for LLM tasks.

Ensures every year in the dataset is represented while prioritising
high-importance papers using:
  - Citation count & influential citations
  - Citation velocity (citations/year — proxy for recency-weighted impact)
  - Venue impact factor proxy (avg citation velocity of peer papers in same venue)
  - Author prominence (authors with many papers / high cumulative citations)
  - Source type bonus (journal > conference > preprint)

The NLP-first approach lets us filter to the most significant articles
*before* expensive LLM processing.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from datetime import datetime

from src.storage.models import Paper

CURRENT_YEAR = datetime.now().year

# Venue type multipliers: journals generally more rigorous than preprints
_VENUE_TYPE_BONUS: dict[str, float] = {
    "journal": 1.2,
    "conference": 1.1,
    "repository": 0.9,
    "preprint": 0.85,
}


def select_papers_for_llm(
    papers: list[Paper],
    max_papers: int = 80,
    require_abstract: bool = True,
) -> list[Paper]:
    """Select a year-balanced, importance-weighted subset of papers.

    Algorithm:
      1. Pre-compute venue impact and author prominence from the full pool.
      2. Score each paper with a multi-signal importance function.
      3. Group papers by year.
      4. Allocate a per-year budget (evenly, with leftovers going to
         years that have more papers).
      5. Within each year, take the top-K by importance score.
      6. If any year has fewer papers than its budget, redistribute
         the remainder to other years proportionally.

    Returns papers sorted by year (ascending) then importance (descending).
    """
    pool = [
        p for p in papers
        if (not require_abstract or p.abstract)
    ]
    if not pool:
        return []

    if len(pool) <= max_papers:
        return sorted(pool, key=lambda p: (p.year or 0, -_importance(p, {}, {})))

    # ── Pre-compute corpus-level signals ──
    venue_impact = _compute_venue_impact(pool)
    author_prominence = _compute_author_prominence(pool)

    # Group by year
    by_year: dict[int, list[Paper]] = defaultdict(list)
    no_year: list[Paper] = []
    for p in pool:
        if p.year:
            by_year[p.year].append(p)
        else:
            no_year.append(p)

    if not by_year:
        # No year info at all — fall back to top-scored
        pool.sort(key=lambda p: _importance(p, venue_impact, author_prominence), reverse=True)
        return pool[:max_papers]

    # Sort within each year by importance
    for year in by_year:
        by_year[year].sort(
            key=lambda p: _importance(p, venue_impact, author_prominence),
            reverse=True,
        )

    years_sorted = sorted(by_year.keys())
    num_years = len(years_sorted)

    # Reserve a small budget for papers with no year
    no_year_budget = min(len(no_year), max(max_papers // 20, 2))
    remaining = max_papers - no_year_budget

    # Initial even allocation
    base_per_year = max(remaining // num_years, 1)
    allocation: dict[int, int] = {y: base_per_year for y in years_sorted}

    # Redistribute from years with fewer papers than budget
    leftover = 0
    for y in years_sorted:
        avail = len(by_year[y])
        if avail < allocation[y]:
            leftover += allocation[y] - avail
            allocation[y] = avail

    # Give leftover to years with surplus papers
    if leftover > 0:
        surplus_years = [y for y in years_sorted if len(by_year[y]) > allocation[y]]
        if surplus_years:
            extra_each = leftover // len(surplus_years)
            extra_rem = leftover % len(surplus_years)
            for i, y in enumerate(surplus_years):
                bonus = extra_each + (1 if i < extra_rem else 0)
                allocation[y] = min(allocation[y] + bonus, len(by_year[y]))

    # Select top papers per year
    selected: list[Paper] = []
    for y in years_sorted:
        selected.extend(by_year[y][: allocation[y]])

    # Add no-year papers (top-scored)
    no_year.sort(
        key=lambda p: _importance(p, venue_impact, author_prominence),
        reverse=True,
    )
    selected.extend(no_year[:no_year_budget])

    # Final sort: year ascending, importance descending
    selected.sort(
        key=lambda p: (p.year or 0, -_importance(p, venue_impact, author_prominence))
    )
    return selected


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------

def _importance(
    paper: Paper,
    venue_impact: dict[str, float],
    author_prominence: dict[str, float],
) -> float:
    """Multi-signal importance score combining citations, venue IF, and author background.

    Components (all normalised via log or bounded):
      1. Citation score:       log(1 + citations) + 5 * log(1 + influential_citations)
      2. Citation velocity:    citations / max(age, 1)  — rewards recent impactful work
      3. Venue impact proxy:   avg citation velocity of papers in same venue (from pool)
      4. Author prominence:    max prominence score among the paper's authors
      5. Venue type bonus:     journal > conference > preprint multiplier
    """
    cites = paper.citations or 0
    influential = paper.influential_citations or 0

    # 1. Citation score (log-dampened to avoid extreme outliers dominating)
    citation_score = math.log1p(cites) + 5.0 * math.log1p(influential)

    # 2. Citation velocity
    age = max(CURRENT_YEAR - (paper.year or CURRENT_YEAR) + 1, 1)
    velocity = cites / age

    # 3. Venue impact factor proxy
    venue_if = 0.0
    if paper.venue and paper.venue in venue_impact:
        venue_if = venue_impact[paper.venue]

    # 4. Author prominence — take the best-known author on the paper
    author_score = 0.0
    for author in paper.get_authors():
        name = author.strip()
        if name and name in author_prominence:
            author_score = max(author_score, author_prominence[name])

    # 5. Venue type bonus
    vtype = (paper.venue_type or "").lower()
    type_mult = _VENUE_TYPE_BONUS.get(vtype, 1.0)

    # Combine: weighted sum, then type multiplier
    # Weights chosen so citation_score dominates but venue/author break ties
    combined = (
        citation_score * 3.0
        + velocity * 2.0
        + venue_if * 1.5
        + author_score * 1.0
    )
    return combined * type_mult


def _compute_venue_impact(papers: list[Paper]) -> dict[str, float]:
    """Compute average citation velocity per venue from the paper pool.

    This serves as a proxy for journal Impact Factor, computed from
    the actual papers in our search results rather than external data.
    """
    venue_velocities: dict[str, list[float]] = defaultdict(list)
    for p in papers:
        if not p.venue or p.citations is None or not p.year:
            continue
        age = max(CURRENT_YEAR - p.year + 1, 1)
        velocity = p.citations / age
        venue_velocities[p.venue].append(velocity)

    return {
        venue: sum(vels) / len(vels)
        for venue, vels in venue_velocities.items()
        if vels
    }


def _compute_author_prominence(papers: list[Paper]) -> dict[str, float]:
    """Score each author by their cumulative citations and paper count in the pool.

    prominence = log(1 + total_citations) + 2 * log(1 + paper_count)

    Authors who appear often and accumulate many citations are considered
    more authoritative, giving their papers an importance boost.
    """
    author_cites: dict[str, int] = defaultdict(int)
    author_count: dict[str, int] = defaultdict(int)

    for p in papers:
        c = p.citations or 0
        for author in p.get_authors():
            name = author.strip()
            if not name:
                continue
            author_cites[name] += c
            author_count[name] += 1

    return {
        name: math.log1p(author_cites[name]) + 2.0 * math.log1p(author_count[name])
        for name in author_cites
    }
