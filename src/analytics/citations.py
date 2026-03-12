"""Citation analytics — h-index, velocity, top-cited papers, top-cited authors."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import datetime

from src.storage.models import Paper

CURRENT_YEAR = datetime.now().year


def compute_citation_velocity(paper: Paper) -> float:
    """Citations per year since publication."""
    if not paper.citations or not paper.year:
        return 0.0
    age = max(CURRENT_YEAR - paper.year + 1, 1)
    return paper.citations / age


def compute_h_index(papers: list[Paper]) -> int:
    """Virtual h-index for the paper set."""
    cites = sorted(
        (p.citations for p in papers if p.citations is not None),
        reverse=True,
    )
    h = 0
    for i, c in enumerate(cites, start=1):
        if c >= i:
            h = i
        else:
            break
    return h


def compute_most_cited_authors(papers: list[Paper], top_n: int = 10) -> list[dict]:
    """Rank authors by cumulative citations across all their papers.

    Returns list of dicts:
        [{"author": str, "total_citations": int, "paper_count": int,
          "avg_citations": float, "top_paper": str}]
    """
    author_cites: defaultdict[str, int] = defaultdict(int)
    author_papers: defaultdict[str, int] = defaultdict(int)
    author_best: dict[str, tuple[str, int]] = {}  # author → (title, cites)

    for p in papers:
        if p.citations is None:
            continue
        for author in p.get_authors():
            name = author.strip()
            if not name:
                continue
            author_cites[name] += p.citations
            author_papers[name] += 1
            prev = author_best.get(name, ("", -1))
            if p.citations > prev[1]:
                author_best[name] = (p.title or "", p.citations)

    ranked = sorted(author_cites.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        {
            "author": name,
            "total_citations": total,
            "paper_count": author_papers[name],
            "avg_citations": round(total / max(author_papers[name], 1), 1),
            "top_paper": author_best.get(name, ("", 0))[0],
        }
        for name, total in ranked
    ]


def compute_top_cited_with_details(papers: list[Paper], top_n: int = 10) -> list[dict]:
    """Return top-cited papers with extended metadata including impact proxy.

    Impact factor proxy = citation_velocity (citations/year) for the paper,
    which approximates journal-level IF when averaged across venue papers.

    Returns list of dicts:
        [{"title": str, "citations": int, "year": int|None,
          "authors": list[str], "venue": str|None, "doi": str|None,
          "citation_velocity": float, "impact_factor_proxy": float}]
    """
    eligible = [p for p in papers if p.citations is not None]
    eligible.sort(key=lambda p: p.citations or 0, reverse=True)

    results = []
    for p in eligible[:top_n]:
        velocity = compute_citation_velocity(p)
        results.append({
            "title": p.title or "",
            "citations": p.citations or 0,
            "year": p.year,
            "authors": p.get_authors()[:5],  # first 5 authors
            "venue": p.venue,
            "doi": p.doi,
            "citation_velocity": round(velocity, 2),
            "impact_factor_proxy": round(velocity, 2),
        })
    return results


def compute_venue_impact(papers: list[Paper], top_n: int = 10) -> list[dict]:
    """Estimate venue-level impact factor from average citation velocity.

    Returns list of dicts:
        [{"venue": str, "paper_count": int, "avg_citation_velocity": float,
          "total_citations": int}]
    """
    venue_data: defaultdict[str, list[float]] = defaultdict(list)
    venue_cites: defaultdict[str, int] = defaultdict(int)

    for p in papers:
        if not p.venue or p.citations is None:
            continue
        velocity = compute_citation_velocity(p)
        venue_data[p.venue].append(velocity)
        venue_cites[p.venue] += p.citations

    results = []
    for venue, velocities in venue_data.items():
        results.append({
            "venue": venue,
            "paper_count": len(velocities),
            "avg_citation_velocity": round(statistics.mean(velocities), 2),
            "total_citations": venue_cites[venue],
        })

    results.sort(key=lambda x: x["avg_citation_velocity"], reverse=True)
    return results[:top_n]


def compute_citation_stats(papers: list[Paper]) -> dict:
    """Compute all citation metrics."""
    cites_list = [p.citations for p in papers if p.citations is not None]
    velocities = [compute_citation_velocity(p) for p in papers if p.citations is not None and p.year]

    cumulative = sum(cites_list) if cites_list else 0
    median = statistics.median(cites_list) if cites_list else 0.0
    avg_velocity = statistics.mean(velocities) if velocities else 0.0
    h_index = compute_h_index(papers)

    # Top cited (legacy format for backward compat)
    papers_with_cites = [(p.title, p.citations or 0) for p in papers if p.citations is not None]
    papers_with_cites.sort(key=lambda x: x[1], reverse=True)
    top_cited = papers_with_cites[:10]

    # Extended: most-cited authors + top papers with details + venue impact
    most_cited_authors = compute_most_cited_authors(papers)
    top_cited_details = compute_top_cited_with_details(papers)
    venue_impact = compute_venue_impact(papers)

    return {
        "cumulative_citations": cumulative,
        "median_citations": median,
        "avg_citation_velocity": round(avg_velocity, 2),
        "h_index_estimate": h_index,
        "top_cited_papers": top_cited,
        "most_cited_authors": most_cited_authors,
        "top_cited_details": top_cited_details,
        "venue_impact": venue_impact,
    }
