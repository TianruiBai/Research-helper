"""Search orchestrator — runs fetchers in parallel and deduplicates results.

Academic fetchers iterate year-by-year within the user's date range to ensure
balanced coverage (APIs tend to bias toward popular/recent papers).
News fetchers query the full range at once.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from typing import AsyncIterator, Callable

from rapidfuzz import fuzz

from src.searcher.base import AbstractFetcher
from src.searcher.arxiv import ArxivFetcher
from src.searcher.semantic_scholar import SemanticScholarFetcher
from src.searcher.openalex import OpenAlexFetcher
from src.searcher.pubmed import PubMedFetcher
from src.searcher.crossref import CrossrefFetcher
from src.searcher.news_google import GoogleNewsFetcher
from src.searcher.news_bing import BingNewsFetcher
from src.storage.models import Paper

logger = logging.getLogger(__name__)

FETCHER_MAP: dict[str, type[AbstractFetcher]] = {
    "arxiv": ArxivFetcher,
    "semantic_scholar": SemanticScholarFetcher,
    "openalex": OpenAlexFetcher,
    "pubmed": PubMedFetcher,
    "crossref": CrossrefFetcher,
    "google_news": GoogleNewsFetcher,
    "bing_news": BingNewsFetcher,
}

# News/web fetchers don't benefit from per-year iteration
NEWS_FETCHERS = {"google_news", "bing_news"}


class SearchOrchestrator:
    def __init__(
        self,
        sources: list[str] | None = None,
        max_results_per_source: int = 200,
        timeout: int = 30,
        title_similarity_threshold: float = 0.92,
    ):
        self._source_names = sources or list(FETCHER_MAP.keys())
        self._max_results = max_results_per_source
        self._timeout = timeout
        self._sim_threshold = title_similarity_threshold

    async def search(
        self,
        query: str,
        year_start: int | None = None,
        year_end: int | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> list[Paper]:
        """Run all fetchers concurrently, deduplicate, return merged papers.

        Academic fetchers iterate year-by-year so every year in the user's
        range is represented.  News fetchers query the full range once.
        """
        fetchers = self._build_fetchers()
        all_papers: list[Paper] = []

        tasks = []
        for fetcher in fetchers:
            is_news = fetcher.name in NEWS_FETCHERS
            if is_news or year_start is None or year_end is None:
                # Single-shot fetch (news or no year range given)
                tasks.append(
                    self._fetch_one(fetcher, query, year_start, year_end, progress_callback)
                )
            else:
                # Per-year academic fetch
                tasks.append(
                    self._fetch_per_year(fetcher, query, year_start, year_end, progress_callback)
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for fetcher, result in zip(fetchers, results):
            if isinstance(result, Exception):
                logger.warning("Fetcher %s failed: %s", fetcher.name, result)
                if progress_callback:
                    progress_callback(f"{fetcher.name}: error", 0)
            else:
                all_papers.extend(result)
                if progress_callback:
                    progress_callback(f"{fetcher.name}: done", len(result))

        # Deduplicate
        deduped = self._deduplicate(all_papers)
        logger.info(
            "Orchestrator: %d raw → %d deduped", len(all_papers), len(deduped)
        )
        return deduped

    def _build_fetchers(self) -> list[AbstractFetcher]:
        fetchers: list[AbstractFetcher] = []
        for name in self._source_names:
            cls = FETCHER_MAP.get(name)
            if cls:
                fetchers.append(cls(timeout=self._timeout))
        return fetchers

    async def _fetch_one(
        self,
        fetcher: AbstractFetcher,
        query: str,
        year_start: int | None,
        year_end: int | None,
        progress_callback: Callable[[str, int], None] | None,
    ) -> list[Paper]:
        if progress_callback:
            progress_callback(f"{fetcher.name}: searching...", 0)
        return await fetcher.fetch_and_normalise(
            query, self._max_results, year_start, year_end
        )

    async def _fetch_per_year(
        self,
        fetcher: AbstractFetcher,
        query: str,
        year_start: int,
        year_end: int,
        progress_callback: Callable[[str, int], None] | None,
    ) -> list[Paper]:
        """Fetch papers year-by-year so every year is represented.

        Allocates max_results evenly across years.  Each single-year fetch
        runs sequentially per fetcher to avoid rate-limiting, but different
        fetchers are parallelised at the caller level.
        """
        num_years = year_end - year_start + 1
        per_year = max(self._max_results // num_years, 10)

        if progress_callback:
            progress_callback(f"{fetcher.name}: searching {num_years} years...", 0)

        all_papers: list[Paper] = []
        for idx, year in enumerate(range(year_start, year_end + 1)):
            # Rate-limit: pause between per-year requests (PubMed allows ~3 req/s
            # without an API key; other APIs have similar limits)
            if idx > 0:
                await asyncio.sleep(1.5)
            try:
                batch = await fetcher.fetch_and_normalise(
                    query, per_year, year, year
                )
                all_papers.extend(batch)
                if progress_callback:
                    progress_callback(
                        f"{fetcher.name}: {year} ({len(batch)} papers)", len(batch)
                    )
            except Exception as e:
                logger.warning("Fetcher %s year %d failed: %s", fetcher.name, year, e)

        logger.info(
            "Fetcher %s per-year: %d papers across %d–%d",
            fetcher.name, len(all_papers), year_start, year_end,
        )
        return all_papers

    def _deduplicate(self, papers: list[Paper]) -> list[Paper]:
        """Remove duplicates using DOI match then fuzzy title matching."""
        seen_dois: dict[str, Paper] = {}
        seen_arxiv: dict[str, Paper] = {}
        unique: list[Paper] = []

        for paper in papers:
            # 1. Exact DOI match
            if paper.doi:
                doi_key = paper.doi.lower().strip()
                if doi_key in seen_dois:
                    self._merge_paper(seen_dois[doi_key], paper)
                    continue
                seen_dois[doi_key] = paper

            # 2. arXiv ID match
            if paper.arxiv_id:
                aid = paper.arxiv_id.strip()
                if aid in seen_arxiv:
                    self._merge_paper(seen_arxiv[aid], paper)
                    continue
                seen_arxiv[aid] = paper

            # 3. Fuzzy title + year match
            is_dup = False
            for existing in unique:
                if paper.year and existing.year and paper.year != existing.year:
                    continue
                ratio = fuzz.ratio(
                    paper.title.lower().strip(),
                    existing.title.lower().strip(),
                ) / 100.0
                if ratio >= self._sim_threshold:
                    self._merge_paper(existing, paper)
                    is_dup = True
                    break

            if not is_dup:
                unique.append(paper)

        return unique

    @staticmethod
    def _merge_paper(target: Paper, source: Paper) -> None:
        """Merge richer data from source into target."""
        if source.abstract and not target.abstract:
            target.abstract = source.abstract
        if source.doi and not target.doi:
            target.doi = source.doi
        if source.arxiv_id and not target.arxiv_id:
            target.arxiv_id = source.arxiv_id
        if source.pmid and not target.pmid:
            target.pmid = source.pmid
        if source.citations is not None:
            if target.citations is None or source.citations > target.citations:
                target.citations = source.citations
        if source.influential_citations is not None and target.influential_citations is None:
            target.influential_citations = source.influential_citations
        if source.funder_names and not target.funder_names:
            target.funder_names = source.funder_names

        # Merge source lists
        t_src = set(json.loads(target.sources or "[]"))
        s_src = set(json.loads(source.sources or "[]"))
        target.sources = json.dumps(sorted(t_src | s_src))
