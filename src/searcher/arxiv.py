"""arXiv API fetcher — uses Atom XML feed."""

from __future__ import annotations

import json
import re
from datetime import datetime

import feedparser
import httpx

from src.searcher.base import AbstractFetcher
from src.storage.models import Paper


class ArxivFetcher(AbstractFetcher):
    name = "arxiv"
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, timeout: int = 30):
        self._timeout = timeout

    # ------------------------------------------------------------------
    # arXiv API query format (per https://info.arxiv.org/help/api/user-manual.html):
    #   - Field prefix `all:` searches title + abstract + authors + categories
    #   - Multi-word phrases must be double-quoted: all:"machine learning"
    #   - Boolean operators MUST be uppercase: AND, OR, ANDNOT
    #   - Phrase-quoted ti:/abs: is overly restrictive → use `all:` instead
    # ------------------------------------------------------------------

    def _build_query(self, query: str) -> str:
        """Convert a free-text or comma-separated query into an arXiv search string.

        Strategy:
          * Comma/semicolon separators → independent keyword groups joined with OR
          * Multi-word groups are phrase-quoted: all:"deep learning"
          * Single words left unquoted: all:transformer
          * Falls back to the raw query in `all:` when nothing else makes sense
        """
        groups = [g.strip() for g in re.split(r"[,;]+", query) if g.strip()]
        if not groups:
            return f"all:{query}"

        parts: list[str] = []
        for g in groups:
            words = g.split()
            if len(words) == 1:
                parts.append(f"all:{words[0]}")
            else:
                # Quoted phrase search within `all:` field
                safe = g.replace('"', "")   # strip any stray quotes
                parts.append(f'all:"{safe}"')

        # Multiple groups → any one of them must match (recall-oriented)
        return " OR ".join(parts)

    async def search(
        self,
        query: str,
        max_results: int = 200,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[dict]:
        search_query = self._build_query(query)
        headers = {"User-Agent": "ResearchFieldIntelTool/1.0 (contact: admin@localhost)"}

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(max_results, 2000),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        async with httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=True, headers=headers
        ) as client:
            resp = await self._request_with_retry(client, "GET", self.BASE_URL, params=params)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)

            # Zero-result fallback: try a plain `all:` search without field qualifiers
            if not feed.entries and search_query != f"all:{query}":
                fallback_params = {**params, "search_query": f"all:{query}"}
                resp2 = await self._request_with_retry(client, "GET", self.BASE_URL, params=fallback_params)
                resp2.raise_for_status()
                feed = feedparser.parse(resp2.text)

        results: list[dict] = []
        for entry in feed.entries:
            year = None
            if hasattr(entry, "published"):
                try:
                    year = int(entry.published[:4])
                except (ValueError, TypeError):
                    pass
            if year_start and year and year < year_start:
                continue
            if year_end and year and year > year_end:
                continue
            results.append({
                "title": entry.get("title", "").replace("\n", " ").strip(),
                "authors": [a.get("name", "") for a in entry.get("authors", [])],
                "year": year,
                "abstract": entry.get("summary", "").replace("\n", " ").strip(),
                "url": entry.get("link", ""),
                "arxiv_id": entry.get("id", "").split("/abs/")[-1] if "/abs/" in entry.get("id", "") else entry.get("id", ""),
                "categories": [t.get("term", "") for t in entry.get("tags", [])],
                "published": entry.get("published", ""),
            })
        return results

    def normalise(self, raw: dict) -> Paper:
        paper_id = Paper.make_id(title=raw.get("title", ""), year=raw.get("year"))
        return Paper(
            id=paper_id,
            doi=None,
            arxiv_id=raw.get("arxiv_id"),
            pmid=None,
            title=raw.get("title", ""),
            authors=json.dumps(raw.get("authors", [])),
            year=raw.get("year"),
            venue="arXiv",
            venue_type="preprint",
            abstract=raw.get("abstract"),
            keywords=json.dumps(raw.get("categories", [])),
            citations=None,
            citation_velocity=None,
            influential_citations=None,
            sources=json.dumps(["arxiv"]),
            url=raw.get("url"),
            fetched_at=datetime.utcnow(),
            is_local=False,
        )
