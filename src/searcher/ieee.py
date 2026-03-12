"""IEEE Xplore API fetcher (requires API key)."""

from __future__ import annotations

import json
import os
from datetime import datetime

import httpx

from src.searcher.base import AbstractFetcher
from src.storage.models import Paper


class IEEEFetcher(AbstractFetcher):
    name = "ieee"
    BASE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

    def __init__(self, timeout: int = 30):
        self._timeout = timeout
        self._api_key = os.environ.get("IEEE_API_KEY", "")

    async def search(
        self,
        query: str,
        max_results: int = 200,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[dict]:
        if not self._api_key:
            return []

        params: dict = {
            "querytext": query,
            "max_records": min(max_results, 200),
            "apikey": self._api_key,
        }
        if year_start:
            params["start_year"] = year_start
        if year_end:
            params["end_year"] = year_end

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await self._request_with_retry(client, "GET", self.BASE_URL, params=params)
            if resp.status_code == 429:
                return []
            resp.raise_for_status()
            data = resp.json()
        return data.get("articles", [])

    def normalise(self, raw: dict) -> Paper:
        doi = raw.get("doi")
        title = raw.get("title", "")
        year_str = raw.get("publication_year", "")
        year = int(year_str) if year_str else None
        paper_id = Paper.make_id(doi=doi, title=title, year=year)

        authors = []
        for a in raw.get("authors", {}).get("authors", []):
            name = a.get("full_name", "")
            if name:
                authors.append(name)

        return Paper(
            id=paper_id,
            doi=doi,
            arxiv_id=None,
            pmid=None,
            title=title,
            authors=json.dumps(authors),
            year=year,
            venue=raw.get("publication_title"),
            venue_type="conference" if "conference" in raw.get("content_type", "").lower() else "journal",
            abstract=raw.get("abstract"),
            keywords=json.dumps(raw.get("index_terms", {}).get("author_terms", {}).get("terms", [])),
            citations=raw.get("citing_paper_count"),
            citation_velocity=None,
            influential_citations=None,
            sources=json.dumps(["ieee"]),
            url=raw.get("html_url"),
            fetched_at=datetime.utcnow(),
            is_local=False,
        )
