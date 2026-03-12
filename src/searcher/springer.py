"""Springer Nature API fetcher (requires API key)."""

from __future__ import annotations

import json
import os
from datetime import datetime

import httpx

from src.searcher.base import AbstractFetcher
from src.storage.models import Paper


class SpringerFetcher(AbstractFetcher):
    name = "springer"
    BASE_URL = "https://api.springernature.com/meta/v2/json"

    def __init__(self, timeout: int = 30):
        self._timeout = timeout
        self._api_key = os.environ.get("SPRINGER_API_KEY", "")

    async def search(
        self,
        query: str,
        max_results: int = 200,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[dict]:
        if not self._api_key:
            return []

        q = f'keyword:"{query}"'
        if year_start and year_end:
            q += f" onlinedatefrom:{year_start}-01-01 onlinedateto:{year_end}-12-31"
        elif year_start:
            q += f" onlinedatefrom:{year_start}-01-01"
        elif year_end:
            q += f" onlinedateto:{year_end}-12-31"

        params = {
            "q": q,
            "s": 1,
            "p": min(max_results, 100),
            "api_key": self._api_key,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await self._request_with_retry(client, "GET", self.BASE_URL, params=params)
            if resp.status_code == 429:
                return []
            resp.raise_for_status()
            data = resp.json()
        return data.get("records", [])

    def normalise(self, raw: dict) -> Paper:
        doi = raw.get("doi")
        title = raw.get("title", "")
        year = None
        pub_date = raw.get("publicationDate", "")
        if pub_date:
            try:
                year = int(pub_date[:4])
            except (ValueError, IndexError):
                pass

        paper_id = Paper.make_id(doi=doi, title=title, year=year)
        authors = [c.get("creator", "") for c in raw.get("creators", []) if c.get("creator")]
        abstract = raw.get("abstract", "")

        return Paper(
            id=paper_id,
            doi=doi,
            arxiv_id=None,
            pmid=None,
            title=title,
            authors=json.dumps(authors),
            year=year,
            venue=raw.get("publicationName"),
            venue_type="journal",
            abstract=abstract if abstract else None,
            keywords=json.dumps([]),
            citations=None,
            citation_velocity=None,
            influential_citations=None,
            sources=json.dumps(["springer"]),
            url=raw.get("url", [{}])[0].get("value") if isinstance(raw.get("url"), list) else None,
            fetched_at=datetime.utcnow(),
            is_local=False,
        )
