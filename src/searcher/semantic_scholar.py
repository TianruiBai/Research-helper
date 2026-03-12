"""Semantic Scholar API fetcher."""

from __future__ import annotations

import json
from datetime import datetime

import httpx

from src.searcher.base import AbstractFetcher
from src.storage.models import Paper


class SemanticScholarFetcher(AbstractFetcher):
    name = "semantic_scholar"
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "paperId,externalIds,title,authors,year,venue,abstract,citationCount,influentialCitationCount,url"

    def __init__(self, timeout: int = 30):
        self._timeout = timeout

    async def search(
        self,
        query: str,
        max_results: int = 200,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[dict]:
        results: list[dict] = []
        offset = 0
        batch_size = min(max_results, 100)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            while len(results) < max_results:
                params: dict = {
                    "query": query,
                    "offset": offset,
                    "limit": batch_size,
                    "fields": self.FIELDS,
                }
                if year_start and year_end:
                    params["year"] = f"{year_start}-{year_end}"
                elif year_start:
                    params["year"] = f"{year_start}-"
                elif year_end:
                    params["year"] = f"-{year_end}"

                resp = await self._request_with_retry(client, "GET", self.BASE_URL, params=params)
                if resp.status_code == 429:
                    break  # retries exhausted
                resp.raise_for_status()
                data = resp.json()
                batch = data.get("data", [])
                if not batch:
                    break
                results.extend(batch)
                offset += len(batch)
                if offset >= data.get("total", 0):
                    break

        return results[:max_results]

    def normalise(self, raw: dict) -> Paper:
        ext_ids = raw.get("externalIds") or {}
        doi = ext_ids.get("DOI")
        arxiv_id = ext_ids.get("ArXiv")
        pmid = ext_ids.get("PubMed")

        paper_id = Paper.make_id(doi=doi, title=raw.get("title", ""), year=raw.get("year"))

        authors = [a.get("name", "") for a in raw.get("authors", []) if a.get("name")]

        return Paper(
            id=paper_id,
            doi=doi,
            arxiv_id=arxiv_id,
            pmid=pmid,
            title=raw.get("title", ""),
            authors=json.dumps(authors),
            year=raw.get("year"),
            venue=raw.get("venue") or None,
            venue_type=None,
            abstract=raw.get("abstract"),
            keywords=json.dumps([]),
            citations=raw.get("citationCount"),
            citation_velocity=None,
            influential_citations=raw.get("influentialCitationCount"),
            sources=json.dumps(["semantic_scholar"]),
            url=raw.get("url"),
            fetched_at=datetime.utcnow(),
            is_local=False,
        )
