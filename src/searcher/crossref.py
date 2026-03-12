"""Crossref API fetcher."""

from __future__ import annotations

import json
from datetime import datetime

import httpx

from src.searcher.base import AbstractFetcher
from src.storage.models import Paper


class CrossrefFetcher(AbstractFetcher):
    name = "crossref"
    BASE_URL = "https://api.crossref.org/works"

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
        rows = min(max_results, 100)
        offset = 0

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            while len(results) < max_results:
                params: dict = {
                    "query": query,
                    "rows": rows,
                    "offset": offset,
                    "sort": "relevance",
                    "order": "desc",
                }
                filt_parts: list[str] = []
                if year_start:
                    filt_parts.append(f"from-pub-date:{year_start}")
                if year_end:
                    filt_parts.append(f"until-pub-date:{year_end}")
                if filt_parts:
                    params["filter"] = ",".join(filt_parts)

                resp = await self._request_with_retry(
                    client, "GET", self.BASE_URL,
                    params=params,
                    headers={"User-Agent": "ResearchFieldIntelTool/1.0 (mailto:user@example.com)"},
                )
                if resp.status_code == 429:
                    break  # retries exhausted
                resp.raise_for_status()
                data = resp.json()
                items = data.get("message", {}).get("items", [])
                if not items:
                    break
                results.extend(items)
                offset += len(items)

        return results[:max_results]

    def normalise(self, raw: dict) -> Paper:
        doi = raw.get("DOI")
        title_parts = raw.get("title", [])
        title = title_parts[0] if title_parts else ""
        year = None
        date_parts = raw.get("published-print", raw.get("published-online", {})).get("date-parts", [[]])
        if date_parts and date_parts[0]:
            try:
                year = int(date_parts[0][0])
            except (ValueError, TypeError, IndexError):
                pass

        paper_id = Paper.make_id(doi=doi, title=title, year=year)

        authors: list[str] = []
        for a in raw.get("author", []):
            family = a.get("family", "")
            given = a.get("given", "")
            if family:
                authors.append(f"{family}, {given}".strip(", "))

        # Abstract (Crossref provides it for some)
        abstract = raw.get("abstract", "")
        if abstract:
            # Strip basic XML/HTML tags
            import re
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()

        venue = None
        container = raw.get("container-title", [])
        if container:
            venue = container[0]

        # Funder information
        funders: list[str] = []
        for f in raw.get("funder", []):
            name = f.get("name", "")
            if name:
                funders.append(name)

        return Paper(
            id=paper_id,
            doi=doi,
            arxiv_id=None,
            pmid=None,
            title=title,
            authors=json.dumps(authors),
            year=year,
            venue=venue,
            venue_type="journal" if raw.get("type") == "journal-article" else raw.get("type"),
            abstract=abstract if abstract else None,
            keywords=json.dumps([]),
            citations=raw.get("is-referenced-by-count"),
            citation_velocity=None,
            influential_citations=None,
            sources=json.dumps(["crossref"]),
            url=raw.get("URL"),
            fetched_at=datetime.utcnow(),
            is_local=False,
            funder_names=json.dumps(funders) if funders else None,
        )
