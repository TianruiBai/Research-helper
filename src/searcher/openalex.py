"""OpenAlex API fetcher."""

from __future__ import annotations

import json
from datetime import datetime

import httpx

from src.searcher.base import AbstractFetcher
from src.storage.models import Paper


class OpenAlexFetcher(AbstractFetcher):
    name = "openalex"
    BASE_URL = "https://api.openalex.org/works"

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
        per_page = min(max_results, 200)
        page = 1

        # Build a richer filter: title.search gives higher weight to title matches
        # while the `search` param covers abstract + keywords too.
        import re as _re
        # Split comma/semicolon separated keywords and also search via title.search
        raw_terms = [t.strip() for t in _re.split(r"[,;]+", query) if t.strip()]
        title_search = raw_terms[0] if raw_terms else query  # use first keyword for title.search

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            while len(results) < max_results:
                params: dict = {
                    "search": query,
                    "per_page": per_page,
                    "page": page,
                    "sort": "relevance_score:desc",
                }
                filt_parts: list[str] = []
                if year_start and year_end:
                    filt_parts.append(f"publication_year:{year_start}-{year_end}")
                elif year_start:
                    filt_parts.append(f"publication_year:>{year_start - 1}")
                elif year_end:
                    filt_parts.append(f"publication_year:<{year_end + 1}")
                if filt_parts:
                    params["filter"] = ",".join(filt_parts)

                resp = await self._request_with_retry(client, "GET", self.BASE_URL, params=params)
                if resp.status_code == 429:
                    break  # retries exhausted
                resp.raise_for_status()
                data = resp.json()
                batch = data.get("results", [])
                if not batch:
                    break
                results.extend(batch)
                page += 1

        return results[:max_results]

    def normalise(self, raw: dict) -> Paper:
        doi = raw.get("doi", "")
        if doi and doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]

        title = raw.get("title") or raw.get("display_name") or ""
        year = raw.get("publication_year")
        paper_id = Paper.make_id(doi=doi or None, title=title, year=year)

        # Authors + affiliations
        authors: list[str] = []
        institutions: list[str] = []
        for authorship in raw.get("authorships", []):
            author_name = authorship.get("author", {}).get("display_name", "")
            if author_name:
                authors.append(author_name)
            for inst in authorship.get("institutions", []):
                iname = inst.get("display_name", "")
                if iname:
                    institutions.append(iname)

        # Venue
        primary_location = raw.get("primary_location") or {}
        source = primary_location.get("source") or {}
        venue = source.get("display_name")
        venue_type = source.get("type")  # e.g. "journal", "repository"

        # Abstract — OpenAlex provides inverted index
        abstract = self._reconstruct_abstract(raw.get("abstract_inverted_index"))

        keywords = [
            kw.get("display_name", "")
            for kw in raw.get("keywords", [])
            if kw.get("display_name")
        ]

        return Paper(
            id=paper_id,
            doi=doi or None,
            arxiv_id=None,
            pmid=None,
            title=title,
            authors=json.dumps(authors),
            year=year,
            venue=venue,
            venue_type=venue_type,
            abstract=abstract,
            keywords=json.dumps(keywords),
            citations=raw.get("cited_by_count"),
            citation_velocity=None,
            influential_citations=None,
            sources=json.dumps(["openalex"]),
            url=raw.get("id"),  # OpenAlex URL
            fetched_at=datetime.utcnow(),
            is_local=False,
        )

    @staticmethod
    def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
        """OpenAlex stores abstracts as inverted index {word: [positions]}."""
        if not inverted_index:
            return None
        word_positions: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_positions)
