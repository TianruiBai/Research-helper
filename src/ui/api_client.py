"""Streamlit API client — calls the FastAPI backend."""

from __future__ import annotations

import json

import httpx


class APIClient:
    """Synchronous wrapper around the FastAPI backend."""

    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url.rstrip("/")
        self._timeout = 300.0  # 5 min for long searches

    # -- helpers ------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        with httpx.Client(timeout=self._timeout) as c:
            r = c.get(f"{self.base_url}{path}", params=params)
            r.raise_for_status()
            return r.json()

    def _post(self, path: str, json_body: dict) -> dict:
        with httpx.Client(timeout=self._timeout) as c:
            r = c.post(f"{self.base_url}{path}", json=json_body)
            r.raise_for_status()
            return r.json()

    def _post_file(self, path: str, filename: str, content: bytes) -> dict:
        with httpx.Client(timeout=self._timeout) as c:
            r = c.post(
                f"{self.base_url}{path}",
                files={"file": (filename, content)},
            )
            r.raise_for_status()
            return r.json()

    def _delete(self, path: str) -> dict:
        with httpx.Client(timeout=self._timeout) as c:
            r = c.delete(f"{self.base_url}{path}")
            r.raise_for_status()
            return r.json()

    # -- endpoints ----------------------------------------------------------

    def get_status(self) -> dict:
        return self._get("/status")

    def search(
        self,
        query: str,
        year_start: int = 2015,
        year_end: int = 2026,
        max_results: int = 200,
        sources: list[str] | None = None,
        web_sources: list[str] | None = None,
    ) -> dict:
        body: dict = {
            "query": query,
            "year_start": year_start,
            "year_end": year_end,
            "max_results_per_source": max_results,
        }
        if sources:
            body["sources"] = sources
        if web_sources is not None:
            body["web_sources"] = web_sources
        return self._post("/search", body)

    def stream_search(
        self,
        query: str,
        year_start: int = 2015,
        year_end: int = 2026,
        max_results: int = 200,
        sources: list[str] | None = None,
        web_sources: list[str] | None = None,
    ):
        """Stream search progress events (NDJSON) and final result payload."""
        body: dict = {
            "query": query,
            "year_start": year_start,
            "year_end": year_end,
            "max_results_per_source": max_results,
        }
        if sources:
            body["sources"] = sources
        if web_sources is not None:
            body["web_sources"] = web_sources

        with httpx.Client(timeout=None) as c:
            with c.stream(
                "POST",
                f"{self.base_url}/search/stream",
                json=body,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield {"type": "progress", "message": line, "count": 0}

    def analyze(
        self,
        query: str = "",
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> dict:
        body: dict = {"query": query}
        if year_start:
            body["year_start"] = year_start
        if year_end:
            body["year_end"] = year_end
        return self._post("/analyze", body)

    def stream_analyze(
        self,
        query: str = "",
        year_start: int | None = None,
        year_end: int | None = None,
    ):
        """Stream analytics pipeline progress events (NDJSON)."""
        body: dict = {"query": query}
        if year_start:
            body["year_start"] = year_start
        if year_end:
            body["year_end"] = year_end

        with httpx.Client(timeout=None) as c:
            with c.stream(
                "POST",
                f"{self.base_url}/analyze/stream",
                json=body,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield {"type": "progress", "message": line, "count": 0}

    def get_library(self, search: str | None = None) -> list:
        params = {"search": search} if search else None
        return self._get("/library", params=params)

    def upload_to_library(self, filename: str, content: bytes) -> dict:
        return self._post_file("/library/upload", filename, content)

    def delete_from_library(self, paper_id: str) -> dict:
        return self._delete(f"/library/{paper_id}")

    def analyze_proposal(
        self, proposal_text: str, reference_query: str | None = None
    ) -> dict:
        body: dict = {"proposal_text": proposal_text}
        if reference_query:
            body["reference_query"] = reference_query
        return self._post("/proposal", body)
