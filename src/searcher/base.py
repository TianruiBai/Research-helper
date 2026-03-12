"""Abstract base class for all API fetchers."""

from __future__ import annotations

import abc
import asyncio
import logging
from datetime import datetime

import httpx

from src.storage.models import Paper

logger = logging.getLogger(__name__)

# Default retry settings
MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0  # seconds
BACKOFF_MULTIPLIER = 2.0


class AbstractFetcher(abc.ABC):
    """Every source fetcher must implement search() and normalise()."""

    name: str = "base"

    @abc.abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 200,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[dict]:
        """Return raw API results as list of dicts."""
        ...

    @abc.abstractmethod
    def normalise(self, raw: dict) -> Paper:
        """Convert a single raw API result dict into a Paper object."""
        ...

    async def fetch_and_normalise(
        self,
        query: str,
        max_results: int = 200,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[Paper]:
        """Run search + normalise. Convenience wrapper."""
        raw_results = await self.search(query, max_results, year_start, year_end)
        papers: list[Paper] = []
        for raw in raw_results:
            try:
                paper = self.normalise(raw)
                papers.append(paper)
            except Exception:
                continue  # skip malformed records
        return papers

    @staticmethod
    async def _request_with_retry(
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        max_retries: int = MAX_RETRIES,
        initial_backoff: float = INITIAL_BACKOFF,
        **kwargs,
    ) -> httpx.Response:
        """Make an HTTP request with exponential backoff on 429 / 5xx errors.

        Respects the ``Retry-After`` header when present.
        """
        backoff = initial_backoff
        last_resp: httpx.Response | None = None

        for attempt in range(max_retries + 1):
            resp = await client.request(method, url, **kwargs)

            if resp.status_code == 429 or resp.status_code >= 500:
                last_resp = resp
                if attempt == max_retries:
                    break
                # Respect Retry-After header if server provides one
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = backoff
                else:
                    wait = backoff
                logger.info(
                    "Fetcher %s: %d on attempt %d, retrying in %.1fs",
                    url, resp.status_code, attempt + 1, wait,
                )
                await asyncio.sleep(wait)
                backoff *= BACKOFF_MULTIPLIER
                continue

            # Success or non-retryable error
            return resp

        # All retries exhausted — return last response (caller can raise)
        assert last_resp is not None
        return last_resp
