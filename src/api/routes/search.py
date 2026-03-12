"""POST /api/v1/search — full search + analytics pipeline."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from src.api import schemas
import src.api.main as _main
from src.config.settings import get_settings
from src.searcher.orchestrator import SearchOrchestrator
from src.storage.models import FieldStats

router = APIRouter()
logger = logging.getLogger(__name__)


def _emit_progress(
    progress_callback,
    message: str,
    count: int = 0,
) -> None:
    if progress_callback:
        progress_callback(message, count)


async def _run_search_impl(
    req: schemas.SearchRequest,
    progress_callback=None,
    token_callback=None,
) -> schemas.SearchResponse:
    """Run search + analytics and return the normal SearchResponse payload."""
    store = _main.store
    pipeline = _main.pipeline
    if store is None or pipeline is None:
        raise HTTPException(503, "Service not ready")

    settings = get_settings()

    _emit_progress(progress_callback, "Starting academic search...")
    orchestrator = SearchOrchestrator(
        sources=req.sources,
        max_results_per_source=req.max_results_per_source,
        timeout=settings.search.timeout_seconds,
        title_similarity_threshold=settings.dedup.title_similarity_threshold,
    )
    papers = await orchestrator.search(
        query=req.query,
        year_start=req.year_start,
        year_end=req.year_end,
        progress_callback=progress_callback,
    )

    _emit_progress(progress_callback, "Starting news/web search...")
    web_articles: list = []
    if req.web_sources:
        web_orchestrator = SearchOrchestrator(
            sources=req.web_sources,
            max_results_per_source=min(req.max_results_per_source, 100),
            timeout=settings.search.timeout_seconds,
            title_similarity_threshold=0.85,
        )
        web_articles = await web_orchestrator.search(
            query=req.query,
            year_start=req.year_start,
            year_end=req.year_end,
            progress_callback=progress_callback,
        )
        logger.info("Web search returned %d news/web articles", len(web_articles))

    all_items = papers + web_articles

    if not all_items:
        raise HTTPException(404, "No papers or articles found for this query")

    _emit_progress(progress_callback, "Saving papers to library...", len(papers))
    if papers:
        store.upsert_papers(papers)

    _emit_progress(progress_callback, "Running analytics pipeline...", len(all_items))
    stats = await pipeline.run(
        all_items,
        query=req.query,
        year_start=req.year_start,
        year_end=req.year_end,
        progress_callback=progress_callback,
        token_callback=token_callback,
    )

    _emit_progress(progress_callback, "Saving analysis session...")
    session_id = store.save_session(
        query=req.query,
        year_start=req.year_start,
        year_end=req.year_end,
        sources_used=req.sources + req.web_sources,
        total_papers=len(all_items),
        stats=stats,
    )

    paper_responses = [
        schemas.PaperResponse(
            id=p.id,
            title=p.title,
            authors=p.get_authors(),
            year=p.year,
            venue=p.venue,
            abstract=p.abstract,
            citations=p.citations,
            doi=p.doi,
            url=p.url,
            sources=p.get_sources(),
        )
        for p in all_items
    ]

    stats_dict = stats.to_dict()
    stats_dict["papers_per_year"] = {
        str(k): v for k, v in stats_dict["papers_per_year"].items()
    }
    stats_dict["year_range"] = list(stats_dict["year_range"])

    _emit_progress(progress_callback, "Completed.", len(all_items))
    return schemas.SearchResponse(
        session_id=session_id,
        papers=paper_responses,
        stats=schemas.FieldStatsResponse(**stats_dict),
    )


@router.post("/search", response_model=schemas.SearchResponse)
async def search(req: schemas.SearchRequest):
    """Run a full search across academic databases, then analyse."""
    return await _run_search_impl(req)


@router.post("/search/stream")
async def search_stream(req: schemas.SearchRequest):
    """Run search + analytics and stream progress events as NDJSON."""
    queue: asyncio.Queue[dict] = asyncio.Queue()

    def progress_callback(message: str, count: int = 0) -> None:
        queue.put_nowait({"type": "progress", "message": message, "count": count})

    def token_callback(text: str) -> None:
        queue.put_nowait({"type": "token", "text": text})

    async def producer() -> None:
        try:
            result = await _run_search_impl(req, progress_callback=progress_callback, token_callback=token_callback)
            await queue.put({"type": "result", "data": jsonable_encoder(result)})
        except Exception as e:  # noqa: BLE001
            await queue.put({"type": "error", "message": str(e)})
        finally:
            await queue.put({"type": "done"})

    async def event_stream():
        producer_task = asyncio.create_task(producer())
        try:
            while True:
                event = await queue.get()
                yield json.dumps(event, ensure_ascii=False) + "\n"
                if event.get("type") == "done":
                    break
        finally:
            if not producer_task.done():
                producer_task.cancel()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
