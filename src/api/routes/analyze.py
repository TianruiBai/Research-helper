"""POST /api/v1/analyze — re-run analytics on existing papers."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from src.api import schemas
import src.api.main as _main

router = APIRouter()
logger = logging.getLogger(__name__)


async def _run_analyze_impl(
    req: schemas.AnalyzeRequest,
    progress_callback=None,
    token_callback=None,
) -> schemas.FieldStatsResponse:
    """Run the analytics pipeline and return a FieldStatsResponse."""
    store = _main.store
    pipeline = _main.pipeline
    if store is None or pipeline is None:
        raise HTTPException(503, "Service not ready")

    if progress_callback:
        progress_callback("Loading papers from database...", 0)

    papers = store.get_papers_by_query(req.query)
    if not papers:
        raise HTTPException(404, "No papers found for this query in the database")

    if progress_callback:
        progress_callback(f"Loaded {len(papers)} papers. Starting analytics...", len(papers))

    stats = await pipeline.run(
        papers,
        query=req.query,
        year_start=req.year_start,
        year_end=req.year_end,
        progress_callback=progress_callback,
        token_callback=token_callback,
    )

    stats_dict = stats.to_dict()
    stats_dict["papers_per_year"] = {
        str(k): v for k, v in stats_dict["papers_per_year"].items()
    }
    stats_dict["year_range"] = list(stats_dict["year_range"])
    return schemas.FieldStatsResponse(**stats_dict)


@router.post("/analyze", response_model=schemas.FieldStatsResponse)
async def analyze(req: schemas.AnalyzeRequest):
    """Run the analytics pipeline on papers already in the database."""
    return await _run_analyze_impl(req)


@router.post("/analyze/stream")
async def analyze_stream(req: schemas.AnalyzeRequest):
    """Run the analytics pipeline and stream progress events as NDJSON."""
    queue: asyncio.Queue[dict] = asyncio.Queue()

    def progress_callback(message: str, count: int = 0) -> None:
        queue.put_nowait({"type": "progress", "message": message, "count": count})

    def token_callback(text: str) -> None:
        queue.put_nowait({"type": "token", "text": text})

    async def producer() -> None:
        try:
            result = await _run_analyze_impl(req, progress_callback=progress_callback, token_callback=token_callback)
            await queue.put({"type": "result", "data": jsonable_encoder(result)})
        except HTTPException as e:
            await queue.put({"type": "error", "message": e.detail})
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
