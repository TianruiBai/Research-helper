"""FastAPI application — main entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.analytics.pipeline import AnalyticsPipeline
from src.config.hardware import HardwareInfo, detect_hardware
from src.config.settings import get_settings
from src.llm.client import LLMClient
from src.llm.model_registry import find_best_available
from src.storage.library_store import LibraryStore
from src.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

# Ensure our application logs are visible at INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
# LLM client: DEBUG for raw-response diagnostics
logging.getLogger("src.llm.client").setLevel(logging.DEBUG)

# Global singletons set during startup
store: SQLiteStore | None = None
library_store: LibraryStore | None = None
pipeline: AnalyticsPipeline | None = None
llm_client: LLMClient | None = None
field_context_client: LLMClient | None = None
hardware_info: HardwareInfo | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global store, library_store, pipeline, llm_client, field_context_client, hardware_info

    settings = get_settings()

    # Initialise storage
    store = SQLiteStore(settings.storage.db_path)
    library_store = LibraryStore(settings.storage.library_db_path)
    logger.info("Storage initialised: %s, %s", settings.storage.db_path, settings.storage.library_db_path)

    # Detect hardware capabilities
    hardware_info = detect_hardware()
    logger.info("Hardware: %s", hardware_info)

    # Decide whether to attempt LLM setup
    auto_detect = settings.llm.auto_detect_hardware
    skip_llm = auto_detect and not hardware_info.llm_capable

    if skip_llm:
        logger.info(
            "Hardware insufficient for LLM (%s) — running heuristic-only mode",
            hardware_info.reason,
        )
        llm_client = None
    else:
        # Initialise LLM client + detect availability
        llm_client = LLMClient(
            model=settings.llm.default_model,
            base_url=settings.llm.ollama_base_url,
            timeout=settings.llm.timeout_seconds,
            web_search=settings.llm.web_search,
        )
        llm_available = await llm_client.health_check()
        if llm_available:
            model_available = await llm_client.is_model_available()
            if not model_available:
                # Try fallback models
                installed = await llm_client.list_models()
                best = await find_best_available(installed)
                if best:
                    llm_client.model = best
                    logger.info("Using fallback LLM model: %s", best)
                else:
                    logger.warning("No suitable LLM model found, running in heuristic mode")
                    llm_client = None
            else:
                logger.info("LLM available: %s @ %s", llm_client.model, llm_client.base_url)
        else:
            logger.info("LLM server not running — using heuristic fallback")
            llm_client = None

    # Initialise secondary (smaller) model client for field-context analysis
    if not skip_llm:
        fc_client = LLMClient(
            model=settings.llm.field_context_model,
            base_url=settings.llm.field_context_base_url,
            timeout=settings.llm.field_context_timeout_seconds,
            web_search=settings.llm.field_context_web_search,
        )
        fc_available = await fc_client.health_check()
        if fc_available:
            field_context_client = fc_client
            logger.info(
                "Field-context LLM available: %s @ %s (web_search=%s)",
                fc_client.model, fc_client.base_url, fc_client.web_search,
            )
        else:
            logger.info("Field-context server not running — will use primary model")
            field_context_client = None

    # Initialise pipeline
    pipeline = AnalyticsPipeline(
        llm_client=llm_client,
        field_context_client=field_context_client,
        use_parallel=settings.llm.max_concurrent_llm_calls > 1,
    )
    await pipeline.check_llm()

    yield  # app runs

    logger.info("Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Research Field Intelligence Tool",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from src.api.routes.search import router as search_router
    from src.api.routes.analyze import router as analyze_router
    from src.api.routes.library import router as library_router
    from src.api.routes.status import router as status_router
    from src.api.routes.proposal import router as proposal_router

    app.include_router(search_router, prefix="/api/v1")
    app.include_router(analyze_router, prefix="/api/v1")
    app.include_router(library_router, prefix="/api/v1")
    app.include_router(status_router, prefix="/api/v1")
    app.include_router(proposal_router, prefix="/api/v1")

    return app


app = create_app()
