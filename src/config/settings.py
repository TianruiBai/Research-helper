"""Application settings — loaded from .env + config.yaml."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

_ROOT = Path(__file__).resolve().parent.parent.parent  # project root


def _load_yaml() -> dict:
    cfg_path = _ROOT / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml = _load_yaml()


class SearchSettings(BaseSettings):
    max_results_per_source: int = 200
    min_citations: int = 0
    year_start: int = 2015
    year_end: int = 2026
    default_sources: list[str] = [
        "arxiv", "semantic_scholar", "openalex", "pubmed", "crossref",
    ]
    timeout_seconds: int = 30
    retry_max: int = 3


class DedupSettings(BaseSettings):
    title_similarity_threshold: float = 0.92


class LLMSettings(BaseSettings):
    default_model: str = "crow-9b-opus"
    ollama_base_url: str = "http://localhost:8080"  # llama-server default; Ollama uses 11434
    max_concurrent_llm_calls: int = 4
    abstract_sample_size: int = 500
    timeout_seconds: int = 180
    temperature_classification: float = 0.3
    temperature_narrative: float = 0.7
    auto_detect_hardware: bool = True
    web_search: bool = True
    # Field-context analysis reuses the primary model; these are kept for
    # backward-compat if a user runs a second server on 8081
    field_context_model: str = "crow-9b-opus"
    field_context_base_url: str = "http://localhost:8081"
    field_context_timeout_seconds: int = 180
    field_context_web_search: bool = True
    field_context_ngl: int = -1


class StorageSettings(BaseSettings):
    db_path: str = "papers.db"
    library_db_path: str = "local_library.db"
    cache_dir: str = ".cache"


class APISettings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8000


class UISettings(BaseSettings):
    port: int = 8501


class Settings(BaseSettings):
    """Master settings — merges .env and config.yaml values."""

    # API keys (from .env only)
    ieee_api_key: str = ""
    springer_api_key: str = ""

    # Sub-settings
    search: SearchSettings = Field(default_factory=SearchSettings)
    dedup: DedupSettings = Field(default_factory=DedupSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    api: APISettings = Field(default_factory=APISettings)
    ui: UISettings = Field(default_factory=UISettings)

    model_config = {"env_file": str(_ROOT / ".env"), "env_file_encoding": "utf-8"}


def _merge_yaml(settings: Settings) -> Settings:
    """Override defaults with values from config.yaml if present."""
    y = _yaml
    if "search" in y:
        for k, v in y["search"].items():
            if hasattr(settings.search, k):
                object.__setattr__(settings.search, k, v)
    if "dedup" in y:
        for k, v in y["dedup"].items():
            if hasattr(settings.dedup, k):
                object.__setattr__(settings.dedup, k, v)
    if "llm" in y:
        for k, v in y["llm"].items():
            if hasattr(settings.llm, k):
                object.__setattr__(settings.llm, k, v)
    if "storage" in y:
        for k, v in y["storage"].items():
            if hasattr(settings.storage, k):
                object.__setattr__(settings.storage, k, v)
    if "api" in y:
        for k, v in y["api"].items():
            if hasattr(settings.api, k):
                object.__setattr__(settings.api, k, v)
    if "ui" in y:
        for k, v in y["ui"].items():
            if hasattr(settings.ui, k):
                object.__setattr__(settings.ui, k, v)
    return settings


def get_settings() -> Settings:
    s = Settings()
    return _merge_yaml(s)
