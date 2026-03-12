"""Pydantic request/response schemas for FastAPI."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# -- Requests ---------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    year_start: int = 2015
    year_end: int = 2026
    max_results_per_source: int = 200
    sources: list[str] = Field(
        default=["arxiv", "semantic_scholar", "openalex", "pubmed", "crossref"]
    )
    web_sources: list[str] = Field(
        default=["google_news", "bing_news"],
        description="News/web sources for public sentiment analysis",
    )


class AnalyzeRequest(BaseModel):
    """Re-run analytics on papers already in the database."""
    query: str = ""
    year_start: int | None = None
    year_end: int | None = None


class ProposalRequest(BaseModel):
    """Proposal analysis — text provided directly."""
    proposal_text: str
    reference_query: str | None = None


# -- Responses ---------------------------------------------------------------

class HardwareGPU(BaseModel):
    name: str = ""
    vram_gb: float = 0.0


class HardwareResponse(BaseModel):
    ram_gb: float = 0.0
    gpus: list[HardwareGPU] = []
    os_name: str = ""
    llm_capable: bool = False
    reason: str = ""


class StatusResponse(BaseModel):
    llm_available: bool
    model_name: str | None = None
    models_available: list[str] = []
    paper_count: int = 0
    library_count: int = 0
    hardware: HardwareResponse | None = None


class PaperResponse(BaseModel):
    id: str
    title: str
    authors: list[str] = []
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    citations: int | None = None
    doi: str | None = None
    url: str | None = None
    sources: list[str] = []

    class Config:
        from_attributes = True


class LibraryPaperResponse(BaseModel):
    id: str
    title: str
    authors: list[str] = []
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    citations: int | None = None
    doi: str | None = None
    url: str | None = None
    file_path: str | None = None

    class Config:
        from_attributes = True


class FieldStatsResponse(BaseModel):
    query: str = ""
    year_range: list[int] = [0, 0]
    total_papers: int = 0
    review_papers: int = 0
    papers_per_year: dict[str, int] = {}
    growth_rate_pct: float = 0.0
    cagr_pct: float = 0.0
    cumulative_citations: int = 0
    avg_citation_velocity: float = 0.0
    median_citations: float = 0.0
    h_index_estimate: int = 0
    top_cited_papers: list[list] = []
    most_cited_authors: list[dict] = []
    top_cited_details: list[dict] = []
    venue_impact: list[dict] = []
    top_venues: list[list] = []
    top_authors: list[list] = []
    country_distribution: dict[str, int] = {}
    industry_ratio: float = 0.0
    interest_score: float = 0.0
    motivation_score: float = 0.0
    confidence_score: float = 0.0
    market_score: float = 0.0
    public_sentiment_score: float = 50.0
    comprehensive_score: float = 0.0
    news_article_count: int = 0
    sentiment_positive_ratio: float = 0.0
    sentiment_negative_ratio: float = 0.0
    sentiment_neutral_ratio: float = 0.0
    sentiment_by_year: dict = {}
    sentiment_by_source: dict = {}
    sentiment_positive_samples: list[dict] | None = None
    sentiment_negative_samples: list[dict] | None = None
    top_themes: list[str] | None = None
    top_funders: list[list] | None = None
    field_narrative: str | None = None
    maturity_label: str | None = None
    # Field awareness
    field_category: str | None = None
    field_display_name: str | None = None
    field_pace: str | None = None
    # Deep field-context analysis (LLM)
    motivation_depth: str | None = None
    confidence_assessment: str | None = None
    market_reality: str | None = None
    velocity_context: str | None = None
    gaps_and_opportunities: list[str] | None = None
    field_specific_risks: list[str] | None = None
    recommended_focus_areas: list[str] | None = None


class SearchResponse(BaseModel):
    session_id: str
    papers: list[PaperResponse]
    stats: FieldStatsResponse


class ProposalAnalysisResponse(BaseModel):
    novelty_score: float = 0.0
    overlapping_papers: list[dict] = []
    gap_clusters: list[str] = []
    recommended_citations: list[str] = []
    narrative: str | None = None
