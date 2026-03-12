"""Data models — Paper, FieldStats, ProposalAnalysis.

Used both as plain dataclasses (in-memory) and as SQLAlchemy ORM models (storage).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


# ---------------------------------------------------------------------------
# SQLAlchemy base
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------
class Paper(Base):
    __tablename__ = "papers"

    id = Column(String, primary_key=True)
    doi = Column(String, nullable=True, index=True)
    arxiv_id = Column(String, nullable=True, index=True)
    pmid = Column(String, nullable=True, index=True)

    title = Column(Text, nullable=False)
    authors = Column(Text, default="[]")  # JSON list
    year = Column(Integer, nullable=True, index=True)
    venue = Column(String, nullable=True)
    venue_type = Column(String, nullable=True)

    abstract = Column(Text, nullable=True)
    keywords = Column(Text, default="[]")  # JSON list

    citations = Column(Integer, nullable=True)
    citation_velocity = Column(Float, nullable=True)
    influential_citations = Column(Integer, nullable=True)

    sources = Column(Text, default="[]")  # JSON list
    url = Column(String, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    is_local = Column(Boolean, default=False)
    file_path = Column(String, nullable=True)

    # LLM-derived (populated after analytics)
    themes = Column(Text, nullable=True)  # JSON list
    motivation_sentences = Column(Text, nullable=True)  # JSON list
    confidence_label = Column(String, nullable=True)
    industry_affiliated = Column(Boolean, nullable=True)
    funder_names = Column(Text, nullable=True)  # JSON list

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def make_id(doi: str | None = None, title: str = "", year: int | None = None) -> str:
        if doi:
            raw = doi.lower().strip()
        else:
            raw = f"{title.lower().strip()}|{year or ''}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def get_authors(self) -> list[str]:
        return json.loads(self.authors) if self.authors else []

    def set_authors(self, val: list[str]) -> None:
        self.authors = json.dumps(val)

    def get_keywords(self) -> list[str]:
        return json.loads(self.keywords) if self.keywords else []

    def set_keywords(self, val: list[str]) -> None:
        self.keywords = json.dumps(val)

    def get_sources(self) -> list[str]:
        return json.loads(self.sources) if self.sources else []

    def set_sources(self, val: list[str]) -> None:
        self.sources = json.dumps(val)

    def get_themes(self) -> list[str]:
        return json.loads(self.themes) if self.themes else []

    def set_themes(self, val: list[str]) -> None:
        self.themes = json.dumps(val)

    def get_motivation_sentences(self) -> list[str]:
        return json.loads(self.motivation_sentences) if self.motivation_sentences else []

    def set_motivation_sentences(self, val: list[str]) -> None:
        self.motivation_sentences = json.dumps(val)

    def get_funder_names(self) -> list[str]:
        return json.loads(self.funder_names) if self.funder_names else []

    def set_funder_names(self, val: list[str]) -> None:
        self.funder_names = json.dumps(val)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "pmid": self.pmid,
            "title": self.title,
            "authors": self.get_authors(),
            "year": self.year,
            "venue": self.venue,
            "venue_type": self.venue_type,
            "abstract": self.abstract,
            "keywords": self.get_keywords(),
            "citations": self.citations,
            "citation_velocity": self.citation_velocity,
            "influential_citations": self.influential_citations,
            "sources": self.get_sources(),
            "url": self.url,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "is_local": self.is_local,
            "themes": self.get_themes(),
            "motivation_sentences": self.get_motivation_sentences(),
            "confidence_label": self.confidence_label,
            "industry_affiliated": self.industry_affiliated,
            "funder_names": self.get_funder_names(),
        }


# ---------------------------------------------------------------------------
# SearchSession
# ---------------------------------------------------------------------------
class SearchSession(Base):
    __tablename__ = "search_sessions"

    id = Column(String, primary_key=True)
    query = Column(String, nullable=False)
    year_start = Column(Integer, nullable=True)
    year_end = Column(Integer, nullable=True)
    sources_used = Column(Text, default="[]")  # JSON list
    total_papers = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    stats_json = Column(Text, nullable=True)  # serialised FieldStats


# ---------------------------------------------------------------------------
# ProposalAnalysis
# ---------------------------------------------------------------------------
class ProposalAnalysis(Base):
    __tablename__ = "proposal_analyses"

    id = Column(String, primary_key=True)
    proposal_text = Column(Text, nullable=False)
    run_at = Column(DateTime, default=datetime.utcnow)
    novelty_score = Column(Float, nullable=True)
    top_overlapping_papers = Column(Text, nullable=True)  # JSON
    gap_clusters = Column(Text, nullable=True)  # JSON list
    recommended_citations = Column(Text, nullable=True)  # JSON list
    llm_narrative = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# FieldStats — a plain dataclass (not persisted as its own table;
# serialised to JSON in SearchSession.stats_json)
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field as dataclass_field


@dataclass
class FieldStats:
    query: str = ""
    year_range: tuple[int, int] = (0, 0)

    # Volume
    total_papers: int = 0
    review_papers: int = 0
    papers_per_year: dict[int, int] = dataclass_field(default_factory=dict)
    growth_rate_pct: float = 0.0
    cagr_pct: float = 0.0

    # Citations
    cumulative_citations: int = 0
    avg_citation_velocity: float = 0.0
    median_citations: float = 0.0
    h_index_estimate: int = 0
    top_cited_papers: list[tuple[str, int]] = dataclass_field(default_factory=list)
    most_cited_authors: list[dict] = dataclass_field(default_factory=list)
    top_cited_details: list[dict] = dataclass_field(default_factory=list)
    venue_impact: list[dict] = dataclass_field(default_factory=list)

    # Structure
    top_venues: list[tuple[str, int]] = dataclass_field(default_factory=list)
    top_authors: list[tuple[str, int]] = dataclass_field(default_factory=list)
    country_distribution: dict[str, int] = dataclass_field(default_factory=dict)
    industry_ratio: float = 0.0

    # Dimension scores
    interest_score: float = 0.0
    motivation_score: float = 0.0
    confidence_score: float = 0.0
    market_score: float = 0.0
    public_sentiment_score: float = 50.0
    comprehensive_score: float = 0.0

    # Sentiment details
    news_article_count: int = 0
    sentiment_positive_ratio: float = 0.0
    sentiment_negative_ratio: float = 0.0
    sentiment_positive_samples: list[dict] | None = None
    sentiment_negative_samples: list[dict] | None = None
    sentiment_neutral_ratio: float = 0.0
    sentiment_by_year: dict | None = None   # {year_str: {positive_count, negative_count, neutral_count, *_ratio}}
    sentiment_by_source: dict | None = None  # {"academic": {...}, "news": {...}}

    # LLM outputs
    top_themes: list[str] | None = None
    top_funders: list[tuple[str, int]] | None = None
    field_narrative: str | None = None
    maturity_label: str | None = None

    # Field-awareness
    field_category: str | None = None
    field_display_name: str | None = None
    field_pace: str | None = None

    # Field-context deep analysis (LLM)
    motivation_depth: str | None = None
    confidence_assessment: str | None = None
    market_reality: str | None = None
    velocity_context: str | None = None
    gaps_and_opportunities: list[str] | None = None
    field_specific_risks: list[str] | None = None
    recommended_focus_areas: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "year_range": list(self.year_range),
            "total_papers": self.total_papers,
            "review_papers": self.review_papers,
            "papers_per_year": self.papers_per_year,
            "growth_rate_pct": self.growth_rate_pct,
            "cagr_pct": self.cagr_pct,
            "cumulative_citations": self.cumulative_citations,
            "avg_citation_velocity": self.avg_citation_velocity,
            "median_citations": self.median_citations,
            "h_index_estimate": self.h_index_estimate,
            "top_cited_papers": self.top_cited_papers,
            "most_cited_authors": self.most_cited_authors,
            "top_cited_details": self.top_cited_details,
            "venue_impact": self.venue_impact,
            "top_venues": self.top_venues,
            "top_authors": self.top_authors,
            "country_distribution": self.country_distribution,
            "industry_ratio": self.industry_ratio,
            "interest_score": self.interest_score,
            "motivation_score": self.motivation_score,
            "confidence_score": self.confidence_score,
            "market_score": self.market_score,
            "public_sentiment_score": self.public_sentiment_score,
            "comprehensive_score": self.comprehensive_score,
            "news_article_count": self.news_article_count,
            "sentiment_positive_ratio": self.sentiment_positive_ratio,
            "sentiment_negative_ratio": self.sentiment_negative_ratio,
            "sentiment_positive_samples": self.sentiment_positive_samples,
            "sentiment_negative_samples": self.sentiment_negative_samples,
            "sentiment_neutral_ratio": self.sentiment_neutral_ratio,
            "sentiment_by_year": self.sentiment_by_year,
            "sentiment_by_source": self.sentiment_by_source,
            "top_themes": self.top_themes,
            "top_funders": self.top_funders,
            "field_narrative": self.field_narrative,
            "maturity_label": self.maturity_label,
            "field_category": self.field_category,
            "field_display_name": self.field_display_name,
            "field_pace": self.field_pace,
            "motivation_depth": self.motivation_depth,
            "confidence_assessment": self.confidence_assessment,
            "market_reality": self.market_reality,
            "velocity_context": self.velocity_context,
            "gaps_and_opportunities": self.gaps_and_opportunities,
            "field_specific_risks": self.field_specific_risks,
            "recommended_focus_areas": self.recommended_focus_areas,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldStats":
        yr = data.get("year_range", [0, 0])
        data["year_range"] = tuple(yr) if isinstance(yr, list) else yr
        # Convert list-of-lists back to list-of-tuples
        for key in ("top_cited_papers", "top_venues", "top_authors", "top_funders"):
            if key in data and data[key] is not None:
                data[key] = [tuple(item) for item in data[key]]
        # Strip unknown keys that may linger from schema changes
        import dataclasses as _dc
        valid = {f.name for f in _dc.fields(cls)}
        data = {k: v for k, v in data.items() if k in valid}
        return cls(**data)


# ---------------------------------------------------------------------------
# Engine helper
# ---------------------------------------------------------------------------
def init_db(db_path: str) -> sessionmaker[Session]:
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
