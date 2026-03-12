"""SQLite CRUD for the main papers database."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import or_
from sqlalchemy.orm import Session

from src.storage.models import (
    FieldStats,
    Paper,
    ProposalAnalysis,
    SearchSession,
    init_db,
)


class SQLiteStore:
    def __init__(self, db_path: str = "papers.db"):
        self._session_factory = init_db(db_path)

    def _session(self) -> Session:
        return self._session_factory()

    # -- Paper CRUD ---------------------------------------------------------

    def upsert_papers(self, papers: list[Paper]) -> int:
        """Insert or update papers. Returns count of new papers inserted."""
        new_count = 0
        with self._session() as session:
            for paper in papers:
                existing = session.get(Paper, paper.id)
                if existing is None:
                    session.add(paper)
                    new_count += 1
                else:
                    # Merge: keep richer data
                    if paper.abstract and not existing.abstract:
                        existing.abstract = paper.abstract
                    if paper.citations and (existing.citations is None or paper.citations > existing.citations):
                        existing.citations = paper.citations
                    if paper.citation_velocity and not existing.citation_velocity:
                        existing.citation_velocity = paper.citation_velocity
                    if paper.influential_citations and not existing.influential_citations:
                        existing.influential_citations = paper.influential_citations
                    # Merge sources
                    old_src = set(json.loads(existing.sources or "[]"))
                    new_src = set(json.loads(paper.sources or "[]"))
                    existing.sources = json.dumps(sorted(old_src | new_src))
            session.commit()
        return new_count

    def get_all_papers(self) -> list[Paper]:
        with self._session() as session:
            papers = list(session.query(Paper).all())
            for p in papers:
                session.expunge(p)
            return papers

    def get_papers_by_query(
        self,
        keyword: str | None = None,
        year_start: int | None = None,
        year_end: int | None = None,
        limit: int = 5000,
    ) -> list[Paper]:
        with self._session() as session:
            q = session.query(Paper)
            if keyword and keyword.strip():
                # Split into individual terms; search title + abstract + keywords (OR logic)
                terms = [
                    t.strip()
                    for t in re.split(r"[\s,;]+", keyword.strip())
                    if len(t.strip()) > 1
                ]
                if terms:
                    conditions = []
                    for term in terms:
                        like = f"%{term}%"
                        conditions.append(Paper.title.ilike(like))
                        conditions.append(Paper.abstract.ilike(like))
                        conditions.append(Paper.keywords.ilike(like))
                    q = q.filter(or_(*conditions))
            if year_start:
                q = q.filter(Paper.year >= year_start)
            if year_end:
                q = q.filter(Paper.year <= year_end)
            papers = list(q.limit(limit).all())
            for p in papers:
                session.expunge(p)
            return papers

    def get_paper_count(self) -> int:
        with self._session() as session:
            return session.query(Paper).count()

    def update_paper(self, paper_id: str, updates: dict[str, Any]) -> bool:
        with self._session() as session:
            paper = session.get(Paper, paper_id)
            if paper is None:
                return False
            for k, v in updates.items():
                if hasattr(paper, k):
                    setattr(paper, k, v)
            session.commit()
            return True

    def delete_all_papers(self) -> int:
        with self._session() as session:
            count = session.query(Paper).delete()
            session.commit()
            return count

    # -- SearchSession CRUD -------------------------------------------------

    def save_session(
        self,
        query: str,
        year_start: int,
        year_end: int,
        sources_used: list[str],
        total_papers: int,
        stats: FieldStats | None = None,
    ) -> str:
        session_id = uuid.uuid4().hex[:16]
        with self._session() as session:
            ss = SearchSession(
                id=session_id,
                query=query,
                year_start=year_start,
                year_end=year_end,
                sources_used=json.dumps(sources_used),
                total_papers=total_papers,
                created_at=datetime.utcnow(),
                stats_json=json.dumps(stats.to_dict()) if stats else None,
            )
            session.add(ss)
            session.commit()
        return session_id

    def get_session(self, session_id: str) -> SearchSession | None:
        with self._session() as session:
            return session.get(SearchSession, session_id)

    def get_all_sessions(self) -> list[SearchSession]:
        with self._session() as session:
            return list(
                session.query(SearchSession)
                .order_by(SearchSession.created_at.desc())
                .all()
            )

    # -- ProposalAnalysis CRUD -----------------------------------------------

    def save_proposal_analysis(self, pa: ProposalAnalysis) -> str:
        with self._session() as session:
            session.add(pa)
            session.commit()
            return pa.id
