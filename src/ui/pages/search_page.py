"""Page 1: Search — run a new academic database search."""

from __future__ import annotations

from datetime import datetime

import re

import pandas as pd
import streamlit as st

from src.ui.api_client import APIClient
from src.ui.components.score_card import render_score_cards, render_sentiment_details
from src.ui.components.trend_chart import (
    citation_distribution_chart,
    export_chart_buttons,
    papers_per_year_chart,
)
from src.ui.components.venue_table import (
    render_themes,
    render_top_authors,
    render_top_venues,
)
from src.reports.charts import (
    sentiment_by_source_chart,
    sentiment_by_year_chart,
)

ALL_SOURCES = [
    "arxiv",
    "semantic_scholar",
    "openalex",
    "pubmed",
    "crossref",
    "ieee",
    "springer",
]

ALL_WEB_SOURCES = [
    "google_news",
    "bing_news",
]


def _render_export_buttons(stats: dict, papers: list[dict]) -> None:
    """Render CSV / JSON / HTML download buttons from search/analysis results."""
    import csv
    import io
    import json as _json

    query_slug = (stats.get("query") or "report")[:40].replace(" ", "_")
    col1, col2, col3 = st.columns(3)

    with col1:
        data = _json.dumps(stats, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            label="⬇️ Stats (JSON)",
            data=data,
            file_name=f"stats_{query_slug}.json",
            mime="application/json",
        )

    with col2:
        if papers:
            buf = io.StringIO()
            fields = ["title", "authors", "year", "venue", "citations", "doi", "url"]
            writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for p in papers:
                row = {k: p.get(k, "") for k in fields}
                if isinstance(row.get("authors"), list):
                    row["authors"] = "; ".join(str(a) for a in row["authors"])
                writer.writerow(row)
            st.download_button(
                label="⬇️ Papers (CSV)",
                data=buf.getvalue().encode("utf-8"),
                file_name=f"papers_{query_slug}.csv",
                mime="text/csv",
            )
        else:
            st.caption("No papers list available.")

    with col3:
        try:
            from src.reports.html_exporter import export_html
            html_bytes = export_html(stats, papers).encode("utf-8")
            st.download_button(
                label="⬇️ Report (HTML)",
                data=html_bytes,
                file_name=f"report_{query_slug}.html",
                mime="text/html",
            )
        except Exception as _e:
            st.caption(f"HTML export unavailable: {_e}")


def render(client: APIClient) -> None:
    st.header("🔍 Search Academic Databases")

    # Initialise session-state defaults once so values persist across page switches
    if "sq_query" not in st.session_state:
        st.session_state["sq_query"] = ""
    if "sq_year_start" not in st.session_state:
        st.session_state["sq_year_start"] = 2015
    if "sq_year_end" not in st.session_state:
        st.session_state["sq_year_end"] = datetime.now().year
    if "sq_max_results" not in st.session_state:
        st.session_state["sq_max_results"] = 200
    if "sq_sources" not in st.session_state:
        st.session_state["sq_sources"] = ["arxiv", "semantic_scholar", "openalex", "pubmed", "crossref"]
    if "sq_web_sources" not in st.session_state:
        st.session_state["sq_web_sources"] = ALL_WEB_SOURCES

    with st.form("search_form"):
        query = st.text_input(
            "Research query",
            placeholder="e.g. transformer architecture, neural networks; attention mechanism",
            help="Separate multiple keywords with commas or semicolons.",
            key="sq_query",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            year_start = st.number_input("From year", 2000, datetime.now().year, key="sq_year_start")
        with col2:
            year_end = st.number_input("To year", 2000, datetime.now().year + 1, key="sq_year_end")
        with col3:
            max_results = st.number_input("Max per source", 50, 1000, step=50, key="sq_max_results")

        sources = st.multiselect(
            "Academic Sources",
            ALL_SOURCES,
            key="sq_sources",
        )

        web_sources = st.multiselect(
            "News / Web Sources",
            ALL_WEB_SOURCES,
            key="sq_web_sources",
            help="Include public news and web articles for sentiment & comprehensive scoring.",
        )

        submitted = st.form_submit_button("🚀 Search & Analyse", use_container_width=True)

    if submitted and query:
        # Normalize keywords: split on , or ; and rejoin with spaces
        keywords = [kw.strip() for kw in re.split(r"[;,]", query) if kw.strip()]
        normalized_query = " ".join(keywords)

        progress_box = st.empty()
        log_box = st.empty()
        llm_header = st.empty()
        llm_box = st.empty()
        logs: list[str] = []
        llm_text = ""
        in_llm_task = False
        result: dict | None = None

        try:
            progress_box.info("Starting search and analysis...")
            for event in client.stream_search(
                query=normalized_query,
                year_start=year_start,
                year_end=year_end,
                max_results=max_results,
                sources=sources,
                web_sources=web_sources,
            ):
                event_type = (event.get("type") or "").lower()
                if event_type == "progress":
                    msg = str(event.get("message", "Working..."))
                    cnt = int(event.get("count", 0) or 0)
                    line = f"- {msg} ({cnt})" if cnt > 0 else f"- {msg}"
                    logs.append(line)
                    logs = logs[-24:]
                    progress_box.info(f"In progress: {msg}")
                    log_box.markdown("\n".join(logs))
                    if msg.startswith("LLM:"):
                        in_llm_task = True
                        llm_text = ""
                        llm_header.caption(f"🤖 {msg}")
                        llm_box.empty()
                    else:
                        if in_llm_task:
                            in_llm_task = False
                            llm_header.empty()
                            llm_box.empty()
                elif event_type == "token":
                    llm_text += event.get("text", "")
                    llm_box.code(llm_text[-3000:], language="json")
                elif event_type == "result":
                    result = event.get("data")
                elif event_type == "error":
                    raise RuntimeError(str(event.get("message", "Unknown error")))
                elif event_type == "done":
                    break

            if not result:
                raise RuntimeError("No result returned from streaming search")

            st.session_state["last_search"] = result
            # Pre-fill dashboard query from search so switching pages works
            st.session_state["dashboard_query"] = normalized_query
            progress_box.empty()
            log_box.empty()
            llm_header.empty()
            llm_box.empty()
            st.success(
                f"Found {len(result['papers'])} papers "
                f"(session: {result['session_id']})"
            )
        except Exception as e:
            progress_box.empty()
            log_box.empty()
            llm_header.empty()
            llm_box.empty()
            st.error(f"Search failed: {e}")
            return

    # Display results if available
    result = st.session_state.get("last_search")
    if result is None:
        st.info("Enter a query above and click Search to begin.")
        return

    stats = result["stats"]
    papers_list = result.get("papers", [])

    # ── Export ──
    with st.expander("📥 Export Results", expanded=False):
        _render_export_buttons(stats, papers_list)

    # Scores
    render_score_cards(stats)
    st.divider()

    # Sentiment details + charts
    if stats.get("sentiment_positive_ratio") is not None:
        render_sentiment_details(stats)

        by_source = stats.get("sentiment_by_source")
        by_year   = stats.get("sentiment_by_year")

        if by_source and (by_source.get("academic") or by_source.get("news")):
            fig_src = sentiment_by_source_chart(
                by_source.get("academic", {}),
                by_source.get("news", {}),
            )
            st.plotly_chart(fig_src, use_container_width=True)
            with st.expander("Export chart", expanded=False):
                export_chart_buttons(fig_src, "sentiment_by_source")

        if by_year:
            fig_yr = sentiment_by_year_chart(by_year)
            st.plotly_chart(fig_yr, use_container_width=True)
            with st.expander("Export chart", expanded=False):
                export_chart_buttons(fig_yr, "sentiment_by_year")

        st.divider()

    # Trend chart + export
    ppy = stats.get("papers_per_year", {})
    if ppy:
        fig_ppy = papers_per_year_chart(ppy, stats.get("query", ""))
        st.plotly_chart(fig_ppy, use_container_width=True)
        with st.expander("Export chart", expanded=False):
            export_chart_buttons(fig_ppy, "publications_per_year")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Papers", stats.get("total_papers", 0))
    col2.metric("H-index (est.)", stats.get("h_index_estimate", 0))
    col3.metric("Growth Rate", f"{stats.get('growth_rate_pct', 0):.1f}%")
    col4.metric("Industry Ratio", f"{stats.get('industry_ratio', 0):.1%}")

    # Maturity & narrative
    maturity = stats.get("maturity_label")
    narrative = stats.get("field_narrative")
    open_questions = stats.get("open_questions", [])
    if maturity:
        label_colour = {"Emerging": "🟡", "Growing": "🟢", "Established": "🔵", "Saturating": "🔴"}.get(maturity, "⚪")
        st.markdown(f"**Field Maturity:** {label_colour} {maturity}")
    if narrative:
        with st.expander("📖 Field Narrative", expanded=False):
            st.markdown(narrative)
    if open_questions:
        with st.expander("❓ Open Research Questions", expanded=False):
            for q in open_questions:
                st.markdown(f"- {q}")

    # Themes
    render_themes(stats.get("top_themes"))
    st.divider()

    # Venues and authors
    col_v, col_a = st.columns(2)
    with col_v:
        render_top_venues(stats.get("top_venues", []))
    with col_a:
        render_top_authors(stats.get("top_authors", []))

    # Top cited + export
    top_cited = stats.get("top_cited_papers", [])
    if top_cited:
        fig_tc = citation_distribution_chart(top_cited)
        st.plotly_chart(fig_tc, use_container_width=True)
        with st.expander("Export chart", expanded=False):
            export_chart_buttons(fig_tc, "top_cited_papers")

    # Papers table
    st.subheader("📄 Papers")
    papers = result.get("papers", [])
    if papers:
        df = pd.DataFrame(papers)
        display_cols = ["title", "authors", "year", "venue", "citations", "doi"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True, hide_index=True)

        # CSV download
        csv_data = df[available].to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv_data,
            file_name=f"papers_{stats.get('query', 'export')}.csv",
            mime="text/csv",
        )
