"""Page 2: Dashboard — visual analytics overview."""

from __future__ import annotations

import streamlit as st

from src.ui.api_client import APIClient
from src.ui.components.score_card import render_score_cards, render_sentiment_details
from src.ui.components.trend_chart import (
    citation_distribution_chart,
    export_chart_buttons,
    growth_rate_chart,
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


def _render_export_buttons(stats: dict, papers: list[dict]) -> None:
    """Render CSV / JSON / HTML download buttons for analysis results."""
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
            st.caption("No papers list in session.")

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
    st.header("📊 Analytics Dashboard")

    # Re-analyse from DB
    # Pre-populate from search if dashboard_query not yet set
    if "dashboard_query" not in st.session_state:
        last = st.session_state.get("last_search")
        if last:
            st.session_state["dashboard_query"] = last.get("stats", {}).get("query", "")

    with st.form("analyze_form"):
        query = st.text_input(
            "Query filter (leave blank for all papers)",
            key="dashboard_query",
        )
        submitted = st.form_submit_button("🔄 Re-analyse", use_container_width=True)

    if submitted:
        progress_box = st.empty()
        log_box = st.empty()
        llm_header = st.empty()
        llm_box = st.empty()
        logs: list[str] = []
        llm_text = ""
        in_llm_task = False
        result: dict | None = None

        try:
            progress_box.info("Starting analytics pipeline...")
            for event in client.stream_analyze(query=query):
                event_type = (event.get("type") or "").lower()
                if event_type == "progress":
                    msg = str(event.get("message", "Working..."))
                    cnt = int(event.get("count", 0) or 0)
                    line = f"- {msg} ({cnt} papers)" if cnt > 0 else f"- {msg}"
                    logs.append(line)
                    logs = logs[-30:]
                    progress_box.info(f"⏳ {msg}")
                    log_box.markdown("\n".join(logs))
                    # When a new LLM task starts, reset the LLM output pane
                    if msg.startswith("LLM:"):
                        in_llm_task = True
                        llm_text = ""
                        llm_header.caption(f"🤖 {msg}")
                        llm_box.empty()
                    else:
                        # Non-LLM step — clear LLM pane
                        if in_llm_task:
                            in_llm_task = False
                            llm_header.empty()
                            llm_box.empty()
                elif event_type == "token":
                    llm_text += event.get("text", "")
                    # Show the most recent 3 000 characters so the box stays readable
                    display = llm_text[-3000:]
                    llm_box.code(display, language="json")
                elif event_type == "result":
                    result = event.get("data")
                elif event_type == "error":
                    raise RuntimeError(str(event.get("message", "Unknown error")))
                elif event_type == "done":
                    break

            if not result:
                raise RuntimeError("No result returned from analytics pipeline")

            st.session_state["dashboard_stats"] = result
            progress_box.empty()
            log_box.empty()
            llm_header.empty()
            llm_box.empty()
            st.success("✅ Analytics complete.")
        except Exception as e:
            progress_box.empty()
            log_box.empty()
            llm_header.empty()
            llm_box.empty()
            st.error(f"Analysis failed: {e}")
            return

    # Try last search stats as fallback
    stats = st.session_state.get("dashboard_stats")
    if stats is None:
        last_search = st.session_state.get("last_search")
        if last_search:
            stats = last_search.get("stats")

    if stats is None:
        st.info("Run a search first or click Re-analyse to generate a dashboard.")
        return

    # ── Export ──
    _papers_for_export = (st.session_state.get("last_search") or {}).get("papers", [])
    with st.expander("📥 Export Results", expanded=False):
        _render_export_buttons(stats, _papers_for_export)

    # Score cards
    render_score_cards(stats)
    st.divider()

    # Charts row 1
    ppy = stats.get("papers_per_year", {})
    if ppy:
        col1, col2 = st.columns(2)
        with col1:
            fig_ppy = papers_per_year_chart(ppy, stats.get("query", ""))
            st.plotly_chart(fig_ppy, use_container_width=True)
            export_chart_buttons(fig_ppy, "publications_per_year")
        with col2:
            fig_gr = growth_rate_chart(ppy)
            st.plotly_chart(fig_gr, use_container_width=True)
            export_chart_buttons(fig_gr, "growth_rate")

    # Summary metrics
    st.subheader("Key Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Papers", stats.get("total_papers", 0))
    m2.metric("Review Papers", stats.get("review_papers", 0))
    m3.metric("H-index", stats.get("h_index_estimate", 0))
    m4.metric("CAGR", f"{stats.get('cagr_pct', 0):.1f}%")
    m5.metric("Median Citations", f"{stats.get('median_citations', 0):.0f}")
    m6.metric("Industry Ratio", f"{stats.get('industry_ratio', 0):.1%}")

    st.divider()

    # Themes
    render_themes(stats.get("top_themes"))
    st.divider()

    # Venues & Authors
    col_v, col_a = st.columns(2)
    with col_v:
        render_top_venues(stats.get("top_venues", []))
    with col_a:
        render_top_authors(stats.get("top_authors", []))

    st.divider()

    # Top cited
    top_cited = stats.get("top_cited_papers", [])
    if top_cited:
        fig_tc = citation_distribution_chart(top_cited)
        st.plotly_chart(fig_tc, use_container_width=True)
        export_chart_buttons(fig_tc, "top_cited_papers")

    # ── Most Cited Authors & Detailed Top Papers ────────────────────────
    most_cited_authors = stats.get("most_cited_authors", [])
    top_cited_details = stats.get("top_cited_details", [])
    venue_impact = stats.get("venue_impact", [])

    if most_cited_authors or top_cited_details:
        st.divider()
        st.subheader("🏅 Most Cited Authors & Papers")

        col_auth, col_pap = st.columns(2)

        with col_auth:
            if most_cited_authors:
                st.markdown("**Top Authors by Citation Count**")
                for i, a in enumerate(most_cited_authors[:10], 1):
                    st.markdown(
                        f"{i}. **{a.get('author', 'Unknown')}** — "
                        f"{a.get('total_citations', 0):,} citations, "
                        f"{a.get('paper_count', 0)} papers"
                    )

        with col_pap:
            if top_cited_details:
                st.markdown("**Top Papers with Impact Details**")
                for i, p in enumerate(top_cited_details[:10], 1):
                    title = p.get("title", "Untitled")
                    cites = p.get("citations", 0)
                    year = p.get("year", "")
                    venue = p.get("venue", "")
                    if_proxy = p.get("impact_factor_proxy", 0)
                    line = f"{i}. **{title}** ({year})"
                    if venue:
                        line += f" — _{venue}_"
                    line += f"  \n   Citations: {cites:,}"
                    if if_proxy:
                        line += f" | IF proxy: {if_proxy:.1f}"
                    st.markdown(line)

    if venue_impact:
        st.divider()
        st.subheader("📍 Venue Impact Ranking")
        for i, v in enumerate(venue_impact[:10], 1):
            st.markdown(
                f"{i}. **{v.get('venue', 'Unknown')}** — "
                f"{v.get('paper_count', 0)} papers, "
                f"avg velocity {v.get('avg_citation_velocity', 0):.1f} cit/yr, "
                f"total {v.get('total_citations', 0):,} citations"
            )

    # ── Sentiment Analysis Section ──────────────────────────────────────
    if stats.get("sentiment_positive_ratio") is not None:
        st.divider()
        render_sentiment_details(stats)

        by_source = stats.get("sentiment_by_source")
        by_year   = stats.get("sentiment_by_year")

        if by_source and (by_source.get("academic") or by_source.get("news")):
            st.subheader("📰 Sentiment by Source Type")
            fig_src = sentiment_by_source_chart(
                by_source.get("academic", {}),
                by_source.get("news", {}),
            )
            st.plotly_chart(fig_src, use_container_width=True)
            export_chart_buttons(fig_src, "sentiment_by_source")

        if by_year:
            st.subheader("📅 Sentiment Trend by Year")
            fig_yr = sentiment_by_year_chart(by_year)
            st.plotly_chart(fig_yr, use_container_width=True)
            export_chart_buttons(fig_yr, "sentiment_by_year")

    # Maturity & narrative
    maturity = stats.get("maturity_label")
    narrative = stats.get("field_narrative")
    open_questions = stats.get("open_questions", [])
    if maturity or narrative:
        st.divider()
        st.subheader("🧠 LLM Field Analysis")
        if maturity:
            label_colour = {"Emerging": "🟡", "Growing": "🟢", "Established": "🔵", "Saturating": "🔴"}.get(maturity, "⚪")
            st.markdown(f"**Field Maturity:** {label_colour} {maturity}")
        if narrative:
            with st.expander("📖 Field Narrative", expanded=True):
                st.markdown(narrative)
        if open_questions:
            with st.expander("❓ Open Research Questions", expanded=False):
                for q in open_questions:
                    st.markdown(f"- {q}")

    # ── Field Awareness & Deep Context Analysis ─────────────────────────
    field_cat = stats.get("field_category")
    field_pace = stats.get("field_pace")
    if field_cat:
        st.divider()
        st.subheader("🌐 Field Awareness")
        pace_icon = {"fast": "⚡", "medium": "🔄", "slow": "🐢"}.get(field_pace or "", "❓")
        display = stats.get("field_display_name", field_cat)
        st.markdown(
            f"**Detected Domain:** {display}  \n"
            f"**Innovation Pace:** {pace_icon} {field_pace or 'unknown'}  "
        )

    # Deep field-context analysis (LLM)
    mot_depth = stats.get("motivation_depth")
    conf_assess = stats.get("confidence_assessment")
    mkt_reality = stats.get("market_reality")
    vel_context = stats.get("velocity_context")
    gaps = stats.get("gaps_and_opportunities", [])
    risks = stats.get("field_specific_risks", [])
    focus = stats.get("recommended_focus_areas", [])

    if any([mot_depth, conf_assess, mkt_reality, vel_context]):
        st.divider()
        st.subheader("🔬 Deep Field-Context Analysis")
        if mot_depth:
            with st.expander("💡 Motivation Depth", expanded=True):
                st.markdown(mot_depth)
        if conf_assess:
            with st.expander("🛡️ Confidence Assessment", expanded=True):
                st.markdown(conf_assess)
        if mkt_reality:
            with st.expander("💰 Market Reality", expanded=True):
                st.markdown(mkt_reality)
        if vel_context:
            with st.expander("🚀 Research Velocity Context", expanded=True):
                st.markdown(vel_context)
        if gaps:
            with st.expander("🎯 Gaps & Opportunities", expanded=False):
                for g in gaps:
                    st.markdown(f"- {g}")
        if risks:
            with st.expander("⚠️ Field-Specific Risks", expanded=False):
                for r in risks:
                    st.markdown(f"- {r}")
        if focus:
            with st.expander("🔍 Recommended Focus Areas", expanded=False):
                for f in focus:
                    st.markdown(f"- {f}")

    # Funders
    funders = stats.get("top_funders")
    if funders:
        st.subheader("Top Funders")
        for name, count in funders[:10]:
            st.write(f"- **{name}**: {count} mentions")
