from __future__ import annotations

import os

import streamlit as st
from openai import OpenAI

from generation import (
    BaselineReport,
    generate_grounded_report,
    generate_unsearched_baseline,
    verify_baseline,
)
from models import Report
from pdf_export import render_pdf
from research import Verification, search_sources, verify_source


st.set_page_config(page_title="Grounded Article Studio", page_icon="G", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Source+Serif+4:wght@400;500;600;700&display=swap');
    :root { --ink: #1b2928; --muted: #657371; --paper: #f4f6f2; --teal: #176b65; --line: #d7dfdb; }
    html, body, [class*="css"] { font-family: 'Source Serif 4', Georgia, serif; color: var(--ink); }
    .stApp { background: radial-gradient(ellipse at 82% 0%, #e5eee8 0%, transparent 34%), var(--paper); }
    [data-testid="stSidebar"] { background: #e9efeb; border-right: 1px solid var(--line); }
    h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; letter-spacing: 0 !important; }
    h1 { font-size: 2.35rem !important; line-height: 1.12 !important; }
    .eyebrow { font: 500 0.72rem 'DM Mono', monospace; letter-spacing: 0; text-transform: uppercase; color: var(--teal); }
    .lede { color: var(--muted); font-size: 1.05rem; max-width: 760px; }
    div.stButton > button[kind="primary"] { background: var(--teal); border: 1px solid var(--teal); border-radius: 4px; min-height: 2.8rem; }
    div.stButton > button[kind="primary"]:hover { background: #10534e; border-color: #10534e; }
    div[data-testid="stDownloadButton"] button { border-radius: 4px; }
    div[data-testid="stTextInput"] input { border-radius: 4px; }
    .source-note { border-top: 1px solid var(--line); padding-top: 0.75rem; margin-top: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _secrets_key() -> str:
    configured = os.getenv("OPENAI_API_KEY", "")
    if configured:
        return configured
    try:
        return str(st.secrets.get("OPENAI_API_KEY", ""))
    except (FileNotFoundError, AttributeError):
        return ""


def _show_paragraphs(paragraphs, references) -> None:
    sources = {source.id: source for source in references}
    for paragraph in paragraphs:
        citations = []
        for source_id in paragraph.source_ids:
            source = sources[source_id]
            first = source.authors[0].split(",", 1)[0].split()[-1]
            if len(source.authors) == 1:
                label = first
            elif len(source.authors) == 2:
                second = source.authors[1].split(",", 1)[0].split()[-1]
                label = f"{first} & {second}"
            else:
                label = f"{first} et al."
            citations.append(f"{label}, {source.year}")
        suffix = f" ({'; '.join(citations)})" if citations else ""
        st.write(paragraph.text + suffix)


def _show_report(report: Report, checks: list[Verification]) -> None:
    st.header(report.title)
    st.caption(f"{len(report.references)} references · APA-style citations · source checks complete")
    st.subheader("Abstract")
    _show_paragraphs(report.abstract, report.references)
    st.subheader("Introduction")
    _show_paragraphs(report.introduction, report.references)
    for section in report.sections:
        st.subheader(section.heading)
        _show_paragraphs(section.paragraphs, report.references)
    st.subheader("Conclusion")
    _show_paragraphs(report.conclusion, report.references)
    st.subheader("References")
    for source in report.references:
        author_text = ", ".join(source.authors)
        link = f"https://doi.org/{source.doi}" if source.doi else str(source.url)
        citation = f"{author_text} ({source.year}). {source.title}."
        if source.container_title:
            citation += f" *{source.container_title}.*"
        citation += f" [DOI or source]({link})"
        st.markdown(citation)
    with st.expander("Reference verification"):
        for source, check in zip(report.references, checks):
            status = "Verified" if check.genuine else "Needs review"
            st.markdown(f"**{status}** · {source.title}")
            st.caption(check.detail)
            st.caption(
                f"Author: {'match' if check.author_matches else 'mismatch'} · "
                f"Title: {'match' if check.title_matches else 'mismatch'} · "
                f"Year: {'match' if check.year_matches else 'mismatch'} · "
                f"DOI/URL: {'match' if check.locator_matches else 'mismatch'}"
            )


def _show_baseline(baseline: BaselineReport, checks: list[Verification]) -> None:
    genuine = sum(check.genuine for check in checks)
    incorrect = len(checks) - genuine
    first, second, third = st.columns(3)
    first.metric("References listed", len(checks))
    second.metric("Genuine and matched", genuine)
    third.metric("Incorrect or unverified", incorrect)
    st.caption("Generated without search. Each listed reference was searched afterward and checked against source metadata.")
    st.subheader(baseline.title)
    st.markdown("**Abstract**")
    st.write(baseline.abstract)
    st.markdown("**Introduction**")
    for paragraph in baseline.introduction:
        st.write(paragraph)
    for section in baseline.sections:
        st.markdown(f"**{section.heading}**")
        for paragraph in section.paragraphs:
            st.write(paragraph)
    st.markdown("**Conclusion**")
    st.write(baseline.conclusion)
    st.markdown("**References**")
    with st.expander("Baseline reference checks", expanded=True):
        for reference, check in zip(baseline.references, checks):
            status = "Genuine" if check.genuine else "Incorrect / unverified"
            citation = f"{', '.join(reference.authors)} ({reference.year}). {reference.title}."
            if reference.url:
                citation += f" [{reference.url}]({reference.url})"
            st.markdown(f"**{status}** · {citation}")
            if reference.doi:
                st.caption(f"DOI: {reference.doi}")
            elif reference.url:
                st.caption(f"URL: {reference.url}")
            st.caption(check.detail)


def _generate(topic: str, api_key: str, model: str, compare: bool) -> None:
    client = OpenAI(api_key=api_key)
    with st.status("Searching and checking source records…", expanded=True) as status:
        sources = search_sources(topic)
        if not sources:
            status.update(label="No verifiable sources found", state="error")
            st.error("Search results did not expose enough source metadata. Try a narrower or more academic topic.")
            return
        st.write(f"Found {len(sources)} source records with verifiable bibliographic metadata.")
        report = generate_grounded_report(client, topic, sources, model)
        checks = [verify_source(source) for source in report.references]
        if not all(check.genuine for check in checks):
            status.update(label="Reference verification needs attention", state="error")
            st.session_state.pop("generated", None)
            st.error("At least one cited reference failed its publisher-page check. No PDF was created.")
            st.session_state["verification_failure"] = list(zip(report.references, checks))
            return

        pdf = render_pdf(report)
        baseline = None
        baseline_checks: list[Verification] = []
        if compare:
            status.update(label="Checking the no-search comparison…", state="running")
            baseline = generate_unsearched_baseline(client, topic, model)
            baseline_checks = verify_baseline(baseline)

        st.session_state["generated"] = {
            "topic": topic,
            "report": report,
            "checks": checks,
            "pdf": pdf,
            "baseline": baseline,
            "baseline_checks": baseline_checks,
        }
        st.session_state.pop("verification_failure", None)
        status.update(label="Article and checks complete", state="complete")


st.sidebar.markdown('<div class="eyebrow">Configuration</div>', unsafe_allow_html=True)
api_key = st.sidebar.text_input("OpenAI API key", value=_secrets_key(), type="password")
model = st.sidebar.text_input("Model", value="gpt-4o-mini")
st.sidebar.caption("The key is used only for this session. Search and reference checks use public web and Crossref endpoints.")

st.markdown('<div class="eyebrow">Research writing · sourced and checked</div>', unsafe_allow_html=True)
st.title("Grounded Article Studio")
st.markdown(
    '<p class="lede">Turn a topic into a concise technical article, with references drawn from live search results and checked against their source records.</p>',
    unsafe_allow_html=True,
)

with st.form("article_form"):
    topic = st.text_input("Article topic", placeholder="e.g. Heat pumps in cold-climate buildings")
    compare = st.checkbox("Compare against a draft generated without web search", value=True)
    submitted = st.form_submit_button("Generate article", type="primary", use_container_width=True)

if submitted:
    if not topic.strip():
        st.error("Enter a topic to continue.")
    elif not api_key.strip():
        st.error("Add an OpenAI API key in the sidebar or set OPENAI_API_KEY.")
    else:
        try:
            _generate(topic.strip(), api_key.strip(), model.strip(), compare)
        except Exception as error:
            st.error(f"Generation stopped: {error}")

if "verification_failure" in st.session_state:
    st.warning("A fresh publisher-page check did not confirm every field. The article is withheld from PDF export.")
    for source, check in st.session_state["verification_failure"]:
        st.write(f"{source.title}: {check.detail}")

if "generated" in st.session_state:
    result = st.session_state["generated"]
    article_tab, comparison_tab = st.tabs(("Article", "No-search comparison"))
    with article_tab:
        left, right = st.columns([5, 1])
        with right:
            st.download_button(
                "Download PDF",
                data=result["pdf"],
                file_name="article.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        _show_report(result["report"], result["checks"])
    with comparison_tab:
        if result["baseline"] is None:
            st.info("The comparison was not requested for this article.")
        else:
            _show_baseline(result["baseline"], result["baseline_checks"])