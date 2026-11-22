import asyncio
import time
import streamlit as st
from pipeline import run_research_pipeline

st.set_page_config(
    page_title="ResearchMind (MCP)",
    layout="wide",
)

if "results" not in st.session_state:
    st.session_state.results = {}
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""

# Header
st.title("ResearchMind — MCP Edition")
st.caption(
    "Multi-agent AI research assistant powered by 4 MCP servers: "
    "Tavily + Wikipedia → Extract / Playwright → Write → Critique → Filesystem"
)
st.divider()

# Input area
topic = st.text_input(
    "Research Topic",
    placeholder="e.g. Rajasthan Royals IPL 2027 retention list and title chances",
    key="topic_input",
)

use_playwright = st.checkbox(
    "Enable Playwright MCP (for JS-heavy pages)",
    value=False,
    help="Launches a headless Chromium browser via Playwright MCP as a fallback for dynamic sites. Slower and uses more tokens.",
)

run_clicked = st.button("Run Research Pipeline", type="primary")

# Run pipeline
if run_clicked:
    query = st.session_state.topic_input.strip()
    if not query:
        st.warning("Please enter a research topic first.")
    else:
        try:
            with st.status("Running MCP research pipeline...", expanded=True) as status:
                st.session_state.results = asyncio.run(
                    run_research_pipeline(
                        topic=query,
                        use_playwright=use_playwright,
                        on_step=st.write,
                    )
                )
                status.update(label="Research complete!", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

# Display results
r = st.session_state.results

if r:
    st.divider()

    if r.get("saved_path"):
        st.success(f"Saved via Filesystem MCP: `{r['saved_path']}`")

    col_report, col_critic = st.columns([2, 1])

    with col_report:
        st.subheader("Final Research Report")
        st.markdown(r["report"])

        st.download_button(
            label="Download Report (.md)",
            data=r["report"],
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown",
        )

    with col_critic:
        st.subheader("Critic Feedback")
        with st.container(border=True):
            st.markdown(r["feedback"])

    st.divider()
    with st.expander("View Raw Search Results"):
        st.text(r["search_results"])

    with st.expander("View Extracted / Scraped Content"):
        st.text(r["scraped_content"])