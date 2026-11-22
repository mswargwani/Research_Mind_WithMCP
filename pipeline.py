import asyncio
from contextlib import AsyncExitStack
from typing import Callable
from langchain_mcp_adapters.tools import load_mcp_tools
from agents import (
    build_reader_agent,
    build_search_agent,
    critic_chain,
    load_mcp_client,
    save_report_via_mcp,
    writer_chain,
)

MAX_RESEARCH_CHARS = 6000


async def run_agent(agent, prompt: str, log: Callable[[str], None] = print) -> str:
    """Invoke a LangChain agent asynchronously, log MCP tool calls, and return final text."""
    result = await agent.ainvoke({"messages": [("user", prompt)]})
    messages = result["messages"]

    called_tools = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            called_tools.append(tc["name"])
        if getattr(msg, "type", None) == "tool" and getattr(msg, "status", None) == "error":
            log(f"   ⚠️ Tool error ({msg.name}): {str(msg.content)[:200]}")

    log(f"   🔧 MCP tools invoked: {called_tools if called_tools else 'NONE'}")
    return messages[-1].content


async def run_research_pipeline(
    topic: str,
    use_playwright: bool = False,
    on_step: Callable[[str], None] | None = None,
) -> dict:
    """Execute the 5-step MCP research pipeline (shared by CLI and Streamlit)."""
    log = on_step or (lambda msg: print(f"\n{msg}"))
    state = {}

    research_servers = ["tavily", "wikipedia"]
    if use_playwright:
        research_servers.append("playwright")

    client = load_mcp_client(selected_servers=[*research_servers, "filesystem"])

    # Open persistent MCP sessions for Search & Reader
    async with AsyncExitStack() as stack:
        tavily_session = await stack.enter_async_context(client.session("tavily"))
        tavily_tools = await load_mcp_tools(tavily_session)

        wiki_session = await stack.enter_async_context(client.session("wikipedia"))
        wiki_tools = await load_mcp_tools(wiki_session)

        playwright_tools = []
        if use_playwright:
            pw_session = await stack.enter_async_context(client.session("playwright"))
            playwright_tools = await load_mcp_tools(pw_session)

        # Step 1 - Search Agent (Tavily + Wikipedia MCP)
        log("**1/5 Search Agent (Tavily + Wikipedia MCP):** Gathering web and encyclopedic sources...")
        search_agent = build_search_agent(tavily_tools, wiki_tools)
        state["search_results"] = await run_agent(
            search_agent,
            f"Search the live web for recent, reliable information about: {topic}",
            log=log,
        )

        # Step 2 - Reader Agent (Tavily Extract + optional Playwright MCP)
        reader_label = "Tavily Extract + Playwright MCP" if use_playwright else "Tavily Extract MCP"
        log(f"**2/5 Reader Agent ({reader_label}):** Extracting deeper content from top URLs...")
        reader_agent = build_reader_agent(tavily_tools, playwright_tools)
        state["scraped_content"] = await run_agent(
            reader_agent,
            f"Based on the following search results about '{topic}', "
            f"pick the 1-2 most relevant https:// URLs and extract deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:3000]}",
            log=log,
        )

    # Step 3 - Writer Chain (pure LLM)
    log("**3/5 Writer Chain:** Drafting the structured research report...")
    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )[:MAX_RESEARCH_CHARS]

    state["report"] = await writer_chain.ainvoke({
        "topic": topic,
        "research": research_combined,
    })

    # Step 4 - Critic Chain (pure LLM)
    log("**4/5 Critic Chain:** Reviewing and scoring the report...")
    state["feedback"] = await critic_chain.ainvoke({
        "report": state["report"]
    })

    # Step 5 - Save to disk via Filesystem MCP
    log("**5/5 Filesystem MCP:** Saving report and critic feedback to `reports/`...")
    async with client.session("filesystem") as fs_session:
        fs_tools = await load_mcp_tools(fs_session)
        state["saved_path"] = await save_report_via_mcp(
            topic=topic,
            report=state["report"],
            feedback=state["feedback"],
            fs_tools=fs_tools,
        )

    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ").strip()
    pw_choice = input("Enable Playwright MCP for JS-heavy pages? (y/N): ").strip().lower() == "y"

    results = asyncio.run(run_research_pipeline(topic, use_playwright=pw_choice))

    print("\n" + "=" * 60)
    print("FINAL REPORT\n" + "=" * 60)
    print(results["report"])
    print("\n" + "=" * 60)
    print("CRITIC FEEDBACK\n" + "=" * 60)
    print(results["feedback"])
    print(f"\nSaved via Filesystem MCP to: {results['saved_path']}")