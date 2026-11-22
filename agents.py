import functools
import json
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# 1. MCP Configuration Loader
def load_mcp_client(selected_servers: list[str] | None = None) -> MultiServerMCPClient:
    """Load server definitions from mcp.json, inject secrets/paths, and handle Windows npx."""
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY is missing in your .env file.")

    raw = (BASE_DIR / "mcp.json").read_text(encoding="utf-8")
    raw = raw.replace("${TAVILY_API_KEY}", tavily_key)
    raw = raw.replace("${REPORTS_DIR}", REPORTS_DIR.as_posix())

    servers = json.loads(raw)["mcpServers"]

    if selected_servers is not None:
        servers = {k: v for k, v in servers.items() if k in selected_servers}

    # On Windows, run npx through cmd /c so Python resolves npx.cmd reliably
    if os.name == "nt":
        for cfg in servers.values():
            if cfg.get("command") == "npx":
                cfg["command"] = "cmd"
                cfg["args"] = ["/c", "npx", *cfg.get("args", [])]

    return MultiServerMCPClient(servers)


# 2. Model Setup (Groq active production models)
agent_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=6,
    timeout=60,
)

writer_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    max_retries=6,
    timeout=120,
)


# 3. Helper: Cap Oversized MCP Tool Outputs (prevents Groq 413 TPM errors)
def cap_mcp_tool(tool, max_chars: int = 3000):
    """Wrap an async MCP tool to truncate oversized text responses."""
    orig_coro = getattr(tool, "coroutine", None)
    if orig_coro is None:
        return tool

    def _trim(content):
        if isinstance(content, str) and len(content) > max_chars:
            return content[:max_chars] + "\n...[truncated to fit token limit]"
        if isinstance(content, list):
            text = "\n".join(
                b.get("text", str(b)) if isinstance(b, dict) else str(b)
                for b in content
            )
            if len(text) > max_chars:
                return [
                    {
                        "type": "text",
                        "text": text[:max_chars] + "\n...[truncated to fit token limit]",
                    }
                ]
        return content

    @functools.wraps(orig_coro)
    async def capped_coro(*args, **kwargs):
        result = await orig_coro(*args, **kwargs)
        # langchain-mcp-adapters returns (content, artifact) tuples
        if isinstance(result, tuple) and len(result) == 2:
            content, artifact = result
            return _trim(content), artifact
        return _trim(result)

    return tool.model_copy(update={"coroutine": capped_coro})


# 4. Tool Filtering & Agents
def build_search_agent(tavily_tools: list, wiki_tools: list):
    """Search Agent powered by Tavily web search + lightweight Wikipedia MCP tools."""
    tavily_search = [
        cap_mcp_tool(t, max_chars=3000)
        for t in tavily_tools
        if "search" in t.name.lower()
    ]
    # Keep only search/summary tools to avoid bloated schemas and full-article dumps
    wiki_light = [
        cap_mcp_tool(t, max_chars=2000)
        for t in wiki_tools
        if any(k in t.name.lower() for k in ("search", "summary"))
    ]
    search_tools = tavily_search + wiki_light

    return create_agent(
        model=agent_llm,
        tools=search_tools,
        system_prompt=(
            "You are a live web researcher. You MUST call the Tavily search tool before "
            "answering—never answer from memory. Make 1-2 tool calls max (Tavily search, "
            "plus Wikipedia summary only if background context helps). "
            "Your final response MUST list 3-5 exact https:// URLs returned by the tools, "
            "each with a brief factual snippet."
        ),
    )


def build_reader_agent(tavily_tools: list, playwright_tools: list | None = None):
    """Reader Agent powered by Tavily Extract, plus read-only Playwright tools when enabled."""
    extract_tools = [
        cap_mcp_tool(t, max_chars=3000)
        for t in tavily_tools
        if "extract" in t.name.lower()
    ]

    # Keep only essential read-only navigation/snapshot tools, capped for Groq's TPM limit
    allowed_pw_keywords = ("navigate", "snapshot", "close")
    pw_tools = [
        cap_mcp_tool(t, max_chars=2500)
        for t in (playwright_tools or [])
        if any(k in t.name.lower() for k in allowed_pw_keywords)
    ]

    reader_tools = extract_tools + pw_tools

    instructions = (
        "You are a content extractor. You MUST call Tavily extract on 1-2 exact https:// URLs "
        "from the search results below—never invent content. Extract concrete facts, dates, "
        "stats, and player/entity names, and include the exact source URLs in your response."
    )
    if pw_tools:
        instructions += (
            " Only use Playwright if Tavily extract fails or returns empty content."
        )

    return create_agent(
        model=agent_llm,
        tools=reader_tools,
        system_prompt=instructions,
    )


# 5. Chains (Pure LLM)
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. Write clear, structured, and factual reports "
        "grounded strictly in the provided research. Do not fabricate URLs or facts.",
    ),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points grounded in the research)
- Conclusion
- Sources (list all exact https:// URLs found in the research)

Be detailed, factual, and professional."""),
])

writer_chain = writer_prompt | writer_llm | StrOutputParser()

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | writer_llm | StrOutputParser()


# 6. Direct Filesystem MCP Saver (0 LLM tokens, guaranteed formatting)
async def save_report_via_mcp(
    topic: str,
    report: str,
    feedback: str,
    fs_tools: list,
) -> str:
    """Write the final report and critic review to reports/ using the Filesystem MCP server."""
    write_tool = next((t for t in fs_tools if t.name == "write_file"), None)
    if write_tool is None:
        raise RuntimeError(
            f"Filesystem MCP 'write_file' tool not found. Available: {[t.name for t in fs_tools]}"
        )

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", topic.lower()).strip("_")[:40] or "report"
    file_path = REPORTS_DIR / f"{slug}_{int(time.time())}.md"

    full_markdown = (
        f"# Research Report: {topic}\n\n"
        f"{report}\n\n"
        f"---\n\n"
        f"## Critic Review\n\n"
        f"{feedback}\n"
    )

    await write_tool.ainvoke({
        "path": file_path.as_posix(),
        "content": full_markdown,
    })
    return str(file_path)