"""MCP server for synapse-ai-context — optimized for Claude Code."""

import json
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP, Context

from .synapse_wrapper import SynapseWrapper
from .utils import synapse_error_handler


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    wrapper = SynapseWrapper()
    try:
        yield {"synapse": wrapper}
    finally:
        await wrapper.cleanup()


mcp = FastMCP("synapse-mcp", lifespan=app_lifespan)


def _get_wrapper(ctx: Context) -> SynapseWrapper:
    return ctx.request_context.lifespan_context["synapse"]


def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


# --- Tools ---


@mcp.tool()
@synapse_error_handler
async def synapse_index(
    project_path: str, full: bool = False, ctx: Context = None
) -> str:
    """Index codebase for semantic search. Auto-initializes on first run, generates INTELLIGENCE.md.
    Use after cloning or major code changes. Incremental by default.
    Noise control: configure `.synapseignore` (or env `SYNAPSE_IGNORE_PATTERNS` / `SYNAPSE_EXTRA_EXCLUDE_DIRS`) before indexing.
    If ignore rules changed, run with `full=true` to rebuild a clean index."""
    wrapper = _get_wrapper(ctx)
    await ctx.report_progress(0, 100)
    result = await wrapper.index(project_path, full=full)
    await ctx.report_progress(100, 100)
    return _json(result)


@mcp.tool()
@synapse_error_handler
async def synapse_search(
    query: str,
    project_path: str,
    limit: int = 5,
    include_context: bool = False,
    ctx: Context = None,
) -> str:
    """Semantic code search by meaning (vector + graph hybrid).
    Use when you don't know exact keywords. For exact matching use Grep."""
    wrapper = _get_wrapper(ctx)
    result = await wrapper.search(
        query, project_path, limit=limit, include_context=include_context
    )
    return _json(result)


@mcp.tool()
@synapse_error_handler
async def synapse_overview(project_path: str, ctx: Context = None) -> str:
    """Project architecture: INTELLIGENCE.md + index stats + file tree.
    Use for high-level understanding. For specific files use Read."""
    wrapper = _get_wrapper(ctx)
    result = await wrapper.overview(project_path)
    return _json(result)


@mcp.tool()
@synapse_error_handler
async def synapse_context(
    file_path: str,
    project_path: str,
    depth: int = 2,
    max_files: int = 10,
    ctx: Context = None,
) -> str:
    """Skeleton interfaces of files related to target via dependency graph.
    Use to see a file's imports/importers without reading each. For the file itself use Read."""
    wrapper = _get_wrapper(ctx)
    result = await wrapper.get_context(
        file_path, project_path, depth=depth, max_files=max_files
    )
    return _json(result)


@mcp.tool()
@synapse_error_handler
async def synapse_graph(
    file_path: str, project_path: str, ctx: Context = None
) -> str:
    """Dependency graph for a file: what it calls and what calls it.
    Use before refactoring to understand impact. For code search use synapse_search."""
    wrapper = _get_wrapper(ctx)
    result = await wrapper.graph(file_path, project_path)
    return _json(result)


@mcp.tool()
@synapse_error_handler
async def synapse_skeleton(file_path: str, ctx: Context = None) -> str:
    """Skeletonize a file: keep signatures and docstrings, replace bodies with '...'.
    Use to understand a file's API without reading full source. For dependencies use synapse_context."""
    wrapper = _get_wrapper(ctx)
    result = await wrapper.skeleton(file_path)
    return _json(result)


@mcp.tool()
@synapse_error_handler
async def synapse_ask(
    query: str,
    project_path: str,
    limit: int = 5,
    think: bool = False,
    ctx: Context = None,
) -> str:
    """Generate context-enriched prompt via vector search + graph expansion.
    Set think=true for Deep Think mode (Chain-of-Thought). For simple search use synapse_search."""
    wrapper = _get_wrapper(ctx)
    result = await wrapper.ask(query, project_path, limit=limit, think=think)
    return _json(result)


@mcp.tool()
@synapse_error_handler
async def synapse_watch_status(
    project_path: str, ctx: Context = None
) -> str:
    """Check file watcher daemon status for a project.
    Shows if background indexing is running. Start/stop watcher via CLI."""
    wrapper = _get_wrapper(ctx)
    result = await wrapper.watch_status(project_path)
    return _json(result)


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
