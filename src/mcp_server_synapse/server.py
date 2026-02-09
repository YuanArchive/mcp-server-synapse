"""MCP server for synapse-ai-context."""

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
    return json.dumps(data, ensure_ascii=False, indent=2)


# --- Tools ---


@mcp.tool()
@synapse_error_handler
async def synapse_init(project_path: str, ctx: Context) -> str:
    """Initialize a synapse project. Creates .synapse/, .context/, .agent/ directories.

    Args:
        project_path: Absolute path to the project directory.
    """
    wrapper = _get_wrapper(ctx)
    result = await wrapper.init_project(project_path)
    return _json(result)


@mcp.tool()
@synapse_error_handler
async def synapse_analyze(
    project_path: str, full: bool = False, ctx: Context = None
) -> str:
    """Index the codebase. Scans files, builds embeddings and dependency graph.
    Uses incremental indexing by default (only changed files).

    Args:
        project_path: Absolute path to the project directory.
        full: Force complete reindex (default: incremental).
    """
    wrapper = _get_wrapper(ctx)
    await ctx.report_progress(0, 100)
    result = await wrapper.analyze(project_path, full=full)
    await ctx.report_progress(100, 100)
    return _json(result)


@mcp.tool()
@synapse_error_handler
async def synapse_search(
    query: str,
    project_path: str,
    hybrid: bool = True,
    limit: int = 10,
    ctx: Context = None,
) -> str:
    """Search the codebase semantically. Finds relevant code using vector similarity
    and dependency graph relationships.

    Args:
        query: Natural language search query.
        project_path: Absolute path to the project directory.
        hybrid: Use hybrid search combining vectors + graph (default: True).
        limit: Maximum number of results (default: 10).
    """
    wrapper = _get_wrapper(ctx)
    result = await wrapper.search(query, project_path, hybrid=hybrid, limit=limit)
    return _json(result)


@mcp.tool()
@synapse_error_handler
async def synapse_context(
    file_path: str,
    project_path: str,
    depth: int = 2,
    max_files: int = 15,
    ctx: Context = None,
) -> str:
    """Build hierarchical context for a file. Returns 3 layers:
    Global (architecture overview), Reference (related files as skeletons),
    and Active (full source of target file).

    Args:
        file_path: Path to the target file (relative to project or absolute).
        project_path: Absolute path to the project directory.
        depth: BFS depth for finding related files (default: 2).
        max_files: Max related files to include (default: 15).
    """
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
    """Get dependency relationships for a file. Shows files that import/are imported
    by the target file.

    Args:
        file_path: Path to the target file.
        project_path: Absolute path to the project directory.
    """
    wrapper = _get_wrapper(ctx)
    result = await wrapper.get_graph(file_path, project_path)
    return _json(result)


@mcp.tool()
@synapse_error_handler
async def synapse_skeleton(file_path: str, ctx: Context = None) -> str:
    """Convert a file to its AST skeleton. Preserves structure (imports, signatures,
    docstrings, type hints) while removing implementation details.

    Args:
        file_path: Absolute path to the file.
    """
    wrapper = _get_wrapper(ctx)
    result = await wrapper.skeletonize(file_path)
    return _json(result)


@mcp.tool()
@synapse_error_handler
async def synapse_ask(
    query: str,
    project_path: str,
    think: bool = False,
    ctx: Context = None,
) -> str:
    """Generate an AI prompt enriched with relevant codebase context.
    Searches for relevant code and builds a structured prompt.

    Args:
        query: The question or task description.
        project_path: Absolute path to the project directory.
        think: Enable chain-of-thought reasoning prefix (default: False).
    """
    wrapper = _get_wrapper(ctx)
    result = await wrapper.ask(query, project_path, think=think)
    return _json(result)


@mcp.tool()
@synapse_error_handler
async def synapse_watch(
    action: str, project_path: str, ctx: Context = None
) -> str:
    """Manage the file watcher for automatic incremental indexing.

    Args:
        action: One of "start", "stop", or "status".
        project_path: Absolute path to the project directory.
    """
    wrapper = _get_wrapper(ctx)
    result = await wrapper.watch(action, project_path)
    return _json(result)


# --- Resources ---


@mcp.resource("synapse://{project_path}/intelligence")
@synapse_error_handler
async def get_intelligence(project_path: str) -> str:
    """Read the INTELLIGENCE.md architecture document for a project."""
    from pathlib import Path

    p = Path(project_path).resolve()
    intel_path = p / ".context" / "INTELLIGENCE.md"
    if not intel_path.exists():
        return _json(
            {
                "error": "INTELLIGENCE.md not found. Run synapse_analyze first."
            }
        )
    return intel_path.read_text()


@mcp.resource("synapse://{project_path}/stats")
@synapse_error_handler
async def get_stats(project_path: str) -> str:
    """Get index statistics for a project (file count, vectors, graph nodes)."""
    from pathlib import Path

    p = Path(project_path).resolve()
    synapse_dir = p / ".synapse"
    if not synapse_dir.is_dir():
        return _json({"error": "Project not initialized."})

    stats = {"project_path": str(p)}

    # File hashes
    hashes_path = synapse_dir / "file_hashes.json"
    if hashes_path.exists():
        import json as json_mod

        hashes = json_mod.loads(hashes_path.read_text())
        stats["indexed_files"] = len(hashes)

    # Graph
    graph_path = synapse_dir / "dependency_graph.gml"
    if graph_path.exists():
        from synapse.graph import CodeGraph

        g = CodeGraph()
        g.load(str(graph_path))
        stats["graph_nodes"] = g.graph.number_of_nodes() if hasattr(g, "graph") else 0
        stats["graph_edges"] = g.graph.number_of_edges() if hasattr(g, "graph") else 0

    # Context JSON
    context_path = synapse_dir / "context.json"
    if context_path.exists():
        import json as json_mod

        ctx_data = json_mod.loads(context_path.read_text())
        stats["analysis_summary"] = ctx_data

    return _json(stats)


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
