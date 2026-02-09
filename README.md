# mcp-server-synapse

**Give your AI agent deep understanding of any codebase.**

[![MCP Server](https://badge.mcpx.dev?type=server&features=tools,resources)](https://modelcontextprotocol.io)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An MCP server that bridges [synapse-ai-context](https://github.com/YuanArchive/synapse-ai-context)
with AI coding assistants — enabling semantic search, dependency graphs,
AST skeletons, and hierarchical context building through natural language.

## What is this?

[synapse-ai-context](https://github.com/YuanArchive/synapse-ai-context) is a code intelligence engine that indexes codebases using embeddings, dependency graphs, and AST analysis. This MCP server wraps its Python API so any MCP-compatible AI client (Claude Code, Cursor, Windsurf, etc.) can use it as a tool — no CLI needed.

## Features

- **Semantic Search** — Find relevant code using vector similarity + dependency graph
- **Hierarchical Context** — 3-layer context: architecture overview → skeletons → full source
- **Dependency Graph** — Visualize import relationships and call chains
- **AST Skeleton** — Strip implementation, keep structure (50%+ token reduction)
- **Incremental Indexing** — Only re-index changed files
- **Apple Silicon Optimized** — Metal (MPS) GPU acceleration for embeddings

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YuanArchive/mcp-server-synapse.git
cd mcp-server-synapse
uv sync
```

### 2. Register with Claude Code

```bash
claude mcp add synapse -- uv run --directory /path/to/mcp-server-synapse mcp-server-synapse
```

### 3. Use

Ask your AI assistant to analyze a project:

> "Use synapse to analyze this project and find code related to authentication."

The server will initialize the project, build indexes, and search through the codebase semantically.

## Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `synapse_init` | Initialize a synapse project (creates `.synapse/`, `.context/`, `.agent/` dirs) | `project_path` |
| `synapse_analyze` | Index the codebase — builds embeddings and dependency graph | `project_path`, `full` (bool) |
| `synapse_search` | Semantic/hybrid search over the codebase | `query`, `project_path`, `hybrid`, `limit` |
| `synapse_context` | Build hierarchical 3-layer context for a file | `file_path`, `project_path`, `depth`, `max_files` |
| `synapse_graph` | Get dependency relationships for a file | `file_path`, `project_path` |
| `synapse_skeleton` | Convert a file to its AST skeleton (signatures + structure only) | `file_path` |
| `synapse_ask` | Generate an AI prompt enriched with relevant codebase context | `query`, `project_path`, `think` |
| `synapse_watch` | Manage file watcher for automatic incremental indexing | `action` (start/stop/status), `project_path` |

## Resources

| URI | Description |
|-----|-------------|
| `synapse://{project_path}/intelligence` | Read the generated `INTELLIGENCE.md` architecture document |
| `synapse://{project_path}/stats` | Get index statistics (file count, graph nodes/edges) |

## Architecture

```
┌─────────────┐    MCP (stdio)    ┌────────────────────┐    Python API    ┌─────────────────────┐
│  AI Client  │ ◄───────────────► │ mcp-server-synapse │ ◄─────────────► │ synapse-ai-context  │
│ (Claude, …) │                   │   (FastMCP Server)  │                 │   (Core Engine)     │
└─────────────┘                   └────────────────────┘                  └─────────────────────┘
                                         │
                                         ├── server.py         → Tool & resource definitions
                                         ├── synapse_wrapper.py → Async wrapper (ThreadPoolExecutor)
                                         └── utils.py          → Path validation, error handling
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNAPSE_BATCH_SIZE` | `6` | ChromaDB upsert batch size |
| `SYNAPSE_MAX_DOC_CHARS` | `8000` | Max document characters for embedding |

## Requirements

- **Python** 3.10 – 3.12
- **macOS** with Apple Silicon recommended (Metal/MPS acceleration for embeddings)
- **uv** for package management

## License

[MIT](LICENSE)
