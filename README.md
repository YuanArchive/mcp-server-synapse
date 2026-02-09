# mcp-server-synapse

**Semantic code intelligence for Claude Code.**

[![MCP Server](https://badge.mcpx.dev?type=server&features=tools)](https://modelcontextprotocol.io)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An MCP server that bridges [synapse-ai-context](https://github.com/YuanArchive/synapse-ai-context)
with Claude Code — providing capabilities that complement its built-in tools:
semantic search, dependency graph traversal, and auto-generated architecture docs.

---

## Why?

Claude Code already has excellent built-in tools (Grep, Glob, Read) for keyword-based
search and file reading. This server focuses on what those tools **can't** do:

| Capability | Built-in | Synapse |
|:-----------|:--------:|:-------:|
| Exact keyword/regex search | Grep | - |
| Find code by **meaning** | - | `synapse_search` |
| Read a specific file | Read | - |
| See a file's **dependency tree** as skeletons | - | `synapse_context` |
| Understand **project architecture** | - | `synapse_overview` |
| Auto-generate **INTELLIGENCE.md** | - | `synapse_index` |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YuanArchive/mcp-server-synapse.git
cd mcp-server-synapse
uv sync
```

### 2. Register with Claude Code

```bash
claude mcp add synapse -- \
  uv run --directory /path/to/mcp-server-synapse mcp-server-synapse
```

### 3. Use

> "Use synapse to index this project and find code related to authentication."

That's it. Claude Code will automatically call the right tools.

---

## Tools

4 tools, designed to minimize token usage while maximizing information density.

### `synapse_index`

Index codebase for semantic search. Auto-initializes on first run, generates INTELLIGENCE.md.

```
project_path  (str)  — Absolute path to the project
full          (bool) — Force full reindex (default: false, incremental)
```

### `synapse_search`

Semantic code search by meaning — vector + dependency graph hybrid.

```
query           (str)  — Natural language description
project_path    (str)  — Absolute path to the project
limit           (int)  — Max results (default: 5)
include_context (bool) — Include dependency info per result (default: false)
```

### `synapse_overview`

Project architecture: INTELLIGENCE.md + index stats + file tree.

```
project_path  (str)  — Absolute path to the project
```

### `synapse_context`

Skeleton interfaces of files related to target via dependency graph.

```
file_path     (str) — Target file path
project_path  (str) — Absolute path to the project
depth         (int) — BFS traversal depth (default: 2)
max_files     (int) — Max related files (default: 10)
```

---

## Architecture

```
┌─────────────┐    MCP (stdio)    ┌────────────────────┐    Python API    ┌─────────────────────┐
│ Claude Code  │ ◄───────────────► │ mcp-server-synapse │ ◄─────────────► │ synapse-ai-context  │
│             │                   │   (FastMCP Server)  │                 │   (Core Engine)     │
└─────────────┘                   └────────────────────┘                  └─────────────────────┘
                                         │
                                         ├── server.py          → 4 tool definitions
                                         ├── synapse_wrapper.py → Async wrapper (ThreadPoolExecutor)
                                         └── utils.py           → Path validation, error handling
```

### Token Optimization (v0.2.1)

Every MCP tool response consumes Claude's context window. v0.2.1 is designed to minimize this:

- **Compact JSON** — No indentation, minimal separators (~25% smaller)
- **Short descriptions** — Tool descriptions cut from ~275 to ~125 tokens
- **Smaller defaults** — Search limit 10 → 5, snippet 800 → 400 chars, max_files 15 → 10
- **No redundant fields** — Removed `relation_type`, `included_files`, compacted `savings`
- **INTELLIGENCE.md truncation** — Capped at 4,000 chars to prevent context overflow

Measured on a real 18-file project:

| Tool | v0.2.0 | v0.2.1 | Reduction |
|:-----|-------:|-------:|----------:|
| `synapse_search` | ~1,500 tok | ~660 tok | **56%** |
| Tool descriptions (per request) | ~275 tok | ~125 tok | **55%** |
| Overall | — | — | **~31%** |

---

## Configuration

| Variable | Default | Description |
|:---------|:--------|:------------|
| `SYNAPSE_BATCH_SIZE` | `6` | ChromaDB upsert batch size |
| `SYNAPSE_MAX_DOC_CHARS` | `8000` | Max document characters for embedding |

## Requirements

- **Python** 3.10 – 3.12
- **macOS** with Apple Silicon recommended (Metal/MPS acceleration for embeddings)
- **uv** for package management

## License

[MIT](LICENSE)
