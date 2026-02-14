"""Async wrapper around synapse-ai-context modules."""

import asyncio
import fnmatch
import gc
import json as json_mod
import os
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

from .utils import validate_project_path


@dataclass
class CachedProject:
    """Cached synapse instances for a project."""

    path: Path
    analyzer: object | None = None  # ProjectAnalyzer
    vector_store: object | None = None  # VectorStore
    code_graph: object | None = None  # CodeGraph
    last_access: float = field(default_factory=time.monotonic)


class SynapseWrapper:
    """Manages per-project synapse instances with async execution."""

    def __init__(self):
        self._cache: dict[str, CachedProject] = {}
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._max_cached_projects = self._read_int_env(
            "SYNAPSE_MAX_CACHED_PROJECTS", default=1, minimum=1
        )
        self._cache_ttl_seconds = self._read_int_env(
            "SYNAPSE_CACHE_TTL_SECONDS", default=600, minimum=0
        )
        self._index_workers = self._read_int_env(
            "SYNAPSE_INDEX_WORKERS", default=2, minimum=1
        )
        self._aggressive_cleanup = self._read_bool_env(
            "SYNAPSE_AGGRESSIVE_CLEANUP", default=True
        )
        self._keep_vector_store_loaded = self._read_bool_env(
            "SYNAPSE_KEEP_VECTOR_STORE_LOADED", default=False
        )
        self._pretruncate_chars = self._read_int_env(
            "SYNAPSE_PRETRUNCATE_CHARS", default=2500, minimum=1
        )
        self._max_symbol_chars = self._read_int_env(
            "SYNAPSE_MAX_SYMBOL_CHARS", default=1200, minimum=1
        )
        self._max_symbols_per_file = self._read_int_env(
            "SYNAPSE_MAX_SYMBOLS_PER_FILE", default=80, minimum=0
        )
        self._force_cpu_embeddings = self._read_bool_env(
            "SYNAPSE_FORCE_CPU_EMBEDDINGS", default=True
        )
        self._configure_runtime_limits()

    async def _run_sync(self, func, *args, **kwargs):
        """Run a sync function in the thread pool executor."""
        loop = asyncio.get_event_loop()

        # Keep MCP stdio channel clean: third-party libraries may print to stdout.
        def _wrapped():
            with redirect_stdout(sys.stderr):
                return func(*args, **kwargs)

        return await loop.run_in_executor(self._executor, _wrapped)

    @staticmethod
    def _read_int_env(name: str, default: int, minimum: int = 0) -> int:
        """Read an integer env var safely, with a minimum bound."""
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            return default
        return max(minimum, value)

    @staticmethod
    def _read_bool_env(name: str, default: bool) -> bool:
        """Read a boolean env var from common true/false strings."""
        raw = os.getenv(name)
        if raw is None:
            return default
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _read_csv_env(name: str) -> list[str]:
        """Read comma-separated env var values, stripping blanks."""
        raw = os.getenv(name, "")
        if not raw:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]

    def _configure_runtime_limits(self) -> None:
        """Apply conservative runtime defaults to reduce peak memory usage."""
        os.environ.setdefault("SYNAPSE_BATCH_SIZE", "2")
        os.environ.setdefault(
            "SYNAPSE_MAX_DOC_CHARS", str(self._pretruncate_chars)
        )
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        if not self._force_cpu_embeddings:
            return

        try:
            import torch

            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and hasattr(mps_backend, "is_available"):
                mps_backend.is_available = lambda: False
        except Exception:
            pass

    def _load_ignore_patterns(self, project_path: Path) -> list[str]:
        """Load indexing ignore patterns from file + env vars."""
        patterns: list[str] = []

        ignore_file_name = os.getenv("SYNAPSE_IGNORE_FILE", ".synapseignore").strip()
        if ignore_file_name:
            ignore_file_path = Path(ignore_file_name)
            if not ignore_file_path.is_absolute():
                ignore_file_path = project_path / ignore_file_path
            if ignore_file_path.is_file():
                for line in ignore_file_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    patterns.append(stripped)

        patterns.extend(self._read_csv_env("SYNAPSE_IGNORE_PATTERNS"))

        # Dedupe while preserving order
        unique_patterns: list[str] = []
        seen = set()
        for pattern in patterns:
            if pattern in seen:
                continue
            seen.add(pattern)
            unique_patterns.append(pattern)
        return unique_patterns

    def _is_ignored_file(
        self, file_path: Path, project_path: Path, patterns: list[str]
    ) -> bool:
        """Check whether a file should be excluded from indexing."""
        try:
            rel_path = file_path.resolve().relative_to(project_path.resolve()).as_posix()
        except ValueError:
            rel_path = file_path.resolve().as_posix()

        parts = rel_path.split("/")
        filename = parts[-1] if parts else rel_path

        for raw_pattern in patterns:
            pattern = raw_pattern.strip()
            if not pattern or pattern.startswith("#"):
                continue
            if pattern.startswith("!"):
                # Negation is currently not supported.
                continue

            pattern = pattern.removeprefix("./")
            is_dir_pattern = pattern.endswith("/")
            core_pattern = pattern.rstrip("/")
            if not core_pattern:
                continue

            if is_dir_pattern:
                if "/" in core_pattern:
                    anchored = core_pattern.lstrip("/")
                    if rel_path == anchored or rel_path.startswith(anchored + "/"):
                        return True
                    if fnmatch.fnmatch(rel_path, f"**/{anchored}/**"):
                        return True
                else:
                    if any(fnmatch.fnmatch(part, core_pattern) for part in parts[:-1]):
                        return True
                continue

            if "/" in core_pattern:
                anchored = core_pattern.lstrip("/")
                if fnmatch.fnmatch(rel_path, anchored):
                    return True
                if fnmatch.fnmatch(rel_path, f"**/{anchored}"):
                    return True
            elif fnmatch.fnmatch(filename, core_pattern):
                return True

        return False

    def _apply_index_filters(self, analyzer: object, project_path: Path) -> list[str]:
        """Apply directory/file ignore filters to analyzer.scan_files."""
        extra_exclude_dirs = self._read_csv_env("SYNAPSE_EXTRA_EXCLUDE_DIRS")
        ignore_patterns = self._load_ignore_patterns(project_path)

        if hasattr(analyzer, "exclude_dirs"):
            for value in extra_exclude_dirs:
                cleaned = value.strip().strip("/")
                if not cleaned:
                    continue
                ignore_patterns.append(f"{cleaned}/")
                # Analyzer exclude_dirs matches by path.parts, so only simple names apply.
                if "/" not in cleaned and cleaned not in analyzer.exclude_dirs:
                    analyzer.exclude_dirs.append(cleaned)

        # Dedupe merged patterns
        merged_patterns: list[str] = []
        seen = set()
        for pattern in ignore_patterns:
            if pattern in seen:
                continue
            seen.add(pattern)
            merged_patterns.append(pattern)

        if not getattr(analyzer, "_synapse_ignore_wrapped", False):
            original_scan_files = analyzer.scan_files

            def _scan_files_with_ignores(*args, **kwargs):
                files = original_scan_files(*args, **kwargs)
                active_patterns = getattr(analyzer, "_synapse_ignore_patterns", [])
                if not active_patterns:
                    return files
                return [
                    file_path
                    for file_path in files
                    if not self._is_ignored_file(
                        file_path, project_path, active_patterns
                    )
                ]

            analyzer.scan_files = _scan_files_with_ignores
            analyzer._synapse_ignore_wrapped = True

        analyzer._synapse_ignore_patterns = merged_patterns
        return merged_patterns

    def _build_ignore_metadata(
        self, project_path: Path, ignore_patterns: list[str]
    ) -> dict:
        """Build compact ignore-config metadata for tool consumers."""
        ignore_file_name = os.getenv("SYNAPSE_IGNORE_FILE", ".synapseignore").strip()
        if not ignore_file_name:
            ignore_file_name = ".synapseignore"

        ignore_file_path = Path(ignore_file_name)
        if not ignore_file_path.is_absolute():
            ignore_file_path = project_path / ignore_file_path

        return {
            "pattern_count": len(ignore_patterns),
            "patterns_preview": ignore_patterns[:8],
            "ignore_file": str(ignore_file_path),
            "ignore_file_exists": ignore_file_path.is_file(),
            "hint": "Use .synapseignore or env: SYNAPSE_IGNORE_PATTERNS,SYNAPSE_EXTRA_EXCLUDE_DIRS",
        }

    def _build_memory_metadata(self) -> dict:
        """Expose active memory controls for diagnostics."""
        return {
            "index_workers": self._index_workers,
            "force_cpu_embeddings": self._force_cpu_embeddings,
            "pretruncate_chars": self._pretruncate_chars,
            "max_symbol_chars": self._max_symbol_chars,
            "max_symbols_per_file": self._max_symbols_per_file,
            "batch_size": os.getenv("SYNAPSE_BATCH_SIZE"),
            "max_doc_chars": os.getenv("SYNAPSE_MAX_DOC_CHARS"),
        }

    def _apply_low_memory_doc_limits(self, analyzer: object) -> None:
        """Patch analyzer aggregation to cap in-memory payload size."""
        if getattr(analyzer, "_synapse_low_memory_wrapped", False):
            return
        if not hasattr(analyzer, "_aggregate_results"):
            return

        original_aggregate_results = analyzer._aggregate_results

        def _aggregate_with_limits(
            this,
            file_path: Path,
            result: dict,
            content: str,
            documents: list,
            metadatas: list,
            ids: list,
        ):
            bounded_content = content
            if (
                isinstance(bounded_content, str)
                and self._pretruncate_chars > 0
                and len(bounded_content) > self._pretruncate_chars
            ):
                bounded_content = bounded_content[: self._pretruncate_chars]

            bounded_result = result
            symbols = (
                result.get("symbols", [])
                if isinstance(result, dict)
                else []
            )
            if symbols:
                bounded_symbols = symbols
                changed = False

                if (
                    self._max_symbols_per_file >= 0
                    and len(bounded_symbols) > self._max_symbols_per_file
                ):
                    bounded_symbols = bounded_symbols[
                        : self._max_symbols_per_file
                    ]
                    changed = True

                if self._max_symbol_chars > 0:
                    rewritten = []
                    for symbol in bounded_symbols:
                        if not isinstance(symbol, dict):
                            rewritten.append(symbol)
                            continue

                        symbol_code = symbol.get("code")
                        if (
                            isinstance(symbol_code, str)
                            and len(symbol_code) > self._max_symbol_chars
                        ):
                            compact_symbol = dict(symbol)
                            compact_symbol["code"] = symbol_code[
                                : self._max_symbol_chars
                            ]
                            rewritten.append(compact_symbol)
                            changed = True
                        else:
                            rewritten.append(symbol)
                    bounded_symbols = rewritten

                if changed:
                    bounded_result = dict(result)
                    bounded_result["symbols"] = bounded_symbols

            return original_aggregate_results(
                file_path,
                bounded_result,
                bounded_content,
                documents,
                metadatas,
                ids,
            )

        analyzer._aggregate_results = types.MethodType(
            _aggregate_with_limits, analyzer
        )
        analyzer._synapse_low_memory_wrapped = True

    def _collect_memory(self) -> None:
        """Run best-effort memory cleanup for Python and torch allocators."""
        if not self._aggressive_cleanup:
            return

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            mps = getattr(torch, "mps", None)
            if (
                mps is not None
                and hasattr(mps, "is_available")
                and mps.is_available()
                and hasattr(mps, "empty_cache")
            ):
                mps.empty_cache()
        except Exception:
            # torch may not be available or cache clear may fail on some backends
            pass

    def _evict_project(self, key: str) -> None:
        """Drop cached state for a project and free associated memory."""
        project = self._cache.pop(key, None)
        if project is None:
            return

        if project.code_graph is not None and hasattr(project.code_graph, "graph"):
            try:
                project.code_graph.graph.clear()
            except Exception:
                pass

        project.analyzer = None
        project.vector_store = None
        project.code_graph = None
        self._collect_memory()

    def _evict_idle_projects(self) -> None:
        """Evict projects that have been idle for longer than TTL."""
        if self._cache_ttl_seconds <= 0:
            return

        now = time.monotonic()
        stale_keys = [
            key
            for key, project in self._cache.items()
            if now - project.last_access > self._cache_ttl_seconds
        ]
        for key in stale_keys:
            self._evict_project(key)

    def _enforce_cache_limit(self, active_key: str) -> None:
        """Keep only the most recently used projects in cache."""
        if len(self._cache) <= self._max_cached_projects:
            return

        candidates = sorted(
            (
                (key, project.last_access)
                for key, project in self._cache.items()
                if key != active_key
            ),
            key=lambda item: item[1],
        )
        for key, _ in candidates:
            if len(self._cache) <= self._max_cached_projects:
                break
            self._evict_project(key)

    def _get_or_create(self, path: str) -> CachedProject:
        """Get or create cached project instances."""
        self._evict_idle_projects()

        key = str(Path(path).resolve())
        if key not in self._cache:
            self._cache[key] = CachedProject(path=Path(key))
        project = self._cache[key]
        project.last_access = time.monotonic()
        self._enforce_cache_limit(active_key=key)
        return project

    def _get_analyzer(self, project: CachedProject):
        """Lazy-create analyzer only when indexing is requested."""
        if project.analyzer is None:
            from synapse.analyzer import ProjectAnalyzer

            project.analyzer = ProjectAnalyzer(str(project.path))
        return project.analyzer

    def _load_stores(
        self,
        project: CachedProject,
        *,
        need_vector_store: bool = True,
        need_graph: bool = True,
    ) -> None:
        """Load vector store and/or graph from .synapse/ directory."""
        synapse_dir = project.path / ".synapse"
        if need_vector_store and project.vector_store is None:
            from synapse.vector_store import VectorStore

            project.vector_store = VectorStore(
                db_path=str(synapse_dir / "db")
            )
        if need_graph and project.code_graph is None:
            from synapse.graph import CodeGraph

            graph_path = synapse_dir / "dependency_graph.gml"
            project.code_graph = CodeGraph(storage_path=graph_path)
            if graph_path.exists():
                project.code_graph.load()

    def _ensure_initialized(self, project_path: Path) -> bool:
        """Initialize .synapse/ if not present. Returns True if freshly created."""
        synapse_dir = project_path / ".synapse"
        if synapse_dir.is_dir():
            return False
        synapse_dir.mkdir(parents=True)
        (synapse_dir / "db").mkdir()
        (project_path / ".context").mkdir(exist_ok=True)
        (project_path / ".agent").mkdir(exist_ok=True)
        return True

    async def index(self, path: str, full: bool = False) -> dict:
        """Initialize (if needed) and index the codebase.

        Auto-creates .synapse/ directories on first run.
        Generates INTELLIGENCE.md after analysis.
        """
        project_path = validate_project_path(path)
        project = self._get_or_create(str(project_path))

        def _index():
            is_fresh = self._ensure_initialized(project_path)
            analyzer = self._get_analyzer(project)
            ignore_patterns = self._apply_index_filters(analyzer, project_path)
            self._apply_low_memory_doc_limits(analyzer)

            if full or is_fresh:
                result = analyzer.analyze(
                    json_output=True, num_workers=self._index_workers
                )
            else:
                result = analyzer.analyze_incremental(
                    json_output=True, num_workers=self._index_workers
                )

            # Reuse analyzer's graph to avoid duplicate loads.
            # Keep vector store cached only when explicitly enabled.
            project.code_graph = getattr(analyzer, "code_graph", None)
            if self._keep_vector_store_loaded:
                project.vector_store = getattr(analyzer, "vector_store", None)
            else:
                project.vector_store = None
            project.analyzer = None
            self._collect_memory()

            # Generate INTELLIGENCE.md
            try:
                self._load_stores(
                    project, need_vector_store=False, need_graph=True
                )
                if project.code_graph and hasattr(project.code_graph, "graph"):
                    from synapse.markdown_gen import MarkdownGenerator

                    # Build analysis summary from context.json if available
                    context_path = project_path / ".synapse" / "context.json"
                    analysis_summary = {}
                    if context_path.exists():
                        analysis_summary = json_mod.loads(
                            context_path.read_text()
                        )

                    gen = MarkdownGenerator(
                        project_path,
                        analysis_summary,
                        project.code_graph.graph,
                    )
                    intel_content = gen.generate()
                    intel_path = project_path / ".context" / "INTELLIGENCE.md"
                    intel_path.parent.mkdir(parents=True, exist_ok=True)
                    intel_path.write_text(intel_content)
            except Exception:
                pass  # INTELLIGENCE.md generation is best-effort

            # Build response
            output = {
                "status": result.get("status", "success"),
                "files_analyzed": result.get(
                    "files_analyzed", result.get("changed_files", 0)
                ),
                "graph_nodes": result.get("graph_nodes", 0),
                "graph_edges": result.get("graph_edges", 0),
                "is_fresh_index": is_fresh,
                "ignore": self._build_ignore_metadata(
                    project_path, ignore_patterns
                ),
                "memory": self._build_memory_metadata(),
            }
            return output

        return await self._run_sync(_index)

    async def search(
        self,
        query: str,
        path: str,
        limit: int = 5,
        include_context: bool = False,
    ) -> dict:
        """Hybrid semantic search over the codebase.

        Always uses vector + graph hybrid search when graph is available.
        Optionally includes dependency info per result.
        """
        project_path = validate_project_path(path)
        if not (project_path / ".synapse").is_dir():
            return {
                "error": "Project not indexed.",
                "hint": "Run synapse_index first.",
            }

        project = self._get_or_create(str(project_path))

        def _search():
            self._load_stores(
                project, need_vector_store=True, need_graph=True
            )

            if project.code_graph is not None:
                from synapse.hybrid_search import HybridSearch

                searcher = HybridSearch(
                    project.vector_store, graph=project.code_graph
                )
                hybrid_result = searcher.search(query, top_k=limit)
                results = []
                for r in hybrid_result.results:
                    entry = {
                        "file": r.node_id,
                        "score": round(r.hybrid_score, 4),
                        "snippet": r.content[:400],
                    }
                    if include_context and project.code_graph:
                        related = project.code_graph.get_related_files(
                            r.node_id, depth=1
                        )
                        entry["dependencies"] = related
                    results.append(entry)
            else:
                raw = project.vector_store.query(query, n_results=limit)
                results = []
                if raw and raw.get("documents"):
                    docs = raw["documents"][0] if raw["documents"] else []
                    metas = (
                        raw["metadatas"][0] if raw.get("metadatas") else []
                    )
                    dists = (
                        raw["distances"][0] if raw.get("distances") else []
                    )
                    for i, doc in enumerate(docs):
                        entry = {
                            "file": (
                                metas[i].get("file_path", "")
                                if i < len(metas)
                                else ""
                            ),
                            "score": (
                                round(1 / (1 + dists[i]), 4)
                                if i < len(dists)
                                else 0
                            ),
                            "snippet": doc[:400],
                        }
                        if include_context and project.code_graph:
                            related = project.code_graph.get_related_files(
                                entry["file"], depth=1
                            )
                            entry["dependencies"] = related
                        results.append(entry)

            return {"query": query, "results": results, "count": len(results)}

        return await self._run_sync(_search)

    async def get_context(
        self,
        file: str,
        path: str,
        depth: int = 2,
        max_files: int = 10,
    ) -> dict:
        """Build dependency-aware reference context for a file.

        Returns skeleton interfaces of related files (imports/imported_by).
        Does NOT include full source or global context — Claude Code handles those.
        """
        project_path = validate_project_path(path)
        if not (project_path / ".synapse").is_dir():
            return {
                "error": "Project not indexed.",
                "hint": "Run synapse_index first.",
            }

        def _context():
            from synapse.context_manager import ContextManager

            manager = ContextManager(
                Path(project_path), depth=depth, max_reference_files=max_files
            )
            result = manager.build_context(Path(file))

            # Parse reference_context into structured entries
            references = []
            if result.reference_context:
                # reference_context is a formatted string with file sections
                current_file = None
                current_skeleton = []
                for line in result.reference_context.split("\n"):
                    if line.startswith("### "):
                        if current_file:
                            references.append(
                                {
                                    "file": current_file,
                                    "skeleton": "\n".join(current_skeleton),
                                }
                            )
                        current_file = line.replace("### ", "").strip()
                        current_skeleton = []
                    else:
                        current_skeleton.append(line)
                if current_file:
                    references.append(
                        {
                            "file": current_file,
                            "skeleton": "\n".join(current_skeleton),
                        }
                    )

            output = {
                "target_file": file,
                "references": references,
                "reference_count": len(references),
            }

            if result.savings:
                orig = getattr(result.savings, "original_tokens", 0)
                opt = getattr(result.savings, "optimized_tokens", 0)
                ratio = getattr(result.savings, "reduction_ratio", 0)
                pct = round(ratio * 100) if ratio <= 1 else round(ratio)
                output["savings"] = f"{orig}\u2192{opt} tokens ({pct}% saved)"

            return output

        return await self._run_sync(_context)

    async def overview(self, path: str, max_intel_chars: int = 4000) -> dict:
        """Get project architecture overview.

        Returns INTELLIGENCE.md content, index stats, and file tree.
        """
        project_path = validate_project_path(path)
        if not (project_path / ".synapse").is_dir():
            return {
                "error": "Project not indexed.",
                "hint": "Run synapse_index first.",
            }

        def _overview():
            result = {"project_path": str(project_path)}

            # INTELLIGENCE.md
            intel_path = project_path / ".context" / "INTELLIGENCE.md"
            if intel_path.exists():
                content = intel_path.read_text()
                if len(content) > max_intel_chars:
                    result["intelligence"] = content[:max_intel_chars] + "\n\n... (truncated)"
                    result["intelligence_truncated"] = True
                else:
                    result["intelligence"] = content
            else:
                result["intelligence"] = None

            # Index stats from file_hashes.json
            hashes_path = project_path / ".synapse" / "file_hashes.json"
            if hashes_path.exists():
                hashes = json_mod.loads(hashes_path.read_text())
                files_data = hashes.get("files", hashes)
                result["indexed_files"] = len(files_data)

            # Graph stats
            graph_path = project_path / ".synapse" / "dependency_graph.gml"
            if graph_path.exists():
                from synapse.graph import CodeGraph

                g = CodeGraph(storage_path=graph_path)
                g.load()
                if hasattr(g, "graph"):
                    result["graph_nodes"] = g.graph.number_of_nodes()
                    result["graph_edges"] = g.graph.number_of_edges()

            # File tree (3 levels)
            result["file_tree"] = self._build_file_tree(project_path, max_depth=3)

            return result

        return await self._run_sync(_overview)

    async def graph(self, file: str, path: str) -> dict:
        """Get dependency graph for a specific file.

        Returns dependencies (calls), dependents (called by), and defines.
        """
        project_path = validate_project_path(path)
        if not (project_path / ".synapse").is_dir():
            return {
                "error": "Project not indexed.",
                "hint": "Run synapse_index first.",
            }

        def _graph():
            from synapse.graph import CodeGraph

            graph_path = project_path / ".synapse" / "dependency_graph.gml"
            code_graph = CodeGraph(storage_path=graph_path)
            if not graph_path.exists():
                return {"error": "Dependency graph not found.", "hint": "Run synapse_index first."}
            code_graph.load()

            target = Path(file).resolve().as_posix()

            # Find node — exact match or suffix fallback
            if target not in code_graph.graph:
                candidates = [
                    n for n in code_graph.graph.nodes
                    if str(n).endswith(file)
                    and code_graph.graph.nodes[n].get("type") == "file"
                ]
                if len(candidates) == 1:
                    target = candidates[0]
                elif len(candidates) > 1:
                    return {"error": "Ambiguous path.", "candidates": candidates}
                else:
                    return {"error": f"File not found in graph: {file}"}

            # Dependencies (outgoing calls edges)
            dependencies = []
            defines = []
            for succ in code_graph.graph.successors(target):
                edge = code_graph.graph.get_edge_data(target, succ)
                if edge and edge.get("type") == "calls":
                    dependencies.append({"file": succ, "symbol": edge.get("symbol", "?")})
                elif edge and edge.get("type") == "defines":
                    node_data = code_graph.graph.nodes[succ]
                    defines.append(node_data.get("name", succ))

            # Dependents (incoming calls edges)
            dependents = []
            for pred in code_graph.graph.predecessors(target):
                edge = code_graph.graph.get_edge_data(pred, target)
                if edge and edge.get("type") == "calls":
                    dependents.append({"file": pred, "symbol": edge.get("symbol", "?")})

            return {
                "target": target,
                "dependencies": dependencies,
                "dependents": dependents,
                "defines": defines,
            }

        return await self._run_sync(_graph)

    async def skeleton(self, file: str) -> dict:
        """Skeletonize a file — keep signatures, replace bodies with '...'.

        AST-based, preserves imports and docstrings.
        """
        file_path = Path(file).resolve()
        if not file_path.exists():
            return {"error": f"File not found: {file}"}

        def _skeleton():
            from synapse.structure.pruner import ASTSkeletonizer

            skeletonizer = ASTSkeletonizer()
            result = skeletonizer.skeletonize_file(file_path)
            return {
                "file": str(file_path),
                "original_lines": result.original_lines,
                "skeleton_lines": result.skeleton_lines,
                "reduction_ratio": round(result.reduction_ratio, 3),
                "skeleton": result.skeleton,
            }

        return await self._run_sync(_skeleton)

    async def ask(
        self,
        query: str,
        path: str,
        limit: int = 5,
        think: bool = False,
    ) -> dict:
        """Generate a context-enriched prompt for the LLM.

        Vector search + graph expansion. Optionally prepend Deep Think prompt.
        """
        project_path = validate_project_path(path)
        if not (project_path / ".synapse").is_dir():
            return {
                "error": "Project not indexed.",
                "hint": "Run synapse_index first.",
            }

        project = self._get_or_create(str(project_path))

        def _ask():
            self._load_stores(
                project, need_vector_store=True, need_graph=True
            )

            # Step 1: Vector search
            context_files = set()
            context_snippets = []

            if project.vector_store:
                results = project.vector_store.query(query, n_results=limit)
                if results and results.get("documents"):
                    documents = results["documents"][0]
                    metadatas = results["metadatas"][0] if results.get("metadatas") else []
                    for i, doc in enumerate(documents):
                        file_path = metadatas[i].get("path", "") if i < len(metadatas) else ""
                        if file_path:
                            context_files.add(file_path)
                        context_snippets.append({
                            "file": file_path,
                            "code": doc[:600],
                        })

            # Step 2: Graph expansion
            graph_linked = []
            if project.code_graph:
                expanded = set()
                for f in context_files:
                    related = project.code_graph.get_related_files(f, depth=1)
                    for r in related:
                        if r not in context_files and not r.startswith("symbol:"):
                            expanded.add(r)
                graph_linked = list(expanded)[:10]

            # Step 3: Build prompt
            prompt_parts = []
            if think:
                from synapse.prompts import DEEP_THINK_PROMPT
                prompt_parts.append(DEEP_THINK_PROMPT.strip())

            prompt_parts.append(f"Question: {query}")
            prompt_parts.append("\n### Retrieved Context:")
            for s in context_snippets:
                prompt_parts.append(f"\nFile: {s['file']}\n```\n{s['code']}\n```")

            if graph_linked:
                prompt_parts.append("\n### Graph-Linked Files:")
                for gl in graph_linked:
                    prompt_parts.append(f"- {gl}")

            output = {
                "query": query,
                "context_snippets": context_snippets,
                "graph_linked_files": graph_linked,
                "think_mode": think,
                "prompt": "\n".join(prompt_parts),
            }
            return output

        return await self._run_sync(_ask)

    async def watch_status(self, path: str) -> dict:
        """Get watcher daemon status for a project."""
        project_path = validate_project_path(path)

        def _status():
            from synapse.watcher import SynapseWatcher

            status = SynapseWatcher.get_status(project_path)
            return status.to_dict()

        return await self._run_sync(_status)

    def _build_file_tree(
        self, root: Path, max_depth: int = 3, _depth: int = 0
    ) -> str:
        """Build a simple file tree string (3 levels max)."""
        if _depth >= max_depth:
            return ""

        skip_dirs = {
            ".git", ".synapse", ".context", ".agent",
            "__pycache__", "node_modules", ".venv", "venv",
            ".tox", ".mypy_cache", ".pytest_cache",
        }

        lines = []
        try:
            entries = sorted(root.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        except PermissionError:
            return ""

        for entry in entries:
            if entry.name.startswith(".") and entry.name in skip_dirs:
                continue
            if entry.name in skip_dirs:
                continue

            prefix = "  " * _depth
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
                sub = self._build_file_tree(entry, max_depth, _depth + 1)
                if sub:
                    lines.append(sub)
            else:
                lines.append(f"{prefix}{entry.name}")

        return "\n".join(lines)

    async def cleanup(self):
        """Shutdown the thread pool executor."""
        for key in list(self._cache.keys()):
            self._evict_project(key)
        self._executor.shutdown(wait=False)
        self._cache.clear()
