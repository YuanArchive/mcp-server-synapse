"""Async wrapper around synapse-ai-context modules."""

import asyncio
import json as json_mod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .utils import validate_project_path


@dataclass
class CachedProject:
    """Cached synapse instances for a project."""

    path: Path
    analyzer: object  # ProjectAnalyzer
    vector_store: object | None = None  # VectorStore
    code_graph: object | None = None  # CodeGraph


class SynapseWrapper:
    """Manages per-project synapse instances with async execution."""

    def __init__(self):
        self._cache: dict[str, CachedProject] = {}
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def _run_sync(self, func, *args, **kwargs):
        """Run a sync function in the thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, lambda: func(*args, **kwargs)
        )

    def _get_or_create(self, path: str) -> CachedProject:
        """Get or create cached project instances."""
        key = str(Path(path).resolve())
        if key not in self._cache:
            from synapse.analyzer import ProjectAnalyzer

            self._cache[key] = CachedProject(
                path=Path(key),
                analyzer=ProjectAnalyzer(key),
            )
        return self._cache[key]

    def _load_stores(self, project: CachedProject) -> None:
        """Load vector store and graph from .synapse/ directory."""
        synapse_dir = project.path / ".synapse"
        if project.vector_store is None:
            from synapse.vector_store import VectorStore

            project.vector_store = VectorStore(
                db_path=str(synapse_dir / "db")
            )
        if project.code_graph is None:
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

            if full or is_fresh:
                result = project.analyzer.analyze(json_output=True)
            else:
                result = project.analyzer.analyze_incremental(json_output=True)

            # Invalidate cached stores after re-indexing
            project.vector_store = None
            project.code_graph = None

            # Generate INTELLIGENCE.md
            try:
                self._load_stores(project)
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
            self._load_stores(project)

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
        self._executor.shutdown(wait=False)
        self._cache.clear()
