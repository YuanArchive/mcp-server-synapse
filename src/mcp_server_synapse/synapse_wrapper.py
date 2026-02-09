"""Async wrapper around synapse-ai-context modules."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .utils import ensure_initialized, validate_project_path


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

    async def init_project(self, path: str) -> dict:
        """Initialize .synapse/, .context/, .agent/ directories."""
        project_path = validate_project_path(path)

        def _init():
            synapse_dir = project_path / ".synapse"
            context_dir = project_path / ".context"
            agent_dir = project_path / ".agent"

            created = []
            for d in [synapse_dir, context_dir, agent_dir]:
                if not d.exists():
                    d.mkdir(parents=True)
                    created.append(str(d.relative_to(project_path)))

            # Create db subdirectory
            db_dir = synapse_dir / "db"
            if not db_dir.exists():
                db_dir.mkdir()

            return {
                "success": True,
                "project_path": str(project_path),
                "created_directories": created,
                "message": (
                    f"Initialized synapse project at {project_path}"
                    if created
                    else "Project already initialized"
                ),
            }

        return await self._run_sync(_init)

    async def analyze(self, path: str, full: bool = False) -> dict:
        """Run codebase indexing (incremental or full)."""
        project_path = validate_project_path(path)
        if not ensure_initialized(path):
            return {
                "error": "Project not initialized. Run synapse_init first."
            }

        project = self._get_or_create(str(project_path))

        def _analyze():
            if full:
                result = project.analyzer.analyze(json_output=True)
            else:
                result = project.analyzer.analyze_incremental(json_output=True)
            # Invalidate cached stores after re-indexing
            project.vector_store = None
            project.code_graph = None
            return result

        return await self._run_sync(_analyze)

    async def search(
        self,
        query: str,
        path: str,
        hybrid: bool = True,
        limit: int = 10,
    ) -> dict:
        """Semantic or hybrid search over the codebase."""
        project_path = validate_project_path(path)
        if not ensure_initialized(path):
            return {
                "error": "Project not initialized. Run synapse_init first."
            }

        project = self._get_or_create(str(project_path))

        def _search():
            self._load_stores(project)
            if hybrid and project.code_graph is not None:
                from synapse.hybrid_search import HybridSearch

                searcher = HybridSearch(
                    project.vector_store, graph=project.code_graph
                )
                hybrid_result = searcher.search(query, top_k=limit)
                results = [
                    {
                        "node_id": r.node_id,
                        "hybrid_score": round(r.hybrid_score, 4),
                        "vector_score": round(r.vector_score, 4),
                        "graph_score": round(r.graph_score, 4),
                        "content": r.content[:500],
                        "relation_type": r.relation_type,
                    }
                    for r in hybrid_result.results
                ]
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
                        results.append(
                            {
                                "node_id": (
                                    metas[i].get("file_path", "")
                                    if i < len(metas)
                                    else ""
                                ),
                                "score": (
                                    round(1 / (1 + dists[i]), 4)
                                    if i < len(dists)
                                    else 0
                                ),
                                "content": doc[:500],
                            }
                        )
            return {"query": query, "results": results, "count": len(results)}

        return await self._run_sync(_search)

    async def get_context(
        self,
        file: str,
        path: str,
        depth: int = 2,
        max_files: int = 15,
    ) -> dict:
        """Build hierarchical context for a file."""
        project_path = validate_project_path(path)
        if not ensure_initialized(path):
            return {
                "error": "Project not initialized. Run synapse_init first."
            }

        def _context():
            from synapse.context_manager import ContextManager

            manager = ContextManager(
                Path(project_path), depth=depth, max_reference_files=max_files
            )
            result = manager.build_context(Path(file))
            output = {
                "global_context": result.global_context,
                "reference_context": result.reference_context,
                "active_context": result.active_context,
                "total_tokens": result.total_tokens,
                "token_breakdown": result.token_breakdown,
                "included_files": result.included_files,
            }
            if result.savings:
                output["savings"] = {
                    "original_tokens": getattr(result.savings, "original_tokens", 0),
                    "optimized_tokens": getattr(result.savings, "optimized_tokens", 0),
                    "reduction_ratio": getattr(result.savings, "reduction_ratio", 0),
                }
            return output

        return await self._run_sync(_context)

    async def get_graph(self, file: str, path: str) -> dict:
        """Get dependency relationships for a file."""
        project_path = validate_project_path(path)
        if not ensure_initialized(path):
            return {
                "error": "Project not initialized. Run synapse_init first."
            }

        project = self._get_or_create(str(project_path))

        def _graph():
            self._load_stores(project)
            if project.code_graph is None:
                return {"error": "Graph not built. Run synapse_analyze first."}
            related = project.code_graph.get_related_files(file)
            return {
                "file": file,
                "related_files": related,
                "count": len(related),
            }

        return await self._run_sync(_graph)

    async def skeletonize(self, file: str) -> dict:
        """Convert a file to its AST skeleton."""
        file_path = Path(file).resolve()
        if not file_path.is_file():
            return {"error": f"File not found: {file}"}

        def _skeleton():
            from synapse.structure.pruner import ASTSkeletonizer

            skeletonizer = ASTSkeletonizer()
            source = file_path.read_text()
            result = skeletonizer.skeletonize(source)
            return {
                "file": str(file_path),
                "skeleton": result.skeleton,
                "original_lines": result.original_lines,
                "skeleton_lines": result.skeleton_lines,
                "reduction_ratio": result.reduction_ratio,
            }

        return await self._run_sync(_skeleton)

    async def ask(self, query: str, path: str, think: bool = False) -> dict:
        """Generate an AI prompt with relevant context."""
        project_path = validate_project_path(path)
        if not ensure_initialized(path):
            return {
                "error": "Project not initialized. Run synapse_init first."
            }

        project = self._get_or_create(str(project_path))

        def _ask():
            self._load_stores(project)
            from synapse.hybrid_search import HybridSearch

            # Search for relevant files
            if project.code_graph is not None:
                searcher = HybridSearch(
                    project.vector_store, graph=project.code_graph
                )
                hybrid_result = searcher.search(query, top_k=5)
                results = hybrid_result.results
                relevant_files = [r.node_id for r in results]
            else:
                raw = project.vector_store.query(query, n_results=5)
                results = []
                relevant_files = []
                if raw and raw.get("documents"):
                    metas = (
                        raw["metadatas"][0] if raw.get("metadatas") else []
                    )
                    relevant_files = [
                        m.get("file_path", "") for m in metas
                    ]

            # Build context from top result
            context_text = ""
            if relevant_files and relevant_files[0]:
                try:
                    from synapse.context_manager import ContextManager

                    manager = ContextManager(
                        Path(project_path), depth=1, max_reference_files=10
                    )
                    ctx = manager.build_context(Path(relevant_files[0]))
                    context_text = ctx.formatted_output
                except Exception:
                    # Fallback to search snippets
                    if hasattr(results[0], "content"):
                        context_text = "\n\n".join(
                            r.content[:500] for r in results[:3]
                        )

            # Build prompt
            think_prefix = (
                "Think step by step before answering.\n\n" if think else ""
            )
            prompt = (
                f"{think_prefix}"
                f"## Context\n\n{context_text}\n\n"
                f"## Question\n\n{query}"
            )

            return {
                "prompt": prompt,
                "relevant_files": relevant_files,
                "token_estimate": len(prompt) // 4,
            }

        return await self._run_sync(_ask)

    async def watch(self, action: str, path: str) -> dict:
        """Manage file watcher (start/stop/status)."""
        project_path = validate_project_path(path)
        if not ensure_initialized(path):
            return {
                "error": "Project not initialized. Run synapse_init first."
            }

        def _watch():
            from synapse.watcher import SynapseWatcher

            if action == "status":
                return SynapseWatcher.get_status(str(project_path))
            elif action == "start":
                watcher = SynapseWatcher(str(project_path))
                watcher.start()
                return {
                    "status": "started",
                    "message": f"Watcher started for {project_path}",
                }
            elif action == "stop":
                import os
                import signal

                status = SynapseWatcher.get_status(str(project_path))
                pid = status.get("pid")
                if pid:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        return {
                            "status": "stopped",
                            "message": f"Watcher (PID {pid}) stopped",
                        }
                    except ProcessLookupError:
                        return {
                            "status": "not_running",
                            "message": "Watcher process not found",
                        }
                return {
                    "status": "not_running",
                    "message": "No watcher is running",
                }
            else:
                return {
                    "error": f"Unknown action: {action}. Use start/stop/status."
                }

        return await self._run_sync(_watch)

    async def cleanup(self):
        """Shutdown the thread pool executor."""
        self._executor.shutdown(wait=False)
        self._cache.clear()
