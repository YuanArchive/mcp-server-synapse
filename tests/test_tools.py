"""Tests for synapse MCP server tools."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mcp_server_synapse.utils import validate_project_path, format_error
from mcp_server_synapse.synapse_wrapper import SynapseWrapper


class TestUtils:
    def test_validate_project_path_valid(self, tmp_path):
        result = validate_project_path(str(tmp_path))
        assert result == tmp_path.resolve()

    def test_validate_project_path_not_exists(self):
        with pytest.raises(ValueError, match="does not exist"):
            validate_project_path("/nonexistent/path")

    def test_validate_project_path_not_dir(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="not a directory"):
            validate_project_path(str(f))

    def test_format_error_basic(self):
        raw = format_error("test error")
        assert "  " not in raw  # compact JSON, no indentation
        result = json.loads(raw)
        assert result == {"error": "test error"}

    def test_format_error_with_hint(self):
        raw = format_error("test error", hint="do this")
        result = json.loads(raw)
        assert result == {"error": "test error", "hint": "do this"}

    def test_format_error_without_hint(self):
        result = json.loads(format_error("test error"))
        assert "hint" not in result


class TestSynapseWrapper:
    @pytest.fixture
    def wrapper(self):
        w = SynapseWrapper()
        yield w
        w._executor.shutdown(wait=False)

    # --- index ---

    @pytest.mark.asyncio
    async def test_index_auto_init(self, wrapper, tmp_path):
        """Index should auto-create .synapse/ on first run."""
        with patch.object(wrapper, "_run_sync") as mock_run:
            mock_run.return_value = {
                "status": "success",
                "files_analyzed": 10,
                "graph_nodes": 5,
                "graph_edges": 3,
                "is_fresh_index": True,
            }
            result = await wrapper.index(str(tmp_path))
            assert result["is_fresh_index"] is True

    @pytest.mark.asyncio
    async def test_index_incremental(self, wrapper, tmp_path):
        """Index on already-initialized project does incremental."""
        (tmp_path / ".synapse").mkdir()
        (tmp_path / ".synapse" / "db").mkdir()
        with patch.object(wrapper, "_run_sync") as mock_run:
            mock_run.return_value = {
                "status": "success",
                "files_analyzed": 2,
                "graph_nodes": 5,
                "graph_edges": 3,
                "is_fresh_index": False,
            }
            result = await wrapper.index(str(tmp_path))
            assert result["is_fresh_index"] is False

    @pytest.mark.asyncio
    async def test_index_invalid_path(self, wrapper):
        with pytest.raises(ValueError):
            await wrapper.index("/nonexistent/path")

    # --- search ---

    @pytest.mark.asyncio
    async def test_search_not_indexed(self, wrapper, tmp_path):
        result = await wrapper.search("test query", str(tmp_path))
        assert "error" in result
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_search_returns_results(self, wrapper, tmp_path):
        (tmp_path / ".synapse").mkdir()
        with patch.object(wrapper, "_run_sync") as mock_run:
            mock_run.return_value = {
                "query": "test",
                "results": [
                    {"file": "a.py", "score": 0.9, "snippet": "code..."},
                ],
                "count": 1,
            }
            result = await wrapper.search("test", str(tmp_path))
            assert result["count"] == 1
            assert result["results"][0]["file"] == "a.py"

    @pytest.mark.asyncio
    async def test_search_include_context(self, wrapper, tmp_path):
        (tmp_path / ".synapse").mkdir()
        with patch.object(wrapper, "_run_sync") as mock_run:
            mock_run.return_value = {
                "query": "test",
                "results": [
                    {
                        "file": "a.py",
                        "score": 0.9,
                        "snippet": "code...",
                        "dependencies": ["b.py", "c.py"],
                    },
                ],
                "count": 1,
            }
            result = await wrapper.search(
                "test", str(tmp_path), include_context=True
            )
            assert "dependencies" in result["results"][0]

    # --- context ---

    @pytest.mark.asyncio
    async def test_context_not_indexed(self, wrapper, tmp_path):
        result = await wrapper.get_context("file.py", str(tmp_path))
        assert "error" in result
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_context_returns_references(self, wrapper, tmp_path):
        (tmp_path / ".synapse").mkdir()
        with patch.object(wrapper, "_run_sync") as mock_run:
            mock_run.return_value = {
                "target_file": "file.py",
                "references": [
                    {"file": "dep.py", "skeleton": "class Foo: ..."},
                ],
                "reference_count": 1,
            }
            result = await wrapper.get_context("file.py", str(tmp_path))
            assert result["reference_count"] == 1
            assert result["references"][0]["file"] == "dep.py"

    # --- overview ---

    @pytest.mark.asyncio
    async def test_overview_not_indexed(self, wrapper, tmp_path):
        result = await wrapper.overview(str(tmp_path))
        assert "error" in result
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_overview_returns_data(self, wrapper, tmp_path):
        (tmp_path / ".synapse").mkdir()
        with patch.object(wrapper, "_run_sync") as mock_run:
            mock_run.return_value = {
                "project_path": str(tmp_path),
                "intelligence": "# Architecture\n...",
                "indexed_files": 10,
                "graph_nodes": 5,
                "graph_edges": 3,
                "file_tree": "src/\n  main.py",
            }
            result = await wrapper.overview(str(tmp_path))
            assert result["intelligence"] is not None
            assert result["indexed_files"] == 10
            assert "file_tree" in result

    @pytest.mark.asyncio
    async def test_overview_no_intelligence(self, wrapper, tmp_path):
        (tmp_path / ".synapse").mkdir()
        with patch.object(wrapper, "_run_sync") as mock_run:
            mock_run.return_value = {
                "project_path": str(tmp_path),
                "intelligence": None,
                "file_tree": "",
            }
            result = await wrapper.overview(str(tmp_path))
            assert result["intelligence"] is None

    # --- cleanup ---

    @pytest.mark.asyncio
    async def test_cleanup(self, wrapper):
        await wrapper.cleanup()
        assert len(wrapper._cache) == 0

    def test_load_ignore_patterns_from_file_and_env(
        self, wrapper, monkeypatch, tmp_path
    ):
        (tmp_path / ".synapseignore").write_text(
            "# comments are ignored\nbuild/\n*.generated.ts\n\n"
        )
        monkeypatch.setenv(
            "SYNAPSE_IGNORE_PATTERNS", "logs/,*.min.js,*.generated.ts"
        )

        patterns = wrapper._load_ignore_patterns(tmp_path)

        assert "build/" in patterns
        assert "*.generated.ts" in patterns
        assert "logs/" in patterns
        assert "*.min.js" in patterns
        # dedupe should keep only one copy
        assert patterns.count("*.generated.ts") == 1

    def test_is_ignored_file_with_directory_and_glob_patterns(
        self, wrapper, tmp_path
    ):
        generated = tmp_path / "generated" / "a.py"
        generated.parent.mkdir(parents=True)
        generated.write_text("x = 1")

        minified = tmp_path / "src" / "vendor.min.js"
        minified.parent.mkdir(parents=True, exist_ok=True)
        minified.write_text("const x=1;")

        keep = tmp_path / "src" / "main.py"
        keep.write_text("print('ok')")

        patterns = ["generated/", "*.min.js"]
        assert wrapper._is_ignored_file(generated, tmp_path, patterns) is True
        assert wrapper._is_ignored_file(minified, tmp_path, patterns) is True
        assert wrapper._is_ignored_file(keep, tmp_path, patterns) is False

    def test_apply_index_filters_wraps_scan_files(
        self, wrapper, monkeypatch, tmp_path
    ):
        (tmp_path / ".synapseignore").write_text("generated/\n")
        monkeypatch.setenv("SYNAPSE_IGNORE_PATTERNS", "*.min.js")
        monkeypatch.setenv("SYNAPSE_EXTRA_EXCLUDE_DIRS", "build")

        files = [
            tmp_path / "generated" / "a.py",
            tmp_path / "src" / "vendor.min.js",
            tmp_path / "src" / "main.py",
        ]
        for file_path in files:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("x = 1")

        class FakeAnalyzer:
            def __init__(self):
                self.exclude_dirs = [".git"]

            def scan_files(self, *args, **kwargs):
                return files

        analyzer = FakeAnalyzer()
        patterns = wrapper._apply_index_filters(analyzer, tmp_path)
        filtered = analyzer.scan_files()

        assert "build" in analyzer.exclude_dirs
        assert "generated/" in patterns
        assert "*.min.js" in patterns
        assert filtered == [tmp_path / "src" / "main.py"]

    def test_get_or_create_lazy_analyzer(self, wrapper, tmp_path):
        project = wrapper._get_or_create(str(tmp_path))
        assert project.analyzer is None

    def test_cache_limit_evicts_old_project(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SYNAPSE_MAX_CACHED_PROJECTS", "1")

        project_a = tmp_path / "project_a"
        project_b = tmp_path / "project_b"
        project_a.mkdir()
        project_b.mkdir()

        local_wrapper = SynapseWrapper()
        local_wrapper._get_or_create(str(project_a))
        local_wrapper._get_or_create(str(project_b))

        assert len(local_wrapper._cache) == 1
        assert str(project_b.resolve()) in local_wrapper._cache
        assert str(project_a.resolve()) not in local_wrapper._cache
        local_wrapper._executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_index_respects_index_worker_setting(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("SYNAPSE_INDEX_WORKERS", "3")

        local_wrapper = SynapseWrapper()
        fake_analyzer = MagicMock()
        fake_analyzer.analyze.return_value = {
            "status": "success",
            "files_analyzed": 1,
            "graph_nodes": 0,
            "graph_edges": 0,
        }
        fake_analyzer.code_graph = None
        fake_analyzer.vector_store = None

        with patch.object(
            local_wrapper, "_get_analyzer", return_value=fake_analyzer
        ), patch.object(local_wrapper, "_ensure_initialized", return_value=True):
            result = await local_wrapper.index(str(tmp_path), full=True)

        fake_analyzer.analyze.assert_called_once_with(
            json_output=True, num_workers=3
        )
        assert result["status"] == "success"
        local_wrapper._executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_index_returns_ignore_metadata(self, tmp_path):
        local_wrapper = SynapseWrapper()
        fake_analyzer = MagicMock()
        fake_analyzer.analyze.return_value = {
            "status": "success",
            "files_analyzed": 1,
            "graph_nodes": 0,
            "graph_edges": 0,
        }
        fake_analyzer.code_graph = None
        fake_analyzer.vector_store = None

        with patch.object(
            local_wrapper, "_get_analyzer", return_value=fake_analyzer
        ), patch.object(
            local_wrapper, "_ensure_initialized", return_value=True
        ), patch.object(
            local_wrapper, "_apply_index_filters", return_value=["logs/", "*.log"]
        ):
            result = await local_wrapper.index(str(tmp_path), full=True)

        assert "ignore" in result
        assert result["ignore"]["pattern_count"] == 2
        assert result["ignore"]["patterns_preview"] == ["logs/", "*.log"]
        assert "ignore_file" in result["ignore"]
        local_wrapper._executor.shutdown(wait=False)


class TestFileTree:
    def test_build_file_tree(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')")
        (tmp_path / "README.md").write_text("# Hello")

        wrapper = SynapseWrapper()
        tree = wrapper._build_file_tree(tmp_path, max_depth=3)
        assert "src/" in tree
        assert "main.py" in tree
        assert "README.md" in tree
        wrapper._executor.shutdown(wait=False)

    def test_build_file_tree_skips_hidden(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "real.py").write_text("x = 1")

        wrapper = SynapseWrapper()
        tree = wrapper._build_file_tree(tmp_path, max_depth=3)
        assert ".git" not in tree
        assert "__pycache__" not in tree
        assert "real.py" in tree
        wrapper._executor.shutdown(wait=False)

    def test_build_file_tree_max_depth(self, tmp_path):
        (tmp_path / "a" / "b" / "c" / "d").mkdir(parents=True)
        (tmp_path / "a" / "b" / "c" / "d" / "deep.py").write_text("x")

        wrapper = SynapseWrapper()
        tree = wrapper._build_file_tree(tmp_path, max_depth=2)
        assert "a/" in tree
        assert "b/" in tree
        # depth 2 means we see a/ and b/ but not c/
        assert "deep.py" not in tree
        wrapper._executor.shutdown(wait=False)
