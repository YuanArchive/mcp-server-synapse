"""Tests for synapse MCP server tools."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from mcp_server_synapse.utils import validate_project_path, ensure_initialized, format_error
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

    def test_ensure_initialized_true(self, tmp_path):
        (tmp_path / ".synapse").mkdir()
        assert ensure_initialized(str(tmp_path)) is True

    def test_ensure_initialized_false(self, tmp_path):
        assert ensure_initialized(str(tmp_path)) is False

    def test_format_error(self):
        result = json.loads(format_error("test error"))
        assert result == {"error": "test error"}


class TestSynapseWrapper:
    @pytest.fixture
    def wrapper(self):
        w = SynapseWrapper()
        yield w
        w._executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_init_project(self, wrapper, tmp_path):
        result = await wrapper.init_project(str(tmp_path))
        assert result["success"] is True
        assert (tmp_path / ".synapse").is_dir()
        assert (tmp_path / ".context").is_dir()
        assert (tmp_path / ".agent").is_dir()

    @pytest.mark.asyncio
    async def test_init_project_already_initialized(self, wrapper, tmp_path):
        (tmp_path / ".synapse").mkdir()
        (tmp_path / ".synapse" / "db").mkdir()
        (tmp_path / ".context").mkdir()
        (tmp_path / ".agent").mkdir()
        result = await wrapper.init_project(str(tmp_path))
        assert result["success"] is True
        assert "already initialized" in result["message"]

    @pytest.mark.asyncio
    async def test_init_project_invalid_path(self, wrapper):
        with pytest.raises(ValueError):
            await wrapper.init_project("/nonexistent/path")

    @pytest.mark.asyncio
    async def test_analyze_not_initialized(self, wrapper, tmp_path):
        result = await wrapper.analyze(str(tmp_path))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_search_not_initialized(self, wrapper, tmp_path):
        result = await wrapper.search("test query", str(tmp_path))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_context_not_initialized(self, wrapper, tmp_path):
        result = await wrapper.get_context("file.py", str(tmp_path))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_graph_not_initialized(self, wrapper, tmp_path):
        result = await wrapper.get_graph("file.py", str(tmp_path))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_skeletonize_file_not_found(self, wrapper):
        result = await wrapper.skeletonize("/nonexistent/file.py")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_ask_not_initialized(self, wrapper, tmp_path):
        result = await wrapper.ask("question", str(tmp_path))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_watch_not_initialized(self, wrapper, tmp_path):
        result = await wrapper.watch("status", str(tmp_path))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_watch_invalid_action(self, wrapper, tmp_path):
        (tmp_path / ".synapse").mkdir()
        with patch("mcp_server_synapse.synapse_wrapper.SynapseWrapper._run_sync") as mock_run:
            mock_run.return_value = {"error": "Unknown action: invalid. Use start/stop/status."}
            result = await wrapper.watch("invalid", str(tmp_path))
            assert "error" in result

    @pytest.mark.asyncio
    async def test_cleanup(self, wrapper):
        await wrapper.cleanup()
        assert len(wrapper._cache) == 0
