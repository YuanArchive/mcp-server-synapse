"""Utility functions for path validation and error handling."""

import functools
import json
from pathlib import Path


def validate_project_path(path: str) -> Path:
    """Validate that path exists and is a directory."""
    p = Path(path).resolve()
    if not p.exists():
        raise ValueError(f"Path does not exist: {p}")
    if not p.is_dir():
        raise ValueError(f"Path is not a directory: {p}")
    return p


def format_error(message: str, hint: str | None = None) -> str:
    """Format error message as JSON string, with optional hint."""
    data = {"error": message}
    if hint:
        data["hint"] = hint
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def synapse_error_handler(func):
    """Decorator to catch synapse exceptions and return MCP-friendly error messages."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValueError as e:
            return format_error(str(e))
        except FileNotFoundError as e:
            return format_error(f"File not found: {e}")
        except Exception as e:
            error_type = type(e).__name__
            if "NotInitialized" in error_type or "ProjectNotInitialized" in error_type:
                return format_error(
                    "Project not initialized.",
                    hint="Run synapse_index first.",
                )
            if "IndexNotFound" in error_type:
                return format_error(
                    "Index not found.",
                    hint="Run synapse_index first.",
                )
            if "ParseError" in error_type:
                return format_error(f"Code parsing failed: {e}")
            if "VectorStoreError" in error_type:
                return format_error(f"Vector store error: {e}")
            if "GraphError" in error_type:
                return format_error(f"Graph error: {e}")
            return format_error(f"{error_type}: {e}")

    return wrapper
