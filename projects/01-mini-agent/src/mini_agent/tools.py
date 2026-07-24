"""Tool registry — the Agent's "hands".

Every tool is a plain Python function with a schema that gets sent to the LLM.
The registry handles registration, schema generation, and execution.
"""

from __future__ import annotations

import ast
import operator
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


# ── Tool definition ──────────────────────────────────────────────


@dataclass
class Tool:
    """A tool that the Agent can call."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters
    func: Callable[..., Any]

    def to_openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema dict."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys()),
                },
            },
        }


# ── Registry ─────────────────────────────────────────────────────


class ToolRegistry:
    """Holds all available tools. Register → get schemas → execute by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schemas for all registered tools."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name with the given arguments.

        Returns the result as a string (suitable for feeding back to the LLM).
        """
        if name not in self._tools:
            return f"Error: tool '{name}' not found. Available: {list(self._tools.keys())}"

        try:
            result = self._tools[name].func(**arguments)
            return str(result)
        except Exception as e:
            return f"Tool error: {type(e).__name__}: {e}"

    def list_tools(self) -> str:
        """Return a human-readable summary of all tools."""
        lines = []
        for tool in self._tools.values():
            params = ", ".join(tool.parameters.keys())
            lines.append(f"  • {tool.name}({params}) — {tool.description}")
        return "\n".join(lines)


# ── Built-in tools ───────────────────────────────────────────────


# Safe math operators for calculator
_SAFE_OPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr_node: ast.AST) -> Any:
    """Recursively evaluate an AST node with only safe operations."""
    if isinstance(expr_node, ast.Constant):
        return expr_node.value
    if isinstance(expr_node, ast.BinOp):
        left = _safe_eval(expr_node.left)
        right = _safe_eval(expr_node.right)
        op_type = type(expr_node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _SAFE_OPS[op_type](left, right)
    if isinstance(expr_node, ast.UnaryOp):
        operand = _safe_eval(expr_node.operand)
        op_type = type(expr_node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _SAFE_OPS[op_type](operand)
    raise ValueError(f"Unsupported expression type: {type(expr_node).__name__}")


def tool_calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Supports: +, -, *, /, //, %, **, parentheses, integers and floats.
    Does NOT support: function calls, variables, imports (blocked for safety).

    Example: "(3 + 5) * 2 ** 3" → "64"
    """
    # Clean the input
    expression = expression.strip()
    # Remove any leading/trailing non-math chars that LLMs sometimes add
    expression = re.sub(r"^[=\s]+", "", expression)

    if not expression:
        return "Error: empty expression"

    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)

        # Format nicely
        if isinstance(result, float):
            if result == int(result):
                result = int(result)
            else:
                result = round(result, 10)

        return str(result)
    except (SyntaxError, ValueError, ZeroDivisionError) as e:
        return f"Error: {e}"


def tool_current_time() -> str:
    """Return the current date and time in a human-readable format."""
    now = datetime.now(timezone.utc)
    return (
        f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"Day of week: {now.strftime('%A')}\n"
        f"ISO format: {now.isoformat()}"
    )


def tool_read_file(file_path: str, max_lines: int = 200) -> str:
    """Read content from a text file.

    Only reads text files up to max_lines (safety limit).
    The file_path can be absolute or relative to the current working directory.

    Returns the file content or an error message.
    """
    if max_lines > 500:
        max_lines = 500  # Hard cap for safety

    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return f"Error: file not found: {path}"

    if not path.is_file():
        return f"Error: not a file: {path}"

    # Check file size (max ~1MB)
    size = path.stat().st_size
    if size > 1_048_576:
        return f"Error: file too large ({size:,} bytes). Max 1 MB."

    try:
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")
        total_lines = len(lines)

        if total_lines > max_lines:
            shown = lines[:max_lines]
            result = "\n".join(shown)
            return f"[Showing {max_lines} of {total_lines} lines]\n\n{result}"
        else:
            return content
    except UnicodeDecodeError:
        return f"Error: cannot read as text (binary file?): {path}"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except OSError as e:
        return f"Error reading file: {e}"


def tool_list_directory(directory: str = ".") -> str:
    """List the contents of a directory.

    Shows files and subdirectories with their sizes.
    """
    path = Path(directory).expanduser().resolve()

    if not path.exists():
        return f"Error: directory not found: {path}"
    if not path.is_dir():
        return f"Error: not a directory: {path}"

    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = [f"Contents of {path}:"]

        for entry in entries:
            if entry.name.startswith("."):
                continue  # Skip hidden files
            if entry.is_dir():
                lines.append(f"  📁 {entry.name}/")
            else:
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1_048_576:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / 1_048_576:.1f} MB"
                lines.append(f"  📄 {entry.name} ({size_str})")

        return "\n".join(lines)
    except PermissionError:
        return f"Error: permission denied: {path}"
    except OSError as e:
        return f"Error listing directory: {e}"


def tool_text_stats(text: str) -> str:
    """Analyze a piece of text: character count, word count, line count.

    Useful when the Agent needs to count or analyze its own output.
    """
    if not text:
        return "Empty text."

    chars = len(text)
    chars_no_space = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
    words = len(text.split())
    lines = text.count("\n") + 1

    return (
        f"Characters: {chars} ({chars_no_space} without whitespace)\n"
        f"Words: {words}\n"
        f"Lines: {lines}"
    )


# ── Build the registry with all built-in tools ────────────────────


def create_registry() -> ToolRegistry:
    """Create and return a ToolRegistry with all built-in tools."""
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="calculator",
            description="Evaluate a mathematical expression safely. "
            "Supports +, -, *, /, //, %, **, and parentheses. "
            "Use this for any arithmetic or numerical calculation.",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. '(300 + 45) * 6 / 2'",
                }
            },
            func=tool_calculator,
        )
    )

    registry.register(
        Tool(
            name="get_current_time",
            description="Get the current UTC date and time. "
            "Use this when you need to know the current time or date.",
            parameters={},
            func=tool_current_time,
        )
    )

    registry.register(
        Tool(
            name="read_file",
            description="Read the contents of a text file from disk. "
            "Use this to inspect files, source code, or documents. "
            "Only works with text files; binary files will error.",
            parameters={
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read (absolute or relative)",
                }
            },
            func=tool_read_file,
        )
    )

    registry.register(
        Tool(
            name="list_directory",
            description="List the contents of a directory. Shows files and subdirectories. "
            "Use this to explore file structure or find files.",
            parameters={
                "directory": {
                    "type": "string",
                    "description": "Directory path (default: current working directory)",
                }
            },
            func=tool_list_directory,
        )
    )

    registry.register(
        Tool(
            name="text_stats",
            description="Count characters, words, and lines in a piece of text. "
            "Use this when asked to count or analyze text.",
            parameters={
                "text": {
                    "type": "string",
                    "description": "The text to analyze",
                }
            },
            func=tool_text_stats,
        )
    )

    return registry
