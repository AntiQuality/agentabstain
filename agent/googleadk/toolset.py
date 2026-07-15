from __future__ import annotations

from typing import Any

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from mcp.types import CallToolResult


class RuntimeMcpToolset(McpToolset):
    def __init__(self, *args: Any, hidden_tool_names: set[str] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._hidden_tool_names = hidden_tool_names or set()

    async def get_tools(self, readonly_context: ReadonlyContext | None = None) -> list[Any]:
        tools = await super().get_tools(readonly_context)
        return [tool for tool in tools if tool.name not in self._hidden_tool_names]

    async def call_hidden_tool(self, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        # The task environment state lives inside the stdio MCP subprocess, so
        # runtime export must reuse the same session manager instead of creating
        # a separate client connection.
        session = await self._mcp_session_manager.create_session()
        return await session.call_tool(name, arguments or {})
