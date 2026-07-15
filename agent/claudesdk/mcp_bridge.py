from __future__ import annotations

import asyncio
import os
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server import Server
from mcp.types import TextContent, Tool as McpTool

from claude_code_sdk import McpSdkServerConfig


class RuntimeMcpBridge:
    """Bridge that connects to a task MCP subprocess, filters hidden tools,
    and exposes the visible tools as an in-process SDK server for Claude SDK.
    """

    def __init__(
        self,
        connection_params: StdioServerParameters,
        hidden_tool_names: set[str] | None = None,
        server_name: str = "runtime",
    ):
        self._connection_params = connection_params
        self._hidden_tool_names = hidden_tool_names or set()
        self._server_name = server_name

        self._session: ClientSession | None = None
        self._tools: list[McpTool] = []
        self._connected = False
        self._connection_task: asyncio.Task[None] | None = None
        self._session_ready = asyncio.Event()
        self._shutdown_event = asyncio.Event()

    async def connect(self) -> None:
        if self._connected:
            return
        self._connection_task = asyncio.create_task(self._maintain_connection())
        try:
            await asyncio.wait_for(self._session_ready.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            if self._connection_task and not self._connection_task.done():
                self._connection_task.cancel()
                try:
                    await self._connection_task
                except asyncio.CancelledError:
                    pass
            raise TimeoutError(f"Timeout connecting to MCP server '{self._server_name}'")
        if not self._connected:
            raise ConnectionError(f"Failed to connect to MCP server '{self._server_name}'")

    async def _maintain_connection(self) -> None:
        try:
            env = {"PATH": os.environ.get("PATH", ""), **(self._connection_params.env or {})}
            params = StdioServerParameters(
                command=self._connection_params.command,
                args=self._connection_params.args,
                env=env,
                cwd=self._connection_params.cwd,
            )
            async with stdio_client(params) as (read, write):
                self._session = ClientSession(read, write)
                async with self._session:
                    await self._session.initialize()
                    response = await self._session.list_tools()
                    self._tools = list(response.tools)
                    self._connected = True
                    self._session_ready.set()
                    await self._shutdown_event.wait()
        except Exception:
            self._session_ready.set()
            raise

    def get_sdk_server_config(self) -> McpSdkServerConfig:
        """Return an in-process SDK server config that proxies visible tools."""
        if not self._connected:
            raise RuntimeError("Bridge is not connected. Call connect() first.")

        server = Server(self._server_name, version="1.0.0")
        visible_tools = [t for t in self._tools if t.name not in self._hidden_tool_names]
        tool_map = {t.name: t for t in visible_tools}

        @server.list_tools()
        async def list_tools() -> list[McpTool]:
            return [
                McpTool(
                    name=t.name,
                    description=t.description or "",
                    inputSchema=t.inputSchema if isinstance(t.inputSchema, dict) else {},
                )
                for t in visible_tools
            ]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            if name not in tool_map:
                raise ValueError(f"Tool '{name}' not found")
            result = await self._session.call_tool(name, arguments=arguments)
            content: list[TextContent] = []
            for item in result.content:
                text = getattr(item, "text", None) or str(item)
                content.append(TextContent(type="text", text=text))
            return content

        return McpSdkServerConfig(type="sdk", name=self._server_name, instance=server)

    async def call_hidden_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call a hidden tool directly via the MCP session."""
        if not self._session or not self._connected:
            raise RuntimeError("Bridge is not connected.")
        return await self._session.call_tool(name, arguments=arguments or {})

    async def close(self) -> None:
        self._shutdown_event.set()
        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()
            try:
                await self._connection_task
            except (asyncio.CancelledError, Exception):
                pass
        self._session = None
        self._connected = False
