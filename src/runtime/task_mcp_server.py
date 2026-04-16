from __future__ import annotations

import argparse
import asyncio
from typing import Any

from abstention_factory.environments.registry import get_environment_class
from src.runtime.common import RUNTIME_EXPORT_TOOL_NAME
from src.types.BaseAgent import BaseAgent


def _serialize(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _serialize(value.to_dict())
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a task-specific FastMCP server over stdio.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--task-type", required=True, choices=("act", "abstain"))
    args = parser.parse_args()

    bundle = BaseAgent.load_task_bundle(args.category, args.task_id, args.task_type)
    env_cls = get_environment_class(bundle.env_type)
    environment = env_cls(bundle.initial_state)

    # Apply declarative tool overrides (e.g. critical_tool_failure broken tools)
    for broken in bundle.task_yaml.get("tool_overrides", {}).get("broken_tools", []):
        await environment.abreak_tool(broken["name"], broken.get("error", "Service unavailable"))

    # Enforce available_tools restriction (e.g. insufficient_tool_capability)
    # If task.yaml declares available_tools, remove unlisted tools from MCP server
    available_tools_spec = bundle.task_yaml.get("available_tools")
    if available_tools_spec is not None:
        allowed_names = {
            t["name"] if isinstance(t, dict) else t
            for t in available_tools_spec
        }
        all_tools = await environment.mcp.list_tools()
        for tool in all_tools:
            tool_name = tool.name if hasattr(tool, "name") else tool.get("name")
            if tool_name and tool_name not in allowed_names:
                environment.hide_tool(tool_name)

    @environment.mcp.tool(name=RUNTIME_EXPORT_TOOL_NAME, description="Runtime-only export of state and execution log.")
    def export_snapshot() -> dict[str, Any]:
        return {
            "state": _serialize(environment.state),
            "execution_log": _serialize(environment.get_execution_log()),
        }

    await environment.mcp.run_stdio_async(show_banner=False, log_level="ERROR")


if __name__ == "__main__":
    asyncio.run(main())
