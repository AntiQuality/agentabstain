from __future__ import annotations

import argparse
import asyncio
from typing import Any

from abstention_factory.environments.multi import build_multi_environment
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
    menv = build_multi_environment(bundle.env_types, bundle.initial_states)

    # Apply declarative tool overrides — tool names in the artifact are
    # already namespaced (e.g. "email.read_email"); abreak_tool splits and
    # routes to the correct sub-env.
    for broken in bundle.task_yaml.get("tool_overrides", {}).get("broken_tools", []):
        await menv.abreak_tool(broken["name"], broken.get("error", "Service unavailable"))

    available_tools_spec = bundle.task_yaml.get("available_tools")
    if available_tools_spec is not None:
        allowed_names = {
            t["name"] if isinstance(t, dict) else t
            for t in available_tools_spec
        }
        for schema in menv.get_tool_schemas():
            if schema["name"] not in allowed_names:
                menv.hide_tool(schema["name"])

    @menv.mcp.tool(name=RUNTIME_EXPORT_TOOL_NAME, description="Runtime-only export of state and execution log.")
    def export_snapshot() -> dict[str, Any]:
        return {
            "state": _serialize(menv.state),
            "execution_log": _serialize(menv.get_execution_log()),
        }

    await menv.mcp.run_stdio_async(show_banner=False, log_level="ERROR")


if __name__ == "__main__":
    asyncio.run(main())
