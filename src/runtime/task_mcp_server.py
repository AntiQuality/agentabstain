from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from abstention_factory.environments.multi import build_multi_environment
from src.runtime.common import RUNTIME_EXPORT_TOOL_NAME
from src.types.BaseAgent import BaseAgent

# Optional state-dump hook for runtimes that don't keep an in-process MCP
# client (e.g. OpenClaw, which spawns the MCP server itself and gives us no
# way to call the hidden export tool). When RUNTIME_STATE_DUMP_PATH is set,
# every tool call writes the current state + execution_log to that file so
# the runtime can read it after the agent finishes.
RUNTIME_STATE_DUMP_PATH_ENV = "RUNTIME_STATE_DUMP_PATH"


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

    dump_path_str = os.environ.get(RUNTIME_STATE_DUMP_PATH_ENV)
    if dump_path_str:
        dump_path = Path(dump_path_str)
        dump_path.parent.mkdir(parents=True, exist_ok=True)

        def _write_dump() -> None:
            payload = {
                "state": _serialize(menv.state),
                "execution_log": _serialize(menv.get_execution_log()),
            }
            tmp = dump_path.with_suffix(dump_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False))
            tmp.replace(dump_path)

        # Hook every successful tool call. menv.mcp is a FastMCP-style server
        # that records calls via its execution log; we want the snapshot
        # written after each tool call so the latest state is always on
        # disk by the time the agent exits.
        original_call = menv.mcp.call_tool

        async def _call_with_dump(*args: Any, **kwargs: Any):
            try:
                return await original_call(*args, **kwargs)
            finally:
                try:
                    _write_dump()
                except Exception:
                    # Never let the dump break the rollout; the runtime can
                    # still call the export tool directly as a fallback.
                    pass

        menv.mcp.call_tool = _call_with_dump  # type: ignore[assignment]
        # Also write an initial snapshot so the runtime has something to
        # read even if the agent calls zero tools.
        _write_dump()

    await menv.mcp.run_stdio_async(show_banner=False, log_level="ERROR")


if __name__ == "__main__":
    asyncio.run(main())
