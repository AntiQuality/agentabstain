"""OpenClaw-driven runtime for the AgentAbstain benchmark.

OpenClaw (https://openclaw.ai) is invoked as a subprocess (`openclaw agent
--local --json`). For each rollout we:

1. Register a per-rollout entry under the global `mcp.servers.<name>`
   slot pointing at our task MCP server, with `RUNTIME_STATE_DUMP_PATH`
   in the server env so the server writes state + execution_log to a
   file we can read after.
2. Patch a per-rollout `agents.list[]` entry that:
   - Pins `model` to the Bedrock id (e.g. `amazon-bedrock/moonshotai.kimi-k2.5`).
   - Sets `tools.profile=minimal` + `tools.alsoAllow=["bundle-mcp"]` so
     the agent only sees our MCP tools (plus session_status, which is
     unavoidable but inert).
   - Uses an isolated empty workspace (no AGENTS.md/SOUL.md/etc.).
   - Sets `skills: []` so no skill bootstrap is injected.
3. Invokes `openclaw agent --local --agent <id> --message <instruction>
   --json --timeout <T>` with a clean env that only carries
   AWS_BEARER_TOKEN_BEDROCK + AWS_REGION (we explicitly omit
   AWS_ACCESS_KEY_ID/SECRET so OpenClaw routes via the Bedrock bearer
   token, leaving the SigV4 keys free for claude code sdk).
4. Reads the state dump file the MCP server wrote, plus the
   `OPENCLAW_TRAJECTORY_DIR/<sessionId>.trajectory.jsonl` file.
5. Cleans up: removes the per-rollout entries from openclaw.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

from src.runtime.common import (
    BatchRunSummary,
    build_runtime_server_args,
    build_task_run_result,
    coerce_final_output,
    default_runtime_export_payload,
    run_batch,
)
from src.types.BaseAgent import BaseAgent, TaskBundle, TaskRunResult


# Default openclaw CLI binary; users on a custom path can override.
OPENCLAW_BIN = os.environ.get("OPENCLAW_BIN", "openclaw")

# Top-level config-key timeouts.
OPENCLAW_DEFAULT_AGENT_TIMEOUT = 300


async def run_openclaw_task(agent: BaseAgent, bundle: TaskBundle, repo_root: Path) -> TaskRunResult:
    artifact_dir = agent.build_artifact_dir(bundle.category, bundle.task_id, bundle.task_type)

    # Per-rollout unique handles. OpenClaw's mcp.servers and agents.list
    # are global, so concurrent rollouts must not collide.
    rollout_id = secrets.token_hex(6)
    mcp_server_name = f"abstain_{rollout_id}"
    agent_id = f"abstain_agent_{rollout_id}"
    sandbox_workspace = Path("/tmp") / f"openclaw_workspace_{rollout_id}"
    sandbox_agent_dir = Path("/tmp") / f"openclaw_agentdir_{rollout_id}"
    state_dump_path = artifact_dir / "state_dump.json"
    trajectory_dir = artifact_dir / "trajectory"

    sandbox_workspace.mkdir(parents=True, exist_ok=True)
    sandbox_agent_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    # Empty out workspace bootstrap files so they inject 0 chars into
    # the system prompt. OpenClaw auto-creates these the first time it
    # touches the workspace; pre-creating empty stubs is the cleanest
    # way to suppress them without hacking the schema.
    for stub in (
        "AGENTS.md", "SOUL.md", "TOOLS.md", "IDENTITY.md", "USER.md",
        "HEARTBEAT.md", "BOOTSTRAP.md",
    ):
        (sandbox_workspace / stub).write_text("")

    final_output: str | None = None
    export_payload: dict[str, Any] | None = None
    run_error: str | None = None
    session_id: str | None = None
    openclaw_meta: dict[str, Any] | None = None

    python_bin = sys.executable
    server_args = build_runtime_server_args(bundle)
    server_payload = {
        "command": python_bin,
        "args": server_args,
        "cwd": str(repo_root),
        "env": {
            "RUNTIME_STATE_DUMP_PATH": str(state_dump_path),
            # Forward PYTHONPATH so the spawned server can import
            # `src.runtime.task_mcp_server` even when openclaw cwd is elsewhere.
            "PYTHONPATH": str(repo_root),
        },
    }

    model_ref = (
        agent.model
        if agent.model.startswith("amazon-bedrock/")
        else f"amazon-bedrock/{agent.model}"
    )

    agent_entry: dict[str, Any] = {
        "id": agent_id,
        "name": agent_id,
        "workspace": str(sandbox_workspace),
        "agentDir": str(sandbox_agent_dir),
        "model": model_ref,
        "skills": [],
        "tools": {
            "profile": "minimal",
            "alsoAllow": ["bundle-mcp"],
        },
        "systemPromptOverride": bundle.task_yaml["system_prompt"],
    }

    try:
        await _register_mcp_server(mcp_server_name, server_payload)
        await _add_agent_entry(agent_entry)

        try:
            run_outcome = await _invoke_openclaw_agent(
                agent_id=agent_id,
                instruction=bundle.task_yaml["instruction"],
                trajectory_dir=trajectory_dir,
                timeout_seconds=max(60, agent.max_turns * 30),
            )
            final_output = coerce_final_output(run_outcome.get("final_output"))
            session_id = run_outcome.get("session_id")
            openclaw_meta = run_outcome.get("meta")
            if run_outcome.get("error"):
                run_error = run_outcome["error"]
        except Exception as exc:
            run_error = f"openclaw agent invocation failed: {exc}"

        try:
            export_payload = _read_state_dump(state_dump_path)
        except Exception as exc:
            if run_error is None:
                raise
            run_error = f"{run_error}; state dump read failed: {exc}"
    finally:
        try:
            await _unregister_mcp_server(mcp_server_name)
        except Exception:
            pass
        try:
            await _remove_agent_entry(agent_id)
        except Exception:
            pass
        # Per-rollout sandboxes are cheap and short-lived; clean up to
        # keep /tmp tidy under high-fanout runs.
        for path in (sandbox_workspace, sandbox_agent_dir):
            try:
                shutil.rmtree(path)
            except Exception:
                pass

    provider_metadata: dict[str, Any] = {
        "provider": "openclaw",
        "session_id": session_id,
        "model": agent.model,
        "model_ref": model_ref,
        "rollout_id": rollout_id,
        "trajectory_dir": str(trajectory_dir),
        "state_dump_path": str(state_dump_path),
        "openclaw_meta": openclaw_meta,
    }

    return build_task_run_result(
        agent=agent,
        bundle=bundle,
        artifact_dir=artifact_dir,
        final_output=final_output,
        export_payload=export_payload,
        run_error=run_error,
        provider_metadata=provider_metadata,
    )


async def _register_mcp_server(name: str, payload: dict[str, Any]) -> None:
    proc = await asyncio.create_subprocess_exec(
        OPENCLAW_BIN, "mcp", "set", name, json.dumps(payload),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"openclaw mcp set failed (rc={proc.returncode}): "
            f"{stderr.decode(errors='replace')[-2000:]}"
        )


async def _unregister_mcp_server(name: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        OPENCLAW_BIN, "mcp", "unset", name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()


async def _add_agent_entry(entry: dict[str, Any]) -> None:
    """Append `entry` to agents.list via `openclaw config patch`.

    OpenClaw's `patch` defaults to merging arrays only when the result
    keeps every existing id (`Refusing to replace agents.list; it would
    remove existing entries: ...`). Since we read the current list and
    write a superset (existing entries minus a same-id stale entry,
    plus our new one), the regular semantics may still trip when our
    entry shares an id with a partial-state leftover from a prior
    crash. Pass `--replace-path agents.list` so the array is replaced
    intentionally.
    """
    current = await _read_agents_list()
    new_list = [e for e in current if e.get("id") != entry["id"]] + [entry]
    patch = {"agents": {"list": new_list}}
    await _config_patch(patch, replace_paths=["agents.list"])


async def _remove_agent_entry(agent_id: str) -> None:
    current = await _read_agents_list()
    pruned = [e for e in current if e.get("id") != agent_id]
    if pruned == current:
        return
    if not pruned:
        # OpenClaw schema requires at least one entry; fall back to a
        # bare `main` placeholder so the file remains valid.
        pruned = [{"id": "main"}]
    await _config_patch({"agents": {"list": pruned}}, replace_paths=["agents.list"])


async def _read_agents_list() -> list[dict[str, Any]]:
    proc = await asyncio.create_subprocess_exec(
        OPENCLAW_BIN, "config", "get", "agents.list", "--json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"openclaw config get agents.list failed (rc={proc.returncode}): "
            f"{stderr.decode(errors='replace')[-1000:]}"
        )
    raw = stdout.decode(errors="replace").strip() or "[]"
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise RuntimeError(f"unexpected agents.list payload: {parsed!r}")
    return parsed


async def _config_patch(patch: dict[str, Any], replace_paths: list[str] | None = None) -> None:
    args = [OPENCLAW_BIN, "config", "patch", "--stdin"]
    for rp in replace_paths or []:
        args.extend(["--replace-path", rp])
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(json.dumps(patch).encode())
    if proc.returncode != 0:
        raise RuntimeError(
            f"openclaw config patch failed (rc={proc.returncode}): "
            f"{stderr.decode(errors='replace')[-2000:]}"
        )


async def _invoke_openclaw_agent(
    agent_id: str,
    instruction: str,
    trajectory_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    # OpenClaw must use AWS_BEARER_TOKEN_BEDROCK only — strip access key
    # creds so SigV4 path doesn't accidentally take over (and so we
    # don't leak them to a child process unnecessarily).
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        env.pop(key, None)
    env["OPENCLAW_TRAJECTORY"] = "1"
    env["OPENCLAW_TRAJECTORY_DIR"] = str(trajectory_dir)

    args = [
        OPENCLAW_BIN, "agent",
        "--local",
        "--agent", agent_id,
        "--message", instruction,
        "--json",
        "--timeout", str(timeout_seconds),
    ]

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds + 60,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {"error": f"openclaw agent timed out after {timeout_seconds + 60}s", "final_output": None}

    if proc.returncode != 0:
        return {
            "error": (
                f"openclaw agent exited rc={proc.returncode}: "
                f"{stderr.decode(errors='replace')[-2000:]}"
            ),
            "final_output": None,
        }

    raw = stdout.decode(errors="replace").strip()
    if not raw:
        return {"error": "openclaw agent produced no JSON output", "final_output": None}

    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"error": f"could not parse openclaw JSON: {exc}; raw={raw[:1000]}", "final_output": None}

    payloads = envelope.get("payloads") or []
    final_output = None
    if payloads and isinstance(payloads, list):
        first = payloads[0]
        if isinstance(first, dict):
            final_output = first.get("text")

    meta = envelope.get("meta") or {}
    session_id = None
    agent_meta = meta.get("agentMeta") if isinstance(meta, dict) else None
    if isinstance(agent_meta, dict):
        session_id = agent_meta.get("sessionId")

    return {
        "final_output": final_output,
        "session_id": session_id,
        "meta": meta,
    }


def _read_state_dump(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"state dump must be a JSON object: {path}")
    if "state" not in parsed or "execution_log" not in parsed:
        raise ValueError(f"state dump missing keys: {path}")
    return {
        "state": parsed["state"],
        "execution_log": parsed["execution_log"],
    }
