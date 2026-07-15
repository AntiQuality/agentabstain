"""OpenClaw-driven runtime for the AgentAbstain benchmark.

OpenClaw (https://openclaw.ai) is invoked as a subprocess (`openclaw agent
--local --json`). For each rollout we:

1. Allocate a per-rollout OpenClaw profile (`--profile abstain_<rollout_id>`)
   so all openclaw state -- agents.list, mcp.servers, sessions, logs --
   lives at `~/.openclaw-abstain_<rollout_id>/` and never overlaps with
   other rollouts or the user's interactive openclaw config. This is
   what makes concurrent rollouts safe: bundle-mcp's plugin loader
   reads `mcp.servers` from the active profile only, so it cannot leak
   tools across rollouts (tools share `pluginId="bundle-mcp"` so any
   global registration would be visible to every concurrent agent via
   the `tools.alsoAllow=["bundle-mcp"]` allow-list, which there's no
   way to scope at the policy layer).
2. Register a `mcp.servers.<name>` entry in this profile pointing at
   our task MCP server, with `RUNTIME_STATE_DUMP_PATH` in the server
   env so the server writes state + execution_log to a file we can
   read after.
3. Patch a per-rollout `agents.list[]` entry that:
   - Pins `model` to the Bedrock id (e.g. `amazon-bedrock/moonshotai.kimi-k2.5`).
   - Sets `tools.profile=minimal` + `tools.alsoAllow=["bundle-mcp"]` so
     the agent only sees our MCP tools (plus session_status, which is
     unavoidable but inert).
   - Uses an isolated empty workspace (no AGENTS.md/SOUL.md/etc.).
   - Sets `skills: []` so no skill bootstrap is injected.
4. Invokes `openclaw --profile <p> agent --local --agent <id>
   --message <instruction> --json --timeout <T>` with a clean env that
   only carries AWS_BEARER_TOKEN_BEDROCK + AWS_REGION (we explicitly
   omit AWS_ACCESS_KEY_ID/SECRET so OpenClaw routes via the Bedrock
   bearer token, leaving the SigV4 keys free for claude code sdk).
5. Reads the state dump file the MCP server wrote, plus the
   `OPENCLAW_TRAJECTORY_DIR/<sessionId>.trajectory.jsonl` file.
6. Cleans up: removes the entire `~/.openclaw-abstain_<rollout_id>/`
   profile dir. No global config touched.
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


def _load_provider_dotenvs() -> None:
    """Pull provider API keys from per-provider dotenv files at import time.

    Some non-Bedrock upstreams (currently OpenRouter) read their key from a
    `.<provider>.env` file the user keeps out of shell rc, e.g.
    `.open_router.env` with `OPENROUTER_API_KEY=...`. We surface them into
    `os.environ` so openclaw's `{source: env, id: ...}` apiKey resolution
    works without each runner script having to `set -a; source ...`.
    Existing exported values win — we never overwrite.
    """
    repo_root = Path(__file__).resolve().parents[2]
    for filename in (".open_router.env",):
        path = repo_root / filename
        if not path.exists():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_provider_dotenvs()

# Top-level config-key timeouts.
OPENCLAW_DEFAULT_AGENT_TIMEOUT = 300

# Short aliases used for the MCP-exposed tool namespace prefix when running
# under openclaw. OpenClaw sanitizes & truncates tool names to 64 chars total
# (`<server>__<env-tool>`) for Bedrock-safety; with the canonical 12-hex
# rollout id our serverName is 20 chars, leaving 42 chars for `<env-tool>`.
# Long env names (e.g. `industrial_and_infrastructure_control` at 37 chars)
# devour that budget and force per-tool collision suffixes (`-ev-2`, `-ev-3`,
# ...) that the model can't reproduce. Aliasing here keeps the agent-visible
# names short and reproducible while internal state keys, execution_log
# entries, and DAG `critical_actions` all keep the canonical full env name.
# Pair this with the 8-hex rollout id below (serverName=16 chars → 46-char
# tool-fragment budget) and every `<alias>.<func>` fits without truncation.
OPENCLAW_ENV_ALIASES: dict[str, str] = {
    "agriculture_and_yield": "agri",
    "astrology_charting": "astro",
    "autonomous_delivery_and_routing": "adr",
    "calendar": "cal",
    "chat_and_group_management": "chat",
    "clinical_records_and_claims": "clinical",
    "compliance_and_contracts": "compl",
    "consumer_banking": "bank",
    "consumer_health_portal": "health",
    "crm_and_company_lookup": "crm",
    "device_privacy_and_focus": "device",
    "disaster_relief_operations": "disaster",
    "document_authoring_and_publication": "docpub",
    "documents_and_analysis": "docs",
    "education_and_campus_portals": "edu",
    "filesystem": "fs",
    "fitness_and_wellness_logs": "fitness",
    "flight_and_travel_management": "travel",
    "gmail_and_email_records": "gmail",
    "home_medication_inventory": "med",
    "identity_credit_and_collections": "identity",
    "industrial_and_infrastructure_control": "industrial",
    "maps_and_navigation": "maps",
    "metrics_and_spreadsheet_analysis": "metrics",
    "notes_and_reference": "notes",
    "personal_profile_and_contacts": "profile",
    "phone_and_messages": "phone",
    "project_logs": "projlogs",
    "retail_orders": "retail",
    "science_and_environment_data": "science",
    "security_and_privacy_admin": "security",
    "smart_home": "smart",
    "social_media_dataset_analysis": "social",
    "spotify": "spotify",
    "store_procurement_and_inventory": "store",
    "system_operations": "sysops",
    "trading_and_portfolio": "trading",
    "vehicle_status_and_control": "vehicle",
    "venmo_and_shared_expenses": "venmo",
    "weather_and_alerts": "weather",
    "web_and_cms": "web",
    "workforce_and_hr": "workforce",
}


async def run_openclaw_task(agent: BaseAgent, bundle: TaskBundle, repo_root: Path) -> TaskRunResult:
    artifact_dir = agent.build_artifact_dir(bundle.category, bundle.task_id, bundle.task_type)

    # 8-hex (vs the previous 12-hex) shortens serverName from 20 to 16 chars,
    # which raises the per-tool name budget from 42 to 46 chars after openclaw's
    # 64-char total cap — enough for every `<alias>.<func>` to fit without the
    # collision-suffix soup we saw with full env names. 8 hex (~4.3B values) is
    # plenty unique within the lifetime of one batch (only simultaneous active
    # rollouts collide-risk; profile dirs are deleted in `finally`).
    rollout_id = secrets.token_hex(4)
    profile_name = f"abstain_{rollout_id}"
    mcp_server_name = f"abstain_{rollout_id}"
    agent_id = f"abstain_agent_{rollout_id}"
    profile_dir = Path.home() / f".openclaw-{profile_name}"
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
            # Enable openclaw-specific MCP-side env aliasing so tool names fit
            # under Bedrock's 64-char cap; see OPENCLAW_ENV_ALIASES above.
            "RUNTIME_ENV_ALIASES_JSON": json.dumps(OPENCLAW_ENV_ALIASES),
        },
    }

    # Honor an explicit `<provider>/<model>` prefix (e.g. `openrouter/...`,
    # `fireworks/...`); only fall back to `amazon-bedrock/` when bare.
    model_ref = agent.model if "/" in agent.model else f"amazon-bedrock/{agent.model}"
    upstream_provider = model_ref.split("/", 1)[0]

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
        # The paper's evaluation campaign ran on openclaw 2026.4.29 (see
        # agent/openclaw/README.md), which is the version this harness targets:
        # npm i -g openclaw@2026.4.29. openclaw >= 2026.7 removed
        # agents.list[].systemPromptOverride from the config schema (per-agent
        # instructions moved to workspace bootstrap files), so newer CLI
        # versions reject this key.
        "systemPromptOverride": bundle.task_yaml["system_prompt"],
    }

    try:
        await _register_mcp_server(profile_name, mcp_server_name, server_payload)
        await _add_agent_entry(profile_name, agent_entry)
        _check_provider_credentials(upstream_provider)

        try:
            run_outcome = await _invoke_openclaw_agent(
                profile_name=profile_name,
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
        # Profile dir is the source of truth for openclaw state. Removing
        # it cleans up agents.list, mcp.servers, sessions, and logs in
        # one shot -- no per-key unset needed and no chance of leaking
        # global state on kill -9.
        for path in (profile_dir, sandbox_workspace, sandbox_agent_dir):
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


def _profile_args(profile_name: str) -> list[str]:
    return [OPENCLAW_BIN, "--profile", profile_name]


async def _register_mcp_server(profile_name: str, name: str, payload: dict[str, Any]) -> None:
    proc = await asyncio.create_subprocess_exec(
        *_profile_args(profile_name), "mcp", "set", name, json.dumps(payload),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"openclaw mcp set failed (rc={proc.returncode}): "
            f"{stderr.decode(errors='replace')[-2000:]}"
        )


# Map openclaw upstream-provider id → env var that openclaw auto-detects
# at request time. Listed here only so we can fail fast at rollout start
# with a clear error instead of waiting for the subprocess to crash.
# OpenClaw's built-in `auth status` already binds these env vars
# automatically — we don't need to patch `models.providers.<name>` (and
# in fact must not, because that path triggers schema validation that
# requires baseUrl + models, while the built-in catalog provides both).
# Bedrock is intentionally absent: it uses AWS_BEARER_TOKEN_BEDROCK /
# SigV4 wired separately in `_invoke_openclaw_agent`.
_PROVIDER_ENV_KEYS: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
}


def _check_provider_credentials(upstream_provider: str) -> None:
    """Fail fast if the env var an upstream provider needs is unset.

    OpenClaw auto-binds `<PROVIDER>_API_KEY` env vars itself, so we just
    confirm the var is in the env we'll forward to the subprocess.
    """
    env_key = _PROVIDER_ENV_KEYS.get(upstream_provider)
    if env_key is None:
        return
    if not os.environ.get(env_key):
        raise RuntimeError(
            f"openclaw upstream provider {upstream_provider!r} requires env var "
            f"{env_key} but it is unset (load .open_router.env or export it)"
        )


async def _add_agent_entry(profile_name: str, entry: dict[str, Any]) -> None:
    """Write the per-rollout agent entry to the per-profile config.

    With per-profile state isolation (`--profile abstain_<rollout_id>`),
    `agents.list` starts empty for each rollout, so we just write
    `[entry]` directly. Pass `--replace-path agents.list` so openclaw's
    array-merge guardrail (which refuses to drop existing ids) doesn't
    kick in if the profile dir was somehow pre-populated.
    """
    patch = {"agents": {"list": [entry]}}
    await _config_patch(profile_name, patch, replace_paths=["agents.list"])


async def _config_patch(
    profile_name: str,
    patch: dict[str, Any],
    replace_paths: list[str] | None = None,
) -> None:
    args = [*_profile_args(profile_name), "config", "patch", "--stdin"]
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
    profile_name: str,
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
        *_profile_args(profile_name),
        "agent",
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

    meta = envelope.get("meta") or {}
    payloads = envelope.get("payloads") or []
    final_output = None
    # OpenClaw's payloads array is the per-turn sequence of assistant
    # outputs in chronological order, so payloads[0] is typically the
    # pre-tool-call preamble ("Let me check...") rather than the answer.
    # `meta.finalAssistantVisibleText` is openclaw's own designation of
    # the rendered final answer; fall back to the last payload, then
    # the first, only if it's missing.
    if isinstance(meta, dict):
        candidate = meta.get("finalAssistantVisibleText")
        if isinstance(candidate, str) and candidate:
            final_output = candidate
    if final_output is None and isinstance(payloads, list) and payloads:
        for entry in reversed(payloads):
            if isinstance(entry, dict):
                text = entry.get("text")
                if isinstance(text, str) and text:
                    final_output = text
                    break


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
