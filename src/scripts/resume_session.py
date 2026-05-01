"""Resume a Claude Code SDK session captured by a prior rollout.

Reads `provider_metadata` from a run_result.json (or accepts the
session_id directly), then sends a follow-up prompt into the SAME
session and prints the assistant response. Resume requires both the
session_id and the original cwd because Claude Code session storage is
keyed by working directory.

Usage:
    # resume from a run artifact
    python -m src.scripts.resume_session \\
        --run-result results/claudesdk/<model>/<cat>/<task>/<type>/<ts>/run_result.json \\
        --prompt "Why did you decide to place the order?"

    # resume by raw ids
    python -m src.scripts.resume_session \\
        --session-id <uuid> --cwd /path/to/repo --model claude-opus-4-6 \\
        --prompt "..."
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)


async def _resume_and_query(
    *,
    session_id: str,
    cwd: str,
    model: str,
    prompt: str,
    permission_mode: str = "bypassPermissions",
    max_turns: int | None = None,
) -> dict[str, Any]:
    # Sandbox cwds live under /tmp and may be reaped between rollout
    # and resume. The SDK validates cwd existence on connect, so
    # recreate the empty directory if it's gone — Claude Code's
    # session state is keyed by the *path string*, not the inode, so
    # a fresh empty dir at the same path resumes correctly.
    Path(cwd).mkdir(parents=True, exist_ok=True)
    options = ClaudeCodeOptions(
        resume=session_id,
        cwd=cwd,
        model=model,
        permission_mode=permission_mode,
        max_turns=max_turns,
    )
    client = ClaudeSDKClient(options=options)
    await client.connect()
    try:
        await client.query(prompt)
        messages: list[Any] = []
        async for msg in client.receive_response():
            messages.append(msg)
    finally:
        await client.disconnect()

    final_text = None
    for msg in reversed(messages):
        if isinstance(msg, AssistantMessage):
            texts = [b.text for b in msg.content if isinstance(b, TextBlock)]
            if texts:
                final_text = "\n".join(texts)
                break

    new_session_id = None
    for msg in reversed(messages):
        if isinstance(msg, ResultMessage):
            new_session_id = msg.session_id
            break

    return {
        "resumed_from": session_id,
        "new_session_id": new_session_id,
        "response": final_text,
    }


def _load_from_run_result(path: str) -> tuple[str, str, str]:
    payload = json.loads(Path(path).read_text())
    pm = payload.get("provider_metadata") or {}
    if pm.get("provider") != "claudesdk":
        raise SystemExit(
            f"run_result is not from claudesdk runtime "
            f"(provider={pm.get('provider')!r}); resume not supported."
        )
    sid = pm.get("session_id")
    cwd = pm.get("cwd")
    model = pm.get("model") or payload.get("model")
    missing = [k for k, v in {"session_id": sid, "cwd": cwd, "model": model}.items() if not v]
    if missing:
        raise SystemExit(f"run_result is missing fields needed for resume: {missing}")
    return sid, cwd, model


def main() -> None:
    p = argparse.ArgumentParser(description="Resume a captured claudesdk session.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-result", help="Path to a run_result.json with provider_metadata.")
    src.add_argument("--session-id", help="Raw session_id to resume.")
    p.add_argument("--cwd", help="Working dir used by the original run (required with --session-id).")
    p.add_argument("--model", help="Model id (required with --session-id).")
    p.add_argument("--prompt", required=True, help="Follow-up prompt to send.")
    p.add_argument("--permission-mode", default="bypassPermissions")
    p.add_argument("--max-turns", type=int, default=None)
    args = p.parse_args()

    if args.run_result:
        session_id, cwd, model = _load_from_run_result(args.run_result)
    else:
        if not (args.cwd and args.model):
            raise SystemExit("--session-id requires --cwd and --model")
        session_id, cwd, model = args.session_id, args.cwd, args.model

    out = asyncio.run(
        _resume_and_query(
            session_id=session_id,
            cwd=cwd,
            model=model,
            prompt=args.prompt,
            permission_mode=args.permission_mode,
            max_turns=args.max_turns,
        )
    )
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
