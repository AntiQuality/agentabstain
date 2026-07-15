# OpenClaw runtime for AgentAbstain

Drives [OpenClaw](https://openclaw.ai) (`openclaw agent --local --json`)
as an agent harness over AWS Bedrock. The motivation is breadth: one
runtime gives us Kimi K2.5, Kimi K2 Thinking, Qwen3 (multiple sizes),
DeepSeek V3.2, GLM 4.7, gpt-oss-20b/120b, Llama 3/4 — all on Bedrock
under a single `AWS_BEARER_TOKEN_BEDROCK`, with a uniform `--json`
envelope and the third-party `openclaw-trajectory` JSONL schema.

## Prerequisites

1. `npm i -g openclaw@2026.4.29` (or any 2026.4.x stable).
2. `openclaw onboard` once (creates `~/.openclaw/openclaw.json`).
3. Generate a Bedrock long-term API key at
   `https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/api-keys`.
4. Put it in a file the runtime can source. We keep it outside `.env`
   to avoid disturbing claudesdk's SigV4 path:
   ```
   # .aws_bearer.env
   AWS_BEARER_TOKEN_BEDROCK=ABSK...
   ```
5. Make sure `.env` carries `AWS_REGION=us-east-1` (already there).

The runtime explicitly unsets `AWS_ACCESS_KEY_ID/SECRET/SESSION_TOKEN`
before invoking openclaw so the bearer-token path is the only one
considered. claudesdk's SigV4 keys are untouched.

## Running

```bash
python -m src.scripts.run_inference \
  --runtime-config src/configs/openclaw_kimi-k2-5.yaml \
  --task-config src/configs/tasks_smoke.yaml
```

Available backbones (all `provider: openclaw` + `model:
amazon-bedrock/<bedrock-id>`):

| Yaml | Model id |
|---|---|
| `openclaw_kimi-k2-5.yaml` | `amazon-bedrock/moonshotai.kimi-k2.5` |
| `openclaw_deepseek-v3-2.yaml` | `amazon-bedrock/deepseek.v3.2` |
| `openclaw_glm-5.yaml` | `amazon-bedrock/zai.glm-5` |
| `openclaw_gpt-oss-120b.yaml` | `amazon-bedrock/openai.gpt-oss-120b-1:0` |
| `openclaw_minimax-m2-5.yaml` | `amazon-bedrock/minimax.minimax-m2.5` |
| `openclaw_deepseek-v4-pro.yaml` | `openrouter/deepseek/deepseek-v4-pro` |

Bedrock models authenticate via `AWS_BEARER_TOKEN_BEDROCK` as above;
the `openrouter/...` prefix routes through OpenRouter instead and reads
`OPENROUTER_API_KEY` from the environment or `.open_router.env`.

## How it works

For each rollout the runtime:

1. **Registers a per-rollout MCP server** in OpenClaw's global
   `mcp.servers.abstain_<id>` slot, pointing at
   `src.runtime.task_mcp_server` with `RUNTIME_STATE_DUMP_PATH`
   pre-set in env so the server writes state + execution_log to
   `<artifact_dir>/state_dump.json` after every tool call.
2. **Patches a one-shot agent** into `agents.list[]` with:
   - `model: amazon-bedrock/<id>`
   - `tools.profile=minimal` + `tools.alsoAllow=["bundle-mcp"]` —
     blocks every built-in (read/write/exec/web_fetch/...) leaving
     only `session_status` (inert) and the per-task MCP tools.
   - `skills: []` so no skill bootstrap is injected.
   - `systemPromptOverride: <task system_prompt>` so the model sees
     **only** the task's system prompt, not OpenClaw's 8k-char
     personal-assistant system prompt.
   - Empty workspace stubs (`AGENTS.md` / `SOUL.md` / `BOOTSTRAP.md` /
     ...) as belt-and-suspenders against bootstrap injection.
3. **Invokes** `openclaw agent --local --agent <id> --message <inst>
   --json --timeout <T>` with `OPENCLAW_TRAJECTORY_DIR=<artifact_dir>/
   trajectory/` so the third-party trajectory JSONL lands in our
   artifact tree.
4. **Reads** the state-dump file, parses the `--json` envelope for
   final text + `agentMeta.sessionId`, builds a `TaskRunResult`.
5. **Cleans up**: removes the rollout's MCP server entry and agent
   entry from `~/.openclaw/openclaw.json`, deletes the workspace.

## Why `bundle-mcp` is the keystone

OpenClaw's tool profiles whitelist by capability id. `minimal` lists
only `session_status`. The `bundle-mcp` virtual capability is what
exposes registered MCP servers to the agent — `coding` and `messaging`
profiles include it, `minimal` does not. Adding it via `alsoAllow`
gives a clean "MCP-only" surface for benchmarks.

## Concurrency caveat

`agents.list[]` and `mcp.servers` are global config keys. The runtime
reads → appends → writes when adding a rollout entry, which is not
atomic across processes. For `--workers 1` (default) this is safe.
For `--workers > 1` we need a coarse-grained lock around the
register/unregister window — TODO.

## Trajectory output

`<artifact_dir>/trajectory/<sessionId>.trajectory.jsonl` is the raw
OpenClaw trajectory (`{"traceSchema": "openclaw-trajectory",
"schemaVersion": 1}`) — one JSON object per line covering
`session.started`, `prompt.submitted`, `context.compiled`,
`tool.call`, `tool.result`, `model.completed`, `session.ended`. We
keep this alongside our own derived `trajectory.json` so reviewers
can audit either layer.
