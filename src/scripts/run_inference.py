from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE importing any agent so provider-routing flags
# (CLAUDE_CODE_USE_BEDROCK=1, AWS_BEARER_TOKEN_BEDROCK, AWS_REGION,
# OPENAI_API_KEY) reach the Claude Code SDK / OpenAI SDK at import
# time. Without this, the SDK subprocess inherits only the shell env
# and falls back to native Anthropic, which rejects Bedrock model
# IDs like `us.anthropic.claude-opus-4-7` with a generic "model may
# not exist" error. `override=True` lets .env values shadow shell
# defaults so the env block in .env is the source of truth.
load_dotenv(override=True)

from agent import ClaudeSDKAgent, GoogleADKAgent, OpenAISDKAgent, OpenClawAgent
from src.runtime.config import InferenceConfig
from src.runtime.common import run_batch

PROVIDER_REGISTRY = {
    "claudesdk": ClaudeSDKAgent,
    "googleadk": GoogleADKAgent,
    "openaisdk": OpenAISDKAgent,
    "openclaw": OpenClawAgent,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run task inference from YAML config.")
    parser.add_argument("--runtime-config", help="Path to the runtime/model YAML config")
    parser.add_argument("--task-config", help="Path to the task-list YAML config")
    parser.add_argument("--multi-run", action="store_true", default=False,
                        help="Allow re-running tasks that already have results (default: skip existing)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of tasks to run concurrently (default: 1 sequential). "
                             "Each concurrent task spawns its own MCP server subprocess; "
                             "tune based on provider rate limits and local RAM.")
    parser.add_argument("--smoke", action="store_true", default=False,
                        help="Run the bundled one-pair smoke test (src/configs/tasks_smoke.yaml) "
                             "to verify the setup end-to-end; stands in for --task-config.")
    args = parser.parse_args()

    if args.smoke and not args.task_config:
        args.task_config = str(Path(__file__).resolve().parents[1] / "configs" / "tasks_smoke.yaml")

    if args.runtime_config and args.task_config:
        config = InferenceConfig.from_files(args.runtime_config, args.task_config)
    else:
        raise SystemExit("Provide --runtime-config and --task-config, or provide --config for a legacy combined file")

    agent_cls = PROVIDER_REGISTRY.get(config.provider)
    if agent_cls is None:
        raise SystemExit(f"Unsupported provider '{config.provider}'. Expected one of {sorted(PROVIDER_REGISTRY)}")

    agent = agent_cls(
        model=config.model,
        temperature=config.temperature,
        max_turns=config.max_turns,
        results_root=config.results_root,
    )
    summary = run_batch(agent, config, multi_run=args.multi_run, workers=args.workers)
    print(json.dumps(summary.to_dict(), indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
