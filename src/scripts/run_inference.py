from __future__ import annotations

import argparse
import json

from agent.openaisdk import OpenAISDKAgent
from src.runtime.config import InferenceConfig
from src.runtime.openaisdk import run_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI SDK task inference from a YAML config.")
    parser.add_argument("--runtime-config", help="Path to the runtime/model YAML config")
    parser.add_argument("--task-config", help="Path to the task-list YAML config")
    args = parser.parse_args()

    if args.runtime_config and args.task_config:
        config = InferenceConfig.from_files(args.runtime_config, args.task_config)
    else:
        raise SystemExit("Provide --runtime-config and --task-config, or provide --config for a legacy combined file")

    agent = OpenAISDKAgent(
        model=config.model,
        temperature=config.temperature,
        max_turns=config.max_turns,
        results_root=config.results_root,
    )
    summary = run_batch(agent, config)
    print(json.dumps(summary.to_dict(), indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
