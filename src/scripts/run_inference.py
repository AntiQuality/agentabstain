from __future__ import annotations

import argparse
import json

from agent.openaisdk import OpenAISDKAgent
from src.runtime.config import InferenceConfig
from src.runtime.openaisdk import run_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI SDK task inference from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the inference YAML config")
    args = parser.parse_args()

    config = InferenceConfig.from_yaml(args.config)
    agent = OpenAISDKAgent(
        model=config.model,
        temperature=config.temperature,
        max_turns=config.max_turns,
        results_root=config.results_root,
    )
    summary = run_batch(agent, config)
    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
