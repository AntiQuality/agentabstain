from __future__ import annotations

import argparse
import json
from pathlib import Path

from abstention_factory.src.utils.file_io import utc_now_iso, write_json
from eval.config import EvaluationConfig
from eval.discovery import discover_run_result_paths
from eval.evaluators import (
    CriticalActionLooseEvaluator,
    CriticalActionStrictEvaluator,
    ResponseLLMJudgeEvaluator,
)
from eval.loader import load_evaluation_bundle
from tqdm import tqdm


DEFAULT_JUDGE_CONFIG_PATH = Path("eval/configs/default.yaml")


def build_evaluators(config: EvaluationConfig):
    return [
        CriticalActionStrictEvaluator(),
        CriticalActionLooseEvaluator(),
        ResponseLLMJudgeEvaluator(config.judge_models),
    ]


def evaluate_provider_model(
    *,
    provider: str,
    model: str,
    results_root: str | Path,
    judge_config_path: str | Path,
    override_judge: bool = False,
) -> list[Path]:
    config = EvaluationConfig.from_yaml(judge_config_path)
    evaluators = build_evaluators(config)
    written_paths: list[Path] = []
    run_result_paths = discover_run_result_paths(results_root, provider, model)
    progress = tqdm(
        run_result_paths,
        desc=f"evaluating {provider}/{model}",
        unit="run",
        leave=True,
    )

    for run_result_path in progress:
        bundle = load_evaluation_bundle(run_result_path, provider=provider, model=model)
        progress.set_postfix_str(
            f"{bundle.category}/{bundle.task_id}/{bundle.task_type}/{bundle.run_id}",
            refresh=False,
        )
        output_path = bundle.artifact_dir / "eval.json"
        existing_payload = None
        if output_path.exists():
            existing_payload = json.loads(output_path.read_text())
        metrics: dict[str, dict] = {}
        for evaluator in evaluators:
            if (
                evaluator.name == "response_llm_judge"
                and not override_judge
                and isinstance(existing_payload, dict)
                and isinstance(existing_payload.get("metrics"), dict)
                and evaluator.name in existing_payload["metrics"]
            ):
                metrics[evaluator.name] = existing_payload["metrics"][evaluator.name]
                continue
            try:
                metrics[evaluator.name] = evaluator.evaluate(bundle)
            except Exception as exc:
                metrics[evaluator.name] = {
                    "pass": None,
                    "error": str(exc),
                }

        eval_payload = {
            "provider": provider,
            "model": model,
            "category": bundle.category,
            "task_id": bundle.task_id,
            "task_type": bundle.task_type,
            "expected_behavior": bundle.task_type,
            "run_id": bundle.run_id,
            "artifact_dir": str(bundle.artifact_dir),
            "task_dir": str(bundle.task_dir),
            "generated_at": utc_now_iso(),
            "run_error": bundle.run_result.get("error"),
            "metrics": metrics,
        }
        write_json(output_path, eval_payload)
        written_paths.append(output_path)

    return written_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all runs for one provider/model pair.")
    parser.add_argument("--provider", required=True, help="Runtime provider, e.g. openaisdk")
    parser.add_argument("--model", required=True, help="Model name under results/{provider}/{model}")
    parser.add_argument("--results-root", default="results", help="Root directory containing runtime results")
    parser.add_argument(
        "--judge-config",
        default=str(DEFAULT_JUDGE_CONFIG_PATH),
        help="YAML config defining judge_models",
    )
    parser.add_argument(
        "--override-judge",
        action="store_true",
        help="Re-run response_llm_judge even if eval.json already contains judge results",
    )
    args = parser.parse_args()

    written_paths = evaluate_provider_model(
        provider=args.provider,
        model=args.model,
        results_root=args.results_root,
        judge_config_path=args.judge_config,
        override_judge=args.override_judge,
    )
    print(
        json.dumps(
            {
                "provider": args.provider,
                "model": args.model,
                "evaluated_runs": len(written_paths),
                "eval_paths": [str(path) for path in written_paths],
            },
            indent=4,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
