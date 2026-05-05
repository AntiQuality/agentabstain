from __future__ import annotations

from pathlib import Path


def discover_run_result_paths(results_root: str | Path, provider: str, model: str) -> list[Path]:
    model_root = Path(results_root) / provider / model
    if not model_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {model_root}")

    # Dedupe via set: defensive against glob returning the same path
    # twice (e.g. when symlinks or merged dirs cause multiple matches).
    # Without dedupe, two threads in the eval ThreadPoolExecutor would
    # process the same run and concurrently write the same eval.json,
    # producing the truncate/interleave corruption pattern.
    run_result_paths = {
        path
        for path in model_root.glob("*/*/*/*/run_result.json")
        if "batch_runs" not in path.parts
    }
    return sorted(run_result_paths)
