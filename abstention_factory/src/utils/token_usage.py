"""Write token usage summaries to disk.

Called at the end of each pipeline stage to persist a JSON record under
`token_usage/`. Each record captures per-model token counts for the
stage that just completed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from abstention_factory.src.utils.call_llm import get_usage_summary, reset_usage

USAGE_DIR = Path("token_usage")


def write_stage_usage(
    stage: str,
    extra: dict | None = None,
    *,
    subdir: str | None = None,
    extra_usage_by_model: dict | None = None,
) -> Path:
    """Snapshot current usage, write to token_usage/[<subdir>/]<stage>_<timestamp>.json,
    and reset.

    `subdir` lets callers route stage outputs into a subfolder (e.g. "rollout",
    "eval") so the top level of `token_usage/` keeps showing the
    factory pipeline (rewrite/aggregate/envgen/taskgen) and does not get
    drowned out by per-rollout/per-eval-batch records.

    `extra_usage_by_model` lets callers fold in token counts that were
    captured outside the in-process call_llm recorder — e.g. when the
    SDK (Claude Code, OpenAI Agents, OpenClaw) returns its own usage
    object and we read it back from disk during a backfill. Format:
    {model: {input_tokens, output_tokens, calls, ...}}. Counts are
    *added* to whatever call_llm has already recorded; arbitrary extra
    keys (cache_*) are summed in too so detailed Claude usage survives.

    Returns the path written.
    """
    summary = get_usage_summary()
    reset_usage()

    if extra_usage_by_model:
        for model, m in extra_usage_by_model.items():
            existing = summary.setdefault(
                model,
                {"input_tokens": 0, "output_tokens": 0, "calls": 0},
            )
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    existing[k] = existing.get(k, 0) + v

    total_input = sum(int(v.get("input_tokens", 0) or 0) for v in summary.values())
    total_output = sum(int(v.get("output_tokens", 0) or 0) for v in summary.values())

    record = {
        "stage": stage,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "by_model": summary,
        "totals": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
        },
    }
    if extra:
        record["extra"] = extra

    target_dir = USAGE_DIR / subdir if subdir else USAGE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    path = target_dir / f"{stage}_{ts}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
