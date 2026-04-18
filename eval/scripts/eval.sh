#!/bin/bash

# Usage: bash eval/scripts/eval.sh [--override-judge] [other eval.runner args]
# Runs eval for 4 models in parallel, logs saved to logs/eval_<timestamp>/

args="$@"

LOG_DIR="logs/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

JOBS=(
  "openaisdk gpt-4o-2024-08-06"
  "openaisdk gpt-5.4-2026-03-05"
  "claudesdk claude-sonnet-4-6"
  "claudesdk claude-haiku-4-5"
)

PIDS=()
for job in "${JOBS[@]}"; do
    provider=$(echo "$job" | cut -d' ' -f1)
    model=$(echo "$job" | cut -d' ' -f2)
    log_file="${LOG_DIR}/${provider}_${model}.log"

    echo "===== Launching eval: $provider / $model ====="
    python -m eval.runner \
      --provider "$provider" \
      --model "$model" \
      $args \
      > "$log_file" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} eval jobs launched. Logs in $LOG_DIR"
echo "Waiting for all jobs to finish..."

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=$((FAIL + 1))
    fi
done

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "All eval jobs completed successfully."
else
    echo "$FAIL eval job(s) failed. Check logs in $LOG_DIR for details."
    exit 1
fi
