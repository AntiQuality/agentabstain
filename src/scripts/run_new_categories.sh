#!/bin/bash
set -e

# Run high_stakes_action and insufficient_tool_capability (first 10 tasks each)
# across 4 models in parallel: gpt-4o, gpt-5.4, claude-sonnet-4-6, claude-haiku-4-5
#
# Usage: bash src/scripts/run_new_categories.sh [--multi-run]

EXTRA_FLAGS=""
for arg in "$@"; do
    if [ "$arg" = "--multi-run" ]; then
        EXTRA_FLAGS="--multi-run"
    fi
done

TASK_CONFIGS=(
    "src/configs/tasks_high_stakes_action.yaml"
    "src/configs/tasks_insufficient_tool_capability.yaml"
)

RUNTIME_CONFIGS=(
    "src/configs/openaisdk_gpt-4o.yaml"
    "src/configs/openaisdk_gpt-5.4.yaml"
    "src/configs/claudesdk_claude-sonnet-4-6.yaml"
    "src/configs/claudesdk_claude-haiku-4-5.yaml"
)

PIDS=()
LOG_DIR="logs/run_new_categories_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

for runtime_cfg in "${RUNTIME_CONFIGS[@]}"; do
    for task_cfg in "${TASK_CONFIGS[@]}"; do
        # Extract model name and category for the log file
        model=$(basename "$runtime_cfg" .yaml | sed 's/^[^_]*_//')
        category=$(basename "$task_cfg" .yaml | sed 's/^tasks_//')
        log_file="${LOG_DIR}/${model}_${category}.log"

        echo "===== Launching: $model / $category ====="
        python -m src.scripts.run_inference \
            --runtime-config "$runtime_cfg" \
            --task-config "$task_cfg" \
            $EXTRA_FLAGS \
            > "$log_file" 2>&1 &
        PIDS+=($!)
    done
done

echo ""
echo "All ${#PIDS[@]} jobs launched. Logs in $LOG_DIR"
echo "Waiting for all jobs to finish..."

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=$((FAIL + 1))
    fi
done

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "All jobs completed successfully."
else
    echo "$FAIL job(s) failed. Check logs in $LOG_DIR for details."
    exit 1
fi
