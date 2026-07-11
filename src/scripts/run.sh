#!/bin/bash
set -e

# Usage: bash src/scripts/run.sh [provider] [--multi-run]
#   provider: openaisdk (default), googleadk, claudesdk, all
#   --multi-run: re-run tasks even if results already exist

PROVIDER=$1
EXTRA_FLAGS=""
for arg in "$@"; do
    if [ "$arg" = "--multi-run" ]; then
        EXTRA_FLAGS="--multi-run"
    fi
done

declare -A PROVIDER_MODELS
PROVIDER_MODELS[openaisdk]="gpt-4o gpt-4.1 gpt-5 gpt-5.4"
PROVIDER_MODELS[googleadk]="gemini-3.1-pro"
PROVIDER_MODELS[claudesdk]="claude-sonnet-4-6 claude-haiku-4-5"

run_provider() {
    local provider="$1"
    local models="${PROVIDER_MODELS[$provider]}"
    if [ -z "$models" ]; then
        echo "Unknown provider: $provider"
        exit 1
    fi
    for model in $models; do
        local config="src/configs/${provider}_${model}.yaml"
        if [ ! -f "$config" ]; then
            echo "Skipping $config (not found)"
            continue
        fi
        echo "===== Running: $provider / $model ====="
        python -m src.scripts.run_inference \
            --runtime-config "$config" \
            --task-config src/configs/tasks.yaml \
            $EXTRA_FLAGS
    done
}

if [ "$PROVIDER" = "all" ]; then
    for p in openaisdk googleadk claudesdk; do
        run_provider "$p"
    done
else
    run_provider "$PROVIDER"
fi