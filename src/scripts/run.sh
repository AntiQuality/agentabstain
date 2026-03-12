#!/bin/bash

models=(
    # "gpt-4.1"
    # "gpt-5"
    "gpt-5.4"
    "gpt-4o"
)
for model in "${models[@]}"; do
    echo "Running experiments for model: $model"
    python -m src.scripts.run_inference \
        --runtime-config src/configs/openaisdk_${model}.yaml \
        --task-config src/configs/tasks.yaml
done