#!/bin/bash

# Extra command line arguments will be passed to the eval runner, e.g. to specify a different task or evaluation configuration.
args="$@"

python -m eval.runner \
  --provider openaisdk \
  --model gpt-5.4-2026-03-05 \
    $args

python -m eval.runner \
  --provider openaisdk \
  --model gpt-4o-2024-08-06 \
    $args

python -m eval.runner \
  --provider googleadk \
  --model gemini-3.1-pro-preview \
    $args