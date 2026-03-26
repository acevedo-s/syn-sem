#!/bin/bash

set -euo pipefail

models=("$@")
if [ ${#models[@]} -eq 0 ]; then
  models=("qwen7b" "deepseek" "gemma12b")
fi

min_token_length=3
n_tokens=1
avg_tokens=0
global_center_flag=1

for model in "${models[@]}"; do
  echo "submitting model=${model} avg_tokens=${avg_tokens} min_token_length=${min_token_length} n_tokens=${n_tokens}"
  sbatch \
    --job-name="norms_${model}_last_token" \
    /home/acevedo/syn-sem/norms/snorms.sh \
    --model "${model}" \
    --avg-tokens "${avg_tokens}" \
    --min-token-length "${min_token_length}" \
    --n-tokens "${n_tokens}" \
    --global-center-flag "${global_center_flag}"
  sleep 0.2
done
