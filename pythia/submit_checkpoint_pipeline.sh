#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 MODEL [MODEL ...]" >&2
  exit 1
fi

repo_root="/home/acevedo/syn-sem"
hf_root="${repo_root}/hf_extract_activations"
pythia_root="${repo_root}/pythia"
last_job_id=""

submit_job() {
  local workdir="$1"
  shift

  local output
  if [ -n "${last_job_id}" ]; then
    output=$(cd "${workdir}" && sbatch --dependency=afterany:"${last_job_id}" "$@")
  else
    output=$(cd "${workdir}" && sbatch "$@")
  fi

  local job_id
  job_id=$(echo "${output}" | awk '{print $4}')
  if [ -z "${job_id}" ]; then
    echo "Could not parse sbatch output: ${output}" >&2
    exit 1
  fi

  echo "${output}"
  last_job_id="${job_id}"
}

submit_model() {
  local model="$1"
  local languages=(arabic chinese english german italian spanish turkish)

  echo "Submitting pipeline for ${model}"

  for language in "${languages[@]}"; do
    submit_job "${hf_root}" sactivations.sh "${language}" sem matching "${model}"
  done
  submit_job "${hf_root}" sactivations.sh english syn matching "${model}"

  for avg_tokens in 0 1; do
    submit_job "${pythia_root}" scompute_sem_averages.sh \
      --model "${model}" \
      --avg-tokens "${avg_tokens}" \
      --min-token-length 3 \
      --n-samples 2018

    submit_job "${pythia_root}" scompute_syn_averages.sh \
      --model "${model}" \
      --avg-tokens "${avg_tokens}" \
      --min-token-length 3 \
      --n-samples 2018
  done

  submit_job "${pythia_root}" snorms.sh \
    --model "${model}" \
    --avg-tokens 0 \
    --min-token-length 3 \
    --n-samples 2018

  submit_job "${pythia_root}" snorms.sh \
    --model "${model}" \
    --avg-tokens 1 \
    --min-token-length 3 \
    --n-samples 2018

  submit_job "${pythia_root}" snorms.sh \
    --model "${model}" \
    --avg-tokens 0 \
    --n-tokens 1 \
    --min-token-length 3 \
    --n-samples 2018
}

for model in "$@"; do
  submit_model "${model}"
done

echo "Final job in chain: ${last_job_id}"
