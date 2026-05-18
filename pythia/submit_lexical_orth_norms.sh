#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

models=(
  "pythia6p9b_step0"
  "pythia6p9b_step512"
  "pythia6p9b_step4000"
  "pythia6p9b_step16000"
  "pythia6p9b_step64000"
  "pythia6p9b_step143000"
)

submit_job() {
  local model="$1"
  local dependency="${2:-}"

  local cmd=(
    sbatch
    --parsable
    --job-name="pnorm_${model}_lexorth"
  )
  if [ -n "${dependency}" ]; then
    cmd+=("--dependency=afterany:${dependency}")
  fi
  cmd+=(
    "${script_dir}/snorms_lexical_orth.sh"
    --model "${model}"
    --avg-tokens 0
    --min-token-length 3
    --n-samples 2018
  )

  "${cmd[@]}"
}

lane_a_prev=""
lane_b_prev=""

for idx in "${!models[@]}"; do
  model="${models[$idx]}"
  if (( idx % 2 == 0 )); then
    job_id="$(submit_job "${model}" "${lane_a_prev}")"
    lane_a_prev="${job_id}"
    lane="A"
  else
    job_id="$(submit_job "${model}" "${lane_b_prev}")"
    lane_b_prev="${job_id}"
    lane="B"
  fi
  echo "${job_id} lane=${lane} model=${model}"
done
