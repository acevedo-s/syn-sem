#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=600G
#SBATCH --output=../log_output/%x.o%j
#SBATCH --error=../log_output/%x.o%j
#SBATCH --qos=mira

set -euo pipefail

export TF_CPP_MIN_LOG_LEVEL=0
export JAX_DEBUG_NANS=True

ARGS=("$@")
echo "${ARGS[@]}"

cd /home/acevedo/syn-sem/retrieval
source /home/acevedo/venv/bin/activate
python3 -u retrieval_semantics.py "${ARGS[@]}"
