#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=1000G
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error
#SBATCH --qos=mira
# SBATCH --reservation=acevedo

ARGS=("$@") # list of commandline inputs
echo "${ARGS[@]}"
# PYTHONWARNINGS="error" python3 -u compute.py "${ARGS[@]}"
python3 -u compute_sem_averages.py "${ARGS[@]}"