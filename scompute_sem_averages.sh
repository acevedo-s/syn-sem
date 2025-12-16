#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=1000G
#SBATCH --output=./log_sem_centroids/%j.txt              # Standard output
#SBATCH --error=./log_sem_centroids/%j.txt               # Standard error
#SBATCH --qos=mira
# SBATCH --reservation=acevedo

ARGS=("$@") # list of commandline inputs
echo "${ARGS[@]}"
# PYTHONWARNINGS="error" python3 -u compute.py "${ARGS[@]}"
source /home/acevedo/venv/bin/activate
python3 -u compute_sem_averages.py "${ARGS[@]}"