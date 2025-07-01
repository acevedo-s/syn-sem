#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --mem=500G
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error
#SBATCH --qos=mira

ARGS=("$@") # list of commandline inputs
echo "${ARGS[@]}"
python3 -u compute.py "${ARGS[@]}"