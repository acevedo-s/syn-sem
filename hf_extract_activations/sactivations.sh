#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=./log_activations/%x.o%j              # Standard output
#SBATCH --error=./log_activations/%x.o%j               # Standard error
#SBATCH --qos=mira
#SBATCH --mem=1000G

source /home/acevedo/venv/bin/activate

ARGS=("$@")
python3 -u extract_activations.py "${ARGS[@]}"