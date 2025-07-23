#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=1000G
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error
#SBATCH --qos=mira
#SBATCH --job-name=activations


# export SGL_ENABLE_RETURN_HIDDEN_STATES=1

srun /home/acevedo/my_sglang/bin/python3 extract_hidden_states.py