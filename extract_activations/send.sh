#!/bin/bash
#SBATCH --nodes=2
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=1500G
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error
#SBATCH --qos=mira

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun /home/rende/mysglang/bin/python3 extract_hidden_states.py