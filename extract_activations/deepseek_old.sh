#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos=mira
#SBATCH --reservation=deepseek2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1      
#SBATCH --gres=gpu:8
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error

ARGS=("$@") # list of commandline inputs
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun --ntasks=2 --nodes=2 /home/rende/mysglang/bin/python3 extract_hidden_states.py "${ARGS[@]}"