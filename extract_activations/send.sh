#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error

# MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
# export MASTER_ADDR
# export WORLD_SIZE=$SLURM_NTASKS
# export RANK=$SLURM_PROCID
# export LOCAL_RANK=$SLURM_LOCALID
srun /home/rende/mysglang/bin/python3 extract_hidden_states.py