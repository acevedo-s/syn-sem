#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos=mira
#SBATCH --reservation=deepseek
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1      
#SBATCH --gres=gpu:8
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error

ARGS=("$@") # list of commandline inputs
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET


srun --ntasks=2 --nodes=2 /home/rende/mysglang/bin/python3 deepseek_extract_hidden_states.py "${ARGS[@]}"