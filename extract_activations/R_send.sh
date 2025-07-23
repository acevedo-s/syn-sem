#!/bin/bash
#SBATCH --nodes=2
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=1000G
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error
#SBATCH --qos=mira
#SBATCH --job-name=activations


# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_ASYNC_ERROR_HANDLING=1

# export OMP_NUM_THREADS=1
# export TORCH_NCCL_BLOCKING_WAIT=1

srun /home/rende/mysglang/bin/python3 extract_hidden_states.py
# srun /home/acevedo/my_sglang/bin/python3 extract_hidden_states.py