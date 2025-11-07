#!/bin/bash
#SBATCH --time=4:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=1000G

#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error
#SBATCH --qos=mira
#SBATCH --job-name=activations

ARGS=("$@") # list of commandline inputs

source /home/acevedo/my_sglang/bin/activate
module load cuda11.8/toolkit/11.8.0 
export CUDA_HOME=/cm/shared/apps/cuda11.8/toolkit/11.8.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export SGLANG_PORT_FILE="/tmp/sglang_port_${SLURM_JOB_ID}.txt"

echo $CUDA_HOME
echo $LD_LIBRARY_PATH
which nvcc

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# pip install -e /home/acevedo/sglang/python
srun  /home/acevedo/my_sglang/bin/python3 qwen_extract_hidden_states.py "${ARGS[@]}"
