#!/bin/bash
#SBATCH --qos=mira
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=1000G
#SBATCH --job-name=_deepseek
#SBATCH --reservation=deepseek
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error

ARGS=("$@") # list of commandline inputs

source /home/acevedo/my_sglang/bin/activate
module load cuda11.8/toolkit/11.8.0 
export CUDA_HOME=/cm/shared/apps/cuda11.8/toolkit/11.8.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo $CUDA_HOME
echo $LD_LIBRARY_PATH
which nvcc

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# pip install -e /home/acevedo/sglang/python
srun  --ntasks=2 /home/acevedo/my_sglang/bin/python3 extract_hidden_states.py "${ARGS[@]}"
