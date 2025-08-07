#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8 # 8 for deepseek
#SBATCH --gres=gpu:8 # 8 for deepseek
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

echo $CUDA_HOME
echo $LD_LIBRARY_PATH
which nvcc

# pip install -e /home/acevedo/sglang/python
srun  /home/acevedo/my_sglang/bin/python3 extract_hidden_states.py "${ARGS[@]}"
