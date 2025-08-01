#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=6:00:00
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

pip install -e /home/acevedo/sglang/python
srun  /home/acevedo/my_sglang/bin/python3 extract_hidden_states.py "${ARGS[@]}"