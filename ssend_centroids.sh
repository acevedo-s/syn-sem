#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=600G
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error
#SBATCH --qos=mira
# SBATCH --reservation=acevedo

export TF_CPP_MIN_LOG_LEVEL=0      # show INFO, WARNING, and ERROR (default hides INFO=2)
export JAX_DEBUG_NANS=True         # show warnings if NaNs/Infs are encountered
# export JAX_LOG_COMPILES=1          # log compilation events


ARGS=("$@") # list of commandline inputs
echo "${ARGS[@]}"
# PYTHONWARNINGS="error" python3 -u compute.py "${ARGS[@]}"
python3 -u II_centroids.py "${ARGS[@]}"