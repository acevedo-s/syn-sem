#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --qos=mira
#SBATCH --job-name=pythia_syn_centers
#SBATCH --output=/home/acevedo/syn-sem/pythia/logs/%x_%j.out
#SBATCH --error=/home/acevedo/syn-sem/pythia/logs/%x_%j.out

source /home/acevedo/venv/bin/activate

cd /home/acevedo/syn-sem/pythia
python3 -u compute_syn_averages.py "$@"
