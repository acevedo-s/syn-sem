import os,sys
sys.path.append('../')
from modelpaths import *

import torch,gc
import sglang as sgl
import pickle
from tqdm import tqdm
from datasets import load_dataset
import time
import socket

def get_slurm_config():
    """Automatically determine tp_size and nnodes from SLURM."""
    try:
        nnodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
        ngpus_total = int(os.environ.get("SLURM_NTASKS", "1"))

        # Assume 1 task per GPU
        tp_size = ngpus_total  # Tensor parallel size = total number of GPUs

        return tp_size, nnodes

    except Exception as e:
        print(f"Failed to detect SLURM config. Defaulting to 1x1. Error: {e}")
        return 1, 1


def get_master_address():
    hostnames = os.popen("scontrol show hostname $SLURM_JOB_NODELIST").read().split()
    return socket.gethostbyname(hostnames[0])

dist_init_addr = f"{get_master_address()}:8000"


def batch_generator(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def main(model_path,
         file_path,
         output_folder_path,
         batch_size,
         n_lines=None,
         tp_size=1,  # number of GPU's
         nnodes=1,  # number of nodes
         ):



    with open(file_path, "r") as f:
        prompts = [line.strip() for line in f]

    if n_lines is not None:
        prompts = prompts[:n_lines]

    NODE_RANK = int(os.environ.get("SLURM_NODEID", 0))

    llm = sgl.Engine(
        model_path=model_path,
        tp_size=tp_size,
        return_hidden_states=True,
        nnodes=nnodes,
        dist_init_addr=dist_init_addr,
        node_rank=NODE_RANK,
        grammar_backend="xgrammar",
        disable_radix_cache=True,
    ) 

    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 2,
    }

    os.makedirs(output_folder_path, exist_ok=True)

    for i, p in enumerate(batch_generator(prompts, batch_size)):
        start = time.time()
        outputs = llm.generate(p, sampling_params=sampling_params)
        with open(f"{output_folder_path}/chunk_{i}.pkl", "wb") as f:
            pickle.dump(outputs, f)

        t_step = time.time() - start
        print(f"iter {i} | t_step = {t_step:.2f}", flush=True)

    llm.shutdown()

# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    
    model = 'deepseek'
    match_var = 'matching'
    tp_size, nnodes = get_slurm_config()
    tp_size = 16
    print(f'{tp_size=}, {nnodes=}')

    for i in [1]:
        model_path = model_paths[model]
        file_path = f"/home/acevedo/syn-sem/datasets/txt/{match_var}/sentences{i}.txt"
        output_folder_path = f"/home/acevedo/syn-sem/datasets/activations/{model}/{match_var}/{i}/"
        n_lines = 2000
        batch_size = 100

        main(model_path=model_path,
            file_path=file_path,
            output_folder_path=output_folder_path,
            batch_size=batch_size,
            n_lines=n_lines,
            tp_size=tp_size,
            nnodes=nnodes,
            )