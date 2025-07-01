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
        gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))

        tp_size = nnodes * gpus_per_node  # Total number of GPUs

        return tp_size, nnodes

    except Exception as e:
        print(f"Failed to detect SLURM config. Defaulting to 1x1. Error: {e}")
        return 1, 1

def get_master_address():
    hostnames = os.popen("scontrol show hostname $SLURM_JOB_NODELIST").read().split()
    return socket.gethostbyname(hostnames[0])

def batch_generator(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def process_file(llm,
                 sampling_params,
                 batch_size,
                 n_lines,
                 IO_paths,
                 ):
    with open(IO_paths["file_path"], "r") as f:
        prompts = [line.strip() for line in f]

    if n_lines is not None:
        prompts = prompts[:n_lines]

    os.makedirs(IO_paths["output_folder_path"], exist_ok=True)

    for i, p in enumerate(batch_generator(prompts, batch_size)):
        start = time.time()
        outputs = llm.generate(p, sampling_params=sampling_params)
        with open(f"{IO_paths['output_folder_path']}/chunk_{i}.pkl", "wb") as f:
            pickle.dump(outputs, f)
        t_step = time.time() - start
        print(f"iter {i} | t_step = {t_step:.2f}", flush=True)
    return

def main(model_path,
         IO_paths_list,
         batch_size,
         n_lines=None,
         tp_size=1,
         nnodes=1,
         ):

    NODE_RANK = int(os.environ.get("SLURM_NODEID", 0))
    dist_init_addr = f"{get_master_address()}:8000"

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

    for IO_paths_id,IO_paths in enumerate(IO_paths_list):
        print(f'processing file {IO_paths_id}')
        process_file(llm,sampling_params,batch_size,n_lines,IO_paths)

    llm.shutdown()

# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    
    model = 'deepseek'
    match_var = 'missmatching'
    tp_size, nnodes = get_slurm_config()
    print(f'{tp_size=}, {nnodes=}')
    model_path = model_paths[model]
    n_lines = 2000
    batch_size = 100


    IO_paths_list = [
        {
            "file_path": f"/home/acevedo/syn-sem/datasets/txt/{match_var}/sentences{i}.txt",
            "output_folder_path": f"/home/acevedo/syn-sem/datasets/activations/{model}/{match_var}/{i}/"
        }
        for i in [0, 1]
    ]

    main(model_path=model_path,
        IO_paths_list=IO_paths_list,
        batch_size=batch_size,
        n_lines=n_lines,
        tp_size=tp_size,
        nnodes=nnodes,
        )