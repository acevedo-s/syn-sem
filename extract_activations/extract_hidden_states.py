import os, sys
sys.path.append('../')
from modelpaths import *

import torch, gc
import sglang as sgl
import pickle
from tqdm import tqdm
from datasets import load_dataset
import time
import socket

def find_free_port() -> int:
    """Ask the OS for a free ephemeral port and return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

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
                 IO_paths):
    with open(IO_paths["file_path"], "r") as f:
        prompts = [line.strip() for line in f]

    if n_lines is not None:
        prompts = prompts[:n_lines]

    os.makedirs(IO_paths["output_folder_path"], exist_ok=True)

    for i, batch in enumerate(batch_generator(prompts, batch_size)):
        start = time.time()
        # request hidden states on each generate call
        outputs = llm.generate(
            batch,
            sampling_params=sampling_params,
            return_hidden_states=True,
        )
        # extract per-layer hidden states for each prompt in batch
        save_dict = {
            'outputs': outputs,
        }
        with open(f"{IO_paths['output_folder_path']}/chunk_{i}.pkl", "wb") as f:
            pickle.dump(save_dict, f) 

        t_step = time.time() - start
        print(f"iter {i} | t_step = {t_step:.2f} s", flush=True)
    return

def main(model_path,
         IO_paths_list,
         batch_size,
         n_lines=None,
         tp_size=1,
         nnodes=1):

    NODE_RANK = int(os.environ.get("SLURM_NODEID", 0))
    # port = find_free_port()
    if NODE_RANK == 0:
        port = find_free_port()
        with open("/tmp/sglang_port.txt", "w") as f:
            f.write(str(port))
    else:
        while not os.path.exists("/tmp/sglang_port.txt"):
            time.sleep(0.1)
        with open("/tmp/sglang_port.txt", "r") as f:
            port = int(f.read())

    dist_init_addr = f"{get_master_address()}:{port}"
    dist_init_addr = f"{get_master_address()}:{port}"
    print(f"Using free port {port} for rendezvous")

    # initialize engine (hidden states requested per-generate)
    llm = sgl.Engine(
        model_path=model_path,
        tp_size=tp_size,
        nnodes=nnodes,
        dist_init_addr=dist_init_addr,
        node_rank=NODE_RANK,
        grammar_backend="xgrammar",
        disable_radix_cache=True,
        enable_return_hidden_states=True,
    )

    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 1,
    }

    for IO_paths in IO_paths_list:
        print(f'processing {IO_paths}')
        process_file(
            llm,
            sampling_params,
            batch_size,
            n_lines,
            IO_paths,
        )

    llm.shutdown()

if __name__ == "__main__":
    model = sys.argv[1] # 
    language = sys.argv[2] # 
    data_var = sys.argv[3] # syn or sem
    match_var = 'matching'

    tp_size, nnodes = get_slurm_config()
    print(f'{tp_size=}, {nnodes=}')
    model_path = model_paths[model]
    n_lines = 2100
    batch_size = 100
    dataset_var = 'second'

    IO_paths_list = [
        {
            "file_path": f"/home/acevedo/syn-sem/datasets/txt/{data_var}/{dataset_var}/{match_var}/{language}/sentences{i}.txt",
            "output_folder_path": f"/home/acevedo/syn-sem/datasets/activations/{data_var}/{dataset_var}/{model}/{match_var}/{language}/{i}/"
        }
        for i in [0, 1]
    ]
    # avoiding computing the activations of english more than once
    if language != 'english' and data_var == 'sem':
        IO_paths_list = IO_paths_list[1:]

    main(
        model_path=model_path,
        IO_paths_list=IO_paths_list,
        batch_size=batch_size,
        n_lines=n_lines,
        tp_size=tp_size,
        nnodes=nnodes,
    )
