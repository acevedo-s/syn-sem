import os
import torch
import sglang as sgl
import pickle
from tqdm import tqdm
from datasets import load_dataset
import time
import socket

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
         ):

    with open(file_path, "r") as f:
        prompts = [line.strip() for line in f]

    if n_lines is not None:
        prompts = prompts[:n_lines]

    # Create an LLM.
    NODE_RANK = int(os.environ.get("SLURM_NODEID", 0))

    llm = sgl.Engine(
        model_path=model_path,
        tp_size=1,
        return_hidden_states=True,
        nnodes=1,
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
    
    for i in range(2):
        model_path = "/home/rende/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
        file_path = f"/home/acevedo/syn-sem/datasets/matching/sentences{i}.txt"
        output_folder_path = f"/home/acevedo/syn-sem/datasets/activations/matching/{i}/"
        n_lines = 10000
        batch_size = 100

        main(model_path=model_path,
            file_path=file_path,
            output_folder_path=output_folder_path,
            batch_size=batch_size,
            n_lines=n_lines,
            )