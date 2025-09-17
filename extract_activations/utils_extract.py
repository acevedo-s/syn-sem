import sys,os
import torch
import sglang as sgl
import pickle
from tqdm import tqdm
import time
import socket
import numpy as np
from transformers import AutoConfig

import numpy as np

def clip_hidden_torch(hidden, alphamin=0.05, alphamax=0.95):
    """
    Clip hidden states of a single sentence (all layers) using PyTorch.

    hidden : torch.Tensor (L, T, E), dtype=torch.bfloat16 or float, on GPU
    Returns: numpy.ndarray of dtype uint16 with bfloat16 bit patterns.
    """
    # Cast to float32 for quantile computation
    hidden_float = hidden.float()
    L, T, E = hidden.shape
    hidden_flat = hidden_float.view(L, T * E)

    qmin = hidden_flat.quantile(alphamin, dim=1, keepdim=True)
    qmax = hidden_flat.quantile(alphamax, dim=1, keepdim=True)

    hidden_clipped = hidden_flat.clamp(min=qmin, max=qmax).view(L, T, E)

    # Convert back to bf16, reinterpret as uint16 for NumPy export
    hidden_clipped_bf16 = hidden_clipped.to(torch.bfloat16)
    hidden_uint16 = hidden_clipped_bf16.view(torch.uint16).cpu().numpy()

    return hidden_uint16




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
