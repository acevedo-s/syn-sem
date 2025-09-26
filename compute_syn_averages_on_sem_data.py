import sys,os
sys.path.append('../')
sys.path.append('../../')

if __name__ == "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import torch
jax.config.update("jax_enable_x64", True)
import numpy as np
from collections import deque
from datetime import datetime
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

from utils import (
                common_group_ids_B_path,
                syn_common_indices_path,
                torch_to_jax, 
                flatten_tokens_features, 
                makefolder,
                depths,
                emb_dims,
                reduce_list_half_preserve_extremes,
                collect_data,
                _compute_and_export_syn_centers,
                )

from geometry import *
from datapaths import *
import argparse
from time import time

def main(
        layers,
        min_token_length, 
        n_files,
        model,
        n_tokens_list,
        output_folder0,
        avg_tokens,
        Nbits_list,
        precision,
        ):
    
    start_time = time()
    centers = 'syn'
    batch_shuffle = 0
   
    all_activations = collect_data(input_paths['english'][model]['matching']['1']['sem'],
                                    min_token_length=min_token_length, 
                                    n_files=n_files,
                                    model_name=model,
                                    avg_tokens=avg_tokens,
                                    )
    syn_common_indices = torch.from_numpy(np.loadtxt(syn_common_indices_path, dtype=int)).long()

    for layer in all_activations:
        all_activations[layer] = all_activations[layer][syn_common_indices]
    print(f'{all_activations[layer].shape=}')

    group_ids_B = jnp.array(np.loadtxt(common_group_ids_B_path, dtype=int),dtype=jnp.int32)
    assert len(group_ids_B) == len(syn_common_indices)
    # unique_group_ids_B, group_counts = jnp.unique(group_ids_B, return_counts=True)

    for n_tokens_id, n_tokens in enumerate(n_tokens_list):
        print(f'{n_tokens=}')
        for Nbits_id, Nbits in enumerate(Nbits_list):
            print(f'{Nbits=}')
            for layer_counter,layer in enumerate(layers):
                print(f'{layer=}')
                activations = all_activations[f"layer_{layer}"] # torch
                if avg_tokens == 0:
                    activations = activations[:,-n_tokens:,:] # torch
                    activations = flatten_tokens_features(activations) #backend agnostic
                act = torch_to_jax(activations,precision)

                centers_folder = makefolder(base=output_folder0+f'similarities/',
                                        create_folder=True,
                                        centers=centers,
                                        Nbits=Nbits,
                                        n_tokens=n_tokens,
                                        avg_tokens=avg_tokens,
                                        batch_shuffle=batch_shuffle,
                                        layer_A=layer,
                                        layer_B=layer,
                                        )
                syn_centers, *_ = _compute_and_export_syn_centers(syn_group_ids_path=group_ids_B_path,
                                                act=act,
                                                centers_folder=centers_folder,
                                                space_index='B')
                print(f'{syn_centers.shape=}')

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ranks")
    parser.add_argument("model", type=str)
    parser.add_argument("min_token_length", type=int)
    parser.add_argument("avg_tokens", type=int, choices=[0,1])

    args = parser.parse_args()

    precision = 32
    match_var = 'matching'
    data_var = 'sem'
    n_files = 21
    global_centering = 0
    min_token_length = args.min_token_length  

    layers = reduce_list_half_preserve_extremes(list(range(1,depths[args.model] + 1)))

    Nbits_list = [0]
    diagonal_constraint = 1
    n_tokens_list = []

    if args.avg_tokens == 0:
        n_tokens_list = np.array([min_token_length, min_token_length // 2],dtype=int)
    else:
        n_tokens_list = [1]
        min_token_length = -1

    print(f'{Nbits_list=}')
    print(f'{args.avg_tokens=}')
    print(f'{diagonal_constraint=}')

    output_folder0 = makefolder(base=f'./results/',
                                create_folder=True,
                                global_centering=global_centering,
                                spaces='AB',
                                similarity_fn=normalized_L2_distance.__name__,
                                precision=precision,
                                language='english',
                                data_var=data_var,
                                modelA=args.model,
                                modelB=args.model,
                                match_var=match_var,
                                n_files=n_files,
                                min_token_length=args.min_token_length,
                                )
    main(
        layers,
        min_token_length, 
        n_files,
        args.model,
        n_tokens_list,
        output_folder0,
        args.avg_tokens,
        Nbits_list,
        precision,
        )


