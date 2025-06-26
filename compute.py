import sys,os
sys.path.append('../')
sys.path.append('../../')

if __name__ == "__main__":
    dbg = int(sys.argv[1])
    if dbg == 1:
        os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

from datetime import datetime
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

from utils import (bf16_torch_to_jax, 
                flatten_tokens_features, 
                list_folder, 
                binarize, 
                clip,
                makefolder,
                depths,
                emb_dims,
                )
from collections import defaultdict
import numpy as np
import pickle
from tqdm import tqdm
import torch
from geometry import *
from datapaths import *
import argparse
from time import time

def reduce_list_half_preserve_extremes(lst):
    """
    Reduces the input list to approximately half its original size,
    preserving the first and last elements, and sampling uniformly
    from the intermediate elements.
    
    Parameters:
    -----------
    lst : list
        The input list to reduce.
    
    Returns:
    --------
    list
        A reduced list with approximately half the points,
        preserving the first and last elements.
    """
    N = len(lst)
    if N <= 2:
        return lst.copy()
    
    half_N = max(N // 2, 2)
    num_points_to_sample = half_N - 2
    
    new_lst = [lst[0]]
    
    if num_points_to_sample > 0:
        # Calculate the indices to sample from intermediates
        step = (N - 2) / (num_points_to_sample + 1)
        intermediate_indices = [int(round(1 + i * step)) for i in range(num_points_to_sample)]
        new_lst.extend([lst[i] for i in intermediate_indices])
    
    new_lst.append(lst[-1])
    return new_lst

def reshuffle_batch_axis(act, key):
    """
    Reshuffles the activations along the batch axis.

    Args:
        act (jnp.ndarray): Activation matrix of shape (batch_size, ...).
        key (jax.random.PRNGKey): A PRNG key for randomness.

    Returns:
        jnp.ndarray: Shuffled activations.
    """
    batch_size = act.shape[0]
    perm = jax.random.permutation(key, batch_size)
    return act[perm]


def collect_data(input_path, 
                 filter_layer=None, 
                 n_loaded_tokens=40, 
                 n_files=10,
                 mask=None,
                #  seed=42,
                 ):
    
    files = list_folder(input_path, desc="chunk_")[:n_files]
    all_hidden_states = defaultdict(list)

    for file in tqdm(files, desc="Collect File"):
        data = pickle.load(open(input_path + "/" + file.name, 'rb'))

        for _, sentence in enumerate(data):
            hidden_states = sentence['meta_info']['hidden_states'][0]
            assert hidden_states.shape[1] >= n_loaded_tokens
            hidden_states = hidden_states[:, -n_loaded_tokens-mask:-mask]

            for layer_idx, layer_tensor in enumerate(hidden_states.split(1, dim=0)):
                if filter_layer is not None:
                    if layer_idx!=filter_layer: continue

                layer_tensor = layer_tensor.squeeze(0)
                all_hidden_states[f"layer_{layer_idx}"].append(layer_tensor)

    for layer, tensors in all_hidden_states.items():
        all_hidden_states[layer] = torch.stack(tensors)
        # print("Layer=", layer, "activations shape= ", all_hidden_states[layer].shape, flush=True)

    return all_hidden_states

def main_ranks(
         layers_A,
         layers_B,
         input_path_A, 
         input_path_B, 
         n_loaded_tokens, 
         mask,
         n_files,
         n_tokens_list,
         output_folder0,
         avg_flags,
         Nbits_list,
         diagonal_constraint,
         method,
         batch_shuffle,
         ):
    start_time = time()
    
    all_activations_A = collect_data(input_path_A,
                                     n_loaded_tokens=n_loaded_tokens, 
                                     n_files=n_files,
                                     mask=mask,
                                     )
    all_activations_B = collect_data(input_path_B,
                                     n_loaded_tokens=n_loaded_tokens, 
                                     n_files=n_files,
                                     mask=mask,
                                     )   

    total_sample_size = all_activations_B["layer_0"].shape[0]

    #dbg:
    print(f'{total_sample_size=}', flush=True)
    
    key_distances = jax.random.key(42)
    key_distances, subkey_distances = jax.random.split(key_distances) 
    
    for n_tokens_id, n_tokens in enumerate(n_tokens_list):
        print(f'{n_tokens=}')
        for Nbits_id, Nbits in enumerate(Nbits_list):
            print(f'{Nbits=}')
            distance_fn = hamming_distance if Nbits else normalized_L2_distance 
            get_ranks = build_get_ranks(key=subkey_distances, 
                                        sample_size=total_sample_size, 
                                        distance_fn=distance_fn,
                                        method=method,
                                        )
            
            for avg_tokens in avg_flags:
                print(f'{avg_tokens=}')

                for A_counter,layer_A in enumerate(layers_A):
                    activations_A = all_activations_A[f"layer_{layer_A}"]

                    for B_counter,layer_B in enumerate(layers_B):
                        activations_B = all_activations_B[f"layer_{layer_B}"]

                        if diagonal_constraint == 1 and layer_B != layer_A:
                            continue
                        act_A = bf16_torch_to_jax(activations_A[:,-n_tokens:,:])
                        act_B = bf16_torch_to_jax(activations_B[:,-n_tokens:,:])

                        if batch_shuffle:
                            print(f'batch_shuffling!!! A')
                            act_A = reshuffle_batch_axis(act_A, jax.random.key(111))

                        if Nbits == 1:
                            if avg_tokens == 1:
                                act_A = act_A.mean(axis=1,keepdims=True)
                                act_B = act_B.mean(axis=1,keepdims=True)
                            act_A = binarize(act_A)
                            act_B = binarize(act_B)

                        elif Nbits == 0:
                            act_A = clip(act_A).astype(jnp.double) # promote them to double to break massive degeneracies due to small precision
                            act_B = clip(act_B).astype(jnp.double)
                            if avg_tokens == 1:
                                act_A = act_A.mean(axis=1,keepdims=True)
                                act_B = act_B.mean(axis=1,keepdims=True)

                        act_A = flatten_tokens_features(act_A)
                        act_B = flatten_tokens_features(act_B)

                        ranks_folder = makefolder(base=output_folder0+f'ranks/{method}/',
                                                    create_folder=True,
                                                    Nbits=Nbits,
                                                    n_tokens=n_tokens,
                                                    avg_tokens=avg_tokens,
                                                    batch_shuffle=batch_shuffle,
                                                    layer_A=layer_A,
                                                    layer_B=layer_B,
                                                    )
                        if method == 'max':
                            R,L = get_ranks(act_A,act_B)
                            np.savez(ranks_folder+"L.npz", x_l=L[0], y_l=L[1])
                        elif method == 'min':
                            R = get_ranks(act_A,act_B)

                        np.savez(ranks_folder+"R.npz", x_ranks=R[0], y_ranks=R[1])

    print(f'this took {(time()-start_time)/60.} m')
    return


def main_compute_II(layers_A,
                layers_B,
                Nbits_list,
                n_tokens_list,
                avg_flags,
                diagonal_constraint,):
    inf_imb = np.zeros(shape=(2,len(layers_A),len(layers_B)))
    II_fn = build_information_imbalance(k=1)

    # R = np.load(ranks_folder+"R.npz").values()
    # inf_imb[:,A_counter,B_counter] = II_fn(R[0],R[1])
    # print(f'{inf_imb[:,A_counter,B_counter]=}')
    # output_filename = "IIs.npy"
    # _save = inf_imb
    return

def main_compute_coeff(layers_A,
                layers_B,
                Nbits_list,
                n_tokens_list,
                avg_flags,
                diagonal_constraint,
                method,
                batch_shuffle,
                ):
    
    print(f'computing observables')
    start_time = time()

    xi = np.zeros(shape=(2,len(layers_A),len(layers_B)))
    corr_coeff = build_corr_coeff()

    for Nbits_id,Nbits in enumerate(Nbits_list):
        print(f'{Nbits=}')
        for avg_id,avg_tokens in enumerate(avg_flags):
            for n_tokens_id,n_tokens in enumerate(n_tokens_list):
                print(f'{n_tokens=}')
                for A_counter,layer_A in enumerate(layers_A):
                    for B_counter,layer_B in enumerate(layers_B):
                        if diagonal_constraint == 1 and layer_B != layer_A:
                            continue
                        ranks_folder = makefolder(base=output_folder0+f'ranks/{method}/',
                                                create_folder=False,
                                                Nbits=Nbits,
                                                n_tokens=n_tokens,
                                                avg_tokens=avg_tokens,
                                                batch_shuffle=batch_shuffle,
                                                layer_A=layer_A,
                                                layer_B=layer_B,
                                                )
                        output_folder = makefolder(base=output_folder0,
                                                create_folder=True,
                                                Nbits=Nbits,
                                                n_tokens=n_tokens,
                                                avg_tokens=avg_tokens,
                                                batch_shuffle=batch_shuffle,
                                                )
                        
                        R_npz = np.load(ranks_folder + "R.npz")
                        L_npz = np.load(ranks_folder + "L.npz")

                        R = (
                            jnp.array(R_npz['x_ranks']),
                            jnp.array(R_npz['y_ranks'])
                        )

                        L = (
                            jnp.array(L_npz['x_l']),
                            jnp.array(L_npz['y_l'])
                        )

                        xi[:,A_counter,B_counter] = corr_coeff(R,L)

                np.save(output_folder+"corr_coeff.npy",xi)
                    
    print(f'this took {(time()-start_time)/60.} m')
    return



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ranks")
    parser.add_argument("dbg",type=int)
    parser.add_argument("min_token_length",type=int)
    parser.add_argument("modelA",type=str) # llama or deepseek
    parser.add_argument("modelB",type=str)
    parser.add_argument("aux_A",type=str) # source or target
    parser.add_argument("aux_B",type=str)
    parser.add_argument("languageA",type=str)
    parser.add_argument("languageB",type=str)
    parser.add_argument("compute_ranks_flag",type=int)
    parser.add_argument("compute_observables_flag",type=int)
    args = parser.parse_args()

    batch_size = 10
    mask = 2 # greater than zero
    n_loaded_tokens = args.min_token_length - mask  # the mask excludes final points and quotes
    method = "max"


    layers_A = list(range(1,depths[args.modelA] + 1))
    layers_B = list(range(1,depths[args.modelB] + 1))

    if 1:
        layers_A = reduce_list_half_preserve_extremes(layers_A)
        layers_B = reduce_list_half_preserve_extremes(layers_B)
    batch_shuffle = 0

    if args.dbg == 0:
        n_files = 500
        if args.min_token_length == 40:
            n_tokens_list = np.array([1,10,20])
        elif args.min_token_length == 100:
            n_tokens_list = np.array([1,5,10,20,30,40,50,60,70,80,90,n_loaded_tokens])
        Nbits_list = [0,1]
        avg_flags = [0]
        diagonal_constraint = 1

    elif args.dbg == 1:
        n_files = 100
        n_tokens_list = np.array([10])
        Nbits_list = [0,1]
        avg_flags = [0]
        diagonal_constraint = 1
    
    if args.dbg == 2: # llama vs deepseek
        n_files = 200
        n_tokens_list = np.array([20])
        # k_list = [10] 
        Nbits_list = [0,1]
        avg_flags = [0]
        diagonal_constraint = 1

    # print(f'{k_list=}')
    print(f'{Nbits_list=}')
    print(f'{avg_flags=}')
    print(f'{diagonal_constraint=}')

    input_path_A = input_paths[args.modelA][args.languageA][args.languageB][args.aux_A]['40']
    input_path_B = input_paths[args.modelB][args.languageA][args.languageB][args.aux_B]['40']
    print("Input path A = ", input_path_A, flush=True)
    print("Input path B = ", input_path_B, flush=True)
    
    output_folder0 = makefolder(base=f'./results/',
                               create_folder=True,
                               modelA=args.modelA,
                               modelB=args.modelB,
                               aux_A=args.aux_A,
                               aux_B=args.aux_B,
                               n_files=n_files,
                               min_token_length=args.min_token_length,
                               languageA=args.languageA,
                               languageB=args.languageB,
                               mask=mask,
                               )
    
    if args.compute_ranks_flag:
        main_ranks(
            layers_A=layers_A,
            layers_B=layers_B,
            input_path_A=input_path_A,
            input_path_B=input_path_B,
            n_loaded_tokens=n_loaded_tokens,
            mask=mask,
            n_files=n_files,
            n_tokens_list=n_tokens_list,
            output_folder0=output_folder0,
            avg_flags=avg_flags,
            Nbits_list=Nbits_list,
            diagonal_constraint=diagonal_constraint,
            method=method,
            batch_shuffle=batch_shuffle,
        )
    if args.compute_observables_flag:
        # main_compute_coeff(layers_A,
        #                 layers_B,
        #                 Nbits_list,
        #                 n_tokens_list,
        #                 avg_flags,
        #                 diagonal_constraint,
        #                 method,
        #                 batch_shuffle,
        #                 )
        main_compute_II(layers_A,
                        layers_B,
                        Nbits_list,
                        n_tokens_list,
                        avg_flags,
                        diagonal_constraint,
                        method,
                        batch_shuffle,
        )
    


