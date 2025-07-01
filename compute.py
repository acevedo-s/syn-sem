import sys,os
sys.path.append('../')
sys.path.append('../../')

if __name__ == "__main__":
    dbg = int(sys.argv[1])
    if dbg == 0:
        os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

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
                reduce_list_half_preserve_extremes,
                collect_data,
                reshuffle_batch_axis
                )

from geometry import *
from datapaths import *
import argparse
from time import time


def main_ranks(
         layers_A,
         layers_B,
         input_path_A, 
         input_path_B, 
         min_token_length, 
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
                                     min_token_length=min_token_length, 
                                     n_files=n_files,
                                     )
    all_activations_B = collect_data(input_path_B,
                                     min_token_length=min_token_length, 
                                     n_files=n_files,
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
                            np.save(os.path.join(ranks_folder, "x_l.npy"), L[0])
                            np.save(os.path.join(ranks_folder, "y_l.npy"), L[1])
                        elif method == 'min':
                            R = get_ranks(act_A,act_B)
                        np.save(os.path.join(ranks_folder, "x_ranks.npy"), R[0])
                        np.save(os.path.join(ranks_folder, "y_ranks.npy"), R[1])
    print(f'ranks took {(time()-start_time)/60.} m')
    return


def main_compute_II(
                output_folder0,
                layers_A,
                layers_B,
                Nbits_list,
                n_tokens_list,
                avg_flags,
                diagonal_constraint,
                method,
                batch_shuffle,
                ratio_jackknife=0.5,
                jack_seeds=5,
                ):
    start_time = time()

    jack_seeds = np.arange(jack_seeds,dtype=int)
    II_fn = build_information_imbalance(k=1)

    for Nbits_id,Nbits in enumerate(Nbits_list):
        print(f'{Nbits=}')
        for avg_id,avg_tokens in enumerate(avg_flags):
            for n_tokens_id,n_tokens in enumerate(n_tokens_list):
                print(f'{n_tokens=}')
                inf_imb = np.zeros(shape=(len(jack_seeds),
                                        2,
                                        len(layers_A),
                                        len(layers_B))
                                )
                inf_imb_std = np.zeros(shape=(inf_imb.shape))

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
                        
                        x_ranks = np.load(os.path.join(ranks_folder, "x_ranks.npy"))
                        y_ranks = np.load(os.path.join(ranks_folder, "y_ranks.npy"))   

                        for jack_seed_id,jack_seed in enumerate(jack_seeds):
                            jack_key = jax.random.key(jack_seed)
                            jack_indices = jax.random.choice(key=jack_key,
                                                             a=x_ranks.shape[0],
                                                             shape=(int(ratio_jackknife*x_ranks.shape[0]),),
                                                             replace=False)
                            R_jack = (x_ranks[jack_indices], y_ranks[jack_indices])

                            _inf_imb,_inf_imb_std = II_fn(R_jack[0],R_jack[1])
                            (inf_imb[jack_seed_id,:,A_counter,B_counter],
                             inf_imb_std[jack_seed_id,:,A_counter,B_counter]) = _inf_imb,_inf_imb_std
                            
                jack_std = inf_imb.std(axis=0)
                inf_imb = inf_imb.mean(axis=0)
                np.save(output_folder+"II.npy",inf_imb)
                np.save(output_folder+"II_jack_std.npy",jack_std)

                    
    print(f'II took {(time()-start_time)/60.} m')
    return


def main_compute_coeff(layers_A,
                layers_B,
                Nbits_list,
                n_tokens_list,
                avg_flags,
                diagonal_constraint,
                method,
                batch_shuffle,
                ratio_jackknife=.8,
                jack_seeds=1,
                ):
    
    print(f'computing corr coeff')
    start_time = time()

    jack_seeds = np.arange(jack_seeds,dtype=int)
    corr_coeff = build_corr_coeff()

    for Nbits_id,Nbits in enumerate(Nbits_list):
        print(f'{Nbits=}')
        for avg_id,avg_tokens in enumerate(avg_flags):
            for n_tokens_id,n_tokens in enumerate(n_tokens_list):
                print(f'{n_tokens=}')
                xi = np.zeros(shape=(len(jack_seeds),2,len(layers_A),len(layers_B)))
                std = np.zeros(shape=(len(jack_seeds),2,len(layers_A),len(layers_B)))

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
                        x_ranks = jnp.array(np.load(os.path.join(ranks_folder, "x_ranks.npy")))
                        y_ranks = jnp.array(np.load(os.path.join(ranks_folder, "y_ranks.npy")))
                        x_l = jnp.array(np.load(os.path.join(ranks_folder, "x_l.npy")))
                        y_l = jnp.array(np.load(os.path.join(ranks_folder, "y_l.npy")))

                        for jack_seed_id,jack_seed in enumerate(jack_seeds):
                            jack_key = jax.random.PRNGKey(jack_seed)
                            jack_indices = jax.random.choice(key=jack_key,
                                                             a=x_ranks.shape[0],
                                                             shape=(int(ratio_jackknife*x_ranks.shape[0]),),
                                                             replace=False)

                            x_ranks_jack = jnp.take(x_ranks, jack_indices, axis=0)
                            y_ranks_jack = jnp.take(y_ranks, jack_indices, axis=0)
                            x_l_jack = jnp.take(x_l, jack_indices, axis=0)
                            y_l_jack = jnp.take(y_l, jack_indices, axis=0)

                            (xi[jack_seed_id,:,A_counter,B_counter],
                            std[jack_seed_id,:,A_counter,B_counter]) = corr_coeff((x_ranks_jack,y_ranks_jack))#,(x_l_jack,y_l_jack))
                            print(corr_coeff((x_ranks_jack,y_ranks_jack)))#,(x_l_jack,y_l_jack)))

                jack_std = xi.std(axis=0)
                xi = xi.mean(axis=0)
                np.save(output_folder+"corr_coeff.npy",xi)
                np.save(output_folder+"corr_coeff_jack_std.npy",jack_std)
                    
    print(f'corr coeff took {(time()-start_time)/60.} m')
    return



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ranks")
    parser.add_argument("dbg",type=int)
    parser.add_argument("min_token_length",type=int)
    parser.add_argument("modelA",type=str) # llama or deepseek
    parser.add_argument("modelB",type=str)
    parser.add_argument("compute_ranks_flag",type=int)
    parser.add_argument("compute_observables_flag",type=int)
    args = parser.parse_args()

    batch_shuffle = 0
    batch_size = 100
    min_token_length = args.min_token_length  # the mask excludes final points and quotes
    method = "max"

    layers_A = list(range(1,depths[args.modelA] + 1))
    layers_B = list(range(1,depths[args.modelB] + 1))

    if 1:
        layers_A = reduce_list_half_preserve_extremes(layers_A)
        layers_B = reduce_list_half_preserve_extremes(layers_B)

    Nbits_list = None
    avg_flags = None
    diagonal_constraint = None
    n_files = None
    n_tokens_list = None
    match_var_list = None

    if args.dbg == 0:
        n_tokens_list = np.array([min_token_length])
        n_files = 20
        Nbits_list = [0]
        avg_flags = [0]
        diagonal_constraint = 1
        match_var_list = ["matching"]

    elif args.dbg == 1:
        n_files = 1
        n_tokens_list = np.array([min_token_length])
        Nbits_list = [0,1]
        avg_flags = [0]
        diagonal_constraint = 1
        match_var_list = ["matching"]

    print(f'{Nbits_list=}')
    print(f'{avg_flags=}')
    print(f'{diagonal_constraint=}')



    for match_var in match_var_list:
        input_path_A = input_paths[args.modelA][match_var]['0']
        input_path_B = input_paths[args.modelB][match_var]['1']

        print("Input path A = ", input_path_A, flush=True)
        print("Input path B = ", input_path_B, flush=True)
        
        output_folder0 = makefolder(base=f'./results/',
                                create_folder=True,
                                modelA=args.modelA,
                                modelB=args.modelB,
                                match_var=match_var,
                                n_files=n_files,
                                min_token_length=args.min_token_length,
                                )
        
        if args.compute_ranks_flag:
            main_ranks(
                layers_A=layers_A,
                layers_B=layers_B,
                input_path_A=input_path_A,
                input_path_B=input_path_B,
                min_token_length=min_token_length,
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
            if method == 'max':
                main_compute_coeff(layers_A,
                                layers_B,
                                Nbits_list,
                                n_tokens_list,
                                avg_flags,
                                diagonal_constraint,
                                method,
                                batch_shuffle,
                                )
            elif method == 'min':
                main_compute_II(
                            output_folder0,
                            layers_A,
                            layers_B,
                            Nbits_list,
                            n_tokens_list,
                            avg_flags,
                            diagonal_constraint,
                            method,
                            batch_shuffle,
                )
        


