import sys,os
sys.path.append('../')
sys.path.append('../../')

if __name__ == "__main__":
    dbg = int(sys.argv[1])
    if dbg == 1:
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
                reshuffle_batch_axis,
                compute_and_subtract_group_averages,
                load_and_subtract_group_averages,
                )

from geometry import *
from datapaths import *
import argparse
from time import time


def main_similarities(
        layers_A,
        layers_B,
        input_path_A, 
        input_path_B,
        group_ids_path,
        min_token_length, 
        n_files,
        n_tokens_list,
        output_folder0,
        avg_flags,
        Nbits_list,
        diagonal_constraint,
        batch_shuffle,
        similarity_fn,
        random_centers_list,
        txt_var,
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
            get_similarities = build_get_similarities(key=subkey_distances, 
                                        sample_size=total_sample_size, 
                                        similarity_fn=similarity_fn,
                                        )
            
            for avg_tokens in avg_flags:
                print(f'{avg_tokens=}')

                for A_counter,layer_A in enumerate(layers_A):
                    activations_A = all_activations_A[f"layer_{layer_A}"]

                    for B_counter,layer_B in enumerate(layers_B):
                        activations_B = all_activations_B[f"layer_{layer_B}"]

                        if diagonal_constraint == 1 and layer_B != layer_A:
                            continue

                        for random_centers in random_centers_list:
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

                            sim_folder = makefolder(base=output_folder0+f'similarities/',
                                                    create_folder=True,
                                                    random_centers=random_centers,
                                                    Nbits=Nbits,
                                                    n_tokens=n_tokens,
                                                    avg_tokens=avg_tokens,
                                                    batch_shuffle=batch_shuffle,
                                                    layer_A=layer_A,
                                                    layer_B=layer_B,
                                                    )
                            
                            if random_centers != -1:
                                if txt_var == 'syn':
                                    act_A = compute_and_subtract_group_averages(group_ids_path,act_A,random_centers,sim_folder,'A')
                                    act_B = compute_and_subtract_group_averages(group_ids_path,act_B,random_centers,sim_folder,'B')
                                elif txt_var == 'sem':
                                    act_A,act_B = load_and_subtract_group_averages(act_A,
                                                                            act_B,
                                                                            sim_folder,
                                                                            group_ids_path,
                                                                            random_centers,
                                                                            )

                            sim_A,sim_B = get_similarities(act_A,act_B)
                            np.save(os.path.join(sim_folder, "sim_A.npy"), sim_A)
                            np.save(os.path.join(sim_folder, "sim_B.npy"), sim_B)
    print(f'similarities took {(time()-start_time)/60.} m')
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
                random_centers_list,
                ratio_jackknife=0.5,
                jack_seeds=1,
                ):
    if jack_seeds == 1:
        ratio_jackknife = 1.0

    start_time = time()

    jack_seeds = np.arange(jack_seeds,dtype=int)
    II_fn = build_information_imbalance(k=1)

    for random_centers in random_centers_list:
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
                            sim_folder = makefolder(base=output_folder0+f'similarities/',
                                                    create_folder=False,
                                                    random_centers=random_centers,
                                                    Nbits=Nbits,
                                                    n_tokens=n_tokens,
                                                    avg_tokens=avg_tokens,
                                                    batch_shuffle=batch_shuffle,
                                                    layer_A=layer_A,
                                                    layer_B=layer_B,
                                                    )
                            output_folder = makefolder(base=output_folder0,
                                                    create_folder=True,
                                                    random_centers=random_centers,
                                                    Nbits=Nbits,
                                                    n_tokens=n_tokens,
                                                    avg_tokens=avg_tokens,
                                                    batch_shuffle=batch_shuffle,
                                                    )
                            
                            sim_A = np.load(os.path.join(sim_folder, "sim_A.npy"))
                            sim_B = np.load(os.path.join(sim_folder, "sim_B.npy"))

                            for jack_seed_id,jack_seed in enumerate(jack_seeds):
                                jack_key = jax.random.key(jack_seed)
                                jack_indices = jax.random.choice(key=jack_key,
                                                                a=sim_A.shape[0],
                                                                shape=(int(ratio_jackknife*sim_A.shape[0]),),
                                                                replace=False)
                                
                                R_jack = mapped_compute_ranks(method)(sim_A[jack_indices, :][:, jack_indices],
                                                                    sim_B[jack_indices, :][:, jack_indices])

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
                random_centers_list,
                ratio_jackknife=.5,
                jack_seeds=1,
                ):

    if jack_seeds == 1:
        ratio_jackknife = 1.0
    
    print(f'computing corr coeff')
    start_time = time()

    jack_seeds = np.arange(jack_seeds,dtype=int)
    corr_coeff = build_corr_coeff_ties()

    for random_centers in random_centers_list:
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
                            sim_folder = makefolder(base=output_folder0+f'similarities/',
                                                    create_folder=False,
                                                    random_centers=random_centers,
                                                    Nbits=Nbits,
                                                    n_tokens=n_tokens,
                                                    avg_tokens=avg_tokens,
                                                    batch_shuffle=batch_shuffle,
                                                    layer_A=layer_A,
                                                    layer_B=layer_B,
                                                    )
                            output_folder = makefolder(base=output_folder0,
                                                    create_folder=True,
                                                    random_centers=random_centers,
                                                    Nbits=Nbits,
                                                    n_tokens=n_tokens,
                                                    avg_tokens=avg_tokens,
                                                    batch_shuffle=batch_shuffle,
                                                    )                            
                            sim_A = np.load(os.path.join(sim_folder, "sim_A.npy"))
                            sim_B = np.load(os.path.join(sim_folder, "sim_B.npy"))


                            for jack_seed_id,jack_seed in enumerate(jack_seeds):
                                jack_key = jax.random.PRNGKey(jack_seed)
                                jack_indices = jax.random.choice(key=jack_key,
                                                                a=sim_A.shape[0],
                                                                shape=(int(ratio_jackknife*sim_A.shape[0]),),
                                                                replace=False)
                                A_ranks, B_ranks = mapped_compute_ranks(method)(sim_A[jack_indices, :][:, jack_indices],
                                                                            sim_B[jack_indices, :][:, jack_indices])
                                A_l,B_l = mapped_compute_ranks(method)(-sim_A[jack_indices, :][:, jack_indices],
                                                                    -sim_B[jack_indices, :][:, jack_indices])

                                (xi[jack_seed_id,:,A_counter,B_counter],
                                std[jack_seed_id,:,A_counter,B_counter]) = corr_coeff((A_ranks,B_ranks),(A_l,B_l))
                                print(corr_coeff((A_ranks,B_ranks),(A_l,B_l)))

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
    parser.add_argument("method",type=str, choices=['max','min'], help="max or min")
    parser.add_argument("txt_var",type=str, choices=['syn','sem'], help="syntax or semantics")
    args = parser.parse_args()

    batch_shuffle = 0
    batch_size = 100
    min_token_length = args.min_token_length  

    layers_A = list(range(1,depths[args.modelA] + 1))
    layers_B = list(range(1,depths[args.modelB] + 1))

    if 1:
        layers_A = reduce_list_half_preserve_extremes(layers_A)
        layers_B = reduce_list_half_preserve_extremes(layers_B)

    Nbits_list = [0]
    avg_flags = [0]
    diagonal_constraint = None
    n_files = None
    n_tokens_list = None
    match_var_list = None
    random_centers_list = None

    if args.dbg == 0:
        n_tokens_list = np.array([min_token_length])
        n_files = 16
        diagonal_constraint = 1
        match_var_list = ["matching"]
        random_centers_list = [1]

    elif args.dbg == 1:
        n_files = 1
        n_tokens_list = np.array([min_token_length])
        diagonal_constraint = 1
        match_var_list = ["matching"]
        random_centers_list = [-1,0]


    print(f'{Nbits_list=}')
    print(f'{avg_flags=}')
    print(f'{diagonal_constraint=}')

    if args.method == 'max':
        similarity_fn = jnp.dot
    elif args.method == 'min':
        similarity_fn = normalized_L2_distance 
    assert 1 not in Nbits_list

    for match_var in match_var_list:
        input_path_A = input_paths[args.modelA][match_var]['0'][args.txt_var]
        input_path_B = input_paths[args.modelB][match_var]['1'][args.txt_var]

        print("Input path A = ", input_path_A, flush=True)
        print("Input path B = ", input_path_B, flush=True)
        
        output_folder0 = makefolder(base=f'./results/',
                                create_folder=True,
                                txt_var=args.txt_var,
                                modelA=args.modelA,
                                modelB=args.modelB,
                                match_var=match_var,
                                n_files=n_files,
                                min_token_length=args.min_token_length,
                                )
        
        group_ids_path = f"/home/acevedo/syn-sem/datasets/txt/{args.txt_var}/second/{match_var}/"

        if args.compute_ranks_flag:
            main_similarities(
                layers_A=layers_A,
                layers_B=layers_B,
                input_path_A=input_path_A,
                input_path_B=input_path_B,
                group_ids_path=group_ids_path,
                min_token_length=min_token_length,
                n_files=n_files,
                n_tokens_list=n_tokens_list,
                output_folder0=output_folder0,
                avg_flags=avg_flags,
                Nbits_list=Nbits_list,
                diagonal_constraint=diagonal_constraint,
                batch_shuffle=batch_shuffle,
                similarity_fn=similarity_fn,
                random_centers_list=random_centers_list,
                txt_var=args.txt_var,
            )
        if args.compute_observables_flag:
            if args.method == 'max':
                main_compute_coeff(layers_A,
                                layers_B,
                                Nbits_list,
                                n_tokens_list,
                                avg_flags,
                                diagonal_constraint,
                                args.method,
                                batch_shuffle,
                                random_centers_list,
                                )
            elif args.method == 'min':
                main_compute_II(
                            output_folder0,
                            layers_A,
                            layers_B,
                            Nbits_list,
                            n_tokens_list,
                            avg_flags,
                            diagonal_constraint,
                            args.method,
                            batch_shuffle,
                            random_centers_list,
                )
        


