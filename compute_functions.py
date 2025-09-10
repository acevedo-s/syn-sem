import sys,os
sys.path.append('../')
sys.path.append('../../')

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from datetime import datetime
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

from utils import (
                precision_map,
                torch_to_jax, 
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
                compute_and_subtract_syn_group_averages,
                load_and_subtract_syn_group_averages,
                load_and_subtract_sem_group_averages,
                set_number_of_languages_list,
                )

from geometry import *
from time import time
from torch import zeros_like

def similarities(
        modelA,
        modelB,
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
        batch_shuffle,
        similarity_fn,
        centers_list,
        data_var,
        center_A_flag,
        center_B_flag,
        zero_activations,
        removal_method,
        precision,
        random_center_type,
        ):
    start_time = time()
    
    all_activations_A = collect_data(input_path_A,
                                     min_token_length=min_token_length, 
                                     n_files=n_files,
                                     model_name=modelA,
                                     )
    all_activations_B = collect_data(input_path_B,
                                     min_token_length=min_token_length, 
                                     n_files=n_files,
                                     model_name=modelB,
                                     )

    total_sample_size = all_activations_B["layer_0"].shape[0]

    #dbg:
    print(f'{total_sample_size=}', flush=True)
    
    key_distances = jax.random.PRNGKey(42)
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
                    if zero_activations: activations_A = zeros_like(activations_A)

                    for B_counter,layer_B in enumerate(layers_B):
                        activations_B = all_activations_B[f"layer_{layer_B}"]

                        if diagonal_constraint == 1 and layer_B != layer_A:
                            continue

                        for centers in centers_list:
                            number_of_languages_list = set_number_of_languages_list(center_A_flag,center_B_flag,centers)
                            for number_of_languages in number_of_languages_list:
                                ### Extra careful with precisions here...
                                act_A = torch_to_jax(activations_A[:,-n_tokens:,:],precision)
                                act_B = torch_to_jax(activations_B[:,-n_tokens:,:],precision)

                                if batch_shuffle:
                                    print(f'batch_shuffling!!! A')
                                    act_A = reshuffle_batch_axis(act_A, jax.random.PRNGKey(111))

                                if Nbits == 1:
                                    if avg_tokens == 1:
                                        act_A = act_A.mean(axis=1,keepdims=True)
                                        act_B = act_B.mean(axis=1,keepdims=True)
                                    act_A = binarize(act_A)
                                    act_B = binarize(act_B)

                                elif Nbits == 0:
                                    act_A = clip(act_A) 
                                    act_B = clip(act_B)
                                    if avg_tokens == 1:
                                        act_A = act_A.mean(axis=1,keepdims=True)
                                        act_B = act_B.mean(axis=1,keepdims=True)

                                act_A = flatten_tokens_features(act_A)
                                act_B = flatten_tokens_features(act_B)

                                sim_folder = makefolder(base=output_folder0+f'similarities/',
                                                        create_folder=True,
                                                        centers=centers,
                                                        Nbits=Nbits,
                                                        n_tokens=n_tokens,
                                                        avg_tokens=avg_tokens,
                                                        batch_shuffle=batch_shuffle,
                                                        layer_A=layer_A,
                                                        layer_B=layer_B,
                                                        )
                                
                                if centers == 'syn':
                                    if data_var == 'syn':
                                        if center_A_flag:
                                            act_A = compute_and_subtract_syn_group_averages(sim_folder,act_A,center_A_flag,'A',removal_method)
                                        if center_B_flag:
                                            act_B = compute_and_subtract_syn_group_averages(sim_folder,act_B,center_B_flag,'B',removal_method)
                                    elif data_var == 'sem':
                                        if center_A_flag != 0 and center_B_flag == center_A_flag:
                                            act_A,act_B = load_and_subtract_syn_group_averages(act_A,act_B,sim_folder,center_A_flag,removal_method,random_center_type)
                                elif centers == 'sem':
                                    if center_A_flag != 0:
                                        act_A = load_and_subtract_sem_group_averages(sim_folder,act_A,data_var,center_A_flag,number_of_languages,removal_method)
                                    if center_B_flag != 0:
                                        act_B = load_and_subtract_sem_group_averages(sim_folder,act_B,data_var,center_B_flag,number_of_languages,removal_method)

                                sim_A,sim_B = get_similarities(act_A,act_B)
                                sim_folder = makefolder(base=sim_folder,
                                                        create_folder=True,
                                                        zero_activations=zero_activations,
                                                        center_A_flag=center_A_flag,
                                                        center_B_flag=center_B_flag,
                                                        number_of_languages=number_of_languages,
                                                        removal_method=removal_method,
                                                        random_center_type=random_center_type,
                                                        )
                                np.save(os.path.join(sim_folder, "sim_A.npy"), sim_A)
                                np.save(os.path.join(sim_folder, "sim_B.npy"), sim_B)
    print(f'similarities took {(time()-start_time)/60.} m')
    return


def compute_II(
                output_folder0,
                layers_A,
                layers_B,
                Nbits_list,
                n_tokens_list,
                avg_flags,
                diagonal_constraint,
                method,
                batch_shuffle,
                centers_list,
                center_A_flag,
                center_B_flag,
                zero_activations,
                removal_method,
                precision,
                random_center_type,
                ratio_jackknife=0.5,
                jack_seeds=1,
                ):
    if jack_seeds == 1:
        ratio_jackknife = 1.0
            
    start_time = time()

    jack_seeds = np.arange(jack_seeds,dtype=int)
    II_fn = build_information_imbalance(k=1)

    for centers in centers_list:
        number_of_languages_list = set_number_of_languages_list(center_A_flag, center_B_flag, centers)
        for number_of_languages in number_of_languages_list:
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

                        II_folder = makefolder(base=output_folder0,
                                                create_folder=True,
                                                centers=centers,
                                                Nbits=Nbits,
                                                n_tokens=n_tokens,
                                                avg_tokens=avg_tokens,
                                                batch_shuffle=batch_shuffle,
                                                zero_activations=zero_activations,
                                                center_A_flag=center_A_flag,
                                                center_B_flag=center_B_flag,
                                                number_of_languages=number_of_languages,
                                                removal_method=removal_method,
                                                random_center_type=random_center_type,
                                                )
                        for A_counter,layer_A in enumerate(layers_A):
                            for B_counter,layer_B in enumerate(layers_B):
                                if diagonal_constraint == 1 and layer_B != layer_A:
                                    continue
                                sim_folder = makefolder(base=output_folder0+f'similarities/',
                                                        create_folder=False,
                                                        centers=centers,
                                                        Nbits=Nbits,
                                                        n_tokens=n_tokens,
                                                        avg_tokens=avg_tokens,
                                                        batch_shuffle=batch_shuffle,
                                                        layer_A=layer_A,
                                                        layer_B=layer_B,
                                                        zero_activations=zero_activations,
                                                        center_A_flag=center_A_flag,
                                                        center_B_flag=center_B_flag,
                                                        number_of_languages=number_of_languages,
                                                        removal_method=removal_method,
                                                        random_center_type=random_center_type,
                                                        )
                                ### carefull with precisions here too...
                                sim_A = jnp.array(np.load(os.path.join(sim_folder, "sim_A.npy"))).astype(precision_map[precision])
                                sim_B = jnp.array(np.load(os.path.join(sim_folder, "sim_B.npy"))).astype(precision_map[precision])

                                for jack_seed_id,jack_seed in enumerate(jack_seeds):
                                    jack_key = jax.random.PRNGKey(jack_seed)
                                    jack_indices = jax.random.choice(key=jack_key,
                                                                    a=sim_A.shape[0],
                                                                    shape=(int(ratio_jackknife*sim_A.shape[0]),),
                                                                    replace=False)
                                    
                                    R_jack = mapped_compute_ranks(method)(sim_A[jack_indices, :][:, jack_indices],
                                                                        sim_B[jack_indices, :][:, jack_indices])

                                    _inf_imb,_inf_imb_std = II_fn(R_jack[0],R_jack[1])
                                    (inf_imb[jack_seed_id,:,A_counter,B_counter],
                                    inf_imb_std[jack_seed_id,:,A_counter,B_counter]) = _inf_imb,_inf_imb_std
                                
                                #to save memory...
                                os.remove(os.path.join(sim_folder, "sim_A.npy"))
                                os.remove(os.path.join(sim_folder, "sim_B.npy"))

                                    
                        jack_std = inf_imb.std(axis=0)
                        inf_imb = inf_imb.mean(axis=0)
                        np.save(II_folder+"II.npy",inf_imb)
                        np.save(II_folder+"II_jack_std.npy",jack_std)

                    
    print(f'II took {(time()-start_time)/60.} m')
    return


# def compute_coeff(layers_A,
#                 layers_B,
#                 Nbits_list,
#                 n_tokens_list,
#                 avg_flags,
#                 diagonal_constraint,
#                 method,
#                 batch_shuffle,
#                 centers_list,
#                 ratio_jackknife=.5,
#                 jack_seeds=1,
#                 ):

#     if jack_seeds == 1:
#         ratio_jackknife = 1.0
    
#     print(f'computing corr coeff')
#     start_time = time()

#     jack_seeds = np.arange(jack_seeds,dtype=int)
#     corr_coeff = build_corr_coeff_ties()

#     for centers in centers_list:
#         for Nbits_id,Nbits in enumerate(Nbits_list):
#             print(f'{Nbits=}')
#             for avg_id,avg_tokens in enumerate(avg_flags):
#                 for n_tokens_id,n_tokens in enumerate(n_tokens_list):
#                     print(f'{n_tokens=}')
#                     xi = np.zeros(shape=(len(jack_seeds),2,len(layers_A),len(layers_B)))
#                     std = np.zeros(shape=(len(jack_seeds),2,len(layers_A),len(layers_B)))

#                     for A_counter,layer_A in enumerate(layers_A):
#                         for B_counter,layer_B in enumerate(layers_B):
#                             if diagonal_constraint == 1 and layer_B != layer_A:
#                                 continue
#                             sim_folder = makefolder(base=output_folder0+f'similarities/',
#                                                     create_folder=False,
#                                                     centers=centers,
#                                                     Nbits=Nbits,
#                                                     n_tokens=n_tokens,
#                                                     avg_tokens=avg_tokens,
#                                                     batch_shuffle=batch_shuffle,
#                                                     layer_A=layer_A,
#                                                     layer_B=layer_B,
#                                                     )
#                             output_folder = makefolder(base=output_folder0,
#                                                     create_folder=True,
#                                                     centers=centers,
#                                                     Nbits=Nbits,
#                                                     n_tokens=n_tokens,
#                                                     avg_tokens=avg_tokens,
#                                                     batch_shuffle=batch_shuffle,
#                                                     )                            
#                             sim_A = np.load(os.path.join(sim_folder, "sim_A.npy"))
#                             sim_B = np.load(os.path.join(sim_folder, "sim_B.npy"))


#                             for jack_seed_id,jack_seed in enumerate(jack_seeds):
#                                 jack_key = jax.random.PRNGKey(jack_seed)
#                                 jack_indices = jax.random.choice(key=jack_key,
#                                                                 a=sim_A.shape[0],
#                                                                 shape=(int(ratio_jackknife*sim_A.shape[0]),),
#                                                                 replace=False)
#                                 A_ranks, B_ranks = mapped_compute_ranks(method)(sim_A[jack_indices, :][:, jack_indices],
#                                                                             sim_B[jack_indices, :][:, jack_indices])
#                                 A_l,B_l = mapped_compute_ranks(method)(-sim_A[jack_indices, :][:, jack_indices],
#                                                                     -sim_B[jack_indices, :][:, jack_indices])

#                                 (xi[jack_seed_id,:,A_counter,B_counter],
#                                 std[jack_seed_id,:,A_counter,B_counter]) = corr_coeff((A_ranks,B_ranks),(A_l,B_l))
#                                 print(corr_coeff((A_ranks,B_ranks),(A_l,B_l)))

#                     jack_std = xi.std(axis=0)
#                     xi = xi.mean(axis=0)
#                     np.save(output_folder+"corr_coeff.npy",xi)
#                     np.save(output_folder+"corr_coeff_jack_std.npy",jack_std)
                        
#     print(f'corr coeff took {(time()-start_time)/60.} m')
#     return