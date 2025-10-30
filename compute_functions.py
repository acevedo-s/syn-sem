import sys,os
sys.path.append('../')
sys.path.append('../../')

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

import shutil
from datetime import datetime
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

from utils import (
                precision_map,
                syn_group_ids_path,
                len_group_ids_path,
                syn_group_id_paths_for_sem_data,
                sem_ids_with_syn_path,
                syn_ids_with_sem_path,
                syn_syn_ids_path,
                torch_to_jax, 
                flatten_tokens_features, 
                makefolder,
                collect_data,
                reshuffle_batch_axis,
                compute_and_subtract_syn_group_averages,
                load_and_subtract_syn_group_averages,
                load_and_subtract_sem_group_averages,
                set_number_of_languages_list,
                set_language_list_permutations,
                set_global_center,
                )
from geometry import *
from corr_coeff_functions import *
from time import time
from torch import zeros_like, from_numpy

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
        avg_tokens,
        Nbits_list,
        diagonal_constraint,
        batch_shuffle,
        similarity_fn,
        centers_var,
        data_var,
        center_A_flag,
        center_B_flag,
        zero_activations,
        removal_method,
        precision,
        spaces,
        global_centering,
        ):
    start_time = time()

    number_of_languages_list = set_number_of_languages_list(center_A_flag,center_B_flag,centers_var)
    language_list_permutations = set_language_list_permutations(center_A_flag,center_B_flag,centers_var)

    all_activations_A = collect_data(input_path_A,
                                     min_token_length=min_token_length, 
                                     n_files=n_files,
                                     model_name=modelA,
                                     avg_tokens=avg_tokens,
                                     )
    if spaces == 'AB':
        all_activations_B = collect_data(input_path_B,
                                        min_token_length=min_token_length, 
                                        n_files=n_files,
                                        model_name=modelB,
                                        avg_tokens=avg_tokens,
                                        )
    elif spaces == 'AA':
        all_activations_B = all_activations_A
        
    if data_var == 'sem' and centers_var == 'syn':
        sem_ids_with_syn = from_numpy(np.loadtxt(sem_ids_with_syn_path,dtype=int)).long() # filtering sem_data to have their syntax group in space A 
        for layer in all_activations_A:
            all_activations_A[layer] = all_activations_A[layer][sem_ids_with_syn]
            all_activations_B[layer] = all_activations_B[layer][sem_ids_with_syn]
        total_sample_size = all_activations_A[next(reversed(all_activations_A))].shape[0]
        # if center_B_flag != 0:
        #     syn_syn_indices = jnp.array(np.loadtxt(syn_syn_ids_path,dtype=int),dtype=jnp.int32) # filtering data to ALSO have their syntax group in space B
        #     total_sample_size = syn_syn_indices.shape[0]
    
    elif data_var == 'syn' and centers_var == 'sem':
        syn_ids_with_sem = from_numpy(np.loadtxt(syn_ids_with_sem_path,dtype=int)).long() # filtering sem_data to have their syntax group in space A 
        for layer in all_activations_A:
            all_activations_A[layer] = all_activations_A[layer][syn_ids_with_sem]
            all_activations_B[layer] = all_activations_B[layer][syn_ids_with_sem]    
        total_sample_size = all_activations_A[next(reversed(all_activations_A))].shape[0]

    else:
        total_sample_size = all_activations_A[next(reversed(all_activations_A))].shape[0]

    print(f'{all_activations_A[next(reversed(all_activations_A))].shape=}')
    print(f'{all_activations_B[next(reversed(all_activations_B))].shape=}')
    
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
            for number_of_languages in number_of_languages_list:
                print(f'{number_of_languages=}')

                for language_list_permutation in language_list_permutations:
                    print(f'{language_list_permutation=}')

                    for A_counter,layer_A in enumerate(layers_A):
                        activations_A = all_activations_A[f"layer_{layer_A}"]
                        if zero_activations: activations_A = zeros_like(activations_A)

                        for B_counter,layer_B in enumerate(layers_B):
                            activations_B = all_activations_B[f"layer_{layer_B}"]

                            if diagonal_constraint == 1 and layer_B != layer_A:
                                continue

                            if avg_tokens == 0:
                                assert n_tokens <= activations_A.shape[1] and n_tokens <= activations_B.shape[1]
                                activations_A = activations_A[:,-n_tokens:,:] # torch
                                activations_B = activations_B[:,-n_tokens:,:]
                                activations_A = flatten_tokens_features(activations_A) # backend agnostic
                                activations_B = flatten_tokens_features(activations_B)                         
                            act_A = torch_to_jax(activations_A,precision)
                            act_B = torch_to_jax(activations_B,precision)

                            if batch_shuffle:
                                print(f'batch_shuffling A')
                                act_A = reshuffle_batch_axis(act_A, jax.random.PRNGKey(111))
                            
                            (act_A, global_center_A) = set_global_center(act_A, global_centering)
                            (act_B, global_center_B) = set_global_center(act_B, global_centering)

                            sim_folder = makefolder(base=output_folder0+f'similarities/',
                                                    create_folder=True,
                                                    centers=centers_var,
                                                    Nbits=Nbits,
                                                    n_tokens=n_tokens,
                                                    avg_tokens=avg_tokens,
                                                    batch_shuffle=batch_shuffle,
                                                    layer_A=layer_A,
                                                    layer_B=layer_B,
                                                    )
                            
                            if centers_var == 'syn':
                                if data_var == 'syn':
                                    if center_A_flag != 0:
                                        act_A = compute_and_subtract_syn_group_averages(sim_folder, act_A, act_B, center_A_flag,'A', removal_method, syn_group_ids_path)
                                    if center_B_flag != 0:
                                        act_B = compute_and_subtract_syn_group_averages(sim_folder, act_B, act_A, center_B_flag,'B', removal_method, syn_group_ids_path) # note that A and B share the syntax here
                                elif data_var == 'sem':
                                    if center_A_flag != 0:
                                        act_A = load_and_subtract_syn_group_averages(act_A, syn_group_id_paths_for_sem_data['A'], sim_folder, center_A_flag, removal_method, global_center_A, 'A')
                                        # centers_folder_B = sim_folder
                                        # act_B = act_B[syn_syn_indices] # first I select them, for those I subtract the centers that I have
                                        # act_B = load_and_subtract_syn_group_averages(act_B,syn_group_id_paths_for_sem_data['B'],centers_folder_B,center_B_flag,removal_method,global_center_B,'B')
                                        # act_A = act_A[syn_syn_indices] # for these, first I subtract their centers and then I subsample them
                            elif centers_var == 'sem':
                                if center_A_flag != 0:
                                    act_A = load_and_subtract_sem_group_averages(sim_folder,act_A,data_var,center_A_flag,number_of_languages,language_list_permutation,removal_method)
                                if center_B_flag != 0:
                                    act_B = load_and_subtract_sem_group_averages(sim_folder,act_B,data_var,center_B_flag,number_of_languages,language_list_permutation,removal_method)


                            sim_A,sim_B = get_similarities(act_A,act_B)
                            sim_folder = makefolder(base=sim_folder,
                                                    create_folder=True,
                                                    zero_activations=zero_activations,
                                                    center_A_flag=center_A_flag,
                                                    center_B_flag=center_B_flag,
                                                    number_of_languages=number_of_languages,
                                                    language_list_permutation=language_list_permutation,
                                                    removal_method=removal_method,
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
                avg_tokens,
                diagonal_constraint,
                method,
                batch_shuffle,
                centers_var,
                center_A_flag,
                center_B_flag,
                zero_activations,
                removal_method,
                precision,
                ratio_jackknife=0.5,
                n_jack_seeds=5,
                ):
    if n_jack_seeds == 1:
        ratio_jackknife = 1.0
            
    start_time = time()

    jack_seeds = np.arange(n_jack_seeds,dtype=int)
    II_fn = build_information_imbalance(k=1)

    number_of_languages_list = set_number_of_languages_list(center_A_flag, center_B_flag, centers_var)
    language_list_permutations = set_language_list_permutations(center_A_flag,center_B_flag,centers_var)
    
    for Nbits_id,Nbits in enumerate(Nbits_list):
        print(f'{Nbits=}')

        for n_tokens_id,n_tokens in enumerate(n_tokens_list):
            print(f'{n_tokens=}')

            for number_of_languages in number_of_languages_list:
                if number_of_languages != None: print(f'{number_of_languages=}') 
                
                for language_list_permutation in language_list_permutations:
                    if language_list_permutation != None: print(f'{language_list_permutation=}')

                    inf_imb = np.zeros(shape=(len(jack_seeds),
                                            2,
                                            len(layers_A),
                                            len(layers_B))
                                    )
                    inf_imb_std = np.zeros(shape=(inf_imb.shape))

                    II_folder = makefolder(base=output_folder0,
                                            create_folder=True,
                                            centers=centers_var,
                                            Nbits=Nbits,
                                            n_tokens=n_tokens,
                                            avg_tokens=avg_tokens,
                                            batch_shuffle=batch_shuffle,
                                            zero_activations=zero_activations,
                                            center_A_flag=center_A_flag,
                                            center_B_flag=center_B_flag,
                                            number_of_languages=number_of_languages,
                                            language_list_permutation=language_list_permutation,
                                            removal_method=removal_method,
                                            )
                    for A_counter,layer_A in enumerate(layers_A):
                        for B_counter,layer_B in enumerate(layers_B):
                            if diagonal_constraint == 1 and layer_B != layer_A:
                                continue
                            sim_folder = makefolder(base=output_folder0+f'similarities/',
                                                    create_folder=False,
                                                    centers=centers_var,
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
                                                    language_list_permutation=language_list_permutation,
                                                    removal_method=removal_method,
                                                    )
                            ### carefull with precisions here too...
                            sim_A = jnp.array(np.load(os.path.join(sim_folder, "sim_A.npy"))).astype(precision_map[precision])
                            sim_B = jnp.array(np.load(os.path.join(sim_folder, "sim_B.npy"))).astype(precision_map[precision])

                            for jack_seed_id,jack_seed in enumerate(jack_seeds):
                                print(f'{jack_seed=}')
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
                            try:
                                shutil.rmtree(sim_folder)
                            except Exception as e:
                                print(f"Could not delete {sim_folder}: {type(e).__name__} - {e}")

                                
                    jack_std = inf_imb.std(axis=0)
                    inf_imb = inf_imb.mean(axis=0)
                    II_path = II_folder + f"II_{ratio_jackknife:.2f}.npy"
                    np.save(II_path, inf_imb)
                    print(f'II saved to {II_path}')
                    if ratio_jackknife < 1.0:
                        np.save(II_folder + f"II_jack_std_{ratio_jackknife:.2f}.npy",jack_std)

                    
    print(f'II took {(time()-start_time)/60.} m')
    return


def compute_coeff(
                output_folder0,
                layers_A,
                layers_B,
                Nbits_list,
                n_tokens_list,
                avg_tokens,
                diagonal_constraint,
                # method,
                batch_shuffle,
                centers_var,
                center_A_flag,
                center_B_flag,
                zero_activations,
                removal_method,
                precision,
                ratio_jackknife=0.5,
                n_jack_seeds=5,
                ):

    master_seed = 9999
    master_key = jax.random.PRNGKey(master_seed)
    keyA, keyB = jax.random.split(master_key)
    average=True

    if n_jack_seeds == 1:
        ratio_jackknife = 1.0
    
    print(f'computing corr coeff')
    start_time = time()

    jack_seeds = np.arange(n_jack_seeds,dtype=int)
    rankdata_2D_ties = build_rankdata_2D_ties()
    corr_coeff = build_corr_coeff_2D_ties(average=average)

    number_of_languages_list = set_number_of_languages_list(center_A_flag, center_B_flag, centers_var)
    language_list_permutations = set_language_list_permutations(center_A_flag,center_B_flag,centers_var)


    for Nbits_id,Nbits in enumerate(Nbits_list):
        print(f'{Nbits=}')

        for n_tokens_id,n_tokens in enumerate(n_tokens_list):
            print(f'{n_tokens=}')

            for number_of_languages in number_of_languages_list:
                if number_of_languages != None: print(f'{number_of_languages=}') 

                for language_list_permutation in language_list_permutations:
                    if language_list_permutation != None: print(f'{language_list_permutation=}')

                    xi = jnp.zeros(shape=(len(jack_seeds),2,len(layers_A),len(layers_B)))
                    std_xi = jnp.zeros_like(xi)
                    # all_xis = np.zeros(())

                    corr_folder = makefolder(
                                base=output_folder0,
                                create_folder=True,
                                centers=centers_var,
                                Nbits=Nbits,
                                n_tokens=n_tokens,
                                avg_tokens=avg_tokens,
                                batch_shuffle=batch_shuffle,
                                zero_activations=zero_activations,
                                center_A_flag=center_A_flag,
                                center_B_flag=center_B_flag,
                                number_of_languages=number_of_languages,
                                language_list_permutation=language_list_permutation,
                                removal_method=removal_method,
                            )  
                    for A_counter,layer_A in enumerate(layers_A):
                        for B_counter,layer_B in enumerate(layers_B):
                            if diagonal_constraint == 1 and layer_B != layer_A:
                                continue
                            sim_folder = makefolder(
                                base=output_folder0+f'similarities/',
                                create_folder=False,
                                centers=centers_var,
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
                                language_list_permutation=language_list_permutation,
                                removal_method=removal_method,
                            )
                          
                            sim_A = jnp.array(np.load(os.path.join(sim_folder, "sim_A.npy"))).astype(precision_map[precision])
                            sim_B = jnp.array(np.load(os.path.join(sim_folder, "sim_B.npy"))).astype(precision_map[precision])


                            for jack_seed_id,jack_seed in enumerate(jack_seeds):
                                jack_key = jax.random.PRNGKey(jack_seed)
                                jack_indices = jax.random.choice(key=jack_key,
                                                                a=sim_A.shape[0],
                                                                shape=(int(ratio_jackknife*sim_A.shape[0]),),
                                                                replace=False)
                                
                                sim_A_jack = sim_A[jack_indices, :][:, jack_indices]
                                sim_B_jack = sim_B[jack_indices, :][:, jack_indices]

                                R_A_jack = rankdata_2D_ties(sim_A_jack,keyA)
                                R_B_jack = rankdata_2D_ties(sim_B_jack,keyB)
                                
                                if average == False:
                                    corr_AB, corr_BA = corr_coeff((R_A_jack,R_B_jack))
                                    np.save(os.path.join(corr_folder, f"all_corr_AB_layer{layer_A}.npy"), corr_AB)
                                    np.save(os.path.join(corr_folder, f"all_corr_BA_layer{layer_A}.npy"), corr_BA)

                                    _mean = jnp.array([corr_AB.mean(), corr_BA.mean()])
                                    _std  = jnp.array([corr_AB.std(), corr_BA.std()])
                                else:
                                    _mean,_std = corr_coeff((R_A_jack,R_B_jack))

                                xi = xi.at[jack_seed_id, :, A_counter, B_counter].set(_mean)
                                std_xi = std_xi.at[jack_seed_id, :, A_counter, B_counter].set(_std)
                                

                    jack_std = xi.std(axis=0)
                    xi = xi.mean(axis=0)
                    np.save(os.path.join(corr_folder, f"corr_coeff_{ratio_jackknife:.2f}.npy"), xi)
                    if ratio_jackknife < 1:
                        np.save(os.path.join(corr_folder, f"corr_coeff_jack_std_{ratio_jackknife:.2f}.npy"), jack_std)

                            
    print(f'corr coeff took {(time()-start_time)/60.} m')
    return