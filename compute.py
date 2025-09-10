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

from utils import (
                makefolder,
                depths,
                emb_dims,
                reduce_list_half_preserve_extremes,

                )

from geometry import *
from datapaths import *
import argparse
from time import time

from compute_functions import (
                similarities,
                compute_II
                )



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ranks")
    parser.add_argument("dbg", type=int)
    parser.add_argument("min_token_length", type=int)
    parser.add_argument("modelA", type=str) # llama, deepseek or qwen
    # parser.add_argument("compute_ranks_flag",type=int)
    # parser.add_argument("compute_observables_flag",type=int)
    parser.add_argument("method", type=str, choices=['max','min'], help="max = corr coeff, min = II")
    parser.add_argument("data_var", type=str, choices=['syn','sem'], help="syntax or semantics")
    parser.add_argument("language", type=str)
    parser.add_argument("center_A_flag", type=int)
    parser.add_argument("center_B_flag", type=int)
    parser.add_argument("zero_activations", type=int)
    parser.add_argument("removal_method", type=str,choices=['projection', 'subtraction', 'none'])
    parser.add_argument("random_center_type", type=str,choices=['permuted', 'shuffled', 'random', 'none'])
    # parser.add_argument("precision",type=int,choices=[16,32,64])
    args = parser.parse_args()

    removal_method = args.removal_method
    if removal_method == 'none':
        removal_method = None
        assert args.center_A_flag == 0 or args.center_B_flag == 0

    random_center_type = args.random_center_type
    if random_center_type == 'none':
        random_center_type = None
        assert args.center_A_flag == 0 or args.center_A_flag == 1
        assert args.center_B_flag == 0 or args.center_B_flag == 1
    
    if args.center_A_flag == 0 and args.center_B_flag == 0:
        random_center_type = None
        removal_method = None

    modelB = args.modelA
    precision = 32
    batch_shuffle = 0
    min_token_length = args.min_token_length  

    layers_A = list(range(1,depths[args.modelA] + 1))
    layers_B = list(range(1,depths[modelB] + 1))

    if 1:
        layers_A = reduce_list_half_preserve_extremes(layers_A)
        layers_B = reduce_list_half_preserve_extremes(layers_B)

    Nbits_list = [0]
    avg_flags = [0]
    diagonal_constraint = 1
    n_files = None
    n_tokens_list = []
    match_var_list = [] # in ['matching','mismatching']
    centers_list = [] # in ['syn','sem',0]
    similarity_fn = None


    if args.data_var == 'sem':
        n_files = 16
    elif args.data_var == 'syn':
        n_files = 21
    n_tokens_list = np.array([min_token_length])
    match_var_list = ["matching"]
    centers_list = ['syn']

    # if args.dbg == 1:
    #     n_files = 2


    print(f'{Nbits_list=}')
    print(f'{avg_flags=}')
    print(f'{diagonal_constraint=}')

    if args.method == 'max':
        similarity_fn = jnp.dot
    elif args.method == 'min':
        similarity_fn = normalized_L2_distance 
    assert 1 not in Nbits_list

    for match_var in match_var_list:
        input_path_A = input_paths['english'][args.modelA][match_var]['0'][args.data_var]
        input_path_B = input_paths[args.language][modelB][match_var]['1'][args.data_var]

        print("Input path A = ", input_path_A, flush=True)
        print("Input path B = ", input_path_B, flush=True)
        
        output_folder0 = makefolder(base=f'./results/',
                                create_folder=True,
                                precision=precision,
                                language=args.language,
                                data_var=args.data_var,
                                modelA=args.modelA,
                                modelB=modelB,
                                match_var=match_var,
                                n_files=n_files,
                                min_token_length=args.min_token_length,
                                )
    
        ### Computation:
        similarities(
            modelA=args.modelA,
            modelB=modelB,
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
            batch_shuffle=batch_shuffle,
            similarity_fn=similarity_fn,
            data_var=args.data_var,
            centers_list=centers_list,
            center_A_flag=args.center_A_flag,
            center_B_flag=args.center_B_flag,
            zero_activations=args.zero_activations,
            removal_method=removal_method,
            precision=precision,
            random_center_type=random_center_type,
        )

        if args.method == 'max':
            pass
            # compute_coeff(layers_A,
            #                 layers_B,
            #                 Nbits_list,
            #                 n_tokens_list,
            #                 avg_flags,
            #                 diagonal_constraint,
            #                 args.method,
            #                 batch_shuffle,
            #                 centers_list,
            #                 )
        elif args.method == 'min':
            compute_II(
                        output_folder0,
                        layers_A,
                        layers_B,
                        Nbits_list,
                        n_tokens_list,
                        avg_flags,
                        diagonal_constraint,
                        args.method,
                        batch_shuffle,
                        centers_list,
                        center_A_flag=args.center_A_flag,
                        center_B_flag=args.center_B_flag,
                        zero_activations=args.zero_activations,
                        removal_method=removal_method,
                        precision=precision,
                        random_center_type=random_center_type
            )
    


