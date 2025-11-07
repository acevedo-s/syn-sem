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
                compute_coeff,
                compute_II,
                )



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ranks")
    parser.add_argument("dbg", type=int)
    parser.add_argument("min_token_length", type=int)
    parser.add_argument("modelA", type=str)
    parser.add_argument("method", type=str, choices=['max','min'], help="max = corr coeff, min = II")
    parser.add_argument("data_var", type=str, choices=['syn','sem'], help="syntax or semantics")
    parser.add_argument("match_var", type=str, choices=['matching','mismatching'])
    parser.add_argument("centers_var", type=str, choices=['syn','sem'])
    parser.add_argument("language", type=str)
    parser.add_argument("center_A_flag", type=int)
    parser.add_argument("center_B_flag", type=int)
    parser.add_argument("zero_activations", type=int)
    parser.add_argument("removal_method", type=str, choices=['projection', 'subtraction', 'none'])
    parser.add_argument("global_centering", type=int, choices=[0,1])
    parser.add_argument("avg_tokens", type=int, choices=[0,1])
    parser.add_argument("similarity_fn", type=str, choices=['L2_distance','normalized_L2_distance'])
    args = parser.parse_args()

    spaces = 'AB'
    Nbits_list = [0]
    diagonal_constraint = 1
    n_files = 21
    n_tokens_list = []
    match_var_list = [] # in ['matching','mismatching']
    similarity_fn = lambda x: x
    input_path_B = ' '
    modelB = args.modelA
    precision = 32
    batch_shuffle = 0

    min_token_length = args.min_token_length
    n_tokens_list = np.array([args.min_token_length])

    removal_method = args.removal_method
    if removal_method == 'none':
        removal_method = None
        assert args.center_A_flag == 0 and args.center_B_flag == 0
    
    if args.center_A_flag == 0 and args.center_B_flag == 0:
        removal_method = None


    layers_A = list(range(1,depths[args.modelA] + 1))
    layers_B = list(range(1,depths[modelB] + 1))

    if 1:
        layers_A = reduce_list_half_preserve_extremes(layers_A)
        layers_B = reduce_list_half_preserve_extremes(layers_B)

    print(f'{Nbits_list=}')
    print(f'{diagonal_constraint=}')

    if args.similarity_fn == 'normalized_L2_distance':
        similarity_fn = normalized_L2_distance
    elif args.similarity_fn == 'L2_distance':
        similarity_fn = L2_distance 
    assert 1 not in Nbits_list

    input_path_A = input_paths['english'][args.modelA][args.match_var]['0'][args.data_var]
    print("Input path A = ", input_path_A, flush=True)
    if spaces == 'AB':
        input_path_B = input_paths[args.language][modelB][args.match_var]['1'][args.data_var]
        print("Input path B = ", input_path_B, flush=True)
    
    output_folder0 = makefolder(base=f'./results/',
                            create_folder=True,
                            global_centering=args.global_centering,
                            spaces=spaces,
                            similarity_fn=similarity_fn.__name__,
                            precision=precision,
                            language=args.language,
                            data_var=args.data_var,
                            modelA=args.modelA,
                            modelB=modelB,
                            match_var=args.match_var,
                            n_files=n_files,
                            min_token_length=min_token_length,
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
        avg_tokens=args.avg_tokens,
        Nbits_list=Nbits_list,
        diagonal_constraint=diagonal_constraint,
        batch_shuffle=batch_shuffle,
        similarity_fn=similarity_fn,
        data_var=args.data_var,
        centers_var=args.centers_var,
        center_A_flag=args.center_A_flag,
        center_B_flag=args.center_B_flag,
        zero_activations=args.zero_activations,
        removal_method=removal_method,
        precision=precision,
        spaces=spaces,
        global_centering=args.global_centering
    )


    compute_coeff(
            output_folder0,
            layers_A,
            layers_B,
            Nbits_list,
            n_tokens_list,
            args.avg_tokens,
            diagonal_constraint,
            # args.method,
            batch_shuffle,
            args.centers_var,
            center_A_flag=args.center_A_flag,
            center_B_flag=args.center_B_flag,
            zero_activations=args.zero_activations,
            removal_method=removal_method,
            precision=precision,
            # n_jack_seeds=n_jack_seeds,
    )
    compute_II(
            output_folder0,
            layers_A,
            layers_B,
            Nbits_list,
            n_tokens_list,
            args.avg_tokens,
            diagonal_constraint,
            args.method,
            batch_shuffle,
            args.centers_var,
            center_A_flag=args.center_A_flag,
            center_B_flag=args.center_B_flag,
            zero_activations=args.zero_activations,
            removal_method=removal_method,
            precision=precision,
            # n_jack_seeds=n_jack_seeds,
    )



