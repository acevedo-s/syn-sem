import sys,os
sys.path.append('../')
sys.path.append('../../')

# if __name__ == "__main__":
    # os.environ["JAX_PLATFORMS"] = "cpu"

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
                )

from geometry import *
from datapaths import *
import argparse
from time import time
import torch


def main(
        layers,
        languages, 
        min_token_length, 
        n_files,
        batch_size,
        model,
        n_tokens_list,
        output_folder0,
        avg_flags,
        Nbits_list,
        data_var,
        ):
    start_time = time()
    centers = 'sem'
    batch_shuffle = 0

    all_activations = []
    
    languages = ['italian','german']
    for language_id,language in enumerate(languages):
        all_activations.append(collect_data(input_paths[language][model]['matching']['1'][data_var],
                                            min_token_length=min_token_length, 
                                            n_files=n_files,
                                            )
        )

    for n_tokens_id, n_tokens in enumerate(n_tokens_list):
        print(f'{n_tokens=}')
        for Nbits_id, Nbits in enumerate(Nbits_list):
            print(f'{Nbits=}')
            for layer_counter,layer in enumerate(layers):
                print(f'{layer=}')
                for avg_tokens in avg_flags:
                    print(f'{avg_tokens=}')
                    T=1 if avg_tokens else min_token_length 
                    semantic_center = jnp.zeros(shape=(len(input_paths),
                                                   n_files*batch_size,
                                                   T*emb_dims[model]),dtype=jnp.double)
                    for language_id in range(len(all_activations)):
                        print(f'processing {languages[language_id]}')
                        activations = all_activations[language_id][f"layer_{layer}"]
                        act = bf16_torch_to_jax(activations[:,-n_tokens:,:])
                        act = clip(act).astype(jnp.double) # promote them to double to break massive degeneracies due to small precision
                        if avg_tokens == 1:
                            act = act.mean(axis=1,keepdims=True)
                        semantic_center = semantic_center.at[language_id].set(flatten_tokens_features(act))
                    semantic_center = jnp.mean(semantic_center,axis=0)

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
                    np.save(os.path.join(centers_folder,f"semantic_centers_{layer}"),semantic_center)


    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ranks")
    parser.add_argument("dbg",type=int)
    parser.add_argument("model",type=str)
    parser.add_argument("min_token_length",type=int)

    args = parser.parse_args()

    data_var = 'sem'
    batch_size = 100
    min_token_length = args.min_token_length  

    layers = list(range(1,depths[args.model] + 1))

    if 1:
        layers = reduce_list_half_preserve_extremes(layers)

    Nbits_list = [0]
    avg_flags = [0]
    diagonal_constraint = None
    n_files = None
    n_tokens_list = None
    match_var = 'matching'

    if args.dbg == 0:
        n_tokens_list = np.array([min_token_length])
        n_files = 16
        diagonal_constraint = 1

    elif args.dbg == 1:
        n_files = 1
        n_tokens_list = np.array([min_token_length])
        diagonal_constraint = 1


    print(f'{Nbits_list=}')
    print(f'{avg_flags=}')
    print(f'{diagonal_constraint=}')

    output_folder0 = makefolder(base=f'./results/',
                                create_folder=True,
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
        languages, 
        min_token_length, 
        n_files,
        batch_size,
        args.model,
        n_tokens_list,
        output_folder0,
        avg_flags,
        Nbits_list,
        data_var,
        )


