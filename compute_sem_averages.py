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

from utils import (
                my_languages,
                precision_map,
                torch_to_jax, 
                flatten_tokens_features, 
                makefolder,
                depths,
                emb_dims,
                reduce_list_half_preserve_extremes,
                collect_data,
                collect_data_hf,
                )

from geometry import *
from datapaths import *
import argparse
from time import time

def main(
        layers,
        languages, 
        min_token_length, 
        n_files,
        model,
        output_folder0,
        avg_tokens,
        ):
    
    start_time = time()
    centers = 'sem'
    batch_shuffle = 0
    Nbits = Nbits_list[0]
    all_activations = []
    n_tokens = min_token_length

    if model != 'gemma12b':
        loading_data_f = collect_data
    else:
        loading_data_f = collect_data_hf

    for language_id,language in enumerate(languages):
        all_activations.append(loading_data_f(input_paths[language][model]['matching']['1'][data_var],
                                            min_token_length=min_token_length, 
                                            n_files=n_files,
                                            model_name=model,
                                            avg_tokens=avg_tokens,
                                            )
        )

    for layer_counter,layer in enumerate(layers):
        print(f'{layer=}')
        centers_folder = makefolder(base=output_folder0+f'semantic_centers/',
                                create_folder=True,
                                centers=centers,
                                Nbits=Nbits,
                                n_tokens=n_tokens,
                                avg_tokens=avg_tokens,
                                batch_shuffle=batch_shuffle,
                                layer_A=layer,
                                layer_B=layer,
                                )
        for language_id in range(len(all_activations)):
            print(f'processing {languages[language_id]}')
            activations = all_activations[language_id][f"layer_{layer}"]
            if avg_tokens == 0: 
                activations = flatten_tokens_features(activations)
            activations = torch_to_jax(activations,precision)
            np.save(os.path.join(centers_folder,f"activations_{language_id}.npy"),activations)
    print(f'this took {(time()-start_time)/60:.1f} m')

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ranks")
    parser.add_argument("dbg", type=int)
    parser.add_argument("model", type=str)
    parser.add_argument("min_token_length", type=int)
    parser.add_argument("avg_tokens", type=int)
    args = parser.parse_args()

    precision = 32
    data_var = 'sem'

    layers = list(range(1,depths[args.model] + 1))
    layers = reduce_list_half_preserve_extremes(layers)

    n_files = 21    
    Nbits_list = [0]
    diagonal_constraint = 1
    match_var = 'matching'
    min_token_length = args.min_token_length  

    output_folder0 = makefolder(base=f'./results/',
                                create_folder=True,
                                global_centering=0,
                                spaces='AB',
                                similarity_fn='none',
                                precision=precision,
                                language='english', # The left-out language
                                data_var=data_var,
                                modelA=args.model,
                                modelB=args.model,
                                match_var=match_var,
                                n_files=n_files,
                                min_token_length=min_token_length,
                                )
    main(
        layers,
        my_languages, 
        min_token_length, 
        n_files,
        args.model,
        output_folder0,
        args.avg_tokens,
        )


