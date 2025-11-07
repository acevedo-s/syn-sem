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
                )

from geometry import *
from datapaths import *
import argparse
from time import time

def rotate_languages(languages:list, language_list_permutation:int):   
  N = language_list_permutation 
  assert N>=0 and N<len(languages)

  _languages = languages[-N:] + languages[:-N]
  return _languages

def main(
        layers,
        languages, 
        min_token_length, 
        n_files,
        model,
        n_tokens_list,
        output_folder0,
        avg_tokens,
        Nbits_list,
        data_var,
        precision,
        language_list_permutation,
        ):
    start_time = time()
    centers = 'sem'
    batch_shuffle = 0

    all_activations = []
    
    for language_id,language in enumerate(languages):
        all_activations.append(collect_data(input_paths[language][model]['matching']['1'][data_var],
                                            min_token_length=min_token_length, 
                                            n_files=n_files,
                                            model_name=model,
                                            avg_tokens=avg_tokens
                                            )
        )
    if avg_tokens == 0:
        B, _, E = all_activations[0][f"layer_{layers[0]}"].shape
    else:    
        B, E = all_activations[0][f"layer_{layers[0]}"].shape
        T = 1

    for n_tokens_id, n_tokens in enumerate(n_tokens_list):
        print(f'{n_tokens=}')
        if avg_tokens == 0: T = n_tokens

        for Nbits_id, Nbits in enumerate(Nbits_list):
            print(f'{Nbits=}')
            for layer_counter,layer in enumerate(layers):
                print(f'{layer=}')
                semantic_center = jnp.zeros(shape=(len(languages),
                                                B,
                                                T*E),dtype=precision_map[precision])
                for language_id in range(len(all_activations)):
                    print(f'processing {languages[language_id]}')
                    activations = all_activations[language_id][f"layer_{layer}"]
                    if avg_tokens == 0: 
                        activations = flatten_tokens_features(activations[:,-n_tokens:,:])
                    activations = torch_to_jax(activations,precision)
                    semantic_center = semantic_center.at[language_id].set(activations)
                    print(f'{activations.shape=}')
                print(f'{semantic_center.shape=}')
                semantic_center = jnp.mean(semantic_center,axis=0)

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
                np.save(os.path.join(centers_folder,f"semantic_centers_{len(languages)}_{language_list_permutation}.npy"),semantic_center)
    print(f'this took {(time()-start_time)/60:.1f} m')

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ranks")
    parser.add_argument("dbg", type=int)
    parser.add_argument("model", type=str)
    parser.add_argument("min_token_length", type=int)
    parser.add_argument("number_of_languages", type=int)
    parser.add_argument("avg_tokens", type=int)
    parser.add_argument("language_list_permutation", type=int)

    args = parser.parse_args()

    precision = 32
    data_var = 'sem'

    layers = list(range(1,depths[args.model] + 1))

    if 1:
        layers = reduce_list_half_preserve_extremes(layers)

    n_files = 21    
    Nbits_list = [0]
    diagonal_constraint = 1
    match_var = 'matching'

    languages = rotate_languages(my_languages,args.language_list_permutation)
    languages = languages[:args.number_of_languages]
    print(f'{languages=}')


    min_token_length = args.min_token_length  
    n_tokens_list = np.array([min_token_length])


    print(f'{Nbits_list=}')
    print(f'{diagonal_constraint=}')

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
        languages, 
        min_token_length, 
        n_files,
        args.model,
        n_tokens_list,
        output_folder0,
        args.avg_tokens,
        Nbits_list,
        data_var,
        precision,
        language_list_permutation=args.language_list_permutation,
        )


