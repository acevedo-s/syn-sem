#!/bin/bash

dbg=0
model="deepseek"
min_token_length=3

# parameter sweeps
num_languages=(1) # from 1 to len(my_languages)
avg_tokens_list=(0 1)
n_permutations_list=(0) # from 0 to len(my_languages)-1


for n_permutations in "${n_permutations_list[@]}"; do
    for number_of_languages in "${num_languages[@]}"; do
        for avg_tokens in "${avg_tokens_list[@]}"; do
            echo "num_languages=$number_of_languages, avg_tokens=$avg_tokens"
            sbatch scompute_sem_averages.sh \
            "$dbg" \
            "$model" \
            "$min_token_length" \
            "$number_of_languages" \
            "$avg_tokens" \
            "$n_permutations" 
            sleep 0.2
    done
  done
done
