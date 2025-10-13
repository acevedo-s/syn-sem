#!/bin/bash

dbg=0
modelA="qwen7b"
method="min"
data_var="syn"
match_var="matching"  # "matching" or "mismatching"
centers_var="syn"
# languages=("spanish" "chinese" "arabic" "german" "italian" "turkish")
languages=('english')
zero_activations=0
center_A_flags=(0 1 -1)
# center_B_flag=0
removal_methods=("projection")
global_centerings=(0)
avg_tokens_list=(0 1)
similarity_fn='normalized_L2_distance'

if [ "$data_var" = "syn" ] || [ "$centers_var" = "syn" ]; then
    min_token_length=6
else
    min_token_length=3
fi

if [ "$dbg" -eq 0 ]; then
    gpu_flag="--gres=gpu:1"
else
    gpu_flag="--gres=gpu:0"
fi


for avg_tokens in "${avg_tokens_list[@]}";do
  for language in "${languages[@]}"; do
    for global_centering in "${global_centerings[@]}"; do
      for center_A_flag in "${center_A_flags[@]}"; do
        center_B_flag=$center_A_flag
        for removal_method in "${removal_methods[@]}"; do
          echo "$language, center_A_flag=$center_A_flag, center_B_flag=$center_B_flag, $removal_method, (dbg=$dbg)"
          sbatch $gpu_flag ssend.sh \
            $dbg \
            $min_token_length \
            $modelA \
            $method \
            $data_var \
            $match_var \
            $centers_var \
            $language \
            $center_A_flag \
            $center_B_flag \
            $zero_activations \
            $removal_method \
            $global_centering \
            $avg_tokens \
            $similarity_fn
          sleep .2
        done
      done
    done
  done
done
