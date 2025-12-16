#!/bin/bash

dbg=0
modelA="qwen7b"
method="min"
data_var="syn"
match_var="matching"  # "matching" or "mismatching"
centers_var="sem"
languages=('english')
zero_activations=0
center_A_flags=(-1)
removal_methods=("projection")
global_centerings=(0)
avg_tokens_list=(0)
similarity_fn='normalized_L2_distance'
batch_shuffle=0

if [ "$data_var" = "syn" ] && [ "$centers_var" = "syn" ] && [ "$batch_shuffle" -eq 0 ]; then
    min_token_length=3
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

        if [ "$data_var" = 'sem' ] && [ "$centers_var" = "syn" ]; then
          center_B_flag=0
        else
          center_B_flag=$center_A_flag
        fi

        for removal_method in "${removal_methods[@]}"; do
          echo "data_var=$data_var, centers_var=$centers_var, center_A_flag=$center_A_flag, center_B_flag=$center_B_flag, $removal_method, (dbg=$dbg)"
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
            $similarity_fn \
            $batch_shuffle
          sleep .2
        done
      done
    done
  done
done
