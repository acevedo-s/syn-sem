#!/bin/bash

# Fixed arguments
dbg=0
min_token_length=6
modelA="qwen7b"
method="min"
data_var="sem"
language="english"
zero_activations=0
random_center_types=("permuted")

flags=(1 -1)
removal_methods=("projection") 

# Decide GPU allocation
if [ "$dbg" -eq 0 ]; then
    gpu_flag="--gres=gpu:1"
else
    gpu_flag="--gres=gpu:0"
fi

for flag in "${flags[@]}"; do
  for removal_method in "${removal_methods[@]}"; do
    for random_center_type in "${random_center_types[@]}"; do
      echo "Submitting with center_A_flag=$flag, center_B_flag=$flag, removal_method=$removal_method, random_center_type=$random_center_type (dbg=$dbg)"
      sbatch $gpu_flag scompute.sh \
        $dbg \
        $min_token_length \
        $modelA \
        $method \
        $data_var \
        $language \
        $flag \
        $flag \
        $zero_activations \
        $removal_method \
        $random_center_type
      sleep .5
    done
  done
done
