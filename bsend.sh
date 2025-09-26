#!/bin/bash

dbg=0
min_token_length=6
modelA="qwen7b"
method="min"
data_var="syn"
centers_var="syn"
# languages=("english" "spanish" "chinese" "arabic" "german" "italian" "turkish")
languages=('english')
zero_activations=0
random_center_types=("shuffled")
center_A_flags=(0)
# center_B_flag=0
removal_methods=("subtraction")
global_centerings=(0)
avg_tokens=1

if [ "$dbg" -eq 0 ]; then
    gpu_flag="--gres=gpu:1"
else
    gpu_flag="--gres=gpu:0"
fi

for language in "${languages[@]}"; do
  for global_centering in "${global_centerings[@]}"; do
    for center_A_flag in "${center_A_flags[@]}"; do
      center_B_flag=$center_A_flag
      for removal_method in "${removal_methods[@]}"; do
        for random_center_type in "${random_center_types[@]}"; do
          echo "$language, center_A_flag=$center_A_flag, center_B_flag=$center_B_flag, $removal_method, random_center_type=$random_center_type (dbg=$dbg)"
          sbatch $gpu_flag ssend.sh \
            $dbg \
            $min_token_length \
            $modelA \
            $method \
            $data_var \
            $centers_var \
            $language \
            $center_A_flag \
            $center_B_flag \
            $zero_activations \
            $removal_method \
            $random_center_type \
            $global_centering \
            $avg_tokens
          sleep .5
        done
      done
    done
  done
done
