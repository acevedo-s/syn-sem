#!/bin/bash

dbg=0
model="qwen7b"
min_token_length=3

avg_tokens_list=(0 1)


for avg_tokens in "${avg_tokens_list[@]}"; do
  echo "avg_tokens=$avg_tokens" 
  sbatch scompute_sem_averages.sh \
  "$dbg" \
  "$model" \
  "$min_token_length" \
  "$avg_tokens" 
  sleep 0.2
done
