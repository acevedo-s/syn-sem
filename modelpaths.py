model_paths = {}
base_path = f'/home/acevedo/LLMs/'

###  models
model_paths["deepseek"] = "/home/rende/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/86518964eaef84e3fdd98e9861759a1384f9c29d"
model_paths["llama8b"] = f"{base_path}LLMs_meta-llama/Meta-Llama-3-8B/"
model_paths['qwen7b'] =  f"{base_path}/qwen2-7b-local/"
model_paths['gemma12b'] = f'{base_path}google/gemma-3-12b-pt/'
model_paths['bert'] = f'{base_path}LLMs_google-bert/bert-base-multilingual-cased/'

repo_ids = {
    'gemma12b' : "google/gemma-3-12b-pt",
}
