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

def get_model_depths():
    from transformers import AutoConfig

    depths = {}
    for name, path in model_paths.items():
        config = AutoConfig.from_pretrained(path)
        num_hidden_layers = None

        if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
            num_hidden_layers = config.text_config.num_hidden_layers
        elif hasattr(config, "num_hidden_layers"):
            num_hidden_layers = config.num_hidden_layers

        if num_hidden_layers is None:
            raise ValueError(f"Could not determine num_hidden_layers for model '{name}'")

        depths[name] = num_hidden_layers

    return depths

if __name__ == "__main__":
    # from transformers import AutoConfig
    # _test_model = "gemma12b"
    # config = AutoConfig.from_pretrained(model_paths[_test_model])
    # # print(config)
    # print(f'{config.text_config.num_hidden_layers=}')
    # # print(f'{config.text_config.=}')
    depths = get_model_depths()
    print(f'{depths=}')

