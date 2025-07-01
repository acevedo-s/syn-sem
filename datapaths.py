input_paths = {}

###  models
input_paths["deepseek"] = {"matching": {},
                           "missmatching":{},
                        }
input_paths["llama"] = {"matching": {},
                        "missmatching":{},
                        }

for model in ['llama','deepseek']:
    for match_var in [ "matching", "missmatching" ]:
        for index in range(2):
            input_paths[model][match_var][f"{index}"] = f"/home/acevedo/syn-sem/datasets/activations/{model}/{match_var}/{index}/"

# ### source languages
# input_paths["deepseek"]["english"] = {"spanish": {"source":{},"target":{}}, 
#                                       "italian": {"source":{},"target":{}}}
# input_paths["deepseek"]["spanish"] = {"italian": {"source":{},"target":{}},}
# input_paths["deepseek"]["german"] = {"hungarian": {"source":{},"target":{}},}

# input_paths["llama"]["english"] = {"spanish": {"source":{},"target":{}}, 
#                                    "italian": {"source":{},"target":{}}}
# input_paths["llama"]["spanish"] = {"italian": {"source":{},"target":{}},}
# input_paths["llama"]["random"] = {}

# ### min_token_length
# input_paths["deepseek"]["english"]["spanish"]['source']['40'] = "/home/laios/activations/opus_english_spanish/deepseek/deepseek_english"
# input_paths["deepseek"]["english"]["spanish"]['target']['40'] = "/home/laios/activations/opus_english_spanish/deepseek/deepseek_spanish"
# input_paths["deepseek"]["english"]["italian"]['source']['40'] = "/home/laios/activations/opus_english_italian/deepseek_source"
# input_paths["deepseek"]["english"]["italian"]['target']['40'] = "/home/laios/activations/opus_english_italian/deepseek_target"

# input_paths["deepseek"]["english"]["spanish"]["source"]['100'] = "/home/laios/activations/opus_english_spanish/deepseek/100_200_tokens/deepseek_source"
# input_paths["deepseek"]["english"]["spanish"]["target"]['100'] = "/home/laios/activations/opus_english_spanish/deepseek/100_200_tokens/deepseek_target"

# input_paths["llama"]["english"]["spanish"]["source"]['20'] = "/home/laios/deepseek-hidden-states/SA/extract_activations/llama_english_from_deepseek_tokenization"
# input_paths["llama"]["english"]["spanish"]["target"]['20'] = "/home/laios/deepseek-hidden-states/SA/extract_activations/llama_spanish_from_deepseek_tokenization"

# input_paths["llama"]["english"]["spanish"]["source"]['40'] = "/home/laios/activations/opus_english_spanish/llama-8b/llama_english"
# input_paths["llama"]["english"]["spanish"]["target"]['40'] = "/home/laios/activations/opus_english_spanish/llama-8b/llama_spanish"
# input_paths["llama"]["english"]["spanish"]["source"]['100'] = "/home/laios/activations/opus_english_spanish/llama-8b/tokens_100_200/llama_english_100"
# input_paths["llama"]["english"]["spanish"]["target"]['100'] = "/home/laios/activations/opus_english_spanish/llama-8b/tokens_100_200/llama_spanish_100"

# input_paths["deepseek"]["spanish"]["italian"]["source"]['40'] = "/home/laios/activations/opus_spanish_italian/deepseek_source"
# input_paths["deepseek"]["spanish"]["italian"]["target"]['40'] = "/home/laios/activations/opus_spanish_italian/deepseek_target"
# input_paths["deepseek"]["german"]["hungarian"]["source"]['40'] = f"/home/laios/activations/opus_german_hungarian/deepseek_german"
# input_paths["deepseek"]["german"]["hungarian"]["target"]['40'] = f"/home/laios/activations/opus_german_hungarian/deepseek_hungarian"
