# input_paths = {}

from collections import defaultdict

input_paths = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

languages = ['english','italian']
models = ['llama', 'deepseek']
match_types = ['matching', 'mismatching']
txt_vars = ['syn', 'sem']
indices = range(2)

for language in languages:
    for model in models:
        for match in match_types:
            for idx in indices:
                input_paths[language][model][match][str(idx)] = {
                    txt_var: f"/home/acevedo/syn-sem/datasets/activations/{txt_var}/second/{model}/{match}/{language}/{idx}/"
                    for txt_var in txt_vars
                }

# Optional: convert back to plain dict
input_paths = {
    lang: {
        model: {
            match: dict(paths) for match, paths in match_dict.items()
        } for model, match_dict in model_dict.items()
    } for lang, model_dict in input_paths.items()
}