# input_paths = {}

from collections import defaultdict

input_paths = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

_languages = ['english','turkish','arabic','italian','spanish','german','chinese']
_models = ['deepseek','qwen7b']
_match_types = ['matching', 'mismatching']
_txt_vars = ['syn', 'sem']
_indices = range(2)

for language in _languages:
    for model in _models:
        for match in _match_types:
            for idx in _indices:
                input_paths[language][model][match][str(idx)] = {
                    txt_var: f"/home/acevedo/syn-sem/datasets/activations/{txt_var}/second/{model}/{match}/{language}/{idx}/"
                    for txt_var in _txt_vars
                }

# Optional: convert back to plain dict
input_paths = {
    lang: {
        model: {
            match: dict(paths) for match, paths in match_dict.items()
        } for model, match_dict in model_dict.items()
    } for lang, model_dict in input_paths.items()
}