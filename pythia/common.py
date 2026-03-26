import json
import math
import sys
from pathlib import Path

from transformers import AutoConfig

REPO_ROOT = Path("/home/acevedo/syn-sem")
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from codex_helpers.helpers import resolve_model_source
from modelpaths import model_paths
from utils import (
    collect_data_hf,
    flatten_tokens_features,
    my_languages,
    reduce_list_half_preserve_extremes,
    sem_ids_with_syn_path,
)


PYTHIA_ROOT = REPO_ROOT / "pythia"
DEFAULT_MODEL = "pythia6p9b_step143000"
ENGLISH = "english"


def activation_dir(language, sample_index, model_name=DEFAULT_MODEL):
    return (
        REPO_ROOT
        / "datasets"
        / "activations"
        / "sem"
        / "second"
        / model_name
        / "matching"
        / language
        / str(sample_index)
    )


def syntax_activation_dir(sample_index, model_name=DEFAULT_MODEL):
    dataset_var = "second" if sample_index == 0 else "third"
    return (
        REPO_ROOT
        / "datasets"
        / "activations"
        / "syn"
        / dataset_var
        / model_name
        / "matching"
        / ENGLISH
        / str(sample_index)
    )


def layer_values(model_name=DEFAULT_MODEL):
    model_source = resolve_model_source(model_name, model_paths)
    config = AutoConfig.from_pretrained(model_source)
    return reduce_list_half_preserve_extremes(list(range(1, config.num_hidden_layers + 1)))


def _validate_n_tokens(avg_tokens, min_token_length, n_tokens):
    if n_tokens is None:
        return min_token_length
    if avg_tokens == 1 and n_tokens != min_token_length:
        raise ValueError(
            f"n_tokens={n_tokens} must equal min_token_length={min_token_length} when avg_tokens=1"
        )
    if n_tokens < 1 or n_tokens > min_token_length:
        raise ValueError(f"n_tokens={n_tokens} must be in [1, {min_token_length}]")
    return n_tokens


def load_activations(input_dir, min_token_length, avg_tokens, n_samples, model_name, n_tokens=None):
    n_tokens = _validate_n_tokens(avg_tokens, min_token_length, n_tokens)
    n_files = None if n_samples is None else math.ceil(n_samples / 100)
    activations = collect_data_hf(
        str(input_dir),
        min_token_length=min_token_length,
        n_files=n_files,
        model_name=model_name,
        avg_tokens=avg_tokens,
    )
    for layer_name, tensor in activations.items():
        if n_samples is not None:
            tensor = tensor[:n_samples]
        if avg_tokens == 0:
            tensor = tensor[:, -n_tokens:, :]
            tensor = flatten_tokens_features(tensor)
        activations[layer_name] = tensor
    return activations


def output_root(model_name, avg_tokens, min_token_length, n_samples, n_tokens=None):
    n_tokens = _validate_n_tokens(avg_tokens, min_token_length, n_tokens)
    root = (
        PYTHIA_ROOT
        / "results"
        / f"model_{model_name}"
        / f"avg_tokens_{avg_tokens}"
        / f"min_token_length_{min_token_length}"
        / f"n_samples_{n_samples}"
    )
    if avg_tokens == 0 and n_tokens != min_token_length:
        root = root / f"n_tokens_{n_tokens}"
    return root


def semantic_centers_root(model_name, avg_tokens, min_token_length, n_samples, n_tokens=None):
    return output_root(model_name, avg_tokens, min_token_length, n_samples, n_tokens=n_tokens) / "semantic_centers"


def syntax_centers_root(model_name, avg_tokens, min_token_length, n_samples, n_tokens=None):
    return output_root(model_name, avg_tokens, min_token_length, n_samples, n_tokens=n_tokens) / "syntax_centers"


def norms_root(model_name, avg_tokens, min_token_length, n_samples, global_center_flag, n_tokens=None):
    return (
        output_root(model_name, avg_tokens, min_token_length, n_samples, n_tokens=n_tokens)
        / "norms"
        / f"global_center_flag_{global_center_flag}"
    )


def save_metadata(output_dir, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
