import argparse
from datetime import datetime

import numpy as np
import torch

from common import (
    activation_dir,
    cosines_root,
    layer_values,
    load_activations,
    save_metadata,
    select_initial_middle_last_layers,
    syntax_activation_dir,
)


DEFAULT_MODEL = "pythia6p9b_step0"
DEFAULT_AVG_TOKENS = 0
DEFAULT_MIN_TOKEN_LENGTH = 3
DEFAULT_N_CHUNKS = 21
DEFAULT_SEED = 0
DEFAULT_GLOBAL_CENTER_FLAG = 1
DEFAULT_N_TOKENS = None


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Pythia cosine summaries for syntax, semantics, and shuffled pairs.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--avg-tokens", type=int, choices=[0, 1], default=DEFAULT_AVG_TOKENS)
    parser.add_argument("--min-token-length", type=int, default=DEFAULT_MIN_TOKEN_LENGTH)
    parser.add_argument("--n-tokens", type=int, default=DEFAULT_N_TOKENS)
    parser.add_argument("--n-chunks", type=int, default=DEFAULT_N_CHUNKS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--global-center-flag", type=int, choices=[0, 1], default=DEFAULT_GLOBAL_CENTER_FLAG)
    return parser.parse_args()


def cosine_similarity(act_a, act_b, eps=1e-8):
    numerator = torch.sum(act_a * act_b, dim=1)
    denominator = torch.linalg.norm(act_a, dim=1) * torch.linalg.norm(act_b, dim=1)
    return numerator / denominator.clamp_min(eps)


def load_pair(input_dir_a, input_dir_b, args):
    n_samples = args.n_chunks * 100
    activations_a = load_activations(
        input_dir=input_dir_a,
        min_token_length=args.min_token_length,
        avg_tokens=args.avg_tokens,
        n_samples=n_samples,
        model_name=args.model,
        n_tokens=args.n_tokens,
    )
    activations_b = load_activations(
        input_dir=input_dir_b,
        min_token_length=args.min_token_length,
        avg_tokens=args.avg_tokens,
        n_samples=n_samples,
        model_name=args.model,
        n_tokens=args.n_tokens,
    )
    return activations_a, activations_b


def pairwise_cosine_stats(activations_a, activations_b, layers):
    means = []
    stds = []
    counts = []
    distributions = []
    for layer in layers:
        act_a = activations_a[f"layer_{layer}"].to(dtype=torch.float32)
        act_b = activations_b[f"layer_{layer}"].to(dtype=torch.float32)
        if act_a.shape != act_b.shape:
            raise ValueError(f"Shape mismatch at layer {layer}: {act_a.shape} vs {act_b.shape}")
        cosine_values = cosine_similarity(act_a, act_b)
        distributions.append(cosine_values.cpu().numpy().astype(np.float32))
        means.append(float(cosine_values.mean().item()))
        stds.append(float(cosine_values.std(unbiased=False).item()))
        counts.append(int(act_a.shape[0]))
    return means, stds, counts, np.stack(distributions, axis=0)


def center_activations(activations):
    centered = {}
    for layer_name, tensor in activations.items():
        tensor = tensor.to(dtype=torch.float32)
        centered[layer_name] = tensor - tensor.mean(dim=0, keepdim=True)
    return centered


def shuffled_copy(activations, seed):
    rng = np.random.default_rng(seed)
    shuffled = {}
    for layer_name, tensor in activations.items():
        permutation = torch.as_tensor(rng.permutation(tensor.shape[0]), dtype=torch.long)
        shuffled[layer_name] = tensor.index_select(0, permutation)
    return shuffled


def main():
    args = parse_args()
    candidate_layers = layer_values(args.model)
    selected_layers = select_initial_middle_last_layers(candidate_layers)
    rel_depths = np.array(selected_layers, dtype=np.float32) / float(candidate_layers[-1])

    output_dir = cosines_root(
        model_name=args.model,
        avg_tokens=args.avg_tokens,
        min_token_length=args.min_token_length,
        n_chunks=args.n_chunks,
        global_center_flag=args.global_center_flag,
        n_tokens=args.n_tokens,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    semantic_dir_a = activation_dir(language="english", sample_index=0, model_name=args.model)
    semantic_dir_b = activation_dir(language="english", sample_index=1, model_name=args.model)
    syntax_dir_a = syntax_activation_dir(sample_index=0, model_name=args.model)
    syntax_dir_b = syntax_activation_dir(sample_index=1, model_name=args.model)

    print(f"selected_layers={selected_layers}")
    print(f"loading semantic pair from {semantic_dir_a} and {semantic_dir_b}")
    semantic_a, semantic_b = load_pair(semantic_dir_a, semantic_dir_b, args)
    if args.global_center_flag:
        print("global-centering semantic activations")
        semantic_a = center_activations(semantic_a)
        semantic_b = center_activations(semantic_b)
    semantic_means, semantic_stds, semantic_counts, semantic_cosines = pairwise_cosine_stats(
        semantic_a,
        semantic_b,
        selected_layers,
    )

    print("computing shuffled semantic baseline")
    shuffled_semantic_b = shuffled_copy(semantic_b, seed=args.seed)
    random_means, random_stds, random_counts, random_cosines = pairwise_cosine_stats(
        semantic_a,
        shuffled_semantic_b,
        selected_layers,
    )
    del semantic_a, semantic_b, shuffled_semantic_b

    print(f"loading syntax pair from {syntax_dir_a} and {syntax_dir_b}")
    syntax_a, syntax_b = load_pair(syntax_dir_a, syntax_dir_b, args)
    if args.global_center_flag:
        print("global-centering syntax activations")
        syntax_a = center_activations(syntax_a)
        syntax_b = center_activations(syntax_b)
    syntax_means, syntax_stds, syntax_counts, syntax_cosines = pairwise_cosine_stats(
        syntax_a,
        syntax_b,
        selected_layers,
    )
    del syntax_a, syntax_b

    np.savez(
        output_dir / "cosines.npz",
        layers=np.array(selected_layers, dtype=np.int32),
        rel_depths=rel_depths,
        syntax_cosines=syntax_cosines,
        syntax_means=np.array(syntax_means, dtype=np.float32),
        syntax_stds=np.array(syntax_stds, dtype=np.float32),
        semantic_cosines=semantic_cosines,
        semantic_means=np.array(semantic_means, dtype=np.float32),
        semantic_stds=np.array(semantic_stds, dtype=np.float32),
        random_cosines=random_cosines,
        random_means=np.array(random_means, dtype=np.float32),
        random_stds=np.array(random_stds, dtype=np.float32),
        syntax_counts=np.array(syntax_counts, dtype=np.int32),
        semantic_counts=np.array(semantic_counts, dtype=np.int32),
        random_counts=np.array(random_counts, dtype=np.int32),
    )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "avg_tokens": args.avg_tokens,
        "min_token_length": args.min_token_length,
        "n_tokens": args.n_tokens if args.n_tokens is not None else args.min_token_length,
        "n_chunks": args.n_chunks,
        "seed": args.seed,
        "global_center_flag": args.global_center_flag,
        "candidate_layers": candidate_layers,
        "selected_layers": selected_layers,
        "semantic_dir_A": str(semantic_dir_a),
        "semantic_dir_B": str(semantic_dir_b),
        "syntax_dir_A": str(syntax_dir_a),
        "syntax_dir_B": str(syntax_dir_b),
        "syntax_counts": syntax_counts,
        "syntax_stds": syntax_stds,
        "semantic_counts": semantic_counts,
        "semantic_stds": semantic_stds,
        "random_counts": random_counts,
        "random_stds": random_stds,
    }
    save_metadata(output_dir, metadata)
    print(output_dir / "cosines.npz")


if __name__ == "__main__":
    main()
