import argparse
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils_activations import (  # noqa: E402
    batched_remove_centroid_projections,
    collect_data,
    collect_data_hf,
    cosine_similarity,
    depths,
    preprocessing_sem_data,
    reduce_list_half_preserve_extremes,
    squared_norm_fraction,
)
from utils import makefolder  # noqa: E402
from datapaths import input_paths  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Compute syn/sem norm fractions.")
    parser.add_argument("--model", required=True, choices=["qwen7b", "deepseek", "gemma12b"])
    parser.add_argument("--data-var", default="sem", choices=["sem"])
    parser.add_argument("--n-files", type=int, default=21)
    parser.add_argument("--min-token-length", type=int, default=3)
    parser.add_argument("--n-tokens", type=int, default=1)
    parser.add_argument("--avg-tokens", type=int, default=0, choices=[0, 1])
    parser.add_argument("--global-center-flag", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


def projection_coefficients(act, syn_centroids, sem_centroids):
    syn_frac = squared_norm_fraction(act, syn_centroids)
    sem_frac = squared_norm_fraction(act, sem_centroids)
    residual_frac = 1.0 - syn_frac - sem_frac
    residual_frac = jnp.clip(residual_frac, 0.0, 1.0)
    return {
        "syn": syn_frac,
        "sem": sem_frac,
        "residual": residual_frac,
    }


def main():
    args = parse_args()

    if args.avg_tokens == 1 and args.n_tokens != args.min_token_length:
        raise ValueError(
            f"n_tokens={args.n_tokens} must equal min_token_length={args.min_token_length} when avg_tokens=1"
        )

    loading_f = collect_data if args.model != "gemma12b" else collect_data_hf
    input_path_A = input_paths["english"][args.model]["matching"]["0"][args.data_var]

    all_activations_A = loading_f(
        input_path_A,
        args.min_token_length,
        args.n_files,
        args.model,
        args.avg_tokens,
    )

    layers = list(range(1, depths[args.model] + 1))
    layer_vals = reduce_list_half_preserve_extremes(layers)

    syn_means = []
    sem_means = []
    residual_means = []
    cos_means = []
    cos_stds = []

    for layer in tqdm(layer_vals, desc="layer_vals"):
        act_A, syn_centroids_A_0, sem_centroids_A, _ = preprocessing_sem_data(
            model_name=args.model,
            all_activations=all_activations_A,
            layer=layer,
            space_index="A",
            global_center_flag=args.global_center_flag,
            min_token_length=args.min_token_length,
            avg_tokens=args.avg_tokens,
            n_tokens=args.n_tokens,
        )

        syn_centroids_A = batched_remove_centroid_projections(
            syn_centroids_A_0,
            jnp.arange(sem_centroids_A.shape[0], dtype=jnp.int32),
            sem_centroids_A,
        )

        cos = np.array(cosine_similarity(syn_centroids_A_0, sem_centroids_A))
        cos_means.append(cos.mean())
        cos_stds.append(cos.std())

        fractions = projection_coefficients(act_A, syn_centroids_A, sem_centroids_A)
        syn_means.append(np.mean(np.array(fractions["syn"])))
        sem_means.append(np.mean(np.array(fractions["sem"])))
        residual_means.append(np.mean(np.array(fractions["residual"])))

    resultsfolder = makefolder(
        base=str(REPO_ROOT / "results" / "norms" / "syn-sem") + "/",
        create_folder=True,
        model_name=args.model,
        avg_tokens=args.avg_tokens,
        min_token_length=args.min_token_length,
        n_tokens=args.n_tokens,
        global_center_flag=args.global_center_flag,
    )
    os.makedirs(resultsfolder, exist_ok=True)

    np.savez(
        os.path.join(resultsfolder, "norms.npz"),
        layer_vals=np.array(layer_vals),
        rel_depths=np.array(layer_vals) / depths[args.model],
        syn_means=np.array(syn_means),
        sem_means=np.array(sem_means),
        residual_means=np.array(residual_means),
        cos_means=np.array(cos_means),
        cos_stds=np.array(cos_stds),
    )
    print(f"results saved at {resultsfolder}")


if __name__ == "__main__":
    main()
