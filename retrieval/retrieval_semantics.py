import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
RESULTS_BASE = str(THIS_DIR / "results" / "recall" / "sem") + "/"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils_activations import *  # noqa: F401,F403
from utils import makefolder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute semantic retrieval recall curves and save them as an .npz file."
    )
    parser.add_argument("--model-name", type=str, default="deepseek")
    parser.add_argument("--n-files", type=int, default=21)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--data-var", type=str, default="sem")
    parser.add_argument("--global-center-flag", type=int, default=1)
    parser.add_argument("--min-token-length", type=int, default=3)
    parser.add_argument("--n-tokens", type=int, default=None)
    parser.add_argument("--k-recall", type=int, default=3)
    parser.add_argument("--avg-tokens", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model_name
    n_files = args.n_files
    precision = args.precision
    data_var = args.data_var
    global_center_flag = args.global_center_flag
    min_token_length = args.min_token_length
    n_tokens = args.n_tokens if args.n_tokens is not None else min_token_length
    k_recall = args.k_recall
    avg_tokens = args.avg_tokens

    if model_name == "gemma12b":
        load_data_f = collect_data_hf
    else:
        load_data_f = collect_data

    input_path_A = input_paths["english"][model_name]["matching"]["0"][data_var]
    input_path_B = input_paths["english"][model_name]["matching"]["1"][data_var]

    all_activations_A = load_data_f(
        input_path_A,
        min_token_length,
        n_files,
        model_name,
        avg_tokens,
    )
    all_activations_B = load_data_f(
        input_path_B,
        min_token_length,
        n_files,
        model_name,
        avg_tokens,
    )

    layers = list(range(1, depths[model_name] + 1))
    layer_vals = reduce_list_half_preserve_extremes(layers)

    recalls_0 = np.empty(len(layer_vals))
    recalls_sem = np.empty_like(recalls_0)
    recalls_syn = np.empty_like(recalls_0)
    recalls_sem_perm = np.empty_like(recalls_0)
    recalls_syn_perm = np.empty_like(recalls_0)

    key_idx = jax.random.PRNGKey(42)

    for layer_id, layer in enumerate(tqdm(layer_vals, desc="layer_vals")):
        act_A, syn_centroids_A, sem_centroids_A, global_center_A = preprocessing_sem_data(
            model_name=model_name,
            all_activations=all_activations_A,
            layer=layer,
            space_index="A",
            global_center_flag=global_center_flag,
            min_token_length=min_token_length,
            avg_tokens=avg_tokens,
            n_tokens=n_tokens,
        )
        act_B, _, _, global_center_B = preprocessing_sem_data(
            model_name=model_name,
            all_activations=all_activations_B,
            layer=layer,
            space_index="B",
            global_center_flag=global_center_flag,
            min_token_length=min_token_length,
            avg_tokens=avg_tokens,
            n_tokens=n_tokens,
            centroids=False,
        )

        _ = precision, global_center_A, global_center_B

        idx = jnp.arange(act_A.shape[0], dtype=jnp.int32)
        permuted_idx = jax.random.permutation(key_idx, act_A.shape[0])

        syn_centroids_A = batched_remove_centroid_projections(
            syn_centroids_A, idx, sem_centroids_A
        )

        cos_matrix_0 = all_cosine_similarities(act_A, act_B)
        recalls_0[layer_id] = float(recall_at_k_jax(cos_matrix_0, k_recall))

        act_A_sem = batched_remove_centroid_projections(act_A, idx, sem_centroids_A)
        cos_matrix_sem = all_cosine_similarities(act_A_sem, act_B)
        recalls_sem[layer_id] = float(recall_at_k_jax(cos_matrix_sem, k_recall))

        act_A_syn = batched_remove_centroid_projections(act_A, idx, syn_centroids_A)
        cos_matrix_syn = all_cosine_similarities(act_A_syn, act_B)
        recalls_syn[layer_id] = float(recall_at_k_jax(cos_matrix_syn, k_recall))

        act_A_sem_perm = batched_remove_centroid_projections(
            act_A, permuted_idx, sem_centroids_A
        )
        cos_matrix_sem_perm = all_cosine_similarities(act_A_sem_perm, act_B)
        recalls_sem_perm[layer_id] = float(recall_at_k_jax(cos_matrix_sem_perm, k_recall))

        act_A_syn_perm = batched_remove_centroid_projections(
            act_A, permuted_idx, syn_centroids_A
        )
        cos_matrix_syn_perm = all_cosine_similarities(act_A_syn_perm, act_B)
        recalls_syn_perm[layer_id] = float(recall_at_k_jax(cos_matrix_syn_perm, k_recall))

    resultsfolder = makefolder(
        base=RESULTS_BASE,
        create_folder=True,
        model_name=model_name,
        avg_tokens=avg_tokens,
        min_token_length=min_token_length,
        global_center_flag=global_center_flag,
        k=k_recall,
    )
    layer_vals_arr = np.array(layer_vals)
    rel_depths = layer_vals_arr / depths[model_name]

    save_path = os.path.join(resultsfolder, f"recall_k{k_recall}.npz")
    np.savez(
        save_path,
        layer_vals=layer_vals_arr,
        rel_depths=rel_depths,
        recalls_0=recalls_0,
        recalls_sem=recalls_sem,
        recalls_syn=recalls_syn,
        recalls_sem_perm=recalls_sem_perm,
        recalls_syn_perm=recalls_syn_perm,
    )
    print(f"Saved results to {save_path}")


if __name__ == "__main__":
    main()
