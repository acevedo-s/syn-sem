import argparse
from datetime import datetime

import numpy as np
import torch

from common import (
    ENGLISH,
    activation_dir,
    layer_values,
    load_activations,
    my_languages,
    norms_root,
    save_metadata,
    sem_ids_with_syn_path,
    semantic_centers_root,
    syntax_centers_root,
)
from utils import syn_group_id_paths_for_sem_data


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Pythia syntax-semantic norm summaries.")
    parser.add_argument("--model", default="pythia6p9b_step143000")
    parser.add_argument("--min-token-length", type=int, default=3)
    parser.add_argument("--avg-tokens", type=int, choices=[0, 1], required=True)
    parser.add_argument("--n-tokens", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--permute-syntax-centroids", type=int, choices=[0, 1], default=0)
    parser.add_argument("--shuffle-tokens", type=int, choices=[0, 1], default=0)
    parser.add_argument("--activation-match-var", default="matching")
    return parser.parse_args()


def projected_squared_norm(act, centroid, eps=1e-8):
    dot = torch.sum(act * centroid, dim=1, keepdim=True)
    centroid_norm_sq = torch.sum(centroid * centroid, dim=1, keepdim=True).clamp_min(eps)
    projection = (dot / centroid_norm_sq) * centroid
    return torch.sum(projection * projection, dim=1)


def squared_norm_fraction(act, centroid, eps=1e-8):
    return projected_squared_norm(act, centroid, eps=eps) / torch.sum(act * act, dim=1).clamp_min(eps)


def slice_last_token_features(tensor, min_token_length, n_tokens):
    if n_tokens is None or n_tokens == min_token_length:
        return tensor
    hidden_size = tensor.shape[-1] // min_token_length
    if hidden_size * min_token_length != tensor.shape[-1]:
        raise ValueError(
            f"Cannot split feature dimension {tensor.shape[-1]} into {min_token_length} token blocks"
        )
    return tensor[..., -hidden_size * n_tokens :]


def load_semantic_centroid(output_dir, layer, device, min_token_length, n_tokens):
    layer_dir = output_dir / f"layer_{layer}"
    activation_paths = [layer_dir / f"activations_{language_id}.npy" for language_id in range(len(my_languages))]
    if all(path.exists() for path in activation_paths):
        stacked = np.stack([np.load(path).astype(np.float32) for path in activation_paths], axis=0)
        mean_np = stacked.mean(axis=0, dtype=np.float32)
        tensor = torch.from_numpy(mean_np).to(device=device, dtype=torch.float32)
        return slice_last_token_features(tensor, min_token_length, n_tokens)

    mean_path = layer_dir / "semantic_centroid_mean.npy"
    if not mean_path.exists():
        raise FileNotFoundError(f"Missing semantic centroids for layer {layer}: {mean_path}")
    tensor = torch.from_numpy(np.load(mean_path)).to(device=device, dtype=torch.float32)
    return slice_last_token_features(tensor, min_token_length, n_tokens)


def load_syntax_centroid(output_dir, layer, device, space_index, min_token_length, n_tokens, sample_ids=None):
    path = output_dir / f"layer_{layer}" / f"syn_centers_{space_index}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing syntax centroids for layer {layer}: {path}")
    centers = torch.from_numpy(np.load(path)).to(device=device, dtype=torch.float32)
    centers = slice_last_token_features(centers, min_token_length, n_tokens)
    if sample_ids is None:
        return centers
    return centers.index_select(0, sample_ids)


def remove_centroid_projections(act, centroids, eps=1e-8):
    proj_coeffs = torch.sum(act * centroids, dim=1, keepdim=True) / (
        torch.sum(centroids * centroids, dim=1, keepdim=True).clamp_min(eps)
    )
    return act - proj_coeffs * centroids


def permuted_index(tensor, permutation):
    return tensor.index_select(0, permutation)


def reshape_token_blocks(tensor, n_token_blocks):
    hidden_size = tensor.shape[1] // n_token_blocks
    if hidden_size * n_token_blocks != tensor.shape[1]:
        raise ValueError(
            f"Cannot split feature dimension {tensor.shape[1]} into {n_token_blocks} token blocks"
        )
    return tensor.reshape(tensor.shape[0], n_token_blocks, hidden_size)


def shuffle_token_blocks(tensor, n_token_blocks, permutation):
    token_blocks = reshape_token_blocks(tensor, n_token_blocks)
    flat_blocks = token_blocks.reshape(-1, token_blocks.shape[-1])
    shuffled_blocks = flat_blocks.index_select(0, permutation)
    return shuffled_blocks.reshape(token_blocks.shape[0], n_token_blocks, token_blocks.shape[-1]).reshape(tensor.shape)


def load_semantic_syntax_alignment(n_samples, device, space_index="A"):
    sem_ids_np = np.loadtxt(sem_ids_with_syn_path, dtype=int)
    syn_group_ids_np = np.loadtxt(syn_group_id_paths_for_sem_data[space_index], dtype=int)
    if len(sem_ids_np) != len(syn_group_ids_np):
        raise ValueError(
            "Semantic-to-syntax alignment files have different lengths: "
            f"{len(sem_ids_np)} vs {len(syn_group_ids_np)}"
        )

    keep_mask = sem_ids_np < n_samples
    sem_ids_np = sem_ids_np[keep_mask]
    syn_group_ids_np = syn_group_ids_np[keep_mask]

    sem_ids = torch.from_numpy(sem_ids_np).to(device=device, dtype=torch.long)
    syn_group_ids = torch.from_numpy(syn_group_ids_np).to(device=device, dtype=torch.long)
    return sem_ids, syn_group_ids


def main():
    args = parse_args()
    global_center_flag = 1
    if args.shuffle_tokens == 1 and args.avg_tokens != 0:
        raise ValueError("--shuffle-tokens is only supported for concatenated representations (avg_tokens=0)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sem_ids, syn_group_ids_for_sem = load_semantic_syntax_alignment(
        args.n_samples,
        device,
        space_index="A",
    )
    layers = layer_values(args.model)
    centers_n_tokens = args.min_token_length if args.avg_tokens == 0 else args.n_tokens
    sem_root = semantic_centers_root(
        args.model,
        args.avg_tokens,
        args.min_token_length,
        args.n_samples,
        n_tokens=centers_n_tokens,
    )
    syn_root = syntax_centers_root(
        args.model,
        args.avg_tokens,
        args.min_token_length,
        args.n_samples,
        n_tokens=centers_n_tokens,
    )
    output_dir = norms_root(
        args.model,
        args.avg_tokens,
        args.min_token_length,
        args.n_samples,
        global_center_flag,
        n_tokens=args.n_tokens,
    )
    if args.activation_match_var != "matching":
        output_dir = output_dir / f"activation_match_var_{args.activation_match_var}"

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "avg_tokens": args.avg_tokens,
        "global_center_flag": global_center_flag,
        "layers": layers,
        "min_token_length": args.min_token_length,
        "model": args.model,
        "n_tokens": args.n_tokens,
        "n_samples": args.n_samples,
        "effective_semantic_syntax_samples": int(sem_ids.numel()),
        "centers_n_tokens": centers_n_tokens,
        "permute_syntax_centroids": args.permute_syntax_centroids,
        "shuffle_tokens": args.shuffle_tokens,
        "activation_match_var": args.activation_match_var,
        "semantic_centers_dir": str(sem_root),
        "syntax_centers_dir": str(syn_root),
        "uses_syn_centroids": True,
    }
    save_metadata(output_dir, metadata)

    english_dir = activation_dir(
        language=ENGLISH,
        sample_index=0,
        model_name=args.model,
        match_var=args.activation_match_var,
    )
    english_activations = load_activations(
        input_dir=english_dir,
        min_token_length=args.min_token_length,
        avg_tokens=args.avg_tokens,
        n_samples=args.n_samples,
        model_name=args.model,
        n_tokens=args.n_tokens,
    )

    print(
        f"[setup] avg_tokens={args.avg_tokens} requested_n_samples={args.n_samples} "
        f"effective_matched_samples={sem_ids.numel()}"
    )
    print(f"[setup] sem_ids.shape={tuple(sem_ids.shape)}")
    print(f"[setup] syn_group_ids_for_sem.shape={tuple(syn_group_ids_for_sem.shape)}")

    sem_means = []
    syn_means = []
    residual_means = []
    sem_stds = []
    syn_stds = []
    residual_stds = []
    sem_abs_means = []
    syn_abs_means = []
    residual_abs_means = []
    total_abs_means = []
    permuted_syn_means = []
    permuted_syn_stds = []
    shuffled_tokens_syn_means = []
    shuffled_tokens_syn_stds = []
    combined_shuffle_syn_means = []
    combined_shuffle_syn_stds = []
    syn_distributions = []
    permuted_syn_distributions = []
    shuffled_tokens_syn_distributions = []
    combined_shuffle_syn_distributions = []
    rel_depths = np.array(layers, dtype=np.float32) / float(max(layers))
    rng = np.random.default_rng(0)
    perm_indices = torch.from_numpy(rng.permutation(sem_ids.numel())).to(device=device, dtype=torch.long)
    n_token_blocks = args.n_tokens if args.n_tokens is not None else args.min_token_length
    token_perm_size = sem_ids.numel() * n_token_blocks
    token_shuffle_indices = torch.from_numpy(rng.permutation(token_perm_size)).to(device=device, dtype=torch.long)

    for layer in layers:
        full_act = english_activations[f"layer_{layer}"].to(device=device, dtype=torch.float32)
        global_center = full_act.mean(dim=0, keepdim=True)
        full_act = full_act - global_center

        act = full_act.index_select(0, sem_ids)
        sem_centroid = load_semantic_centroid(
            sem_root,
            layer,
            device,
            min_token_length=args.min_token_length,
            n_tokens=args.n_tokens,
        ).index_select(0, sem_ids)
        sem_centroid = sem_centroid - global_center
        unique_syn_centers = load_syntax_centroid(
            syn_root,
            layer,
            device,
            space_index="A",
            min_token_length=args.min_token_length,
            n_tokens=args.n_tokens,
        )
        # centers_A are the crossed syntax centroids for semantic space A:
        # they were computed from syntax split B, not from the evaluated samples.
        syn_centroid_raw = unique_syn_centers.index_select(0, syn_group_ids_for_sem)
        syn_centroid_raw = syn_centroid_raw - global_center
        syn_centroid = remove_centroid_projections(syn_centroid_raw, sem_centroid)

        print(
            f"[layer {layer}] full_act.shape={tuple(full_act.shape)} "
            f"act.shape={tuple(act.shape)} sem_centroid.shape={tuple(sem_centroid.shape)} "
            f"unique_syn_centers.shape={tuple(unique_syn_centers.shape)} "
            f"syn_centroid_raw.shape={tuple(syn_centroid_raw.shape)}"
        )

        act_norm_sq = torch.sum(act * act, dim=1)
        syn_norm_sq = projected_squared_norm(act, syn_centroid)
        sem_norm_sq = projected_squared_norm(act, sem_centroid)
        residual_norm_sq = torch.clamp(act_norm_sq - syn_norm_sq - sem_norm_sq, min=0.0)
        syn_fraction = syn_norm_sq / act_norm_sq.clamp_min(1e-8)
        sem_fraction = sem_norm_sq / act_norm_sq.clamp_min(1e-8)
        residual_fraction = residual_norm_sq / act_norm_sq.clamp_min(1e-8)
        permuted_syn_mean = np.nan
        permuted_syn_std = np.nan
        shuffled_tokens_syn_mean = np.nan
        shuffled_tokens_syn_std = np.nan
        combined_shuffle_syn_mean = np.nan
        combined_shuffle_syn_std = np.nan
        permuted_syn_distribution = np.full(sem_ids.numel(), np.nan, dtype=np.float32)
        shuffled_tokens_syn_distribution = np.full(sem_ids.numel(), np.nan, dtype=np.float32)
        combined_shuffle_syn_distribution = np.full(sem_ids.numel(), np.nan, dtype=np.float32)

        if args.permute_syntax_centroids == 1:
            permuted_syn_centroid_raw = permuted_index(syn_centroid_raw, perm_indices)
            permuted_syn_centroid = remove_centroid_projections(permuted_syn_centroid_raw, sem_centroid)
            permuted_syn_norm_sq = projected_squared_norm(act, permuted_syn_centroid)
            permuted_syn_fraction = permuted_syn_norm_sq / act_norm_sq.clamp_min(1e-8)
            permuted_syn_mean = permuted_syn_fraction.mean().item()
            permuted_syn_std = permuted_syn_fraction.std(unbiased=False).item()
            permuted_syn_distribution = permuted_syn_fraction.detach().cpu().numpy().astype(np.float32)

        if args.shuffle_tokens == 1:
            # Destroy token-wise syntax structure by rebuilding each concatenated syntax centroid
            # from a global shuffle of its token blocks, while leaving activations untouched.
            shuffled_tokens_syn_centroid_raw = shuffle_token_blocks(
                syn_centroid_raw,
                n_token_blocks,
                token_shuffle_indices,
            )
            shuffled_tokens_syn_centroid = remove_centroid_projections(
                shuffled_tokens_syn_centroid_raw,
                sem_centroid,
            )
            shuffled_tokens_syn_norm_sq = projected_squared_norm(act, shuffled_tokens_syn_centroid)
            shuffled_tokens_syn_fraction = shuffled_tokens_syn_norm_sq / act_norm_sq.clamp_min(1e-8)
            shuffled_tokens_syn_mean = shuffled_tokens_syn_fraction.mean().item()
            shuffled_tokens_syn_std = shuffled_tokens_syn_fraction.std(unbiased=False).item()
            shuffled_tokens_syn_distribution = (
                shuffled_tokens_syn_fraction.detach().cpu().numpy().astype(np.float32)
            )

        if args.permute_syntax_centroids == 1 and args.shuffle_tokens == 1:
            combined_shuffle_syn_centroid_raw = shuffle_token_blocks(
                permuted_syn_centroid_raw,
                n_token_blocks,
                token_shuffle_indices,
            )
            combined_shuffle_syn_centroid = remove_centroid_projections(
                combined_shuffle_syn_centroid_raw,
                sem_centroid,
            )
            combined_shuffle_syn_norm_sq = projected_squared_norm(act, combined_shuffle_syn_centroid)
            combined_shuffle_syn_fraction = combined_shuffle_syn_norm_sq / act_norm_sq.clamp_min(1e-8)
            combined_shuffle_syn_mean = combined_shuffle_syn_fraction.mean().item()
            combined_shuffle_syn_std = combined_shuffle_syn_fraction.std(unbiased=False).item()
            combined_shuffle_syn_distribution = (
                combined_shuffle_syn_fraction.detach().cpu().numpy().astype(np.float32)
            )

        syn_means.append(syn_fraction.mean().item())
        sem_means.append(sem_fraction.mean().item())
        residual_means.append(residual_fraction.mean().item())
        syn_stds.append(syn_fraction.std(unbiased=False).item())
        sem_stds.append(sem_fraction.std(unbiased=False).item())
        residual_stds.append(residual_fraction.std(unbiased=False).item())
        syn_abs_means.append(syn_norm_sq.mean().item())
        sem_abs_means.append(sem_norm_sq.mean().item())
        residual_abs_means.append(residual_norm_sq.mean().item())
        total_abs_means.append(act_norm_sq.mean().item())
        permuted_syn_means.append(permuted_syn_mean)
        permuted_syn_stds.append(permuted_syn_std)
        shuffled_tokens_syn_means.append(shuffled_tokens_syn_mean)
        shuffled_tokens_syn_stds.append(shuffled_tokens_syn_std)
        combined_shuffle_syn_means.append(combined_shuffle_syn_mean)
        combined_shuffle_syn_stds.append(combined_shuffle_syn_std)
        syn_distributions.append(syn_fraction.detach().cpu().numpy().astype(np.float32))
        permuted_syn_distributions.append(permuted_syn_distribution)
        shuffled_tokens_syn_distributions.append(shuffled_tokens_syn_distribution)
        combined_shuffle_syn_distributions.append(combined_shuffle_syn_distribution)

        print(
            f"layer={layer} syn_mean={syn_means[-1]:.6f} sem_mean={sem_means[-1]:.6f} residual_mean={residual_means[-1]:.6f} "
            f"syn_std={syn_stds[-1]:.6f} "
            f"syn_abs_mean={syn_abs_means[-1]:.6f} sem_abs_mean={sem_abs_means[-1]:.6f} residual_abs_mean={residual_abs_means[-1]:.6f} "
            f"permuted_syn_mean={permuted_syn_means[-1]:.6f} shuffled_tokens_syn_mean={shuffled_tokens_syn_means[-1]:.6f} "
            f"combined_shuffle_syn_mean={combined_shuffle_syn_means[-1]:.6f}"
        )

    np.savez(
        output_dir / "norms.npz",
        layer_vals=np.array(layers, dtype=np.int32),
        rel_depths=rel_depths,
        syn_means=np.array(syn_means, dtype=np.float32),
        sem_means=np.array(sem_means, dtype=np.float32),
        residual_means=np.array(residual_means, dtype=np.float32),
        syn_stds=np.array(syn_stds, dtype=np.float32),
        sem_stds=np.array(sem_stds, dtype=np.float32),
        residual_stds=np.array(residual_stds, dtype=np.float32),
        syn_abs_means=np.array(syn_abs_means, dtype=np.float32),
        sem_abs_means=np.array(sem_abs_means, dtype=np.float32),
        residual_abs_means=np.array(residual_abs_means, dtype=np.float32),
        total_abs_means=np.array(total_abs_means, dtype=np.float32),
        permuted_syn_means=np.array(permuted_syn_means, dtype=np.float32),
        permuted_syn_stds=np.array(permuted_syn_stds, dtype=np.float32),
        shuffled_tokens_syn_means=np.array(shuffled_tokens_syn_means, dtype=np.float32),
        shuffled_tokens_syn_stds=np.array(shuffled_tokens_syn_stds, dtype=np.float32),
        combined_shuffle_syn_means=np.array(combined_shuffle_syn_means, dtype=np.float32),
        combined_shuffle_syn_stds=np.array(combined_shuffle_syn_stds, dtype=np.float32),
        syn_dists=np.stack(syn_distributions).astype(np.float32),
        permuted_syn_dists=np.stack(permuted_syn_distributions).astype(np.float32),
        shuffled_tokens_syn_dists=np.stack(shuffled_tokens_syn_distributions).astype(np.float32),
        combined_shuffle_syn_dists=np.stack(combined_shuffle_syn_distributions).astype(np.float32),
        has_permuted_syn_centroids=np.array(args.permute_syntax_centroids == 1),
        has_shuffled_tokens=np.array(args.shuffle_tokens == 1),
        has_combined_shuffle=np.array(args.permute_syntax_centroids == 1 and args.shuffle_tokens == 1),
        has_syn_centroids=np.array(True),
    )
    print(f"norm results written to {output_dir / 'norms.npz'}")


if __name__ == "__main__":
    main()
