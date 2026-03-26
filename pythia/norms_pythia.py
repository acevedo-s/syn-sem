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


def cosine_similarity(act, centroid, eps=1e-8):
    numerator = torch.sum(act * centroid, dim=1)
    denominator = torch.linalg.norm(act, dim=1) * torch.linalg.norm(centroid, dim=1)
    return numerator / denominator.clamp_min(eps)


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
        "semantic_centers_dir": str(sem_root),
        "syntax_centers_dir": str(syn_root),
        "uses_syn_centroids": True,
    }
    save_metadata(output_dir, metadata)

    english_dir = activation_dir(language=ENGLISH, sample_index=0, model_name=args.model)
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
    sem_abs_means = []
    syn_abs_means = []
    residual_abs_means = []
    total_abs_means = []
    cos_means = []
    cos_stds = []
    rel_depths = np.array(layers, dtype=np.float32) / float(max(layers))

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
        syn_centroid_raw = unique_syn_centers.index_select(0, syn_group_ids_for_sem)
        syn_centroid_raw = syn_centroid_raw - global_center
        syn_centroid = remove_centroid_projections(syn_centroid_raw, sem_centroid)

        print(
            f"[layer {layer}] full_act.shape={tuple(full_act.shape)} "
            f"act.shape={tuple(act.shape)} sem_centroid.shape={tuple(sem_centroid.shape)} "
            f"unique_syn_centers.shape={tuple(unique_syn_centers.shape)} "
            f"syn_centroid_raw.shape={tuple(syn_centroid_raw.shape)}"
        )

        cos = cosine_similarity(syn_centroid_raw, sem_centroid)
        act_norm_sq = torch.sum(act * act, dim=1)
        syn_norm_sq = projected_squared_norm(act, syn_centroid)
        sem_norm_sq = projected_squared_norm(act, sem_centroid)
        residual_norm_sq = torch.clamp(act_norm_sq - syn_norm_sq - sem_norm_sq, min=0.0)
        syn_fraction = syn_norm_sq / act_norm_sq.clamp_min(1e-8)
        sem_fraction = sem_norm_sq / act_norm_sq.clamp_min(1e-8)
        residual_fraction = residual_norm_sq / act_norm_sq.clamp_min(1e-8)

        syn_means.append(syn_fraction.mean().item())
        sem_means.append(sem_fraction.mean().item())
        residual_means.append(residual_fraction.mean().item())
        syn_abs_means.append(syn_norm_sq.mean().item())
        sem_abs_means.append(sem_norm_sq.mean().item())
        residual_abs_means.append(residual_norm_sq.mean().item())
        total_abs_means.append(act_norm_sq.mean().item())
        cos_means.append(cos.mean().item())
        cos_stds.append(cos.std(unbiased=False).item())

        print(
            f"layer={layer} syn_mean={syn_means[-1]:.6f} sem_mean={sem_means[-1]:.6f} residual_mean={residual_means[-1]:.6f} "
            f"syn_abs_mean={syn_abs_means[-1]:.6f} sem_abs_mean={sem_abs_means[-1]:.6f} residual_abs_mean={residual_abs_means[-1]:.6f}"
        )

    np.savez(
        output_dir / "norms.npz",
        layer_vals=np.array(layers, dtype=np.int32),
        rel_depths=rel_depths,
        syn_means=np.array(syn_means, dtype=np.float32),
        sem_means=np.array(sem_means, dtype=np.float32),
        residual_means=np.array(residual_means, dtype=np.float32),
        syn_abs_means=np.array(syn_abs_means, dtype=np.float32),
        sem_abs_means=np.array(sem_abs_means, dtype=np.float32),
        residual_abs_means=np.array(residual_abs_means, dtype=np.float32),
        total_abs_means=np.array(total_abs_means, dtype=np.float32),
        cos_means=np.array(cos_means, dtype=np.float32),
        cos_stds=np.array(cos_stds, dtype=np.float32),
        has_syn_centroids=np.array(True),
    )
    print(f"norm results written to {output_dir / 'norms.npz'}")


if __name__ == "__main__":
    main()
