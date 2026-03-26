import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PYTHIA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PYTHIA_ROOT.parent
for path in (str(PYTHIA_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from common import (  # noqa: E402
    ENGLISH,
    activation_dir,
    layer_values,
    load_activations,
    my_languages,
    sem_ids_with_syn_path,
    semantic_centers_root,
    syntax_centers_root,
)
from utils import syn_group_id_paths_for_sem_data  # noqa: E402


DEFAULT_FINAL_MODEL = "pythia6p9b_step143000"


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Pythia lexical/syntax/semantic norm summaries.")
    parser.add_argument("--model", default=DEFAULT_FINAL_MODEL)
    parser.add_argument("--min-token-length", type=int, default=3)
    parser.add_argument("--avg-tokens", type=int, choices=[0, 1], required=True)
    parser.add_argument("--n-tokens", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=2018)
    parser.add_argument("--global-center-flag", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


def lexical_output_root(model_name, avg_tokens, min_token_length, n_samples, n_tokens):
    root = (
        PYTHIA_ROOT
        / "lexical_norms"
        / "results"
        / f"model_{model_name}"
        / f"avg_tokens_{avg_tokens}"
        / f"min_token_length_{min_token_length}"
        / f"n_samples_{n_samples}"
    )
    if avg_tokens == 0 and n_tokens not in (None, min_token_length):
        root = root / f"n_tokens_{n_tokens}"
    return root


def lexical_norms_root(model_name, avg_tokens, min_token_length, n_samples, global_center_flag, n_tokens):
    return (
        lexical_output_root(model_name, avg_tokens, min_token_length, n_samples, n_tokens)
        / "norms"
        / f"global_center_flag_{global_center_flag}"
    )


def save_metadata(output_dir, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def projected_squared_norm(act, centroid, eps=1e-8):
    dot = torch.sum(act * centroid, dim=1, keepdim=True)
    centroid_norm_sq = torch.sum(centroid * centroid, dim=1, keepdim=True).clamp_min(eps)
    projection = (dot / centroid_norm_sq) * centroid
    return torch.sum(projection * projection, dim=1)


def remove_centroid_projections(act, centroids, eps=1e-8):
    proj_coeffs = torch.sum(act * centroids, dim=1, keepdim=True) / (
        torch.sum(centroids * centroids, dim=1, keepdim=True).clamp_min(eps)
    )
    return act - proj_coeffs * centroids


def cosine_similarity(act, centroid, eps=1e-8):
    numerator = torch.sum(act * centroid, dim=1)
    denominator = torch.linalg.norm(act, dim=1) * torch.linalg.norm(centroid, dim=1)
    return numerator / denominator.clamp_min(eps)


def slice_last_token_features(tensor, min_token_length, n_tokens):
    if n_tokens is None or n_tokens == min_token_length:
        return tensor
    if tensor.shape[-1] % min_token_length != 0:
        # Centers stored under an n_tokens-specific directory may already be sliced.
        return tensor
    hidden_size = tensor.shape[-1] // min_token_length
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sem_ids, syn_group_ids_for_sem = load_semantic_syntax_alignment(args.n_samples, device, space_index="A")
    layers = layer_values(args.model)

    sem_root = semantic_centers_root(
        args.model,
        args.avg_tokens,
        args.min_token_length,
        args.n_samples,
        n_tokens=args.n_tokens,
    )
    syn_root = syntax_centers_root(
        args.model,
        args.avg_tokens,
        args.min_token_length,
        args.n_samples,
        n_tokens=args.n_tokens,
    )
    output_dir = lexical_norms_root(
        args.model,
        args.avg_tokens,
        args.min_token_length,
        args.n_samples,
        args.global_center_flag,
        args.n_tokens,
    )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "avg_tokens": args.avg_tokens,
        "global_center_flag": args.global_center_flag,
        "layers": layers,
        "min_token_length": args.min_token_length,
        "model": args.model,
        "n_tokens": args.n_tokens,
        "n_samples": args.n_samples,
        "effective_semantic_syntax_samples": int(sem_ids.numel()),
        "semantic_centers_dir": str(sem_root),
        "syntax_centers_dir": str(syn_root),
        "lexical_basis": "layer_0 activations in the same representation",
        "decomposition_order": ["lex", "sem", "syn", "residual"],
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

    lex_means = []
    syn_means = []
    sem_means = []
    residual_means = []
    lex_abs_means = []
    syn_abs_means = []
    sem_abs_means = []
    residual_abs_means = []
    total_abs_means = []
    lex_cos_means = []
    lex_cos_stds = []
    rel_depths = np.array(layers, dtype=np.float32) / float(max(layers))

    lexical_full = english_activations["layer_0"].to(device=device, dtype=torch.float32)
    lexical_global_center = lexical_full.mean(dim=0, keepdim=True) if args.global_center_flag else 0.0
    lexical_full = lexical_full - lexical_global_center
    lexical_basis = lexical_full.index_select(0, sem_ids)

    for layer in layers:
        full_act = english_activations[f"layer_{layer}"].to(device=device, dtype=torch.float32)
        global_center = full_act.mean(dim=0, keepdim=True) if args.global_center_flag else 0.0
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

        # Lexical component is the sample-aligned layer-0 direction in the same representation.
        lex_centroid = lexical_basis
        sem_centroid = remove_centroid_projections(sem_centroid, lex_centroid)
        syn_centroid = remove_centroid_projections(syn_centroid_raw, lex_centroid)
        syn_centroid = remove_centroid_projections(syn_centroid, sem_centroid)

        act_norm_sq = torch.sum(act * act, dim=1)
        lex_norm_sq = projected_squared_norm(act, lex_centroid)
        sem_norm_sq = projected_squared_norm(act, sem_centroid)
        syn_norm_sq = projected_squared_norm(act, syn_centroid)
        residual_norm_sq = torch.clamp(act_norm_sq - lex_norm_sq - sem_norm_sq - syn_norm_sq, min=0.0)

        lex_fraction = lex_norm_sq / act_norm_sq.clamp_min(1e-8)
        sem_fraction = sem_norm_sq / act_norm_sq.clamp_min(1e-8)
        syn_fraction = syn_norm_sq / act_norm_sq.clamp_min(1e-8)
        residual_fraction = residual_norm_sq / act_norm_sq.clamp_min(1e-8)

        lex_means.append(lex_fraction.mean().item())
        sem_means.append(sem_fraction.mean().item())
        syn_means.append(syn_fraction.mean().item())
        residual_means.append(residual_fraction.mean().item())
        lex_abs_means.append(lex_norm_sq.mean().item())
        sem_abs_means.append(sem_norm_sq.mean().item())
        syn_abs_means.append(syn_norm_sq.mean().item())
        residual_abs_means.append(residual_norm_sq.mean().item())
        total_abs_means.append(act_norm_sq.mean().item())

        lex_cos = cosine_similarity(act, lex_centroid)
        lex_cos_means.append(lex_cos.mean().item())
        lex_cos_stds.append(lex_cos.std(unbiased=False).item())

        print(
            f"layer={layer} lex_mean={lex_means[-1]:.6f} syn_mean={syn_means[-1]:.6f} "
            f"sem_mean={sem_means[-1]:.6f} residual_mean={residual_means[-1]:.6f}"
        )

    np.savez(
        output_dir / "lexical_norms.npz",
        layer_vals=np.array(layers, dtype=np.int32),
        rel_depths=rel_depths,
        lex_means=np.array(lex_means, dtype=np.float32),
        syn_means=np.array(syn_means, dtype=np.float32),
        sem_means=np.array(sem_means, dtype=np.float32),
        residual_means=np.array(residual_means, dtype=np.float32),
        lex_abs_means=np.array(lex_abs_means, dtype=np.float32),
        syn_abs_means=np.array(syn_abs_means, dtype=np.float32),
        sem_abs_means=np.array(sem_abs_means, dtype=np.float32),
        residual_abs_means=np.array(residual_abs_means, dtype=np.float32),
        total_abs_means=np.array(total_abs_means, dtype=np.float32),
        lex_cos_means=np.array(lex_cos_means, dtype=np.float32),
        lex_cos_stds=np.array(lex_cos_stds, dtype=np.float32),
    )
    print(f"lexical norm results written to {output_dir / 'lexical_norms.npz'}")


if __name__ == "__main__":
    main()
