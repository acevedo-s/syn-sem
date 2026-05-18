import argparse
from datetime import datetime

import numpy as np
import torch

from common import (
    ENGLISH,
    activation_dir,
    layer_values,
    load_activations,
    output_root,
    save_metadata,
    semantic_centers_root,
    syntax_centers_root,
)
from norms_pythia import (
    load_semantic_centroid,
    load_semantic_syntax_alignment,
    load_syntax_centroid,
    projected_squared_norm,
    remove_centroid_projections,
    reshape_token_blocks,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute tokenwise lexical-orthogonalized syntax norms.")
    parser.add_argument("--model", default="pythia6p9b_step143000")
    parser.add_argument("--min-token-length", type=int, default=3)
    parser.add_argument("--avg-tokens", type=int, choices=[0], required=True)
    parser.add_argument("--n-tokens", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=2018)
    parser.add_argument("--activation-match-var", default="matching")
    return parser.parse_args()


def lexical_orth_root(model_name, avg_tokens, min_token_length, n_samples, global_center_flag, n_tokens=None):
    return (
        output_root(model_name, avg_tokens, min_token_length, n_samples, n_tokens=n_tokens)
        / "norms_lexical_orth"
        / f"global_center_flag_{global_center_flag}"
    )


def remove_tokenwise_projections(act, basis, n_token_blocks, eps=1e-8):
    act_blocks = reshape_token_blocks(act, n_token_blocks)
    basis_blocks = reshape_token_blocks(basis, n_token_blocks)
    dot = torch.sum(act_blocks * basis_blocks, dim=2, keepdim=True)
    basis_norm_sq = torch.sum(basis_blocks * basis_blocks, dim=2, keepdim=True).clamp_min(eps)
    projection_blocks = (dot / basis_norm_sq) * basis_blocks
    residual_blocks = act_blocks - projection_blocks
    return residual_blocks.reshape(act.shape)


def main():
    args = parse_args()
    global_center_flag = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sem_ids, syn_group_ids_for_sem = load_semantic_syntax_alignment(args.n_samples, device, space_index="A")
    layers = layer_values(args.model)
    centers_n_tokens = args.min_token_length

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
    output_dir = lexical_orth_root(
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
        "activation_match_var": args.activation_match_var,
        "semantic_centers_dir": str(sem_root),
        "syntax_centers_dir": str(syn_root),
        "control": "tokenwise_layer0_lexical_orthogonalization_on_activations",
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

    lexical_token_blocks = args.n_tokens if args.n_tokens is not None else args.min_token_length
    lexical_full = english_activations["layer_0"].to(device=device, dtype=torch.float32)
    lexical_global_center = lexical_full.mean(dim=0, keepdim=True)
    lexical_full = lexical_full - lexical_global_center
    lexical_basis = lexical_full.index_select(0, sem_ids)

    lexical_orth_syn_means = []
    lexical_orth_syn_stds = []
    lexical_orth_syn_abs_means = []
    lexical_orth_total_abs_means = []
    lexical_orth_syn_distributions = []
    rel_depths = np.array(layers, dtype=np.float32) / float(max(layers))

    print(
        f"[setup] avg_tokens={args.avg_tokens} requested_n_samples={args.n_samples} "
        f"effective_matched_samples={sem_ids.numel()}"
    )
    print(f"[setup] sem_ids.shape={tuple(sem_ids.shape)}")
    print(f"[setup] syn_group_ids_for_sem.shape={tuple(syn_group_ids_for_sem.shape)}")

    for layer in layers:
        full_act = english_activations[f"layer_{layer}"].to(device=device, dtype=torch.float32)
        global_center = full_act.mean(dim=0, keepdim=True)
        full_act = full_act - global_center

        act = full_act.index_select(0, sem_ids)
        lexical_orth_act = remove_tokenwise_projections(act, lexical_basis, lexical_token_blocks)
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

        lexical_orth_act_norm_sq = torch.sum(lexical_orth_act * lexical_orth_act, dim=1)
        lexical_orth_syn_norm_sq = projected_squared_norm(lexical_orth_act, syn_centroid)
        lexical_orth_syn_fraction = lexical_orth_syn_norm_sq / lexical_orth_act_norm_sq.clamp_min(1e-8)

        lexical_orth_syn_means.append(lexical_orth_syn_fraction.mean().item())
        lexical_orth_syn_stds.append(lexical_orth_syn_fraction.std(unbiased=False).item())
        lexical_orth_syn_abs_means.append(lexical_orth_syn_norm_sq.mean().item())
        lexical_orth_total_abs_means.append(lexical_orth_act_norm_sq.mean().item())
        lexical_orth_syn_distributions.append(lexical_orth_syn_fraction.detach().cpu().numpy().astype(np.float32))

        print(
            f"layer={layer} lexical_orth_syn_mean={lexical_orth_syn_means[-1]:.6f} "
            f"lexical_orth_syn_std={lexical_orth_syn_stds[-1]:.6f} "
            f"lexical_orth_syn_abs_mean={lexical_orth_syn_abs_means[-1]:.6f}"
        )

    np.savez(
        output_dir / "lexical_orth_norms.npz",
        layer_vals=np.array(layers, dtype=np.int32),
        rel_depths=rel_depths,
        lexical_orth_syn_means=np.array(lexical_orth_syn_means, dtype=np.float32),
        lexical_orth_syn_stds=np.array(lexical_orth_syn_stds, dtype=np.float32),
        lexical_orth_syn_abs_means=np.array(lexical_orth_syn_abs_means, dtype=np.float32),
        lexical_orth_total_abs_means=np.array(lexical_orth_total_abs_means, dtype=np.float32),
        lexical_orth_syn_dists=np.stack(lexical_orth_syn_distributions).astype(np.float32),
        has_lexical_orthogonalized_activations=np.array(True),
        has_syn_centroids=np.array(True),
    )
    print(f"lexical control results written to {output_dir / 'lexical_orth_norms.npz'}")


if __name__ == "__main__":
    main()
