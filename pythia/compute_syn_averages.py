import argparse
from datetime import datetime

import jax.numpy as jnp
import numpy as np
import torch

from common import (
    layer_values,
    load_activations,
    save_metadata,
    syntax_activation_dir,
    syntax_centers_root,
)
from utils import _compute_and_export_syn_centers, syn_group_ids_path


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Pythia syntax centroids.")
    parser.add_argument("--model", default="pythia6p9b_step143000")
    parser.add_argument("--min-token-length", type=int, default=3)
    parser.add_argument("--avg-tokens", type=int, choices=[0, 1], required=True)
    parser.add_argument("--n-tokens", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    layers = layer_values(args.model)
    n_syntax_samples = len(np.loadtxt(syn_group_ids_path, dtype=int))
    output_dir = syntax_centers_root(
        args.model,
        args.avg_tokens,
        args.min_token_length,
        args.n_samples,
        n_tokens=args.n_tokens,
    )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "layers": layers,
        "avg_tokens": args.avg_tokens,
        "min_token_length": args.min_token_length,
        "model": args.model,
        "n_tokens": args.n_tokens,
        "n_samples": args.n_samples,
        "n_syntax_samples": n_syntax_samples,
        "space_A_dir": str(syntax_activation_dir(sample_index=0, model_name=args.model)),
        "space_B_dir": str(syntax_activation_dir(sample_index=1, model_name=args.model)),
    }
    save_metadata(output_dir, metadata)

    input_dir_A = syntax_activation_dir(sample_index=0, model_name=args.model)
    input_dir_B = syntax_activation_dir(sample_index=1, model_name=args.model)

    print(f"layers={layers}")
    print(f"loading space A from {input_dir_A}")
    print(f"loading space B from {input_dir_B}")
    print(f"writing to {output_dir}")

    activations_A = load_activations(
        input_dir=input_dir_A,
        min_token_length=args.min_token_length,
        avg_tokens=args.avg_tokens,
        n_samples=n_syntax_samples,
        model_name=args.model,
        n_tokens=args.n_tokens,
    )
    activations_B = load_activations(
        input_dir=input_dir_B,
        min_token_length=args.min_token_length,
        avg_tokens=args.avg_tokens,
        n_samples=n_syntax_samples,
        model_name=args.model,
        n_tokens=args.n_tokens,
    )

    for layer in layers:
        layer_dir = output_dir / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        act_A = jnp.array(
            activations_A[f"layer_{layer}"].to(dtype=torch.float32).cpu().numpy(),
            dtype=jnp.float32,
        )
        act_B = jnp.array(
            activations_B[f"layer_{layer}"].to(dtype=torch.float32).cpu().numpy(),
            dtype=jnp.float32,
        )

        _compute_and_export_syn_centers(syn_group_ids_path, act_B, str(layer_dir), "A")
        _compute_and_export_syn_centers(syn_group_ids_path, act_A, str(layer_dir), "B")

    print("syntax centroids exported")


if __name__ == "__main__":
    main()
