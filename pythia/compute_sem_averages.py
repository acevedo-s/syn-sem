import argparse
from datetime import datetime

import numpy as np
import torch

from common import (
    activation_dir,
    layer_values,
    load_activations,
    my_languages,
    save_metadata,
    semantic_centers_root,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Pythia semantic centroids.")
    parser.add_argument("--model", default="pythia6p9b_step143000")
    parser.add_argument("--min-token-length", type=int, default=3)
    parser.add_argument("--avg-tokens", type=int, choices=[0, 1], required=True)
    parser.add_argument("--n-tokens", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    layers = layer_values(args.model)
    output_dir = semantic_centers_root(
        args.model,
        args.avg_tokens,
        args.min_token_length,
        args.n_samples,
        n_tokens=args.n_tokens,
    )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "languages": my_languages,
        "layers": layers,
        "avg_tokens": args.avg_tokens,
        "min_token_length": args.min_token_length,
        "model": args.model,
        "n_tokens": args.n_tokens,
        "n_samples": args.n_samples,
    }
    save_metadata(output_dir, metadata)

    print(f"layers={layers}")
    print(f"writing to {output_dir}")

    for language_id, language in enumerate(my_languages):
        input_dir = activation_dir(language=language, sample_index=1, model_name=args.model)
        print(f"[{language_id}] loading {language} from {input_dir}")
        layer_to_acts = load_activations(
            input_dir=input_dir,
            min_token_length=args.min_token_length,
            avg_tokens=args.avg_tokens,
            n_samples=args.n_samples,
            model_name=args.model,
            n_tokens=args.n_tokens,
        )

        for layer in layers:
            activations = layer_to_acts[f"layer_{layer}"].to(dtype=torch.float32)
            layer_dir = output_dir / f"layer_{layer}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            activations_np = activations.cpu().numpy().astype(np.float32)
            np.save(layer_dir / f"activations_{language_id}.npy", activations_np)

            mean_path = layer_dir / "semantic_centroid_mean.npy"
            if mean_path.exists():
                running_sum = np.load(mean_path) * language_id
                mean_np = (running_sum + activations_np) / float(language_id + 1)
            else:
                mean_np = activations_np
            np.save(mean_path, mean_np.astype(np.float32))

    print("semantic centroids exported")


if __name__ == "__main__":
    main()
