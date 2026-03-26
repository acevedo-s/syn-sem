import json
import os
from glob import glob
from pathlib import Path

import torch
from safetensors.torch import save_file


def resolve_model_source(model_name, model_paths):
    model_source = model_paths[model_name]

    if os.path.exists(os.path.join(model_source, "config.json")):
        return model_source

    snapshot_candidates = sorted(
        glob(os.path.join(model_source, "models--*", "snapshots", "*"))
    )
    for candidate in snapshot_candidates:
        if os.path.exists(os.path.join(candidate, "config.json")):
            return candidate

    raise FileNotFoundError(
        f"Could not resolve a valid local model snapshot for {model_name!r} "
        f"from {model_source!r}"
    )


def has_safetensors_weights(model_source):
    return (
        os.path.exists(os.path.join(model_source, "model.safetensors"))
        or os.path.exists(os.path.join(model_source, "model.safetensors.index.json"))
    )


def convert_snapshot_to_safetensors(snapshot_dir):
    snapshot_dir = Path(snapshot_dir).resolve()
    index_path = snapshot_dir / "pytorch_model.bin.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing shard index: {index_path}")

    with index_path.open("r", encoding="utf-8") as f:
        index_data = json.load(f)

    bin_filenames = sorted(set(index_data["weight_map"].values()))
    safe_name_map = {}

    for bin_name in bin_filenames:
        bin_path = snapshot_dir / bin_name
        safe_name = bin_name.replace("pytorch_model", "model").replace(".bin", ".safetensors")
        safe_path = snapshot_dir / safe_name

        print(f"Converting {bin_path.name} -> {safe_path.name}")
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        save_file(state_dict, str(safe_path), metadata={"format": "pt"})
        safe_name_map[bin_name] = safe_name

    safe_index = {
        "metadata": index_data.get("metadata", {}),
        "weight_map": {
            tensor_name: safe_name_map[bin_name]
            for tensor_name, bin_name in index_data["weight_map"].items()
        },
    }
    safe_index_path = snapshot_dir / "model.safetensors.index.json"
    with safe_index_path.open("w", encoding="utf-8") as f:
        json.dump(safe_index, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {safe_index_path}")
