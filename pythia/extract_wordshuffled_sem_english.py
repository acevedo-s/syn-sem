import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/home/acevedo/syn-sem")
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(REPO_ROOT / "hf_extract_activations") not in sys.path:
    sys.path.append(str(REPO_ROOT / "hf_extract_activations"))

from codex_helpers.helpers import has_safetensors_weights, resolve_model_source
from hf_extract_activations.utils_extract import export, extract, load_lines
from modelpaths import model_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Pythia semantic activations from word-shuffled English inputs.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-lines", type=int, default=2100)
    parser.add_argument("--chunk-size", type=int, default=100)
    return parser.parse_args()


def shuffle_words(text, seed):
    words = text.split()
    if len(words) <= 1:
        return text
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(words))
    return " ".join(words[i] for i in perm)


def main():
    args = parse_args()
    model_source = resolve_model_source(args.model, model_paths)
    print(f"{args.model=}")
    print(f"{model_source=}")

    io_paths = []
    for sample_index in [0, 1]:
        io_paths.append(
            {
                "input_path": REPO_ROOT / "datasets" / "txt" / "sem" / "second" / "matching" / "english" / f"sentences{sample_index}.txt",
                "output_folder": REPO_ROOT / "datasets" / "activations" / "sem" / "second" / args.model / "matching_wordshuffled" / "english" / str(sample_index),
            }
        )

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model_load_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "torch_dtype": "auto",
    }
    if has_safetensors_weights(model_source):
        model_load_kwargs["use_safetensors"] = True

    model_obj = AutoModelForCausalLM.from_pretrained(
        model_source,
        **model_load_kwargs,
    )
    model_obj.config.output_hidden_states = True
    model_obj.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = next(model_obj.parameters()).dtype
    print(f"Model dtype: {model_dtype}")

    for io_cfg in io_paths:
        io_cfg["output_folder"].mkdir(parents=True, exist_ok=True)
        sentences = load_lines(str(io_cfg["input_path"]))[: args.n_lines]
        shuffled_sentences = [shuffle_words(text, seed=idx) for idx, text in enumerate(sentences)]
        export(
            sentences=shuffled_sentences,
            out_dir=str(io_cfg["output_folder"]),
            tokenizer=tokenizer,
            model_obj=model_obj,
            device=device,
            model_dtype=model_dtype,
            model_id=args.model,
            extract_fn=extract,
            chunk_size=args.chunk_size,
        )
        print(f"Export complete for {io_cfg['output_folder']}")


if __name__ == "__main__":
    main()
