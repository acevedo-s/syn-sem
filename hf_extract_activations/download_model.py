import os
from pathlib import Path
from utils_extract import base_path_models
from modelpaths import repo_ids

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
    AutoConfig,
)


def download_model(repo_id: str, cache_dir: str | os.PathLike):
    """
    Download HF model + tokenizer and save them in `cache_dir`.

    `repo_id` is the HF id, e.g.:
      - "CohereForAI/aya-101"
      - "bert-base-uncased"
      - "meta-llama/Llama-3-8B-Instruct"
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Inspect config to pick the right class
    cfg = AutoConfig.from_pretrained(repo_id)

    if cfg.model_type in {"t5", "mt5"}:
        # Aya-101 falls here (mt5-xxl architecture)
        ModelCls = AutoModelForSeq2SeqLM
    elif "bert" in cfg.model_type:
        ModelCls = AutoModel          # encoder-only BERTs
    else:
        ModelCls = AutoModelForCausalLM  # decoder-only LMs

    print(f"→ Downloading {repo_id} into {cache_dir}")

    model = ModelCls.from_pretrained(
        repo_id,
        cache_dir=str(cache_dir),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        cache_dir=str(cache_dir),
        use_fast=True,
    )

    model.save_pretrained(cache_dir)
    tokenizer.save_pretrained(cache_dir)
    return model, tokenizer


model_name = 'gemma12b'
download_dir = base_path_models + f'{repo_ids[model_name]}/'

download_model(repo_ids[model_name], download_dir)
