import sys,os
import pickle
import torch
from time import time 
from tqdm import tqdm

base_path_models = f'/home/acevedo/LLMs/'


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def clip_hidden_torch(hidden, alphamin, alphamax, dtype=None):
    """
    Always clip activations and return storage-optimized format.

    hidden: (L, T, E) tensor on GPU
    dtype: original model dtype (torch.bfloat16, torch.float16, torch.float32)

    return:
        - uint16 numpy array (bit patterns) for bf16/fp16
        - float32 numpy array for fp32
    """

    # Step 1: always quantize in full precision
    hidden_float = hidden.float()
    L, T, E = hidden.shape

    hidden_flat = hidden_float.view(L, T * E)
    qmin = hidden_flat.quantile(alphamin, dim=1, keepdim=True)
    qmax = hidden_flat.quantile(alphamax, dim=1, keepdim=True)

    clipped = hidden_flat.clamp(min=qmin, max=qmax).view(L, T, E)

    # Step 2: convert back to model dtype
    clipped = clipped.to(dtype)

    # Step 3: storage format rules
    if dtype in (torch.bfloat16, torch.float16):
        # reinterpret bits into uint16 array (lossless reversible)
        arr = clipped.view(torch.uint16).cpu().numpy()
    else:
        # fp32 → store as float array
        arr = clipped.cpu().numpy()

    return arr

def extract(text, model, tokenizer, device, model_dtype,
            alphamin=0.05, alphamax=0.95):
    """
    Run the model on a sentence, extract hidden states, clip them,
    and return a compact activation record.

    Args:
        text (str): input sentence
        model: HF model with output_hidden_states enabled
        tokenizer: HF tokenizer
        device: torch.device("cuda") or cpu
        model_dtype: dtype of the model (torch.bfloat16/fp16/fp32)
        alphamin, alphamax: clipping quantiles

    Returns:
        dict with text, tokens, and clipped activations
    """
    with torch.no_grad():
        # Tokenize and move input to device
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}

        # Forward pass
        outputs = model(**enc, output_hidden_states=True)

        # Hidden shape: (layers, batch, tokens, embedding)
        hidden = torch.stack(outputs.hidden_states, dim=0)[:, 0, :, :]

        # Clip
        activations_clipped = clip_hidden_torch(
            hidden,
            alphamin=alphamin,
            alphamax=alphamax,
            dtype=model_dtype
        )

        return {
            "text": text,
            "tokens": enc["input_ids"].cpu(),
            "activations": activations_clipped,
        }
    

def export(
    sentences,
    out_dir,
    tokenizer,
    model_obj,
    device,
    model_dtype,
    model_id,
    extract_fn,
    chunk_size=100,
):
    buffer = []
    chunk_index = 0
    t0 = time()


    for i, text in enumerate(tqdm(sentences, desc='extracting activations')):

        record = extract_fn(
            text=text,
            model=model_obj,
            tokenizer=tokenizer,
            device=device,
            model_dtype=model_dtype,
        )

        buffer.append(record)

        # Flush chunk
        if len(buffer) >= chunk_size or i == len(sentences) - 1:
            file_path = os.path.join(out_dir, f"chunk_{chunk_index}.pkl")

            with open(file_path, "wb") as f:
                pickle.dump(
                    {
                        "model": model_id,
                        "model_dtype": str(model_dtype),
                        "samples": buffer,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            buffer = []
            chunk_index += 1

    print(f"{chunk_index} chunks took {(time()-t0)/60:.2f} min")
