import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Load Qwen ===
qwen_name = "Qwen/Qwen-7B"
# Note: Ensure you have flash_attn installed or remove trust_remote_code if not needed, 
# but Qwen usually requires it.
qwen_tok = AutoTokenizer.from_pretrained(qwen_name, trust_remote_code=True)
qwen = AutoModelForCausalLM.from_pretrained(
    qwen_name, trust_remote_code=True, device_map="auto"
).eval()

# === Function to read translations ===
def load_translations(filepath, n_languages=None):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if n_languages is not None:
        lines = lines[:n_languages]
    print(f"Loaded {len(lines)} translations from {filepath}")
    for i, t in enumerate(lines, 1):
        print(f"  [{i}] {t}")
    return lines

# [markdown]
# # Multi layer 

#
# === 1. Efficiently Build T for ALL layers ===
def build_all_layers_T(filepath, n_languages=None):
    """
    Computes a steering vector for EVERY layer simultaneously.
    Returns a tensor of shape [Num_Layers, Hidden_Dim].
    """
    translations = load_translations(filepath, n_languages)
    
    # Storage for all layers: List of Tensors
    # We will accumulate vectors here then average them
    accumulated_vecs = [] 

    print(f"Extracting activations for {len(translations)} sentences...")

    for i, sentence in enumerate(translations):
        inputs = qwen_tok(sentence, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # output_hidden_states=True returns a tuple of all layer outputs
            out = qwen(**inputs, output_hidden_states=True)
        
        # out.hidden_states is a tuple of len(layers + 1). 
        # Index 0 is embeddings. Indices 1 to N are the transformer layers.
        # We slice [1:] to get the actual layer outputs.
        all_layers_hidden = out.hidden_states[1:] 
        
        # Stack layers into tensor: [Num_Layers, Batch(1), Seq, Dim]
        stacked = torch.stack(all_layers_hidden).squeeze(1) 
        
        # Take last token: [Num_Layers, Dim]
        vec = stacked[:, -1, :]
            
        accumulated_vecs.append(vec)

    # Stack all sentences: [Num_Sentences, Num_Layers, Dim]
    all_sentences_tensor = torch.stack(accumulated_vecs)
    
    # Average across sentences (dim 0) to get Centroid: [Num_Layers, Dim]
    T_all = all_sentences_tensor.mean(dim=0)
    
    # Normalize each layer's vector independently
    # T_all.norm(dim=1, keepdim=True) gives norms of shape [Num_Layers, 1]
    T_all = T_all / (T_all.norm(dim=1, keepdim=True) + 1e-8)
    
    print(f"Built T_all. Shape: {tuple(T_all.shape)} (Layers, Dim)")
    return T_all.to(device)


#
# === 2. Multi-Layer Steering ===
def generate_with_multilayer_steering(
    input_en: str, 
    T_all: torch.Tensor, 
    max_new_tokens=40,
    start_layer_idx: int = 0,
    end_layer_idx: int = None
):
    inputs = qwen_tok(input_en, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    if hasattr(qwen, "model"):
        layers = qwen.model.decoder.layers
    else:
        layers = qwen.transformer.h
    
    num_layers = len(layers)
    if end_layer_idx is None:
        end_layer_idx = num_layers

    if T_all.shape[0] != num_layers:
        raise ValueError(f"T_all has {T_all.shape[0]} layers, but model has {num_layers}.")
    if not (0 <= start_layer_idx < end_layer_idx <= num_layers):
        raise ValueError(
            f"Invalid layer indices. Must be 0 <= start({start_layer_idx}) < end({end_layer_idx}) <= {num_layers}"
        )

    print(f"Steering layers in range: {start_layer_idx} to {end_layer_idx - 1}")

    def make_hook(T_layer):
        def _hook(module, args, output):
            # Unpack tuple if necessary
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            # Prefill vs decode:
            #   prefill: (batch, seq_len_prompt, d)
            #   decode:  (batch, 1, d)
            if hidden_states.shape[1] == 1:
                # decoding step -> do NOT steer
                return output

            # Clone to avoid in-place issues
            hidden_states = hidden_states.clone()

            # steer *all* prompt tokens, not just the last one
            # hidden_states: (B, S, D)
            # T_vec:         (D,)
            T_vec = T_layer.to(hidden_states.dtype)           # (D,)
            T_vec_b = T_vec.view(1, 1, -1)                    # (1, 1, D) for broadcasting

            # dot: (B, S, 1) = ⟨h_t, T⟩ for each token t
            dot = (hidden_states * T_vec_b).sum(dim=-1, keepdim=True)
            # proj: (B, S, D) = ⟨h_t, T⟩ T
            proj = dot * T_vec_b
            # remove T component from every token
            hidden_states = hidden_states - proj

            if rest is not None:
                return (hidden_states,) + rest
            return hidden_states

        return _hook

    hooks = []
    for i in range(start_layer_idx, end_layer_idx):
        layer = layers[i]
        T_l = T_all[i]
        h = layer.register_forward_hook(make_hook(T_l))
        hooks.append(h)

    try:
        with torch.no_grad():
            output_ids = qwen.generate(**inputs, max_new_tokens=max_new_tokens)
    finally:
        for h in hooks:
            h.remove()

    return qwen_tok.decode(output_ids[0], skip_special_tokens=True)

# === Run Test ===
if __name__ == "__main__":
    # Settings
    translations_path = "/home/acevedo/syn-sem/datasets/txt/steering/translations.txt" 
    input_sentence = "Alessandro went for a hike on Sunday"
    n_languages = 5
    max_new_tokens = 40
    
    # 1. Build vectors for ALL layers
    # This returns a tensor of shape [Num_Layers, Hidden_Dim]
    T_all_layers = build_all_layers_T(translations_path, n_languages=n_languages)

    # Calculate layer indices for the second half
    num_layers = T_all_layers.shape[0]
    start = num_layers * 1 // 4
    end = num_layers * 3 // 4
    
    print("\n--- Generating Baseline ---")
    for _ in range(10):
        # 2. Baseline
        baseline = qwen.generate(**qwen_tok(input_sentence, return_tensors="pt").to(device), max_new_tokens=max_new_tokens)
        print(qwen_tok.decode(baseline[0], skip_special_tokens=True))
    print("\n--- Generating with Multi-Layer Steering ---")
    print(f"Steering layers in range: {start} to {end}")

    for _ in range(10):
        # 3. Multi-Layer Steered ()
        steered = generate_with_multilayer_steering(
            input_sentence, 
            T_all_layers, 
            max_new_tokens=max_new_tokens,
            start_layer_idx=start,
            end_layer_idx=end
        )
        print(steered)


