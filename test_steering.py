import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. Load models ===
# Translation model: e.g. mBART or Marian
# You can pick a model that supports en→es and en→zh
# Here, as placeholder:
trans_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE").to(device)
trans_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE")

# For Chinese, maybe a separate model; you might need two translation models.
# (For simplicity, using same model here; you’d replace with a proper en→zh model.)

# Qwen model
qwen_name = "Qwen/Qwen-7B"  # or whichever variant you want
qwen_tok = AutoTokenizer.from_pretrained(qwen_name, trust_remote_code=True)
qwen = AutoModelForCausalLM.from_pretrained(qwen_name, trust_remote_code=True, device_map="auto").eval()

# === 2. helper: translate and get Qwen activation ===

def translate_and_get_hidden(input_en: str, tgt_lang: str, layer_idx: int):
    """
    Translate input_en → target language, then run Qwen and return
    the hidden activations at layer layer_idx (shape: [seq_len, hidden_dim]).
    """
    # translate
    # you may want to use different translation models per target
    tok = trans_tok
    model = trans_model
    # e.g. add language prefix if needed
    t = tok(input_en, return_tensors="pt", padding=True).to(device)
    out = model.generate(**t, num_beams=4, max_length=2 * t["input_ids"].shape[1])
    translation = tok.batch_decode(out, skip_special_tokens=True)[0]
    # now feed to Qwen
    q = qwen_tok(translation, return_tensors="pt").to(device)
    # define hook to capture hidden at that layer
    hidden_cache = {}
    def hook_fn(module, inp, out):
        # out is (batch, seq_len, hidden_dim)
        hidden_cache["h"] = out.detach()[0]  # take batch=0
    # get number of layers
    # assume qwen.transformer.h is list of layers (you may need introspection)
    layers = qwen.model.decoder.layers if hasattr(qwen, "model") else qwen.transformer.h
    handle = layers[layer_idx].register_forward_hook(hook_fn)
    # forward pass
    with torch.no_grad():
        qwen(**q)  # not generating, just feed through
    handle.remove()
    # hidden_cache["h"] is shape [seq_len, hidden_dim]
    return hidden_cache["h"], translation

# === 3. Build T vector for a given input_en ===

def build_centroid_T(input_en: str, central_layer: int):
    """
    Returns T vector (hidden_dim,) computed as average over translations.
    We'll generate 2 Spanish, 2 Chinese translations (you can extend).
    """
    Ts = []
    # Spanish translations (2 variants)
    for _ in range(2):
        h_es, _ = translate_and_get_hidden(input_en, tgt_lang="es", layer_idx=central_layer)
        Ts.append(h_es.mean(dim=0))
    # Chinese translations (2 variants)
    for _ in range(2):
        h_zh, _ = translate_and_get_hidden(input_en, tgt_lang="zh", layer_idx=central_layer)
        Ts.append(h_zh.mean(dim=0))
    # average
    T = torch.stack(Ts, dim=0).mean(dim=0)  # shape [hidden_dim]
    # optionally normalize T
    T = T / (T.norm() + 1e-8)
    return T

# === 4. Run Qwen on English input with intervention ===

def generate_with_steering(input_en: str, T: torch.Tensor, central_layer: int, max_new_tokens=20):
    """
    Generate from Qwen on original English, but intervene at the last token's
    hidden at central_layer (subtracting projection onto T).
    """
    q = qwen_tok(input_en, return_tensors="pt").to(device)
    input_ids = q["input_ids"]
    attention_mask = q["attention_mask"]
    # Hook that modifies only the hidden of the *last token* at that layer
    def steering_hook(module, inp, out):
        # out: (batch, seq_len, hidden_dim)
        h = out  # alias
        # get last token hidden
        last = h[:, -1, :]  # shape (batch, hidden_dim)
        # subtract projection: proj = (last ⋅ T) T
        # assume T is on same device
        proj = (last * T).sum(dim=-1, keepdim=True) * T  # shape (batch, hidden_dim)
        new_last = last - proj
        # replace back
        h2 = h.clone()
        h2[:, -1, :] = new_last
        return h2

    layers = qwen.model.decoder.layers if hasattr(qwen, "model") else qwen.transformer.h
    handle = layers[central_layer].register_forward_hook(steering_hook)

    # now generate (with modification)
    with torch.no_grad():
        out_ids = qwen.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
    handle.remove()

    return qwen_tok.batch_decode(out_ids, skip_special_tokens=True)[0]

# === 5. Full pipeline: for an English sentence ===

def steering_pipeline(input_en: str, central_layer: int = None):
    # pick central layer: e.g. half
    total_layers = len(qwen.model.decoder.layers) if hasattr(qwen, "model") else len(qwen.transformer.h)
    if central_layer is None:
        central_layer = total_layers // 2
    print("Using central layer:", central_layer)
    T = build_centroid_T(input_en, central_layer).to(device)
    print("T norm:", T.norm().item())
    # baseline generation
    baseline = qwen.generate(**qwen_tok(input_en, return_tensors="pt").to(device), max_new_tokens=20)
    baseline_text = qwen_tok.batch_decode(baseline, skip_special_tokens=True)[0]
    steered = generate_with_steering(input_en, T, central_layer, max_new_tokens=20)
    return baseline_text, steered

# === 6. Example usage ===

if __name__ == "__main__":
    inp = "The weather tomorrow will be"
    base, steered = steering_pipeline(inp)
    print("Baseline:", base)
    print("Steered:", steered)
