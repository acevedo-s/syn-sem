import sys,os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from time import time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils_extract import export, extract, load_lines
from modelpaths import *
from codex_helpers.helpers import has_safetensors_weights, resolve_model_source



if __name__ == '__main__':
    print(f'{sys.argv=}')
    model_name = sys.argv[4] if len(sys.argv) > 4 else 'gemma12b'
    language = sys.argv[1] # 
    data_var = sys.argv[2] # syn or sem
    match_var = sys.argv[3] # 'matching' or 'mismatching'
    model_source = resolve_model_source(model_name, model_paths)
    print(f'{model_name=}')
    print(f'{model_source=}')

    n_lines = 2100
    chunk_size = 100

    default_dataset_var = 'second'
    use_third_for_syn_B = (
        language == 'english'
        and data_var == 'syn'
        and match_var == 'matching'
    )

    IO_paths_list = []
    for i in [0, 1]:
        dataset_var = 'third' if use_third_for_syn_B and i == 1 else default_dataset_var
        IO_paths_list.append(
            {
                "input_path": f"/home/acevedo/syn-sem/datasets/txt/{data_var}/{dataset_var}/{match_var}/{language}/sentences{i}.txt",
                "output_folder": f"/home/acevedo/syn-sem/datasets/activations/{data_var}/{dataset_var}/{model_name}/{match_var}/{language}/{i}/"
            }
        )

    # For semantic data, English sentences0 are shared across languages.
    if language != 'english' and data_var == 'sem':
        IO_paths_list = IO_paths_list[1:]

    for IO_paths in IO_paths_list:
        os.makedirs(IO_paths['output_folder'], exist_ok=True)
        input_sentences = load_lines(IO_paths['input_path'])[:n_lines]

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
        # model_obj.to(device)

        model_dtype = next(model_obj.parameters()).dtype
        print(f"Model dtype: {model_dtype}")

        export(
                sentences=input_sentences,
                out_dir=IO_paths['output_folder'],
                tokenizer=tokenizer,
                model_obj=model_obj,
                device=device,
                model_dtype=model_dtype,
                model_id=model_name,
                extract_fn=extract,
                chunk_size=chunk_size,
            )

        print("Export complete.")
