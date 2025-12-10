import sys,os
sys.path.append('../')
from time import time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils_extract import export, extract, load_lines
from modelpaths import *



if __name__ == '__main__':
    print(f'{sys.argv=}')
    model_name = 'gemma12b' 
    language = sys.argv[1] # 
    data_var = sys.argv[2] # syn or sem
    match_var = sys.argv[3] # 'matching' or 'mismatching'

    n_lines = 2100
    chunk_size = 100

    dataset_var = 'second'

    IO_paths_list = [
        {
            "input_path": f"/home/acevedo/syn-sem/datasets/txt/{data_var}/{dataset_var}/{match_var}/{language}/sentences{i}.txt",
            "output_folder": f"/home/acevedo/syn-sem/datasets/activations/{data_var}/{dataset_var}/{model_name}/{match_var}/{language}/{i}/"
        }
        for i in [0, 1]
    ]
    # avoiding computing the activations of english original sentences more than once
    condition_1 = language != 'english' and data_var == 'sem'
    condition_2 = language == 'english' and data_var == 'syn' and dataset_var == 'third' 
    if condition_1 or condition_2:
        IO_paths_list = IO_paths_list[1:]
    
    if dataset_var == 'third':
        IO_paths_list = IO_paths_list[:1]

    for IO_paths in IO_paths_list:
        os.makedirs(IO_paths['output_folder'], exist_ok=True)
        input_sentences = load_lines(IO_paths['input_path'])[:n_lines]

        tokenizer = AutoTokenizer.from_pretrained(repo_ids[model_name])
        model_obj = AutoModelForCausalLM.from_pretrained(repo_ids[model_name],
                                                        cache_dir=model_paths[model_name],
                                                        device_map="auto",
                                                        low_cpu_mem_usage=True,
                                                        torch_dtype="auto",
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
