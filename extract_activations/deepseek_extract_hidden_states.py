import os, sys
sys.path.append('../')
from modelpaths import model_paths
from utils_extract import *

def process_file(
                model_name,
                llm,
                sampling_params,
                batch_size,
                n_lines,
                IO_paths):
    
    config = AutoConfig.from_pretrained(model_paths[model_name], trust_remote_code=True)
    model_dtype = config.torch_dtype
    
    with open(IO_paths["file_path"], "r") as f:
        prompts = [line.strip() for line in f]

    if n_lines is not None:
        prompts = prompts[:n_lines]

    os.makedirs(IO_paths["output_folder_path"], exist_ok=True)

    for i, batch in enumerate(batch_generator(prompts, batch_size)):
        start = time.time()
        # request hidden states on each generate call
        outputs = llm.generate(
            batch,
            sampling_params=sampling_params,
            # return_hidden_states=True,
        )

        # Clip hidden states per sentence
        for output in outputs:
            hidden = output['meta_info']['hidden_states'][0]  # (L, T, E)
            hidden_clipped = clip_hidden_torch(torch.as_tensor(hidden,dtype=model_dtype))
            # Convert to uint16 for storage
            output['meta_info']['hidden_states'][0] = hidden_clipped

        # extract per-layer hidden states for each prompt in batch
        save_dict = {
            'outputs': outputs,
        }
        with open(f"{IO_paths['output_folder_path']}/chunk_{i}.pkl", "wb") as f:
            pickle.dump(save_dict, f) 

        t_step = time.time() - start
        print(f"iter {i} | t_step = {t_step:.2f} s", flush=True)
    return

def main(model_name,
         IO_paths_list,
         batch_size,
         n_lines,
         tp_size,
         nnodes):

    NODE_RANK = int(os.environ.get("SLURM_NODEID", 0))
    
    if nnodes == 1:
        port = find_free_port()
    else:
        port = int(os.environ["MASTER_PORT"])
    dist_init_addr = f"{get_master_address()}:{port}"

    print(f"Using free port {port} for rendezvous")

    # initialize engine (hidden states requested per-generate)
    model_path = model_paths[model_name]

    llm = sgl.Engine(
        model_path=model_path,
        tp_size=tp_size,
        nnodes=nnodes,
        dist_init_addr=dist_init_addr,
        node_rank=NODE_RANK,
        grammar_backend="xgrammar",
        disable_radix_cache=True,
        return_hidden_states=True,
    )

    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 1,
    }

    for IO_paths in IO_paths_list:
        print(f'processing {IO_paths}')
        process_file(
            model_name,
            llm,
            sampling_params,
            batch_size,
            n_lines,
            IO_paths,
        )

    llm.shutdown()



if __name__ == "__main__":
    print(f'{sys.argv=}')
    model_name = 'deepseek' # 
    language = 'english' #sys.argv[1] # 
    data_var = 'syn' #sys.argv[2] # syn or sem
    match_var = 'matching'#sys.argv[3] # 'matching' or 'mismatching'
    dataset_var = 'third'

    tp_size, nnodes = get_slurm_config()
    print(f'{tp_size=}, {nnodes=}')
    n_lines = 2100
    batch_size = 100

    IO_paths_list = [
        {
            "file_path": f"/home/acevedo/syn-sem/datasets/txt/{data_var}/{dataset_var}/{match_var}/{language}/sentences{i}.txt",
            "output_folder_path": f"/home/acevedo/syn-sem/datasets/activations/{data_var}/{dataset_var}/{model_name}/{match_var}/{language}/{i}/"
        }
        for i in [0, 1]
    ]
    
    # avoiding computing the activations of english original sentences more than once
    condition_1 = language != 'english' and data_var == 'sem'
    condition_2 = language == 'english' and data_var == 'syn' and dataset_var == 'third' 
    if condition_1 or condition_2:
        IO_paths_list = IO_paths_list[1:]

    main(
        model_name=model_name,
        IO_paths_list=IO_paths_list,
        batch_size=batch_size,
        n_lines=n_lines,
        tp_size=tp_size,
        nnodes=nnodes,
    )
