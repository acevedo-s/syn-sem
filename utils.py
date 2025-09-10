import re,os,sys
from pathlib import Path
import torch
import jax.numpy as jnp
from collections import defaultdict
import pickle
from tqdm import tqdm
import torch
from einops import rearrange
import jax
import numpy as np
from time import time
from modelpaths import *
from transformers import AutoConfig

precision_map = {
                    16: jnp.bfloat16,   # or jnp.float16 if you really want IEEE half
                    32: jnp.float32,
                    64: jnp.float64,
                }

def get_num_hidden_layers(model_dir: str) -> int:
    """
    Load model config from the given directory and return the number of hidden layers.
    """
    config = AutoConfig.from_pretrained(model_dir)
    return config.num_hidden_layers

def get_hidden_size(model_dir: str) -> int:
    """
    Load model config from the given directory and return the embedding (hidden) size.
    """
    config = AutoConfig.from_pretrained(model_dir)
    return config.hidden_size

def add_model_metadata(depths: dict, emb_dims: dict, model_name: str, model_paths: dict) -> None:
    """
    Add the number of hidden layers and embedding size for a model to the given dictionaries.

    Args:
        depths (dict): Dictionary to update with model depth.
        emb_dims (dict): Dictionary to update with embedding size.
        model_name (str): Key to use in the dictionaries (e.g., 'qwen').
        model_paths (dict): Dictionary mapping model names to local paths.
    """
    config = AutoConfig.from_pretrained(model_paths[model_name])
    depths[model_name] = config.num_hidden_layers
    emb_dims[model_name] = config.hidden_size


depths = {}
emb_dims = {}
models = ['qwen7b', 'llama8b', 'deepseek',]

for model in models:
  add_model_metadata(depths, emb_dims, model, model_paths)

# batch_sizes = {'deepseek':100,
#                'llama':100,
#                'qwen':100}


def extract_index(file):
    match = re.search(r'chunk_(\d+)\.pkl', file.name)
    return int(match.group(1)) if match else float('inf')  # push non-matching files to the end

def list_folder(path, desc="chunk_"):
    folder = Path(path)

    #* collect all the files starting with *chunk_* in the input_path
    files = sorted(
        [f for f in folder.iterdir() if f.name.startswith(desc)],
        key=extract_index)

    return files

def torch_to_jax(tensor,precision):
    assert tensor.dtype == torch.bfloat16
    intermediate_tensor = tensor.view(torch.uint16).numpy()
    tensor = jnp.array(intermediate_tensor).astype(precision_map[precision])
    return tensor

def flatten_tokens_features(tensor):
    tensor = rearrange(tensor, 'batch tokens embed -> batch (tokens embed)')
    return tensor

def binarize(a):
    a = jnp.sign(a).astype(jnp.int8)
    a = a.at[a == 0].set(-1)
    return a

@jax.jit
def get_quantiles(a,alphamin,alphamax):
  qmin = jnp.quantile(a,q=alphamin,axis=1) #(a.shape[0],)
  qmax = jnp.quantile(a,q=alphamax,axis=1)
  return qmin,qmax

@jax.jit
def jclip(act, alphamin, alphamax):
    if act.ndim == 3:  # static Python check, safe in JIT
        B, T, E = act.shape
        act_flat = jnp.reshape(act, (B, T*E))
        qmin, qmax = get_quantiles(act_flat, alphamin, alphamax)
        act_clipped = jnp.minimum(jnp.maximum(act_flat, qmin[:, None]), qmax[:, None])
        return jnp.reshape(act_clipped, (B, T, E))
    else:
        qmin, qmax = get_quantiles(act, alphamin, alphamax)
        return jnp.minimum(jnp.maximum(act, qmin[:, None]), qmax[:, None])


def clip(act,
         alphamin = 0.05,
         alphamax = 0.95,
         verbose = False,
         ):
    if verbose:
      start = time()

    act = jclip(act,alphamin,alphamax)
    
    if verbose:
      print(f'clipping took {(time()-start)/60.} m')

    return act

def makefolder(base='./',
               create_folder=False,
               float_precision=5,
               **kwargs,
               ):
  folder = base
  for key, value in kwargs.items():
    if value != None:
      if isinstance(value,float) == True:
        folder += key + f'_{value:.{float_precision}f}/'
      else:
        folder += key + f'_{value}/'
  if create_folder:
    os.makedirs(folder,exist_ok=True)
  return folder

def get_acronym(language):
  if language == 'spanish':
    acronym = 'es'
  elif language == 'german':
     acronym = 'de'
  elif language == 'italian':
    acronym = 'it'
  elif language == 'english':
    acronym = 'en'
  elif language == 'russian':
    acronym = 'ru'
  elif language == 'hungarian':
    acronym = 'hu'
  return acronym

def reduce_list_half_preserve_extremes(lst):
    """
    Reduces the input list to approximately half its original size,
    preserving the first and last elements, and sampling uniformly
    from the intermediate elements.
    
    Parameters:
    -----------
    lst : list
        The input list to reduce.
    
    Returns:
    --------
    list
        A reduced list with approximately half the points,
        preserving the first and last elements.
    """
    N = len(lst)
    if N <= 2:
        return lst.copy()
    
    half_N = max(N // 2, 2)
    num_points_to_sample = half_N - 2
    
    new_lst = [lst[0]]
    
    if num_points_to_sample > 0:
        # Calculate the indices to sample from intermediates
        step = (N - 2) / (num_points_to_sample + 1)
        intermediate_indices = [int(round(1 + i * step)) for i in range(num_points_to_sample)]
        new_lst.extend([lst[i] for i in intermediate_indices])
    
    new_lst.append(lst[-1])
    return new_lst

def reshuffle_batch_axis(act, key):
    """
    Reshuffles the activations along the batch axis.

    Args:
        act (jnp.ndarray): Activation matrix of shape (batch_size, ...).
        key (jax.random.PRNGKey): A PRNG key for randomness.

    Returns:
        jnp.ndarray: Shuffled activations.
    """
    batch_size = act.shape[0]
    perm = jax.random.permutation(key, batch_size)
    return act[perm]

def collect_data(input_path, 
                 min_token_length, 
                 n_files,
                 model_name,
                 ):
    ### activations dtype
    config = AutoConfig.from_pretrained(model_paths[model_name])
    model_dtype = config.torch_dtype
    print(f'{model_name} dtype: {model_dtype}')
    start_time = time()
    files = list_folder(input_path, desc="chunk_")[:n_files]
    all_hidden_states = defaultdict(list)

    for file in tqdm(files, desc="Collect File"):
        with open(os.path.join(input_path, file.name), 'rb') as f:
            outputs = pickle.load(f)['outputs']  # list of batch_size "outputs"
        for output_id,output in enumerate(outputs):
            hidden = output['meta_info']['all_hidden_states'][0]  # shape (L, sentence_length, E)
            hidden = torch.as_tensor(hidden,dtype=model_dtype)
            assert hidden.shape[1] >= min_token_length, f"{file.name=},{output_id=},{hidden.shape=}"
            hidden = hidden[:, -min_token_length:, :]  

            for i in range(hidden.shape[0]):  # Loop over layers directly
                all_hidden_states[f"layer_{i}"].append(hidden[i])

    # Stack once after collection
    for layer in all_hidden_states:
        all_hidden_states[layer] = torch.stack(all_hidden_states[layer])
    
    print(f'{all_hidden_states["layer_0"].shape=}')
    print(f'importing took {(time()-start_time)/60.} m')

    return all_hidden_states


def _collect_data(input_path, min_token_length, n_files, model_name):
    config = AutoConfig.from_pretrained(model_paths[model_name])
    model_dtype = config.torch_dtype
    print(f'{model_name} dtype: {model_dtype}')
    start_time = time()
    files = list_folder(input_path, desc="chunk_")[:n_files]

    # --- Peek at the first file to get sizes
    with open(os.path.join(input_path, files[0].name), 'rb') as f:
        first_outputs = pickle.load(f)['outputs']
    batch_size = len(first_outputs)
    first_hidden = torch.as_tensor(
        first_outputs[0]['meta_info']['all_hidden_states'][0],
        dtype=model_dtype
    )
    n_layers, _, hidden_dim = first_hidden.shape

    # --- Preallocate assuming all files are full
    max_samples = n_files * batch_size
    all_hidden_states = {
        f"layer_{i}": torch.empty(
            (max_samples, min_token_length, hidden_dim),
            dtype=model_dtype
        )
        for i in range(n_layers)
    }

    # --- Fill batch-wise
    ptr = 0
    for file in tqdm(files, desc="Collect File"):
        with open(os.path.join(input_path, file.name), 'rb') as f:
            outputs = pickle.load(f)['outputs']

        # Convert entire batch to tensor of shape (batch_size, n_layers, seq_len, hidden_dim)
        batch_hidden = torch.stack([
            torch.as_tensor(
                o['meta_info']['all_hidden_states'][0], dtype=model_dtype
            )[:, -min_token_length:, :]
            for o in outputs
        ])  # shape: (batch_size, n_layers, min_token_length, hidden_dim)

        b = batch_hidden.shape[0]  # actual batch size (last batch might be smaller)

        # Copy layer-wise in one shot
        for i in range(n_layers):
            all_hidden_states[f"layer_{i}"][ptr:ptr+b] = batch_hidden[:, i, :, :]
        ptr += b

    # --- Trim unused rows
    for i in range(n_layers):
        all_hidden_states[f"layer_{i}"] = all_hidden_states[f"layer_{i}"][:ptr]

    print(f'{all_hidden_states["layer_0"].shape=}')
    print(f'importing took {(time()-start_time)/60.} m')

    return all_hidden_states

def compute_and_subtract_syn_group_averages(sim_folder,
                                            act,
                                            center_flag,
                                            space_index,
                                            removal_method,
                                            seed = 1234,
                                            ):
  centers_folder = sim_folder

  assert len(act.shape) == 2
  assert space_index == 'A' or space_index == 'B'
  assert removal_method in ['subtraction','projection']

  try: 
    centers = jnp.array(np.load(os.path.join(centers_folder, f'syn_centers_{space_index}.npy'))).astype(act.dtype) #(num_groups,E)
    all_indices = jnp.array(np.loadtxt(centers_folder + f'syn_all_indices.txt',dtype=int),dtype=jnp.int32)
    all_counts = jnp.array(np.loadtxt(centers_folder + f'syn_all_counts.txt',dtype=int),dtype=jnp.int32)
    print(f'centers imported from {os.path.join(centers_folder, f"syn_centers_{space_index}.npy")}')
  except:
    centers, all_indices, all_counts = _compute_and_export_syn_centers(act, centers_folder, space_index)

  if center_flag == -1:
    key_center = jax.random.PRNGKey(seed)
    centers = jax.random.permutation(key_center,centers)
    #key_center, subkey_center = jax.random.split(key_center)
  
  #TODO: optimize with jax.jit
  for group_index in range(centers.shape[0]):
    dynamic_indices = all_indices[group_index][:all_counts[group_index]]
    center = centers[group_index]
    act = _remove_syn_group_average(act, dynamic_indices, center, removal_method)
  return act

@jax.jit
def _compute_syn_center(act, group_index, all_group_ids):
    mask = (all_group_ids == group_index)          # (n_samples,)
    center = jnp.mean(act, where=mask[:,None], axis=0)  # broadcasting mask to (n_samples, E)
    indices = jnp.nonzero(mask, size=act.shape[0])[0]  
    counts = jnp.sum(mask)
    return center, indices, counts

def _compute_and_export_syn_centers(act, centers_folder, space_index):
  syn_group_ids_path = "/home/acevedo/syn-sem/datasets/txt/syn/second/matching/english/group_ids.txt"
  all_group_ids = jnp.array(np.loadtxt(syn_group_ids_path),dtype=jnp.int32)
  unique_groups_indices = jnp.unique(all_group_ids)

  assert len(all_group_ids) == act.shape[0]
  assert unique_groups_indices.min() == 0 and unique_groups_indices.max() == len(unique_groups_indices) - 1
  
  centers, all_indices, all_counts = jax.vmap(_compute_syn_center,in_axes=(None,0,None))(act,unique_groups_indices,all_group_ids)  

  np.save(os.path.join(centers_folder,f"syn_centers_{space_index}"),centers)
  np.savetxt(os.path.join(centers_folder,f"syn_all_indices.txt"), all_indices, fmt='%d')
  np.savetxt(os.path.join(centers_folder,f"syn_all_counts.txt"), all_counts, fmt='%d')
  print(f'centers saved at {os.path.join(centers_folder,f"syn_centers_{space_index}.npy")}')

  return centers, all_indices, all_counts

def _remove_syn_group_average(act, dynamic_indices, center, removal_method:str):
    """
    act: (n_samples,T*E)
    dynamic_indices: (n_samples_at_given_group)
    center:(T*E)
    """
    if removal_method == 'subtraction':
      act = act.at[dynamic_indices].add(-center)
    elif removal_method == 'projection':
      proj_coeffs = (act[dynamic_indices] @ center) / (center @ center) #(len(indices),)
      projections = jnp.outer(proj_coeffs,center) #(len(indices),E)
      act = act.at[dynamic_indices].add(-projections)

    return act

def load_and_subtract_syn_group_averages(act_A,
                                        act_B,
                                        sim_folder,
                                        center_flag,
                                        removal_method, # 'subtraction' or 'projection'
                                        random_center_type,
                                        ):
  print(f'loading and subtracting syn group averages')
  # original_labels = jnp.array(np.loadtxt("/home/acevedo/syn-sem/datasets/txt/sem/second/matching/english/original_labels.txt").astype(int))
  group_ids_path = "/home/acevedo/syn-sem/datasets/txt/sem/second/matching/english/group_ids.txt"
  all_group_ids = jnp.array(np.loadtxt(group_ids_path).astype(int)) # (n_samples,)
  assert len(all_group_ids) == act_A.shape[0] == act_B.shape[0]

  if center_flag == -1:
    key = jax.random.PRNGKey(9999)  # Change seed for different permutations
    if random_center_type == 'permuted':
      unique_groups_indices = jnp.unique(all_group_ids)
      shuffled_groups = jax.random.permutation(key, unique_groups_indices)
      mapping = dict(zip(unique_groups_indices.tolist(), shuffled_groups.tolist()))
      all_group_ids = jnp.array([mapping[g] for g in all_group_ids.tolist()])
    elif random_center_type == 'shuffled':
      all_group_ids = jax.random.permutation(key, all_group_ids)

  centers_folder = sim_folder.replace("data_var_sem", "data_var_syn")
  centers_folder = centers_folder.replace("n_files_16", "n_files_21")

  centers = jnp.mean(jnp.stack([jnp.array(np.load(centers_folder+f'syn_centers_A.npy')),
                                jnp.array(np.load(centers_folder+f'syn_centers_B.npy'))]),
                    axis=0).astype(act_A.dtype) #(num_groups,E)

  act_A = remove_syn_group_averages(act_A, 
                                centers, 
                                all_group_ids, 
                                removal_method)
  # act_B = remove_syn_group_averages(act_B, 
  #                               centers, 
  #                               all_group_ids, 
  #                               removal_method)


  return act_A,act_B 

def remove_syn_group_averages(act, centers, all_group_ids, removal_method):
    
  for group_index in range(centers.shape[0]):
    dynamic_indices = jnp.where(all_group_ids==group_index)[0]
    center = centers[group_index]
    act = _remove_syn_group_average(act, dynamic_indices, center, removal_method)
  return act

def load_and_subtract_sem_group_averages(sim_folder,act,data_var,center_flag,number_of_languages,removal_method):
  centers_folder = re.sub(r'language_[^/]+', 'language_english', sim_folder)
  centers_folder = re.sub(r'data_var_syn', 'data_var_sem', centers_folder)  
  semantic_centers = jnp.array(np.load(centers_folder+f'semantic_centers_{number_of_languages}.npy'),dtype=act.dtype) #(num_sentences,E)

  if data_var == 'syn':
    print('TODO::::')
    sys.exit() # TODO: check
    semantic_labels_file = '/home/acevedo/syn-sem/datasets/txt/syn/second/matching/english/semantic_labels.txt'
    indices = jnp.array(np.loadtxt(semantic_labels_file,dtype=int,unpack=True)[0])
  elif data_var == 'sem':
    indices = jnp.arange(act.shape[0],dtype=jnp.int32)
  if center_flag == -1:
    key_centers = jax.random.PRNGKey(999)
    indices = jax.random.permutation(key_centers,indices)
  
  if removal_method == 'subtraction':
    act = remove_sem_group_center(act, indices, semantic_centers)
  elif removal_method == 'projection':
    act = remove_sem_group_projections(act, indices, semantic_centers)
  return act

@jax.jit
def remove_sem_group_center(act, indices, semantic_centers):

  act = act - semantic_centers[indices]
  return  act

@jax.jit
def remove_sem_group_projections(act,indices,semantic_centers):

  proj_coeffs = (jnp.einsum("ij,ij->i", act[indices], semantic_centers[indices])) / (jnp.einsum("ij,ij->i", semantic_centers[indices], semantic_centers[indices]) + 1E-8) #(len(indices),)
  projections = proj_coeffs[:, None] * semantic_centers[indices]  # (len(indices), E)
  act = act - projections

  return act


def set_number_of_languages_list(center_A_flag,center_B_flag,centers):
    
    number_of_languages_list = [None]
    
    if center_A_flag != 0 or center_B_flag != 0:
        if centers == 'sem':
            number_of_languages_list = list(range(4,4+1))

    return number_of_languages_list