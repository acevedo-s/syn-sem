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


my_languages = ['spanish','chinese', 'german', 'arabic', 'italian', 'turkish']

precision_map = {
                    16: jnp.bfloat16,
                    32: jnp.float32,
                    64: jnp.float64,
                }

removal_method_map = {
                      'subtraction':0,
                      'projection':1,
                      }

len_group_ids_path = "/home/acevedo/syn-sem/datasets/txt/syn/second/mismatching/english/group_ids.txt"
syn_group_ids_path = '/home/acevedo/syn-sem/datasets/txt/syn/second/matching/english/group_ids.txt' 
syn_group_id_paths_for_sem_data = {'A' : "/home/acevedo/syn-sem/datasets/txt/sem/second/matching/english/all_group_ids_A.txt",
                                   'B' : "/home/acevedo/syn-sem/datasets/txt/sem/second/matching/english/all_group_ids_B.txt",
                                  }
syn_common_indices_path = "/home/acevedo/syn-sem/datasets/txt/sem/second/matching/english/syn_common_indices_B.txt"
common_group_ids_B_path = "/home/acevedo/syn-sem/datasets/txt/sem/second/matching/english/common_group_ids_B.txt"
sem_ids_with_syn_path = "/home/acevedo/syn-sem/datasets/txt/sem/second/matching/english/sem_ids_with_syn.txt"
syn_ids_with_sem_path = "/home/acevedo/syn-sem/datasets/txt/syn/second/matching/english/syn_ids_with_sem.txt"
sem_centers_ids_path = "/home/acevedo/syn-sem/datasets/txt/syn/second/matching/english/sem_centers_ids.txt"
syn_syn_ids_path = "/home/acevedo/syn-sem/datasets/txt/sem/second/matching/english/syn_syn_indices.txt"

def get_syn_centroids_folder(sim_folder):
  centers_folder = sim_folder.replace("data_var_sem", "data_var_syn")
  return centers_folder

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
models = ['qwen7b', 'deepseek',]

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

# def torch_to_jax(tensor,precision):
#     assert tensor.dtype == torch.bfloat16
#     intermediate_tensor = tensor.view(torch.uint16).numpy()
#     tensor = jnp.array(intermediate_tensor).astype(precision_map[precision])
#     return tensor

def torch_to_jax(tensor: torch.Tensor, precision:int):
    """Convert torch.bfloat16 tensor to JAX bfloat16/float32 safely."""
    assert tensor.dtype == torch.bfloat16
    # Raw bits as uint16
    np_uint16 = tensor.view(torch.uint16).cpu().numpy()
    jax_uint16 = jnp.array(np_uint16, dtype=jnp.uint16)
    # Reinterpret bits -> bfloat16
    jax_bf16 = jax.lax.bitcast_convert_type(jax_uint16, jnp.bfloat16)
    # Cast if requested
    return jax_bf16.astype(precision_map[precision])

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
  B, T, E = act.shape
  act_flat = jnp.reshape(act, (B, T*E))
  qmin, qmax = get_quantiles(act_flat, alphamin, alphamax)
  act_clipped = jnp.minimum(jnp.maximum(act_flat, qmin[:, None]), qmax[:, None])
  return jnp.reshape(act_clipped, (B, T, E))


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

def reduce_list_half_preserve_extremes(lst):
    """
    Reduces the input list to approximately half its original size,
    preserving the first and last elements, and sampling uniformly
    from the intermediate elements.
    """
    N = len(lst)
    if N <= 2:
        return lst.copy()

    half_N = max(N // 2, 2)
    # use linspace for evenly spaced *float* indices, then floor and deduplicate
    indices = np.linspace(0, N - 1, num=half_N, endpoint=True)
    indices = np.unique(np.floor(indices).astype(int))
    if indices[-1] != N - 1:
        indices = np.append(indices, N - 1)
    return [lst[i] for i in indices]


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
                 avg_tokens,
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
            if model_name == 'deepseek':
              hidden = output['meta_info']['hidden_states'][0]  # shape (L, sentence_length, E)
            else:
              hidden = output['meta_info']['all_hidden_states'][0]  # shape (L, sentence_length, E)
            hidden = torch.from_numpy(hidden).view(torch.bfloat16)
            assert hidden.shape[1] >= min_token_length, f"{file.name=},{output_id=},{hidden.shape=}"
            hidden = hidden[:, -min_token_length:, :]  
            if avg_tokens == 1:
              hidden = torch.mean(hidden.to(torch.float32),dim=1).to(model_dtype)

            for i in range(hidden.shape[0]):  # Loop over layers directly
                all_hidden_states[f"layer_{i}"].append(hidden[i])

    # Stack once after collection
    for layer in all_hidden_states:
        all_hidden_states[layer] = torch.stack(all_hidden_states[layer])
    
    print(f'{all_hidden_states["layer_0"].shape=}')
    print(f'importing took {(time()-start_time)/60.} m')

    return all_hidden_states

def compute_and_subtract_syn_group_averages(sim_folder,
                                            act,
                                            center_flag,
                                            space_index,
                                            removal_method:str,
                                            syn_group_ids_path,
                                            ):
  centers_folder = sim_folder

  assert len(act.shape) == 2
  assert space_index == 'A' or space_index == 'B'
  assert removal_method in ['subtraction','projection']

  all_syn_group_ids = jnp.array(np.loadtxt(syn_group_ids_path).astype(jnp.int32)) # (n_syn_samples,)
  (
  unique_syn_centroids, # (n_groups,n_features)
  all_indices, # (n_groups, n_samples)
  all_counts, # (n_groups, 1)
    ) = _compute_and_export_syn_centers(syn_group_ids_path, act, centers_folder, space_index)
  expanded_syn_centroids = unique_syn_centroids[all_syn_group_ids] # (n_samples,E)

  indices = jnp.arange(act.shape[0],dtype=jnp.int32)
  if center_flag == -1:
    key_centers = jax.random.PRNGKey(np.random.randint(1E5))
    indices = jax.random.permutation(key_centers,indices)

  expanded_group_counts = get_syntax_expanded_counts(unique_syn_centroids,all_syn_group_ids)
  loo_expanded_syn_centroids = (expanded_group_counts[:,None] * expanded_syn_centroids - act) / (expanded_group_counts[:,None] - 1)
  
  if removal_method == 'subtraction':
    act = batched_subtract_centroids(act,indices,loo_expanded_syn_centroids)
  elif removal_method == 'projection':
    act = batched_remove_centroid_projections(act,indices,loo_expanded_syn_centroids)
  return act

def get_syntax_expanded_counts(unique_syn_centroids, expanding_indices):
  unique_syn_group_ids, unique_syn_group_counts = jnp.unique(expanding_indices,return_counts=True)
  assert unique_syn_group_ids.max() == unique_syn_centroids.shape[0] - 1
  assert unique_syn_group_ids.min() == 0
  expanded_group_counts = unique_syn_group_counts[expanding_indices] #(n_samples,)
  return expanded_group_counts

@jax.jit
def _compute_syn_centroid(act, group_index, all_group_ids):
    mask = (all_group_ids == group_index)                 # (n_samples,)
    centroid = jnp.mean(act, where=mask[:,None], axis=0)  # broadcasting mask to (n_samples, E)
    indices = jnp.nonzero(mask, size=act.shape[0])[0]     # where the matches are 
    counts = jnp.sum(mask)                                # how many they are
    return centroid, indices, counts

def _compute_and_export_syn_centers(syn_group_ids_path, act, centers_folder, space_index):
  all_group_ids = jnp.array(np.loadtxt(syn_group_ids_path),dtype=jnp.int32)
  unique_groups_indices = jnp.unique(all_group_ids)

  assert len(all_group_ids) == act.shape[0]
  assert unique_groups_indices.min() == 0 and unique_groups_indices.max() == len(unique_groups_indices) - 1
  
  centers, all_indices, all_counts = jax.vmap(_compute_syn_centroid,in_axes=(None,0,None))(act,unique_groups_indices,all_group_ids)  

  np.save(os.path.join(centers_folder, f"syn_centers_{space_index}"), centers)
  np.savetxt(os.path.join(centers_folder, f"syn_all_indices_{space_index}.txt"), all_indices, fmt='%d')
  np.savetxt(os.path.join(centers_folder, f"syn_all_counts_{space_index}.txt"), all_counts, fmt='%d')
  print(f'centers saved at {os.path.join(centers_folder, f"syn_centers_{space_index}.npy")}')

  return centers, all_indices, all_counts

def load_syn_group_averages(act,
                            group_ids_path,
                            centers_folder,
                            center_flag,
                            global_center,
                            space_index,
                            ):
  all_group_ids = jnp.array(np.loadtxt(group_ids_path).astype(int)) # (n_samples,)
  assert len(all_group_ids) == act.shape[0]

  if center_flag == -1:
    print(f'permuting_group_ids')
    key = jax.random.PRNGKey(np.random.randint(1E5))  
    all_group_ids = jax.random.permutation(key, all_group_ids)

  if global_center != None:
    centers_folder = centers_folder.replace("global_centering_1","global_centering_0")

  centers = jnp.array(np.load(centers_folder+f'syn_centers_{space_index}.npy')).astype(act.dtype) #(num_groups,E)

  if global_center != None: 
    centers = centers - jnp.broadcast_to(global_center,centers.shape)

  return centers, all_group_ids

def load_and_subtract_syn_group_averages(act,
                                        group_ids_path,
                                        sim_folder,
                                        center_flag,
                                        removal_method:str, # 'subtraction' or 'projection'
                                        global_center,
                                        space_index,
                                        ):
  
  print(f'loading and subtracting syn group averages')

  syn_centroids_folder = get_syn_centroids_folder(sim_folder)

  (unique_syn_centroids, #(n_groups,E)
   syn_group_ids_for_sem, #(n_samples,)
   ) = load_syn_group_averages(act,
                              group_ids_path,
                              syn_centroids_folder,
                              center_flag,
                              global_center,
                              space_index,
                              )
  expanded_syn_centroids = unique_syn_centroids[syn_group_ids_for_sem] # (n_samples,E)

  ### I have to use the counting of the original syntax data to do LOO properly
  expanded_group_counts = get_syntax_expanded_counts(unique_syn_centroids,syn_group_ids_for_sem)
  loo_expanded_syn_centroids = (expanded_group_counts[:,None] * expanded_syn_centroids - act) / (expanded_group_counts[:,None] - 1)

  indices = jnp.arange(act.shape[0],dtype=jnp.int32)
  if center_flag == -1:
    key_centers = jax.random.PRNGKey(np.random.randint(1E5))
    indices = jax.random.permutation(key_centers,indices)
  else:
    semantic_centroids = load_sem_centroids(sim_folder,number_of_languages=6,language_list_permutation=0).astype(act.dtype) #(num_sentences,E)
    loo_expanded_syn_centroids = batched_remove_centroid_projections(loo_expanded_syn_centroids,indices,semantic_centroids)

  if removal_method == 'subtraction':
    act = batched_subtract_centroids(act,indices,loo_expanded_syn_centroids)
  elif removal_method == 'projection':
    act = batched_remove_centroid_projections(act,indices,loo_expanded_syn_centroids)
  return act

def load_sem_centroids(sim_folder,number_of_languages,language_list_permutation):

  centers_folder = sim_folder
  # centers_folder = re.sub(r'language_[^/]+', 'language_english', centers_folder)
  centers_folder = re.sub(r'data_var_syn', 'data_var_sem', centers_folder)  
  centers_folder = re.sub(r'centers_syn', 'centers_sem', centers_folder)  
  centers_folder = re.sub(r'similarity_fn_[^/]+', 'similarity_fn_none', centers_folder)
  centers_folder = re.sub(r'similarities','semantic_centers',centers_folder)
  centers_folder = re.sub(r'batch_shuffle_1','batch_shuffle_0',centers_folder)

  semantic_centers = jnp.array(np.load(centers_folder+f'semantic_centers_{number_of_languages}_{language_list_permutation}.npy')) #(num_sentences,E)

  return semantic_centers # careful about precision

def load_and_subtract_sem_group_averages(sim_folder,
                                         act,
                                         data_var,
                                         center_flag,
                                         number_of_languages,
                                         language_list_permutation,
                                         removal_method,
                                         batch_shuffle=False,
                                         seed_batch_shuffle=None,
                                         ):

  semantic_centroids = load_sem_centroids(sim_folder,number_of_languages,language_list_permutation).astype(act.dtype) #(num_sem_sentences,E)
  if batch_shuffle:
    semantic_centroids = reshuffle_batch_axis(semantic_centroids, jax.random.PRNGKey(seed_batch_shuffle))
  if data_var == 'syn':
    sem_center_ids = jnp.array(np.loadtxt(sem_centers_ids_path,dtype=int),dtype=jnp.int32)
    semantic_centroids = semantic_centroids[sem_center_ids] # this alignes centers to syntax data

  assert (act.shape == semantic_centroids.shape)

  indices = jnp.arange(act.shape[0],dtype=jnp.int32)
  if center_flag == -1:
    key_centers = jax.random.PRNGKey(np.random.randint(1E5))
    indices = jax.random.permutation(key_centers,indices)
  
  if removal_method == 'subtraction':
    act = batched_subtract_centroids(act, indices, semantic_centroids)
  elif removal_method == 'projection':
    act = batched_remove_centroid_projections(act, indices, semantic_centroids)
  return act

@jax.jit
def batched_subtract_centroids(act, indices, semantic_centers):

  act = act - semantic_centers[indices]
  return  act

@jax.jit
def batched_remove_centroid_projections(act,indices,centroids):

  proj_coeffs = (jnp.einsum("ij,ij->i", act[indices], centroids[indices])) / (jnp.einsum("ij,ij->i", centroids[indices], centroids[indices]) + 1E-8) #(len(indices),)
  projections = proj_coeffs[:, None] * centroids[indices]  # (len(indices), E)
  act = act - projections

  return act


def set_number_of_languages_list(center_A_flag, center_B_flag, centers_var):
    
    number_of_languages_list = [None]

    if center_A_flag != 0 or center_B_flag != 0:
      if centers_var == 'sem':
        number_of_languages_list = [6] #list(range(1,len(my_languages)+1))

    return number_of_languages_list

def set_language_list_permutations(center_A_flag, center_B_flag, centers_var):
  
  language_list_permutations = [None]

  if center_A_flag != 0 or center_B_flag != 0:
    if centers_var == 'sem':
       language_list_permutations = [0] #list(range(0,len(my_languages)))

  return language_list_permutations

def set_global_center(act, global_centering:int):
    
    if global_centering:
        global_center = jnp.mean(act,axis=0) # (T*E,)
        act = act - jnp.broadcast_to(global_center,act.shape) # (B,T*E)
    else:
        global_center = None

    return act, global_center