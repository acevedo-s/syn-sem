import re,os
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

depths = {"deepseek":61,
          "llama":32}
emb_dims = {"deepseek":7168,
          "llama":4096}
batch_sizes = {'deepseek':10,
               'llama':100}

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

def bf16_torch_to_jax(tensor):
    import jax.numpy as jnp
    intermediate_tensor = tensor.view(torch.uint16)
    tensor = jnp.array(intermediate_tensor).view('bfloat16')
    return tensor

def flatten_tokens_features(tensor):
    tensor = rearrange(tensor, 'batch tokens embed -> batch (tokens embed)')
    return tensor

def binarize(a):
    a = jnp.sign(a).astype(jnp.int8)
    a = a.at[a == 0].set(-1)
    return a

def get_quantiles(a,alphamin,alphamax):
  qmin = jnp.quantile(a,q=alphamin,axis=1)
  qmax = jnp.quantile(a,q=alphamax,axis=1)
  return qmin,qmax

def clip(act,
         alphamin = 0.05,
         alphamax = 0.95):
    if len(act.shape)==3:
      reshape = True
    else:
      reshape = False

    if reshape:
      B,T,E = act.shape
      act = jnp.reshape(act,shape=(B,T*E))
      
    qmin,qmax = get_quantiles(act,alphamin,alphamax)
    act = jnp.clip(act.T,min=qmin,max=qmax).T

    if reshape:
      act = jnp.reshape(act,shape=(B,T,E))
    return act

def makefolder(base='./',
               create_folder=False,
               precision=5,
               **kwargs,
               ):
  folder = base
  for key, value in kwargs.items():
    if value != None:
      if isinstance(value,float) == True:
        folder += key + f'_{value:.{precision}f}/'
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
                 filter_layer=None, 
                 min_token_length=5, 
                 n_files=10,
                 ):
    start_time = time()
    files = list_folder(input_path, desc="chunk_")[:n_files]
    all_hidden_states = defaultdict(list)

    for file in tqdm(files, desc="Collect File"):
        with open(os.path.join(input_path, file.name), 'rb') as f:
            data = pickle.load(f)
        for _, sentence in enumerate(data):
            hidden_states = sentence['meta_info']['hidden_states'][0]
            assert hidden_states.shape[1] >= min_token_length
            hidden_states = hidden_states[:, -min_token_length:]

            for layer_idx, layer_tensor in enumerate(hidden_states.split(1, dim=0)):
                if filter_layer is not None:
                    if layer_idx!=filter_layer: continue

                layer_tensor = layer_tensor.squeeze(0)
                all_hidden_states[f"layer_{layer_idx}"].append(layer_tensor)

    for layer, tensors in all_hidden_states.items():
        all_hidden_states[layer] = torch.stack(tensors)
        # print("Layer=", layer, "activations shape= ", all_hidden_states[layer].shape, flush=True)
    print(f'importing took {(time()-start_time)/60.} m')

    return all_hidden_states

# def substract_group_averages(input_path,data,random_centers):

#   assert len(data.shape) == 2

#   if random_centers:
#     key_center = jax.random.PRNGKey(422)
#     avg_norm = jnp.linalg.norm(data,axis=1).mean()

#   all_group_ids = jnp.array(np.loadtxt(input_path + 'group_ids.txt').astype(int))[:data.shape[0]]
#   unique_groups_indices,counts = jnp.unique(all_group_ids,return_counts=True)

#   centers = jnp.zeros(shape=(len(unique_groups_indices), 
#                              data.shape[1]), 
#                       dtype=data.dtype)

#   for group_index,group_id in enumerate(unique_groups_indices):
#     mask_indices = jnp.where(all_group_ids==group_id)[0]
#     if random_centers == False:
#       centers = centers.at[group_index].set(jnp.sum(data[mask_indices],axis=0) / counts[group_index])
#     else:
#       key_center, subkey = jax.random.split(key_center)
#       vec = jax.random.normal(subkey, shape=(data.shape[1],))
#       vec *= avg_norm / jnp.linalg.norm(vec)
#       centers = centers.at[group_index].set(vec)
#     data = data.at[mask_indices].add(-centers[group_index])
  
#   return data, centers

def compute_and_subtract_group_averages(input_path,data,random_centers,output_folder,space_index):
  print(f'subtracting group averages, random_centers={random_centers}')

  assert len(data.shape) == 2
  assert space_index == 'A' or space_index == 'B'

  if random_centers:
    key_center = jax.random.PRNGKey(422)

  all_group_ids = jnp.array(np.loadtxt(input_path + 'group_ids.txt').astype(int))
  # assert len(all_group_ids) == data.shape[0]
  unique_groups_indices,counts = jnp.unique(all_group_ids,return_counts=True)

  centers = jnp.zeros(shape=(len(unique_groups_indices), 
                             data.shape[1]), 
                             dtype=data.dtype)


  for group_index,group_id in enumerate(unique_groups_indices):
    assert group_index == group_id

    if random_centers == False:
      indices = jnp.where(all_group_ids==group_id)[0]
      centers = centers.at[group_index].set(jnp.sum(data[indices],axis=0) / counts[group_index])
    else:
      key_center, subkey = jax.random.split(key_center)
      N_average = data.shape[0] // len(unique_groups_indices)
      indices = jax.random.choice(subkey, data.shape[0], 
                                      shape=(N_average,), 
                                      replace=False)
      centers = centers.at[group_index].set(jnp.sum(data[indices],axis=0) / N_average)
    data = data.at[indices].add(-centers[group_index])

  np.save(os.path.join(output_folder,f"centers_{space_index}"),centers)
  return data

def load_and_subtract_group_averages(act_A,
                                     act_B,
                                     sim_folder,
                                     group_ids_path,
                                     random_centers,
                                     ):
  original_labels = np.loadtxt("/home/acevedo/syn-sem/datasets/txt/sem/second/matching/original_labels.txt").astype(int)
  group_ids = jnp.array(np.loadtxt(group_ids_path + 'group_ids.txt').astype(int))
  unique_groups_indices = jnp.unique(group_ids)

  if random_centers == 0:
    # I have to pick the <syntax> centers
    centers_folder = sim_folder.replace("txt_var_sem", "txt_var_syn")
    centers_folder = centers_folder.replace("n_files_16", "n_files_21")
    centers = jnp.mean(jnp.stack([jnp.array(np.load(centers_folder+f'centers_A.npy')),
                                jnp.array(np.load(centers_folder+f'centers_B.npy'))]),
                      axis=0)
    
  if random_centers == 1:
    key_center = jax.random.PRNGKey(422)
    # centers = reshuffle_batch_axis(centers,subkey)
    centers = jnp.zeros(shape=(len(unique_groups_indices), 
                              act_A.shape[1]), 
                              dtype=act_A.dtype)
    for group_index,group_id in enumerate(unique_groups_indices):
      assert group_index == group_id
      key_center, subkey = jax.random.split(key_center)
      N_average = 2*act_A.shape[0] // len(unique_groups_indices)
      random_indices = jax.random.choice(subkey, 2*act_A.shape[0], 
                                      shape=(N_average,), 
                                      replace=False)
      centers = centers.at[group_index].set(jnp.sum(jnp.concatenate([act_A,act_B],axis=0)[random_indices],axis=0) / N_average)

  indices_A = jnp.where(original_labels == 0)[0]
  indices_B = jnp.where(original_labels == 1)[0]
  act_A = act_A.at[indices_A].set(act_A[indices_A]-centers[group_ids[indices_A]])
  act_B = act_B.at[indices_B].set(act_B[indices_B]-centers[group_ids[indices_B]])
  return act_A,act_B 
