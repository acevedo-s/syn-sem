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
    
    files = list_folder(input_path, desc="chunk_")[:n_files]
    all_hidden_states = defaultdict(list)

    for file in tqdm(files, desc="Collect File"):
        data = pickle.load(open(input_path + "/" + file.name, 'rb'))

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

    return all_hidden_states

