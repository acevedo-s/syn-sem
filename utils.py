import re,os
from pathlib import Path
import torch
import jax.numpy as jnp

from einops import rearrange

depths = {"deepseek":61,
          "llama":32}
emb_dims = {"deepseek":7168,
          "llama":4096}

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

