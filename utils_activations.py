import os
os.environ["JAX_PLATFORMS"] = 'cpu'
import jax.numpy as jnp
import jax 
from functools import partial
from tqdm import tqdm
import numpy as np
from torch import from_numpy
from geometry import normalized_L2_distance
from datapaths import * 
import matplotlib.pyplot as plt
from utils import (
                  syn_group_id_paths_for_sem_data,
                  sem_centers_ids_path,
                  sem_ids_with_syn_path,
                  syn_ids_with_sem_path,
                  syn_group_ids_path,
                  collect_data,
                  torch_to_jax,
                  flatten_tokens_features,
                  depths,
                  reduce_list_half_preserve_extremes,
                  load_and_subtract_syn_group_averages,
                  load_sem_centroids,
                  get_syn_centroids_folder,
                  load_syn_group_averages,
                  batched_remove_centroid_projections,
                  get_syntax_expanded_counts,
                  load_and_subtract_sem_group_averages
                  )


def preprocessing_sem_data(
                  model_name,
                  all_activations, 
                  layer, 
                  space_index, 
                  global_center_flag, 
                  min_token_length,
                  avg_tokens, 
                  n_tokens, 
                  precision=32, 
                  verbose=False,
                  centroids=True,
                  ):

  # loading data
  _all_activations = all_activations[f"layer_{layer}"]
  if avg_tokens == 0:
    _all_activations = flatten_tokens_features(_all_activations)
  
  act = torch_to_jax(_all_activations,precision)

  # globally centering data [with all samples]
  if global_center_flag:
      global_center = jnp.mean(act,axis=0)
      act = act - jnp.broadcast_to(global_center,act.shape)
  else:
      global_center = None

  sem_ids = jnp.array(np.loadtxt(sem_ids_with_syn_path,dtype=int),dtype=jnp.int32) # filtering data to have their syntax group in space A 
  act = act[sem_ids]

  if centroids:
    sim_folder = f"/home/acevedo/syn-sem/results/global_centering_0/spaces_AB/similarity_fn_normalized_L2_distance/precision_32/language_english/data_var_syn/modelA_{model_name}/modelB_{model_name}/match_var_matching/n_files_21/min_token_length_{min_token_length}/similarities/centers_syn/Nbits_0/n_tokens_{n_tokens}/avg_tokens_{avg_tokens}/batch_shuffle_0/layer_A_{layer}/layer_B_{layer}/"

    # loading semantic_centroids
    sem_centroids = load_sem_centroids(sim_folder,number_of_languages=6,language_list_permutation=0).astype(act.dtype) #(num_sentences,E)

    # keeping data with syn_centroids
    sem_centroids = sem_centroids[sem_ids]

    # loading syntax_centroids
    syn_centroids_folder = get_syn_centroids_folder(sim_folder)
    # syn_centroids = jnp.array(np.load(os.path.join(sim_folder, f'syn_centers_{space_index}.npy'))).astype(jnp.float32) #(num_groups,E)
    
    (unique_syn_centroids, # (n_groups,E)
    syn_group_ids_for_sem, #(n_samples_sem_with_syn,)
    ) = load_syn_group_averages(act,
                                syn_group_id_paths_for_sem_data[space_index],
                                syn_centroids_folder,
                                None,
                                None,
                                space_index,
                                )

    syn_centroids = unique_syn_centroids[syn_group_ids_for_sem] #(n_samples_sem_with_syn,)
    expanded_group_counts = get_syntax_expanded_counts(unique_syn_centroids,syn_group_ids_for_sem)
    syn_centroids = (expanded_group_counts[:,None] * syn_centroids - act) / (expanded_group_counts[:,None] - 1) # loo syn_centers

    # global_centering syntax centroids...
    if global_center_flag: 
        syn_centroids = syn_centroids - jnp.broadcast_to(global_center,syn_centroids.shape)
        sem_centroids = sem_centroids - jnp.broadcast_to(global_center,sem_centroids.shape)
    
    if verbose:
        print(f'{act.shape=}')
        print(f'{sem_centroids.shape=}')
        print(f'{syn_centroids.shape=}')
        print(f'{global_center.shape=}')
  else:
    syn_centroids = None
    sem_centroids = None
  return act, syn_centroids, sem_centroids, global_center

def preprocessing_syn_data(
                           model_name,
                           all_activations,
                           global_center_flag,
                           space_index,
                           layer,
                           avg_tokens,
                           n_tokens,
                           min_token_length,
                           syn_ids_with_sem,
                           precision=32,
                           ):

  _all_activations = all_activations[f"layer_{layer}"]
  if avg_tokens == 0:
    _all_activations = flatten_tokens_features(_all_activations)
  act = torch_to_jax(_all_activations,precision)

  ### Syntax centroids
  sim_folder = f"/home/acevedo/syn-sem/results/global_centering_0/spaces_AB/similarity_fn_normalized_L2_distance/precision_32/language_english/data_var_syn/modelA_{model_name}/modelB_{model_name}/match_var_matching/n_files_21/min_token_length_{min_token_length}/similarities/centers_syn/Nbits_0/n_tokens_{n_tokens}/avg_tokens_{avg_tokens}/batch_shuffle_0/layer_A_{layer}/layer_B_{layer}/"
  (unique_syn_centroids, #(n_groups,E)
   syn_group_ids, #(n_syn_samples,)
   ) = load_syn_group_averages(act,
                              syn_group_ids_path,
                              sim_folder,
                              None,
                              None,
                              space_index,
                              )
  expanded_syn_centroids = unique_syn_centroids[syn_group_ids] # (n_syn_samples,E)

  ### Syn data with semantic centroids
  act = act[syn_ids_with_sem]
  expanded_syn_centroids = expanded_syn_centroids[syn_ids_with_sem]

  # globally centering data [with all samples]
  if global_center_flag:
      global_center = jnp.mean(act,axis=0)
      act = act - jnp.broadcast_to(global_center,act.shape)
      expanded_syn_centroids = expanded_syn_centroids - jnp.broadcast_to(global_center,expanded_syn_centroids.shape)
  else:
      global_center = None

  if space_index == 'A':
    sem_centroids = load_sem_centroids(sim_folder,number_of_languages=6,language_list_permutation=0).astype(act.dtype) #(num_sem_sentences,E)
    sem_center_ids = jnp.array(np.loadtxt(sem_centers_ids_path,dtype=int),dtype=jnp.int32)
    sem_centroids = sem_centroids[sem_center_ids] # this alignes centers to syntax data

  else:
    sem_centroids = None

  if layer == 1:
    print(f'{act.shape=}')
    print(f'{expanded_syn_centroids.shape=}')
    if sem_centroids is not None:
      print(f'{sem_centroids.shape=}')

  return act, expanded_syn_centroids, sem_centroids, global_center

@jax.jit
def cosine_similarity(act_A, act_B, eps=1e-8):
    """
    Compute row-wise cosine similarity between two activation matrices.

    Args:
        act_A (jnp.ndarray): shape (N, D), activations from space A
        act_B (jnp.ndarray): shape (N, D), activations from space B
        eps (float): small constant to avoid division by zero

    Returns:
        jnp.ndarray: shape (N,), cosine similarities for each row pair
    """
    numerator = jnp.sum(act_A * act_B, axis=1)
    denominator = jnp.linalg.norm(act_A, axis=1) * jnp.linalg.norm(act_B, axis=1)
    return numerator / (denominator + eps)

@jax.jit
def all_cosine_similarities(act_A, act_B, eps=1e-8):
    """
    Compute all-pairs cosine similarities between the rows of act_A and act_B.

    Args:
        act_A: (N, D)
        act_B: (M, D)
        eps: small constant to avoid division by zero

    Returns:
        (N, M) array where entry (i, j) is cos_sim(act_A[i], act_B[j])
    """
    # Dot products between all row pairs: (N, D) @ (D, M) -> (N, M)
    numerator = act_A @ act_B.T

    # Row-wise norms
    norm_A = jnp.linalg.norm(act_A, axis=1, keepdims=True)  # (N, 1)
    norm_B = jnp.linalg.norm(act_B, axis=1, keepdims=True)  # (M, 1)

    # Broadcast to (N, M)
    denominator = norm_A * norm_B.T

    return numerator / (denominator + eps)

@partial(jax.jit, static_argnums=(1,))
def recall_at_k_jax(cos_matrix, k):
    N, M = cos_matrix.shape
    
    # top_k works along the last axis; returns values, indices
    _, topk_idx = jax.lax.top_k(cos_matrix, k)  # (N, k)

    targets = jnp.arange(N)[:, None]  # (N, 1)
    hits = (topk_idx == targets)
    return hits.any(axis=1).mean()

def squared_norm_fraction(act, centroid, eps=1e-8):
    # elementwise dot
    dot = jnp.sum(act * centroid, axis=1, keepdims=True)
    centroid_norm_sq = jnp.sum(centroid * centroid, axis=1, keepdims=True) + eps
    proj = (dot / centroid_norm_sq) * centroid
    frac = jnp.sum(proj**2, axis=1) / (jnp.sum(act**2, axis=1) + eps)
    return frac  # shape: (batch,)

