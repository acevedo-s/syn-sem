import os,sys
os.environ["JAX_PLATFORMS"] = 'cpu'
import jax.numpy as jnp
import jax 
import numpy as np
from utils import (
                  syn_group_id_paths_for_sem_data,
                  sem_ids_with_syn_path,
                  syn_group_ids_path,
                  collect_data,
                  torch_to_jax,
                  flatten_tokens_features,
                  depths,
                  reduce_list_half_preserve_extremes,
                  load_sem_centroids,
                  get_syn_centroids_folder,
                  load_syn_group_averages,
                  get_syntax_expanded_counts,
                  batched_remove_centroid_projections,
                  reshuffle_batch_axis,
                   )
from geometry import (
                      build_get_similarities,
                      normalized_L2_distance,
                      build_information_imbalance,
                      mapped_compute_ranks,
                      )
from datapaths import * 
import matplotlib.pyplot as plt


rcpsize = 20
plt.rcParams['xtick.labelsize']= rcpsize
plt.rcParams['ytick.labelsize']=rcpsize
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = rcpsize
plt.rcParams.update({'figure.autolayout': True})
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['ggplot']['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['seaborn-v0_8']['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['seaborn-v0_8-dark-palette']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

markers = ['p','o','h','^','s','*']
_linestyles = ['-','--','dotted']
plot_id = 0
from time import time

@jax.jit
def add_tiny_noise(sim_X, key, eps=1e-6):
    # compute relative scale (std of sim_X)
    sigma = eps * jnp.std(sim_X)
    noise = jax.random.normal(key, shape=sim_X.shape) * sigma
    return sim_X + noise


def cosine_similarity(a, b, eps=1e-8):
    """Compute cosine similarity per row between two (N, D) matrices."""
    dot = jnp.sum(a * b, axis=1)
    norm_a = jnp.linalg.norm(a, axis=1)
    norm_b = jnp.linalg.norm(b, axis=1)
    return dot / (norm_a * norm_b + eps)

def preprocessing(all_activations, 
                  layer, 
                  space_index, 
                  global_center_flag, 
                  avg_tokens, 
                  n_tokens, 
                  syn_centroids_flag,
                  loo_flag,
                  precision=32, 
                  verbose=False,
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

  if syn_centroids_flag:
    sem_ids = jnp.array(np.loadtxt(sem_ids_with_syn_path,dtype=int),dtype=jnp.int32) # filtering data to have their syntax group in space A 
    act = act[sem_ids]

  sem_centroids = None
  syn_centroids = None
  sim_folder = f"/home/acevedo/syn-sem/results/global_centering_0/spaces_AB/similarity_fn_normalized_L2_distance/precision_32/language_english/data_var_syn/modelA_{model_name}/modelB_{model_name}/match_var_matching/n_files_21/min_token_length_{min_token_length}/similarities/centers_syn/Nbits_0/n_tokens_{n_tokens}/avg_tokens_{avg_tokens}/batch_shuffle_0/layer_A_{layer}/layer_B_{layer}/"

  # loading semantic_centroids
  sem_centroids = load_sem_centroids(sim_folder,number_of_languages=6,language_list_permutation=0).astype(act.dtype) #(num_sentences,E)

  if syn_centroids_flag:
    # keeping data with syn_centroids
    sem_centroids = sem_centroids[sem_ids]

    # loading syntax_centroids
    syn_centroids_folder = get_syn_centroids_folder(sim_folder)    
    (unique_syn_centroids, # (n_groups,E)
    syn_group_ids_for_sem, #(n_samples_sem_with_syn,)
    ) = load_syn_group_averages(act,
                                syn_group_id_paths_for_sem_data[space_index],
                                syn_centroids_folder,
                                None,
                                None,
                                space_index,
                                )
    syn_centroids = unique_syn_centroids[syn_group_ids_for_sem] #(n_samples_sem_with_syn,E)
    if loo_flag:
      ### I have to use the counting of the original syntax data to do LOO properly
      expanded_group_counts = get_syntax_expanded_counts(unique_syn_centroids,syn_group_ids_for_sem)
      syn_centroids = (expanded_group_counts[:,None] * syn_centroids - act) / (expanded_group_counts[:,None] - 1) # loo syn_centroids

  ### 
  # sem_centroids = reshuffle_batch_axis(sem_centroids, jax.random.PRNGKey(111))

  # global_centering syntax centroids...
  if global_center_flag: 
    if syn_centroids_flag:
      syn_centroids = syn_centroids - jnp.broadcast_to(global_center,syn_centroids.shape)
    sem_centroids = sem_centroids - jnp.broadcast_to(global_center,sem_centroids.shape)
  
  if verbose:
    print(f'{act.shape=}')
    if sem_centroids != None: print(f'{sem_centroids.shape=}')
    if syn_centroids != None: print(f'{syn_centroids.shape=}')
    if global_center!= None: print(f'{global_center.shape=}')

  return act, syn_centroids, sem_centroids, global_center

letter_to_index_dict = {
                      'A':'0',
                      'B':'1',
                      }

n_files = 21
model_name = 'qwen7b'
precision = 32
data_var = 'sem'
global_center_flag = 0
min_token_length = 3
n_tokens = min_token_length
space = sys.argv[1]
avg_tokens = int(sys.argv[2])
loo_flag = int(sys.argv[3])
input_path = input_paths['english'][model_name]['matching'][letter_to_index_dict[space]][data_var]
syn_centroids_flag = 1

all_activations = collect_data(input_path, 
                                min_token_length, 
                                n_files,
                                model_name,
                                avg_tokens,
                                )

layers = list(range(1, depths[model_name] + 1))
layer_vals = reduce_list_half_preserve_extremes(layers)

II_fn = build_information_imbalance(k=1)
key_distances = jax.random.PRNGKey(42)
key_distances, subkey_distances = jax.random.split(key_distances)

cos_means = []
cos_stds = []
inf_imb = []
sem_inf_imb = []
syn_inf_imb = []

start = time()

verbose = True
for enum_layer_id,layer in enumerate(layer_vals):
  act, syn_centroids, sem_centroids, global_center = preprocessing(
      all_activations, 
      layer, 
      space_index='A', # I only have syntax for A.
      global_center_flag=global_center_flag,
      avg_tokens=avg_tokens,
      n_tokens=n_tokens,
      syn_centroids_flag=syn_centroids_flag,
      loo_flag=loo_flag,
      verbose=verbose,
  )
  verbose = False
  sample_size = act.shape[0]
  if enum_layer_id == 0:
    get_similarities = build_get_similarities(key=subkey_distances, 
                                            sample_size=sample_size, 
                                            similarity_fn=normalized_L2_distance,
                                            )
  # syn_centroids = batched_remove_centroid_projections(syn_centroids,jnp.arange(sem_centroids.shape[0],dtype=jnp.int32),sem_centroids)

  if syn_centroids_flag and space == 'A':
    cos = np.array(cosine_similarity(act,syn_centroids))
    cos_means.append(cos.mean())
    cos_stds.append(cos.std())

    ### centroids_inf_imb
    sim_X, sim_Y = get_similarities(syn_centroids, sem_centroids)
    key = jax.random.PRNGKey(np.random.randint(0,1e6))

    sim_X = add_tiny_noise(sim_X, key)
    key = jax.random.PRNGKey(np.random.randint(0,1e6))
    sim_Y = add_tiny_noise(sim_Y, key)
    
    R_II = mapped_compute_ranks(method="min")(sim_X, sim_Y)
    _inf_imb, _inf_imb_std = II_fn(R_II[0], R_II[1])
    inf_imb.append(_inf_imb)

    # ###syn_inf_imb
    # sim_X, sim_Y = get_similarities(act, syn_centroids)
    # R_II = mapped_compute_ranks(method="min")(sim_X, sim_Y)
    # _inf_imb, _inf_imb_std = II_fn(R_II[0], R_II[1])
    # syn_inf_imb.append(_inf_imb)

  # ### sem_inf_imb
  # sim_X, sim_Y = get_similarities(act, sem_centroids)
  # R_II = mapped_compute_ranks(method="min")(sim_X, sim_Y)
  # _inf_imb, _inf_imb_std = II_fn(R_II[0], R_II[1])
  # sem_inf_imb.append(_inf_imb)

print(f'inf_imb took {(time()-start)/60.:.2f} m')


output_dir = f"results/centroids_correlations/loo_{loo_flag}/"
os.makedirs(output_dir, exist_ok=True)
print(f'{cos_means=}')
print(f'{cos_stds=}')
np.savetxt(os.path.join(output_dir,f'cos_similarities_{space}_Ns_{sample_size}_avg_{avg_tokens}_{model_name}.txt'),np.array([cos_means,cos_stds]).T)

if syn_centroids_flag:
  np.savetxt(os.path.join(output_dir,f'inf_imb_centroids_{space}_Ns_{sample_size}_avg_{avg_tokens}_{model_name}.txt'),np.array(inf_imb))
#   np.savetxt(os.path.join(output_dir,f'inf_imb_syn_Ns_{sample_size}_avg_{avg_tokens}_{model_name}.txt'),np.array(syn_inf_imb))
# np.savetxt(os.path.join(output_dir,f'inf_imb_sem_{space}_Ns_{sample_size}_avg_{avg_tokens}_{model_name}.txt'),np.array(sem_inf_imb))