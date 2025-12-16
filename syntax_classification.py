from utils_activations import *
from utils_syn_classifying import *
from utils import makefolder

n_files = 21
model_name = 'qwen7b'
precision = 32
data_var = 'syn'
global_center_flag = 1
min_token_length = 3
avg_tokens = 0
n_tokens = min_token_length 
lambda_l2 = 10.0

normalization_flag = 1
shuffled_control = 0

layers = list(range(1, depths[model_name] + 1))
layer_vals = reduce_list_half_preserve_extremes(layers)
layer_vals = reduce_list_half_preserve_extremes(layer_vals)  # double halving

syntax_labels = jnp.array(np.loadtxt(syn_group_ids_path).astype(int))  # (n_samples,)
syn_ids_with_sem = jnp.array(np.loadtxt(syn_ids_with_sem_path,dtype=int),dtype=jnp.int32) # filtering syn_data to have their semantic centroid in space A 
syntax_labels = syntax_labels[syn_ids_with_sem]


input_path_A = input_paths['english'][model_name]['matching']['0'][data_var]
all_activations_A = collect_data(
    input_path_A,
    min_token_length,
    n_files,
    model_name,
    avg_tokens,
)

input_path_B = input_path_A.replace('second', 'third').replace('0', '1')
all_activations_B = collect_data(
    input_path_B,
    min_token_length,
    n_files,
    model_name,
    avg_tokens,
)

# --------- PREALLOCATE STORAGE ARRAYS ---------
n_layers = len(layer_vals)

layer_indices   = np.empty(n_layers, dtype=np.int32)
accs_A    = np.empty(n_layers, dtype=np.float32)
accs_B     = np.empty(n_layers, dtype=np.float32)
syn_ablated_accs_A  = np.empty(n_layers, dtype=np.float32)
syn_ablated_accs_B  = np.empty(n_layers, dtype=np.float32)
sem_ablated_accs_A  = np.empty(n_layers, dtype=np.float32)

# --------- MAIN LOOP ---------
for i, layer in enumerate(layer_vals):
    act_A, syn_centroids_A, sem_centroids_A, global_center_A = preprocessing_syn_data(
        model_name=model_name,
        all_activations=all_activations_A,
        global_center_flag=global_center_flag,
        space_index='A',
        layer=layer,
        avg_tokens=avg_tokens,
        n_tokens=n_tokens,
        min_token_length=min_token_length,
        syn_ids_with_sem=syn_ids_with_sem,
    )
    act_B, syn_centroids_B, _, global_center_B = preprocessing_syn_data(
        model_name=model_name,
        all_activations=all_activations_B,
        global_center_flag=global_center_flag,
        space_index='B',
        layer=layer,
        avg_tokens=avg_tokens,
        n_tokens=n_tokens,
        min_token_length=min_token_length,
        syn_ids_with_sem=syn_ids_with_sem,
    )

    assert syntax_labels.shape[0] == act_A.shape[0]
    assert act_A.shape == act_B.shape

    # indices for batched_remove_centroid_projections
    idx = jnp.arange(syn_centroids_A.shape[0], dtype=jnp.int32)
    if shuffled_control:
        key_centers = jax.random.PRNGKey(11)
        idx = jax.random.permutation(key_centers,idx)

    ### Here I don't remove from the sentax vector its projection on the semantic vector because I did not do that in the training set, for which I don't have the semantic vectors. 
    syn_ablated_A = batched_remove_centroid_projections(act_A, idx, syn_centroids_A)
    syn_ablated_B = batched_remove_centroid_projections(act_B, idx, syn_centroids_B)
    sem_ablated_A = batched_remove_centroid_projections(act_A, idx, sem_centroids_A)

    # ---------- OPTIONAL NORMALIZATION (IN-PLACE LOGIC) ----------
    if normalization_flag:
        act_A         = l2_normalize(act_A)
        act_B         = l2_normalize(act_B)
        syn_ablated_A = l2_normalize(syn_ablated_A)
        syn_ablated_B = l2_normalize(syn_ablated_B)
        sem_ablated_A = l2_normalize(sem_ablated_A)

    W = fit_linear_classifier_closed_form(
        act_B, # B, since for it I don't have the semantic centroids
        syntax_labels,
        num_classes=None,
        l2_reg=lambda_l2,
        add_bias=True,
    )

    # ---------- STORE INTO PREALLOCATED ARRAYS ----------
    layer_indices[i]  = layer
    accs_A[i]    = float(accuracy(act_A, syntax_labels, W, add_bias=True))
    accs_B[i]    = float(accuracy(act_B, syntax_labels, W, add_bias=True))
    syn_ablated_accs_A[i] = float(accuracy(syn_ablated_A, syntax_labels, W, add_bias=True))
    syn_ablated_accs_B[i] = float(accuracy(syn_ablated_B, syntax_labels, W, add_bias=True))
    sem_ablated_accs_A[i] = float(accuracy(sem_ablated_A, syntax_labels, W, add_bias=True))
    print(f"layer={layer:3d}")


### Saving

resultsfolder = makefolder(base='./results/syntax_classification/',
                           create_folder=True,
                           model_name=model_name,
                           avg_tokens=avg_tokens,
                           normalization_flag=normalization_flag,
                           shuffled_control=shuffled_control
                           )

results_path  = os.path.join(resultsfolder, 'results.npz')

np.savez(
    results_path,
    layer_indices=layer_indices,
    accs_A=accs_A,
    accs_B=accs_B,
    syn_ablated_accs_A=syn_ablated_accs_A,
    syn_ablated_accs_B=syn_ablated_accs_B,
    sem_ablated_accs_A=sem_ablated_accs_A,
    lambda_l2=lambda_l2,
)

print(f"Saved results to {results_path}")
