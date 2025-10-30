import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.stats import rankdata
import numpy as np
from functools import partial
from jax import random

def hamming_distance(x, y):
    return jnp.count_nonzero(x != y)

def modified_L2_distance(x, y, significant_digits=2):
    u = (x-y)
    normalized_distance = jnp.sqrt(jnp.sum(u*u) / (jnp.sum(x*x) * jnp.sum(y*y)))
    scale = 10**significant_digits
    return (scale*normalized_distance).astype(jnp.int32)

def normalized_L2_distance(x, y):
    x /= jnp.linalg.norm(x)
    y /= jnp.linalg.norm(y)
    u = (x-y)
    return jnp.sqrt(jnp.sum(u*u))

def L2_distance(x, y):
    u = (x-y)
    return jnp.sqrt(jnp.sum(u*u))

def pairwise_similarities(similarity_fn, xs, ys):
  return jax.vmap(lambda x: jax.vmap(lambda y: similarity_fn(x, y))(xs))(ys).T

def compute_ranks(x_dist,y_dist,method):
    x_rank = rankdata(x_dist, method=method)
    y_rank = rankdata(y_dist, method=method)
    return x_rank,y_rank

def mapped_compute_ranks(method):
    return jax.vmap(partial(compute_ranks,method=method))

# def build_get_ranks(key, sample_size, similarity_fn, method):
#     """

#     Args:
#        key: the JAX random key.
#        X: sets from which we compute the information imbalance.
#        Y: sets to which we compute the information imbalance.
#        similarity_fn: function to compute the similarity in spaces X and Y.
#        method: "max" for correlation coefficient, "min" for II and neighborhood overlap. 
#     """
#     indices_rows,indices_columns = separate_samples(key,sample_size)
#     if method == 'max':
#         def _get_ranks(X, Y):
#             assert X.shape[0] == Y.shape[0], "Sample size must be equal across X and Y."
#             d_X = pairwise_similarities(similarity_fn, X[indices_rows], X[indices_columns])
#             d_Y = pairwise_similarities(similarity_fn, Y[indices_rows], Y[indices_columns])
#             R = mapped_compute_ranks(method)(d_X,d_Y)
#             L = mapped_compute_ranks(method)(-d_X,-d_Y)
#             return R,L
        
#     if method == 'min':
#         def _get_ranks(X, Y):
#             assert X.shape[0] == Y.shape[0], "Sample size must be equal across X and Y."
#             d_X = pairwise_similarities(similarity_fn, X[indices_rows], X[indices_columns])
#             d_Y = pairwise_similarities(similarity_fn, Y[indices_rows], Y[indices_columns])
#             R = mapped_compute_ranks(method)(d_X,d_Y)
#             return R
    
#     return jax.jit(_get_ranks)

def build_get_similarities(key, sample_size, similarity_fn):
    """

    Args:
       key: the JAX random key.
       X: sets from which we compute the information imbalance.
       Y: sets to which we compute the information imbalance.
       similarity_fn: function to compute the similarity in spaces X and Y.
    """
    indices_rows,indices_columns = separate_samples(key,sample_size)
    def _get_similarities(X, Y):
        assert X.shape[0] == Y.shape[0], "Sample size must be equal across X and Y."
        sim_X = pairwise_similarities(similarity_fn, X[indices_rows], X[indices_columns])
        sim_Y = pairwise_similarities(similarity_fn, Y[indices_rows], Y[indices_columns])
        return sim_X,sim_Y
        
    
    return jax.jit(_get_similarities)

def get_relative_ranks(x_rank, y_rank, k):
    """ k = 1 corresponds to first neighbours"""
    rel_ranks = jnp.where(x_rank <= k, y_rank - 1, -1)
    return (jnp.sum(rel_ranks, where=rel_ranks != -1), 
            jnp.sum(rel_ranks**2, where=rel_ranks != -1), # for std
            jnp.asarray(rel_ranks != -1).sum()
            )

def mapped_relative_ranks(k):
    return jax.vmap(partial(get_relative_ranks,k=k))

def separate_samples(key, sample_size):
    n_row_samples = int(sample_size // 2)
    n_col_samples = int(sample_size - n_row_samples)
    
    # n_row_samples = jnp.array(500) if sample_size >= 2500 else jnp.floor(0.2 * sample_size).astype(int)
    # n_col_samples = (sample_size - n_row_samples).astype(int)

    
    key, subkey = random.split(key)
    indices_rows = random.choice(key=subkey, a=sample_size, shape=(n_row_samples,), replace=False)

    remaining_indices = jnp.setdiff1d(jnp.arange(sample_size), indices_rows)
    key, subkey = random.split(key)
    indices_columns = random.choice(key=subkey, a=remaining_indices, shape=(n_col_samples,), replace=False)
    
    return indices_rows,indices_columns


def build_information_imbalance(k=1):

    def _normalizing(x,max_rank):

        return 2. * x / (max_rank - 2.) # -2 because ranks start at 0 and end at max_rank - 1

    def _compute(relative_ranks,relative_ranks2,num_neighbours,max_rank):

        mean = jnp.sum(relative_ranks) / jnp.sum(num_neighbours)
        std = jnp.sqrt((jnp.sum(relative_ranks2) / jnp.sum(num_neighbours)) - mean**2)

        return _normalizing(mean,max_rank), _normalizing(std,max_rank)
    
    def _information_imbalance(ranks_X,ranks_Y,k=k):

        assert ranks_X.shape == ranks_Y.shape
        max_rank = ranks_X.shape[1]

        relative_ranks, relative_ranks2, num_neighbours = mapped_relative_ranks(k)(ranks_X,ranks_Y)
        inf_imb, std = _compute(relative_ranks,relative_ranks2,num_neighbours,max_rank)
        del relative_ranks, relative_ranks2, num_neighbours

        reciprocal_ranks, reciprocal_ranks2, reciprocal_num_neighbours = mapped_relative_ranks(k)(ranks_Y,ranks_X)
        reciprocal_inf_imb,reciprocal_std = _compute(reciprocal_ranks,reciprocal_ranks2, reciprocal_num_neighbours, max_rank)

        return jnp.array([inf_imb,reciprocal_inf_imb]),jnp.array([std,reciprocal_std])
    return jax.jit(_information_imbalance)

### Platonic-functions

def _compute_neighbourhood_overlap(x_rank, y_rank,k):
    x_nns = jnp.where(x_rank <= k,1,0).astype(jnp.int8) # due to degeneracies there might be more than k points here
    y_nns = jnp.where(y_rank <= k,1,0).astype(jnp.int8)

    alignment = jnp.float32(jnp.bitwise_and(x_nns,y_nns).sum()) / ((x_nns.sum()+y_nns.sum())/2) # defined like this, it is always <= 1.

    rel_ranks_xy = jnp.where(x_nns.astype(jnp.bool), y_rank - 1, -1)
    rel_ranks_yx = jnp.where(y_nns.astype(jnp.bool), x_rank - 1, -1)

    return (alignment, 
            jnp.sum(rel_ranks_xy, where=rel_ranks_xy != -1), 
            jnp.asarray(rel_ranks_xy != -1).sum(),
            jnp.sum(rel_ranks_yx, where=rel_ranks_yx != -1), 
            jnp.asarray(rel_ranks_yx != -1).sum(),
            )

def mapped_compute_neighbourhood_overlaps(k):
    return jax.vmap(partial(_compute_neighbourhood_overlap,k=k))

def build_mutual_k_NN_alignment(k=10):
    """
    """
    
    def mutual_k_NN_alignment(ranks_X,ranks_Y,k=k):

        assert ranks_X.shape == ranks_Y.shape
        
        (alignment, 
        relative_ranks_XY,
        num_neighbors_X, 
        relative_ranks_YX, 
        num_neighbors_Y) = mapped_compute_neighbourhood_overlaps(k)(ranks_X,ranks_Y)

        alignment = alignment.mean()
        inf_imb_XY = 2. * (jnp.sum(relative_ranks_XY) / jnp.sum(num_neighbors_X)) / (ranks_X.shape[1] - 1.)
        inf_imb_YX = 2. * (jnp.sum(relative_ranks_YX) / jnp.sum(num_neighbors_Y)) / (ranks_X.shape[1] - 1.)

        return alignment, inf_imb_XY, inf_imb_YX
    
    return jax.jit(mutual_k_NN_alignment)