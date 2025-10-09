import sys,os
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np 

@jax.jit
def rankdata_1D(x):
    """
    Compute ranks of elements in `x` (1-based) in a fully JIT-compatible way.
    Equivalent to scipy.stats.rankdata(x, method='ordinal').

    Parameters
    ----------
    x : array_like, shape (..., N)
        Input array to rank along the last axis.

    Returns
    -------
    ranks : array_like, same shape as `x`
        1-based ranks along the last axis.
    """
    # Sort and invert permutation
    sort_idx = jnp.argsort(x, axis=-1, stable=True)
    ranks0 = jnp.argsort(sort_idx, axis=-1, stable=True)
    return ranks0 + 1

def build_rankdata_2D():
    """
    Returns a JIT-compiled batched version of rankdata_1D
    that accepts arrays of shape (batch, n) and computes
    ranks (1-based) independently for each row.
    """
    rankdata_2D = jax.jit(jax.vmap(rankdata_1D, in_axes=0, out_axes=0))
    return rankdata_2D


# 2D batched version
rankdata_2D = jax.jit(jax.vmap(rankdata_1D, in_axes=0, out_axes=0))

@jax.jit
def rankdata_1D_ties(x, key, eps_scale=1e-10):
    """
    Compute 1D ranks (1-based) with random tie-breaking.
    Equivalent to scipy.stats.rankdata(x, method='ordinal'),
    but adds a tiny random jitter to break ties uniformly at random.

    Args:
        x : array_like, shape (n,)
            Input array to rank.
        key : jax.random.PRNGKey
            Random key for tie-breaking noise.
        eps_scale : float
            Magnitude of random jitter added for tie-breaking (default 1e-10).

    Returns:
        ranks : array_like, shape (n,)
            1-based ranks.
    """
    eps = eps_scale * jax.random.normal(key, shape=x.shape, dtype=jnp.float64)
    x_noisy = x + eps
    sort_idx = jnp.argsort(x_noisy, stable=True)
    ranks0 = jnp.argsort(sort_idx, stable=True)
    return ranks0 + 1


# Batched (2D) version using vmap
def build_rankdata_2D_ties(eps_scale=1e-10):
    """
    Returns a batched version of rankdata_1D_ties that accepts (batch, n)
    and a base PRNG key, generating one subkey per batch.
    """
    def rankdata_2D_ties(x, key):
        # Split key into subkeys per batch
        batch_size = x.shape[0]
        subkeys = jax.random.split(key, batch_size)
        return jax.vmap(lambda xi, ki: rankdata_1D_ties(xi, ki, eps_scale))(x, subkeys)
    return jax.jit(rankdata_2D_ties)


@jax.jit
def relative_ranks_1D(rX, rY):
    """
    Compute relative ranks for a single sample (1D arrays)

    Args:
        rX, rY: shape (n,)
    Returns:
        relX, relY: shape (n,)
    """
    order_X = jnp.argsort(rX, stable=True)
    relY = rY[order_X]

    order_Y = jnp.argsort(rY, stable=True)
    relX = rX[order_Y]

    return relX, relY


relative_ranks_2D = jax.jit(jax.vmap(relative_ranks_1D, in_axes=(0, 0), out_axes=(0, 0)))


@jax.jit
def get_xis_1D(relative_ranks_1D):
    """
    Compute xi for a single sample (1D array of relative ranks)
    
    Args:
        relative_ranks_1D: shape (n,)
    Returns:
        xi: scalar
    """
    n = jnp.double(relative_ranks_1D.shape[0])
    S = (jnp.abs(jnp.diff(relative_ranks_1D)).sum()).astype(jnp.double)
    xi = jnp.double(1.0) - (jnp.double(3.0) * S) / (n**2 - jnp.double(1.0))
    return xi


get_xis_2D = jax.vmap(get_xis_1D, in_axes=0, out_axes=0)


def build_corr_coeff_1D():
    """
    Returns a function that computes xi correlation between two 1D ranked arrays (rX, rY).
    """
    def _corr_coef(R):
        rX, rY = R
        # Compute relative ranks
        relX, relY = relative_ranks_1D(rX, rY)
        # Compute xi in both directions
        xi_xy = get_xis_1D(relY)  # xi(X -> Y)
        xi_yx = get_xis_1D(relX)  # xi(Y -> X)
        return xi_xy, xi_yx

    return jax.jit(_corr_coef)

build_corr_coeff_2D = lambda average=True: jax.vmap(build_corr_coeff_1D(), in_axes=0, out_axes=0)

@jax.jit
def get_xis_1D_ties(rel_ranks_Y):
    """
    Compute tie-corrected xi for a single sample (1D array of relative ranks),
    following Equation (1.1) in Chatterjee (2019).

    Args:
        rel_ranks_Y : array_like, shape (n,)
            Relative ranks of Y with respect to X.

    Returns:
        xi : scalar
            Tie-corrected Chatterjee's xi.
    """
    n = rel_ranks_Y.shape[0]

    # Sum of absolute differences of consecutive ranks
    S = jnp.abs(jnp.diff(rel_ranks_Y)).sum()

    # Compute l_i using negation trick
    l = rankdata_1D(-rel_ranks_Y)

    # Denominator: sum_i l_i * (n - l_i)
    denominator = jnp.sum(l * (n - l))

    # Final tie-corrected xi with n/2 prefactor
    xi = 1.0 - (n / 2) * S / denominator
    return xi


get_xis_2D_ties = jax.jit(jax.vmap(get_xis_1D_ties, in_axes=0, out_axes=0))


def build_corr_coeff_2D_ties(average=True):
    """
    Returns a JIT-compiled function that computes Chatterjee's xi for 2D batched inputs,
    using the pre-defined get_xis_2D_ties.

    Input:
        R = (rX_batch, rY_batch), each of shape (batch, n)
    Output:
        if average=True:
            (mean_XY, mean_YX), (std_XY, std_YX)
        else:
            xi_XY, xi_YX of shape (batch,)
    """

    @jax.jit
    def corr_fn(R):
        rX_batch, rY_batch = R

        # Compute relative ranks batchwise
        relX_batch, relY_batch = relative_ranks_2D(rX_batch, rY_batch)

        # Compute xi’s using tie-corrected batched function
        xi_XY = get_xis_2D_ties(relY_batch)
        xi_YX = get_xis_2D_ties(relX_batch)

        if average:
            mean = jnp.array([xi_XY.mean(), xi_YX.mean()])
            std  = jnp.array([xi_XY.std(), xi_YX.std()])
            return mean, std
        else:
            return xi_XY, xi_YX

    return corr_fn



####### "2D functions" for a batch of vector data (like the distance matrix)

# @jax.jit
# def relative_ranks_2D(rX, rY):
#     """
#     JIT-compatible computation for batched input of shape (batch, n)
#     """
#     order_X = jnp.argsort(rX, axis=1, stable=True)
#     order_Y = jnp.argsort(rY, axis=1, stable=True)
#     relY = jnp.take_along_axis(rY, order_X, axis=1)
#     relX = jnp.take_along_axis(rX, order_Y, axis=1)
#     return relX, relY


# @jax.jit
# def get_xis_2D(relative_ranks_2D):
#     """
#     Compute xi for batched relative ranks of shape (batch, n)
#     Returns vector of xi values of shape (batch,)
#     """
#     n = relative_ranks_2D.shape[1]
#     S = jnp.abs(jnp.diff(relative_ranks_2D, axis=1)).sum(axis=1)
#     xi = 1.0 - (3.0 * S) / (n**2 - 1.0)
#     return xi

# def build_corr_coeff_2D(average=True):
#     """
#     Returns a JIT-compiled function that computes xi correlation for batched inputs.

#     Args:
#         average (bool): if True, returns (mean_xi, std_xi) across the batch;
#                         if False, returns xi per sample.

#     Returns:
#         function R -> (xis_XY, xis_YX) or ((mean_XY, mean_YX), (std_XY, std_YX))
#         R = (rX_batch, rY_batch) with shape (batch, n)
#     """
#     def _corr_coef(R):
#         rX_batch, rY_batch = R

#         # Compute relative ranks per sample
#         relX_batch, relY_batch = relative_ranks_2D(rX_batch, rY_batch)

#         # Compute xi per sample
#         xis_XY = get_xis_2D(relY_batch)  # xi(X -> Y)
#         xis_YX = get_xis_2D(relX_batch)  # xi(Y -> X)

#         if average:
#             mean = jnp.array([xis_XY.mean(), xis_YX.mean()])
#             std  = jnp.array([xis_XY.std(), xis_YX.std()])
#             return mean, std
#         else:
#             return xis_XY, xis_YX

#     return jax.jit(_corr_coef)




# @jax.jit
# def get_xis_2D_ties(rel_ranks_Y):
#     """
#     Compute tie-corrected xi for batched 2D relative ranks.
#     Equation below 1.1 in Chatterjee 2019.
    
#     Parameters:
#         rel_ranks_Y: (batch, n)
#     Returns:
#         xi: (batch,)
#     """
#     n = rel_ranks_Y.shape[1]

#     # Sum of absolute differences of consecutive ranks
#     S = jnp.abs(jnp.diff(rel_ranks_Y, axis=1)).sum(axis=1)

#     # Compute l_i using negation trick
#     l = rankdata_2D(-rel_ranks_Y)

#     # Denominator: sum_i l_i * (n - l_i)
#     denominator = jnp.sum(l * (n - l), axis=1)

#     # Final tie-corrected xi with n/2 prefactor
#     xi = 1.0 - (n / 2) * S / denominator
#     return xi


# def build_corr_coeff_2D_ties(average=True):
#     """
#     Returns a JIT-compiled function that computes Chatterjee's xi for 2D batched inputs,
#     fully tie-corrected, with optional averaging across the batch.

#     Input: R = (rX_batch, rY_batch), each of shape (batch, n)
#     Output:
#         if average=True: (mean_XY, mean_YX), (std_XY, std_YX)
#         else: xi_XY, xi_YX of shape (batch,)
#     """
#     @jax.jit
#     def corr_fn(R):
#         rX_batch, rY_batch = R
#         # Compute relative ranks
#         order_X = jnp.argsort(rX_batch, axis=1, stable=True)
#         order_Y = jnp.argsort(rY_batch, axis=1, stable=True)
#         relY = jnp.take_along_axis(rY_batch, order_X, axis=1)
#         relX = jnp.take_along_axis(rX_batch, order_Y, axis=1)
#         # Compute tie-corrected xis
#         xi_XY = get_xis_2D_ties(relY)
#         xi_YX = get_xis_2D_ties(relX)

#         if average:
#             mean = jnp.array([xi_XY.mean(), xi_YX.mean()])
#             std  = jnp.array([xi_XY.std(), xi_YX.std()])
#             return mean, std
#         else:
#             return xi_XY, xi_YX

#     return corr_fn


