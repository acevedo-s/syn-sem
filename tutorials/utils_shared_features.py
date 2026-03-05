import sys,os
sys.path.append('../')
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

def correlated_gaussian_batch(key, Ns, N, alpha):
    """
    Generates x and y, where x is a subset of y's dimensions. 
    This simulates y = x ++ noise, where ++ means adding independent features (either noise or not)
    """
    key, key_y = jax.random.split(key)
    
    # Independent Gaussian samples
    y = jax.random.normal(key_y, (Ns, N))
    
    # Create mask: True where index/N < alpha
    rel_depth = jnp.linspace(0, 1, N, endpoint=False)
    mask = (rel_depth < alpha).astype(bool)
    n_shared = jnp.sum(mask)
    
    x = jnp.array(y[:,:n_shared])
    
    return x, y
