import jax.numpy as jnp


def projection_coefficients_lex(act, syn_centroids, sem_centroids, lex_centroids):
    """Compute norm fractions explained by lexical, syntactic, and semantic centroids."""

    def _squared_norm_fraction(act_, centroid):
        dot = jnp.sum(act_ * centroid, axis=1, keepdims=True)
        centroid_norm_sq = jnp.sum(centroid * centroid, axis=1, keepdims=True) + 1e-8
        proj = (dot / centroid_norm_sq) * centroid
        return jnp.sum(proj**2, axis=1) / (jnp.sum(act_**2, axis=1) + 1e-8)

    lex_frac = _squared_norm_fraction(act, lex_centroids)
    syn_frac = _squared_norm_fraction(act, syn_centroids) if syn_centroids is not None else 0.0
    sem_frac = _squared_norm_fraction(act, sem_centroids) if sem_centroids is not None else 0.0
    residual_frac = 1.0 - lex_frac - syn_frac - sem_frac

    return {
        "lex": lex_frac,
        "syn": syn_frac,
        "sem": sem_frac,
        "residual": residual_frac,
    }
