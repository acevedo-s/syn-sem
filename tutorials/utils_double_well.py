import jax
import jax.numpy as jnp
import numpy as np


def double_well(x):
    """Asymmetric double-well curve."""
    return x ** 4 - 2.4 * x ** 2 + 0.7 * x + 0.1


def find_global_minimum_double_well():
    """Return global minimum G=[x*, y*] of double_well on R."""
    critical_points = np.roots([4.0, 0.0, -4.8, 0.7])
    critical_points = np.sort(critical_points[np.isclose(critical_points.imag, 0.0)].real)

    def second_derivative(x):
        return 12.0 * x ** 2 - 4.8

    minima_x = critical_points[second_derivative(critical_points) > 0.0]
    minima_y = np.array([double_well(x) for x in minima_x])
    global_min_idx = int(np.argmin(minima_y))

    x_global = minima_x[global_min_idx]
    y_global = minima_y[global_min_idx]
    return jnp.array([x_global, y_global])


def build_transform(theta_deg=35.0, translation=(1.2, -0.8)):
    """Return (M, t, T) where T(points)=points@M^T+t."""
    theta = np.deg2rad(theta_deg)
    R_reflect = jnp.array([[-1.0, 0.0], [0.0, 1.0]])
    R_rotate = jnp.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    M = R_rotate @ R_reflect
    t = jnp.array(translation)

    def T(points):
        return points @ M.T + t

    return M, t, T


def generate_anchor_data(
    master_key,
    Ns=500,
    x_min=-2.0,
    x_max=2.0,
    noise_variance=0.05,
    theta_deg=35.0,
    translation=(1.2, -0.8),
):
    """Generate anchors A and transformed noisy anchors A_prime."""
    key_x, key_eps = jax.random.split(master_key, 2)

    X = jax.random.uniform(key_x, (Ns, 1), minval=x_min, maxval=x_max)
    Y = double_well(X)
    A = jnp.concatenate([X, Y], axis=1)

    _, _, T = build_transform(theta_deg=theta_deg, translation=translation)
    A_prime_clean = T(A)

    noise_scale = np.sqrt(noise_variance)
    epsilon_prime = jax.random.normal(key_eps, (Ns, 1))
    A_prime = jnp.concatenate(
        [A_prime_clean[:, 0:1], A_prime_clean[:, 1:2] + noise_scale * epsilon_prime], axis=1
    )

    return A, A_prime
