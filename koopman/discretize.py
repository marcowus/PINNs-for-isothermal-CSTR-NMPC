import numpy as np
from scipy.linalg import expm

def discretize_generator(K, L, dt):
    """
    Compute A = exp(K*dt) and B = integral(exp(K*tau)*L) dtau
    using block matrix exponential.
    """
    n_z = K.shape[0]
    n_u = L.shape[1]

    # Form block matrix
    # [K L]
    # [0 0]
    M = np.zeros((n_z + n_u, n_z + n_u))
    M[:n_z, :n_z] = K
    M[:n_z, n_z:] = L

    # Matrix exponential
    ExpM = expm(M * dt)

    # Extract A and B
    A = ExpM[:n_z, :n_z]
    B = ExpM[:n_z, n_z:]

    return A, B

def compute_error_bound(K, dt, max_residual):
    """
    Compute w_bar based on residual.
    |w_k| <= dt * exp(|K|*dt) * max_residual
    """
    norm_K = np.linalg.norm(K, 2)
    w_bar = dt * np.exp(norm_K * dt) * max_residual
    return w_bar
