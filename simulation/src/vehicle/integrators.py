from utils.math import normalize_quat


def euler_step(dynamics_fn, X, U, dt):
    X_new =  X + dynamics_fn(X, U) * dt
    X_new[..., 3:7] = normalize_quat(X_new[..., 3:7]) # quat normalization
    return X_new

def rk4_step(dynamics_fn, X, U, dt):
    k1 = dynamics_fn(X, U)
    k2 = dynamics_fn(X + dt / 2 * k1, U)
    k3 = dynamics_fn(X + dt / 2 * k2, U)
    k4 = dynamics_fn(X + dt * k3, U)
    
    X_new = X + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    X_new[..., 3:7] = normalize_quat(X_new[..., 3:7]) # quat normalization
    return X_new
    