import torch

from utils.math import normalize_quat
from vehicle.base_vehicle import BaseVehicle

def euler_step(dynamics_fn: BaseVehicle.dynamics, X: torch.tensor, U: torch.tensor, t: torch.tensor, dt: float) -> torch.tensor:
    X_new =  X + dynamics_fn(X, U, t) * dt
    X_new[..., 3:7] = normalize_quat(X_new[..., 3:7]) # quat normalization
    return X_new

def rk4_step(dynamics_fn: BaseVehicle.dynamics, X: torch.tensor, U: torch.tensor, t: torch.tensor, dt: float) -> torch.tensor:
    k1 = dynamics_fn(X, U, t)
    k2 = dynamics_fn(X + dt / 2 * k1, U, t)
    k3 = dynamics_fn(X + dt / 2 * k2, U, t)
    k4 = dynamics_fn(X + dt * k3, U, t)
    
    X_new = X + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    X_new[..., 3:7] = normalize_quat(X_new[..., 3:7]) # quat normalization
    return X_new
    