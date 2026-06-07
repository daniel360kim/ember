import torch
from vehicle.integrators import *


def dynamics_1d(X, U):
    """1D vertical dynamics packed into a 13D state."""
    g = 9.81
    m = 0.5
    v_z = X[..., 9]
    a_z = -g - (U / m) * v_z * torch.abs(v_z)
    
    X_dot = torch.zeros_like(X)
    X_dot[..., 2] = v_z
    X_dot[..., 9] = a_z
    return X_dot
    
def simulate(drag_coeff, m=0.5, v_0 = 500.0, delta_t = 0.001, T = 200):
    v = torch.tensor(v_0)
    x = torch.zeros(1)
    g = 9.81
    N = int(T / delta_t)
    for n in range(N):
        a = -g - (drag_coeff / m) * v * torch.abs(v)
        v = v + a * delta_t
        x = x + v * delta_t
    return x



    


