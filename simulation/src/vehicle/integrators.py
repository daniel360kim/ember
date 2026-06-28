from typing import Callable, Optional

import torch

DynamicsFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
ProjectFn = Callable[[torch.Tensor], torch.Tensor]


def euler_step(
    dynamics_fn: DynamicsFn,
    X: torch.Tensor,
    U: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    project: Optional[ProjectFn] = None,
) -> torch.Tensor:
    X_new = X + dynamics_fn(X, U, t) * dt
    return project(X_new) if project is not None else X_new


def rk4_step(
    dynamics_fn: DynamicsFn,
    X: torch.Tensor,
    U: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    project: Optional[ProjectFn] = None,
) -> torch.Tensor:
    k1 = dynamics_fn(X, U, t)
    k2 = dynamics_fn(X + dt / 2 * k1, U, t + dt / 2)
    k3 = dynamics_fn(X + dt / 2 * k2, U, t + dt / 2)
    k4 = dynamics_fn(X + dt * k3, U, t + dt)

    X_new = X + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return project(X_new) if project is not None else X_new
