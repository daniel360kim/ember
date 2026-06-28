"""Tests for the euler/rk4 integrators.

These are decoupled from any specific vehicle: they use trivial analytic
dynamics so they check only the stepper itself -- accuracy, the optional
projection hook, and (critically for a diffsim) that gradients flow through
a step without an in-place write zeroing them out.

The integrator no longer knows the state layout: manifold projection (e.g.
quaternion renormalization) is supplied as a `project` callable, mirroring
BaseVehicle.project.
"""

import math
import torch

from vehicle.integrators import euler_step, rk4_step
from vehicle.state import S
from utils.math import quat_identity, normalize_quat

DT = 1e-2


def _project(X):
    # Same contract as BaseVehicle.project: renormalize the quaternion, out-of-place.
    q = normalize_quat(X[..., S.ORI])
    return torch.cat([X[..., :S.ORI.start], q, X[..., S.ORI.stop:]], dim=-1)


def _upright(extra=None):
    """A 15D state with an identity quaternion so renormalization is a no-op."""
    X = torch.zeros(15)
    X[S.ORI] = quat_identity()
    if extra is not None:
        extra(X)
    return X


# --- constant-rate dynamics integrate exactly ---

def _constant_climb(X, U, t):
    # constant vertical velocity packed into the position-z derivative
    xd = torch.zeros_like(X)
    xd[..., 2] = 5.0
    return xd

def test_euler_constant_rate_exact():
    X1 = euler_step(_constant_climb, _upright(), None, torch.tensor(0.0), DT)
    torch.testing.assert_close(X1[2], torch.tensor(5.0 * DT))

def test_rk4_equals_euler_for_state_independent_dynamics():
    # When X_dot does not depend on X, all RK4 stages coincide -> same as Euler.
    Xe = euler_step(_constant_climb, _upright(), None, torch.tensor(0.0), DT)
    Xr = rk4_step(_constant_climb, _upright(), None, torch.tensor(0.0), DT)
    torch.testing.assert_close(Xe, Xr)


# --- quaternion stays on the unit manifold ---

def _spin(X, U, t):
    # drive a quaternion derivative so the step pushes it off the manifold
    xd = torch.zeros_like(X)
    xd[S.ORI] = torch.tensor([0.3, 0.0, 0.0, 0.0])
    return xd

def test_project_renormalizes_quat_euler():
    X1 = euler_step(_spin, _upright(), None, torch.tensor(0.0), DT, project=_project)
    torch.testing.assert_close(X1[S.ORI].norm(), torch.tensor(1.0), atol=1e-5, rtol=0)

def test_without_project_quat_drifts_off_manifold():
    # No projection hook -> the integrator is pure arithmetic and the quaternion
    # is free to leave the unit manifold.
    X1 = euler_step(_spin, _upright(), None, torch.tensor(0.0), DT)
    assert X1[S.ORI].norm().item() > 1.0

def test_quat_renormalized_rk4():
    X1 = rk4_step(_spin, _upright(), None, torch.tensor(0.0), DT, project=_project)
    torch.testing.assert_close(X1[S.ORI].norm(), torch.tensor(1.0), atol=1e-5, rtol=0)


# --- rk4 is more accurate than euler on a stiff-ish decay ---

def _decay(X, U, t):
    # dx/dt = -x on the position-x channel; analytic solution is x0 * e^{-t}
    xd = torch.zeros_like(X)
    xd[..., 0] = -X[..., 0]
    return xd

def test_rk4_beats_euler_on_decay():
    T, N = 1.0, 100
    dt = T / N

    def rollout(step):
        X = _upright(lambda X: X.__setitem__(0, 1.0))
        for _ in range(N):
            X = step(_decay, X, None, torch.tensor(0.0), dt)
        return X[0].item()

    exact = math.exp(-T)
    err_euler = abs(rollout(euler_step) - exact)
    err_rk4 = abs(rollout(rk4_step) - exact)
    assert err_rk4 < err_euler


# --- gradients flow through a step (the diffsim safety net) ---

def _identity_dynamics(X, U, t):
    return X  # every output dim depends on X, so a real graph forms

def test_grad_flows_through_euler_step():
    X = _upright()
    X.requires_grad_(True)
    X_next = euler_step(_identity_dynamics, X, None, torch.tensor(0.0), DT)
    X_next.sum().backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()

def test_grad_flows_through_rk4_step():
    X = _upright()
    X.requires_grad_(True)
    X_next = rk4_step(_identity_dynamics, X, None, torch.tensor(0.0), DT)
    X_next.sum().backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
