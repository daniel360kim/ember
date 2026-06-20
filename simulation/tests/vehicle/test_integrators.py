"""Phase 0 deliverable: verify Euler/RK4 on the real 13D state.

Conserved-quantity tests use the torque-free rigid body from the mini-project.
The autograd test is decoupled (trivial dynamics) so it checks ONLY that the
step function is safe to backprop through -- it fails loudly if the quaternion
renormalization is still an in-place write.
"""

import math
import torch

from scripts.free_spin_demo import (
    make_free_body, make_X0, rollout,
    kinetic_energy, ang_momentum_mag, quat_norm, relative_drift,
)
from vehicle.integrators import euler_step, rk4_step


I = torch.tensor([1.0, 2.0, 3.0])     # distinct -> genuine tumble
DT, N = 2e-3, 1000                    # 2 s


def _traj(step, omega0=(0.1, 7.0, 0.0)):
    return rollout(step, make_free_body(I), make_X0(omega0), DT, N)


# --- quaternion stays on the manifold ---------------------------------------

def test_quat_norm_preserved_rk4():
    nq = quat_norm(_traj(rk4_step))
    torch.testing.assert_close(nq, torch.ones_like(nq), atol=1e-5, rtol=0)

def test_quat_norm_preserved_euler():
    nq = quat_norm(_traj(euler_step))
    torch.testing.assert_close(nq, torch.ones_like(nq), atol=1e-5, rtol=0)


# --- energy & angular momentum conserved (RK4) ------------------------------

def test_energy_conserved_rk4():
    T = kinetic_energy(_traj(rk4_step)[:, 10:13], I)
    assert relative_drift(T) < 1e-3

def test_momentum_conserved_rk4():
    L = ang_momentum_mag(_traj(rk4_step)[:, 10:13], I)
    assert relative_drift(L) < 1e-3


# --- RK4 must beat Euler on energy ------------------------------------------

def test_rk4_beats_euler_on_energy():
    drift_rk4 = relative_drift(kinetic_energy(_traj(rk4_step)[:, 10:13], I))
    drift_eul = relative_drift(kinetic_energy(_traj(euler_step)[:, 10:13], I))
    assert drift_rk4 < drift_eul


# --- the test Phase 0 actually exists for: gradients flow through a step -----

def _trivial_dynamics(X, U=None):
    # every output dim depends on X, so a real graph forms through the step
    return X

def test_grad_flows_through_euler_step():
    X = torch.randn(13, requires_grad=True)
    X_next = euler_step(_trivial_dynamics, X, None, DT)
    X_next.sum().backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()

def test_grad_flows_through_rk4_step():
    X = torch.randn(13, requires_grad=True)
    X_next = rk4_step(_trivial_dynamics, X, None, DT)
    X_next.sum().backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
