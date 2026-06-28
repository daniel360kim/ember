"""Tests for the assembled Atlas vehicle dynamics.

These exercise the full dynamics(X, U, t) -- component wrenches, mass model, the
RigidBody core, the ground clamp, and gimbal dynamics wired together. The
`vehicle` fixture builds the real F15 Atlas from configs/vehicles/atlas.yaml.
"""

import torch
import pytest

from vehicle.state import S
from utils.math import quat_identity


def _rest_state(altitude=0.0):
    """A vehicle at rest with an upright (identity) orientation."""
    X = torch.zeros(1, 15)
    X[..., S.ORI] = quat_identity()
    X[..., 2] = altitude
    return X


ZERO_U = torch.zeros(1, 2)


def _t(value):
    return torch.full((1, 1), value)


# --- shape / structure ---

def test_dynamics_output_shape(vehicle):
    xd = vehicle.dynamics(_rest_state(), ZERO_U, _t(0.5))
    assert xd.shape == (1, 15)


def test_first_three_derivatives_are_velocity(vehicle):
    X = _rest_state(altitude=50.0)
    X[..., S.VEL] = torch.tensor([1.0, 2.0, 3.0])
    xd = vehicle.dynamics(X, ZERO_U, _t(0.5))
    torch.testing.assert_close(xd[..., S.POS], torch.tensor([[1.0, 2.0, 3.0]]))


# --- physics sanity ---

def test_accelerates_up_during_burn(vehicle):
    # Airborne mid-burn: F15 thrust exceeds weight -> positive vertical accel.
    xd = vehicle.dynamics(_rest_state(altitude=50.0), ZERO_U, _t(0.5))
    assert xd[..., 9].item() > 0.0


def test_freefall_after_burnout(vehicle):
    # High above ground, at rest, after burnout: only gravity acts.
    xd = vehicle.dynamics(_rest_state(altitude=100.0), ZERO_U, _t(10.0))
    assert xd[..., 9].item() == pytest.approx(-9.80665, abs=1e-3)


def test_ground_clamp_prevents_sinking(vehicle):
    # On the ground, at rest, after burnout: net downward accel clamped to >= 0.
    xd = vehicle.dynamics(_rest_state(altitude=0.0), ZERO_U, _t(10.0))
    assert xd[..., 9].item() >= 0.0


def test_output_is_finite(vehicle):
    X = _rest_state(altitude=50.0)
    X[..., S.ANG_VEL] = torch.tensor([0.1, -0.2, 0.05])
    xd = vehicle.dynamics(X, torch.tensor([[0.3, -0.2]]), _t(0.5))
    assert torch.isfinite(xd).all()


@pytest.mark.xfail(
    reason="ground clamp writes deriv.acceleration[..., 2] in place, breaking autograd; "
           "remove this marker once the clamp is moved out-of-place",
    strict=True,
)
def test_dynamics_is_differentiable(vehicle):
    X = _rest_state(altitude=50.0).requires_grad_(True)
    xd = vehicle.dynamics(X, torch.tensor([[0.3, -0.2]]), _t(0.5))
    xd.sum().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()


# --- characterization (regression lock) ---
#
# Frozen output of dynamics() for a fixed, nontrivial state. This pins current
# behavior so the upcoming StateLayout / RigidBodyVehicle refactors (Phase 2)
# can prove they are behavior-preserving. If the physics intentionally changes,
# regenerate this snapshot.
DYNAMICS_SNAPSHOT = torch.tensor([[
    0.0, 0.0, 30.0,
    0.05000000074505806, -0.10000000149011612, 0.02500000037252903, 0.0,
    0.3907409906387329, -0.19538027048110962, 5.200069427490234,
    -2.6844937801361084, -5.362321853637695, -1.162557161649147e-08,
    -0.007103476673364639, 0.3893447518348694,
]])


def test_dynamics_characterization(vehicle):
    X = torch.zeros(1, 15)
    X[..., S.ORI] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    X[..., 2] = 50.0
    X[..., 9] = 30.0
    X[..., S.ANG_VEL] = torch.tensor([0.1, -0.2, 0.05])
    X[..., S.GIMBAL_ANGLE] = torch.tensor([0.02, -0.01])
    U = torch.tensor([[0.3, -0.2]])
    xd = vehicle.dynamics(X, U, _t(0.5))
    torch.testing.assert_close(xd, DYNAMICS_SNAPSHOT, atol=1e-5, rtol=1e-5)
