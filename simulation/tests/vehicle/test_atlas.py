import torch
import numpy as np
import pytest

from vehicle.atlas_v1 import Atlas, G


@pytest.fixture
def atlas():
    return Atlas()


def _rest_state():
    X = torch.zeros(13)
    X[6] = 1.0  # unit quaternion (w = 1)
    return X


# --- get_mass ---

def test_mass_is_vehicle_plus_motor(atlas):
    t = torch.tensor(0.0)
    expected = atlas.vehicle_mass + atlas.motor.get_mass(t)
    torch.testing.assert_close(atlas.get_mass(t), expected)

def test_mass_decreases_during_burn(atlas):
    m0 = atlas.get_mass(torch.tensor(0.0))
    m1 = atlas.get_mass(torch.tensor(10.0))
    assert m1 < m0


# --- dynamics ---

def test_dynamics_output_shape(atlas):
    xd = atlas.dynamics(_rest_state(), None, torch.tensor(0.0))
    assert xd.shape == (13,)

def test_dynamics_first_three_are_velocity(atlas):
    X = _rest_state()
    X[7:10] = torch.tensor([1.0, 2.0, 3.0])
    xd = atlas.dynamics(X, None, torch.tensor(0.0))
    torch.testing.assert_close(xd[0:3], torch.tensor([1.0, 2.0, 3.0]))

def test_dynamics_accelerates_up_at_ignition(atlas):
    # at t=0 the F15 thrust exceeds weight -> positive vertical acceleration
    xd = atlas.dynamics(_rest_state(), None, torch.tensor(0.0))
    assert xd[9].item() > 0.0

def test_dynamics_freefall_without_thrust(atlas):
    # airborne and stationary after burnout -> only gravity acts
    X = _rest_state()
    X[2] = 100.0  # high above the ground so the ground clamp does not apply
    xd = atlas.dynamics(X, None, torch.tensor(10.0))
    assert xd[9].item() == pytest.approx(-G, abs=1e-3)

def test_dynamics_ground_clamp_prevents_sinking(atlas):
    # on the ground, at rest, after burnout: net downward accel is clamped to >= 0
    X = _rest_state()
    xd = atlas.dynamics(X, None, torch.tensor(10.0))
    assert xd[9].item() >= 0.0

def test_dynamics_no_angular_acceleration(atlas):
    xd = atlas.dynamics(_rest_state(), None, torch.tensor(0.0))
    torch.testing.assert_close(xd[10:13], torch.zeros(3))


# --- get_drag ---

def test_drag_opposes_velocity(atlas):
    v = torch.tensor([0.0, 0.0, 10.0])
    drag = atlas.get_drag(v)
    assert drag[2].item() < 0.0

def test_drag_zero_at_rest(atlas):
    drag = atlas.get_drag(torch.zeros(3))
    torch.testing.assert_close(drag, torch.zeros(3))

def test_drag_scales_with_speed_squared(atlas):
    d1 = torch.norm(atlas.get_drag(torch.tensor([0.0, 0.0, 10.0])))
    d2 = torch.norm(atlas.get_drag(torch.tensor([0.0, 0.0, 20.0])))
    assert d2.item() == pytest.approx(4.0 * d1.item(), rel=1e-3)

def test_drag_magnitude_matches_formula(atlas):
    v = torch.tensor([0.0, 0.0, 7.0])
    speed = 7.0
    area = np.pi * atlas.vehicle_radius ** 2
    expected = 0.5 * atlas.air_density * speed ** 2 * atlas.drag_coeff * area
    assert torch.norm(atlas.get_drag(v)).item() == pytest.approx(expected, rel=1e-3)
