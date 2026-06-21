import torch
import pytest

from vehicle.components.motor import Motor


@pytest.fixture
def motor():
    return Motor()


# --- defaults / metadata ---

def test_default_is_f15(motor):
    assert motor.name == 'F15'
    assert motor.total_impulse == pytest.approx(49.6)
    assert motor.total_mass == pytest.approx(0.103)
    assert motor.prop_mass == pytest.approx(0.060)


# --- get_thrust ---

def test_thrust_zero_before_ignition(motor):
    assert motor.get_thrust(torch.tensor(-1.0)).item() == 0.0

def test_thrust_zero_after_burnout(motor):
    assert motor.get_thrust(torch.tensor(10.0)).item() == 0.0

def test_thrust_positive_during_burn(motor):
    t = torch.linspace(0.0, 3.0, 50)
    assert torch.all(motor.get_thrust(t) > 0.0)

def test_thrust_preserves_shape(motor):
    t = torch.linspace(0.0, 4.0, 17)
    assert motor.get_thrust(t).shape == t.shape

def test_thrust_scalar_input(motor):
    out = motor.get_thrust(torch.tensor(1.0))
    assert out.ndim == 0


# --- get_cumulative_impulse ---

def test_impulse_zero_at_start(motor):
    assert motor.get_cumulative_impulse(torch.tensor(0.0)).item() == pytest.approx(0.0, abs=1e-4)

def test_impulse_saturates_at_total(motor):
    assert motor.get_cumulative_impulse(torch.tensor(10.0)).item() == pytest.approx(49.6, abs=1e-3)

@pytest.mark.xfail(
    reason="cumulative impulse drops ~1.9 N*s at the t~=3.25 segment boundary "
    "(antiderivative offsets discontinuous) -- bug",
    strict=True,
)
def test_impulse_monotonic_nondecreasing(motor):
    t = torch.linspace(0.0, 4.0, 200)
    imp = motor.get_cumulative_impulse(t)
    diffs = imp[1:] - imp[:-1]
    assert torch.all(diffs >= -1e-3)

@pytest.mark.xfail(
    reason="thrust polynomials integrate to ~32.6 N*s over the burn, not the "
    "rated total_impulse of 49.6 N*s -- thrust curve / impulse mismatch",
    strict=True,
)
def test_impulse_matches_thrust_integral(motor):
    # numerically integrate thrust and compare to the analytic cumulative impulse
    t = torch.linspace(0.0, 3.4, 20001)
    dt = (t[1] - t[0]).item()
    numeric = torch.trapz(motor.get_thrust(t), dx=dt)
    analytic = motor.get_cumulative_impulse(torch.tensor(3.4))
    assert numeric.item() == pytest.approx(analytic.item(), rel=1e-2)


# --- get_mass ---

def test_mass_full_at_start(motor):
    assert motor.get_mass(torch.tensor(0.0)).item() == pytest.approx(0.103, abs=1e-4)

def test_mass_burnout_is_dry_mass(motor):
    # after burnout all propellant is gone -> total_mass - prop_mass
    assert motor.get_mass(torch.tensor(10.0)).item() == pytest.approx(0.043, abs=1e-4)

@pytest.mark.xfail(
    reason="mass jumps up ~0.0023 kg at the t~=3.25 boundary, inherited from the "
    "non-monotonic cumulative impulse -- bug",
    strict=True,
)
def test_mass_monotonic_nonincreasing(motor):
    t = torch.linspace(0.0, 4.0, 200)
    mass = motor.get_mass(t)
    diffs = mass[1:] - mass[:-1]
    assert torch.all(diffs <= 1e-4)

def test_mass_never_below_dry_mass(motor):
    t = torch.linspace(0.0, 10.0, 500)
    assert torch.all(motor.get_mass(t) >= 0.043 - 1e-5)

def test_mass_preserves_shape(motor):
    t = torch.linspace(0.0, 4.0, 13)
    assert motor.get_mass(t).shape == t.shape
