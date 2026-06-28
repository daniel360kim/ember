"""Integration test: a short end-to-end rollout of the F15 Atlas flight."""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: avoid opening a GUI during tests

from runners.simulation_runner import SimulationRunner
from runners.history import SimulationHistory
from policy.pid import PID

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs" / "vehicles" / "atlas.yaml")


def _run(duration=0.5, dt=0.01, policy_dt=0.01):
    policy = PID(dt=policy_dt)
    return SimulationRunner(duration, dt, CONFIG_PATH, policy, policy_dt).run()


def test_run_returns_history():
    assert isinstance(_run(), SimulationHistory)

def test_run_produces_states():
    assert len(_run().states) > 0

def test_rocket_leaves_the_ground():
    positions = _run(duration=1.0).get_position_history()  # (N, batch, 3)
    apogee = np.max(positions[:, 0, 2])
    assert apogee > 0.0

def test_trajectory_is_finite():
    positions = _run().get_position_history()
    assert np.isfinite(positions).all()

def test_mass_decreases_over_flight():
    mass = _run().get_extra_history("total_mass")
    assert mass[-1] < mass[0]

def test_starts_near_origin():
    positions = _run().get_position_history()  # (N, batch, 3)
    # first recorded state is one step in, so it should still be very near origin
    assert abs(positions[0, 0, 2]) < 1.0


def test_policy_dt_must_be_multiple_of_dt():
    import pytest
    with pytest.raises(ValueError):
        SimulationRunner(0.1, dt=0.003, config_path=CONFIG_PATH, policy=PID(), policy_dt=0.01)
