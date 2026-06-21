"""Integration test: a short end-to-end rollout of the F15 Atlas flight."""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: avoid opening a GUI during tests

from runners.test_runner import TestRunner
from runners.history import SimulationHistory


def _run(duration=2.0, dt=0.01):
    return TestRunner(duration, dt).run()


def test_run_returns_history():
    assert isinstance(_run(), SimulationHistory)

def test_run_produces_states():
    assert len(_run().states) > 0

def test_rocket_leaves_the_ground():
    positions = _run().get_position_history()
    apogee = np.max(positions[:, 2])
    assert apogee > 0.0

def test_trajectory_is_finite():
    positions = _run().get_position_history()
    assert np.isfinite(positions).all()

def test_mass_decreases_over_flight():
    mass = _run().get_mass_history()
    assert mass[-1] < mass[0]

def test_starts_at_origin():
    positions = _run().get_position_history()
    # first recorded state is one step in, so it should still be very near origin
    assert abs(positions[0, 2]) < 1.0
