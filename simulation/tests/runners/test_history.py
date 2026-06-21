import numpy as np
import torch
import pytest

from runners.history import SimulationHistory
from vehicle.state import State
from utils.math import quat_identity


def _state(i):
    return State(
        position=torch.tensor([float(i), 0.0, 2.0 * i]),
        orientation_quat=quat_identity(),
        velocity=torch.tensor([0.0, 0.0, float(i)]),
        angular_velocity=torch.tensor([0.0, 0.0, 0.1 * i]),
        mass=torch.tensor(1.0 - 0.01 * i),
    )


@pytest.fixture
def history():
    h = SimulationHistory()
    for i in range(5):
        h.add(_state(i))
    return h


def test_empty_history_has_no_states():
    assert SimulationHistory().states == []

def test_add_appends_states(history):
    assert len(history.states) == 5

def test_position_history_shape(history):
    pos = history.get_position_history()
    assert isinstance(pos, np.ndarray)
    assert pos.shape == (5, 3)

def test_position_history_values(history):
    pos = history.get_position_history()
    np.testing.assert_allclose(pos[:, 2], [0.0, 2.0, 4.0, 6.0, 8.0])

def test_velocity_history_shape(history):
    assert history.get_velocity_history().shape == (5, 3)

def test_quaternion_history_shape(history):
    assert history.get_quaternion_history().shape == (5, 4)

def test_angular_velocity_history_shape(history):
    assert history.get_angular_velocity_history().shape == (5, 3)

def test_euler_history_shape(history):
    assert history.get_orientation_euler_history().shape == (5, 3)

def test_euler_history_identity_is_zero(history):
    # all states use the identity quaternion -> zero euler angles
    np.testing.assert_allclose(history.get_orientation_euler_history(), 0.0, atol=1e-6)

def test_mass_history_shape(history):
    mass = history.get_mass_history()
    assert mass.shape == (5,)

def test_mass_history_values(history):
    mass = history.get_mass_history()
    np.testing.assert_allclose(mass, [1.0, 0.99, 0.98, 0.97, 0.96], rtol=1e-5)


def test_history_detaches_gradients():
    # a state carrying gradient-tracking tensors must still be convertible to numpy
    h = SimulationHistory()
    pos = torch.zeros(3, requires_grad=True)
    h.add(State(
        position=pos,
        orientation_quat=quat_identity(),
        velocity=torch.zeros(3),
        angular_velocity=torch.zeros(3),
        mass=torch.tensor(1.0),
    ))
    np.testing.assert_allclose(h.get_position_history(), np.zeros((1, 3)))
