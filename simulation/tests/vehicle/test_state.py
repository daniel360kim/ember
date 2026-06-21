import torch
import pytest

from vehicle.state import State


def test_from_tensor_slices_components():
    X = torch.arange(13).float()
    mass = torch.tensor(1.5)
    s = State.from_tensor(X, mass)

    torch.testing.assert_close(s.position, torch.tensor([0., 1., 2.]))
    torch.testing.assert_close(s.orientation_quat, torch.tensor([3., 4., 5., 6.]))
    torch.testing.assert_close(s.velocity, torch.tensor([7., 8., 9.]))
    torch.testing.assert_close(s.angular_velocity, torch.tensor([10., 11., 12.]))
    torch.testing.assert_close(s.mass, mass)


def test_from_tensor_component_shapes():
    s = State.from_tensor(torch.zeros(13), torch.tensor(1.0))
    assert s.position.shape == (3,)
    assert s.orientation_quat.shape == (4,)
    assert s.velocity.shape == (3,)
    assert s.angular_velocity.shape == (3,)


def test_from_tensor_is_a_view():
    # slicing should reference the original storage, not copy it
    X = torch.zeros(13)
    s = State.from_tensor(X, torch.tensor(1.0))
    X[0] = 42.0
    assert s.position[0].item() == 42.0


@pytest.mark.xfail(reason="to_tensor stacks components of unequal length (3 vs 4) -- bug", strict=True)
def test_to_tensor_roundtrip():
    X = torch.arange(13).float()
    s = State.from_tensor(X, torch.tensor(1.0))
    torch.testing.assert_close(s.to_tensor(), X)
