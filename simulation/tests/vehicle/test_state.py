import torch
import pytest

from vehicle.state import State, S


def test_layout_slices_are_contiguous_and_cover_15():
    slices = [S.POS, S.ORI, S.VEL, S.ANG_VEL, S.GIMBAL_ANGLE]
    starts = [s.start for s in slices]
    stops = [s.stop for s in slices]
    assert starts[0] == 0
    assert stops[-1] == 15
    # each slice begins where the previous ended -- no gaps, no overlap
    assert starts[1:] == stops[:-1]


def test_from_tensor_slices_components():
    X = torch.arange(15).float()
    extras = {"total_mass": torch.tensor(1.5)}
    s = State.from_tensor(X, extras)

    torch.testing.assert_close(s.position, torch.tensor([0., 1., 2.]))
    torch.testing.assert_close(s.orientation_quat, torch.tensor([3., 4., 5., 6.]))
    torch.testing.assert_close(s.velocity, torch.tensor([7., 8., 9.]))
    torch.testing.assert_close(s.angular_velocity, torch.tensor([10., 11., 12.]))
    torch.testing.assert_close(s.gimbal_angle, torch.tensor([13., 14.]))
    assert s.extras is extras


def test_from_tensor_component_shapes():
    s = State.from_tensor(torch.zeros(15), extras={})
    assert s.position.shape == (3,)
    assert s.orientation_quat.shape == (4,)
    assert s.velocity.shape == (3,)
    assert s.angular_velocity.shape == (3,)
    assert s.gimbal_angle.shape == (2,)


def test_from_tensor_is_a_view():
    # slicing should reference the original storage, not copy it
    X = torch.zeros(15)
    s = State.from_tensor(X, extras={})
    X[0] = 42.0
    assert s.position[0].item() == 42.0


@pytest.mark.xfail(reason="to_tensor stacks components of unequal length (3 vs 4) -- bug", strict=True)
def test_to_tensor_roundtrip():
    X = torch.arange(15).float()
    s = State.from_tensor(X, extras={})
    torch.testing.assert_close(s.to_tensor().squeeze(0), X)
