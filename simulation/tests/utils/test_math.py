import torch
import math
from utils.math import *


# --- quat_identity ---

def test_identity():
    q = quat_identity()
    v = torch.tensor([1., 0., 0.])
    torch.testing.assert_close(quat_rotate(q, v), v)

def test_identity_shape():
    q = quat_identity()
    assert q.shape == (4,)

def test_identity_values():
    q = quat_identity()
    torch.testing.assert_close(q, torch.tensor([0., 0., 0., 1.]))


# --- quat_inv ---

def test_inv_identity():
    q = quat_identity()
    torch.testing.assert_close(quat_inv(q), q)

def test_inv_negates_xyz():
    q = torch.tensor([1., 2., 3., 4.])
    qi = quat_inv(q)
    torch.testing.assert_close(qi, torch.tensor([-1., -2., -3., 4.]))

def test_inv_double_inverse():
    q = torch.tensor([0.5, 0.5, 0.5, 0.5])
    torch.testing.assert_close(quat_inv(quat_inv(q)), q)


# --- quat_mul ---

def test_mul_identity_left():
    q = torch.tensor([0., 0., math.sin(math.pi/4), math.cos(math.pi/4)])
    result = quat_mul(quat_identity(), q)
    torch.testing.assert_close(result, q)

def test_mul_identity_right():
    q = torch.tensor([0., 0., math.sin(math.pi/4), math.cos(math.pi/4)])
    result = quat_mul(q, quat_identity())
    torch.testing.assert_close(result, q)

def test_mul_q_inv_is_identity():
    q = torch.tensor([0., 0., math.sin(math.pi/4), math.cos(math.pi/4)])
    result = quat_mul(q, quat_inv(q))
    torch.testing.assert_close(result, quat_identity(), atol=1e-6, rtol=0)

def test_mul_not_commutative():
    q1 = torch.tensor([1., 0., 0., 0.])
    q2 = torch.tensor([0., 1., 0., 0.])
    assert not torch.allclose(quat_mul(q1, q2), quat_mul(q2, q1))


# --- quat_rotate ---

def test_quat_rotate_1():
    q = torch.tensor([0., 0., 1., 0.])
    v = torch.tensor([1., 0., 0.])
    torch.testing.assert_close(quat_rotate(q, v), torch.tensor([-1., 0., 0.]))

def test_quat_rotate_2():
    q = torch.tensor([0., 0., math.sin(math.pi/4), math.cos(math.pi/4)])
    v = torch.tensor([1., 0., 0.])
    torch.testing.assert_close(quat_rotate(q, v), torch.tensor([0., 1., 0.]), atol=1e-6, rtol=0)

def test_rotate_preserves_length():
    q = torch.tensor([0., 0., math.sin(math.pi/4), math.cos(math.pi/4)])
    v = torch.tensor([1., 2., 3.])
    result = quat_rotate(q, v)
    torch.testing.assert_close(result.norm(), v.norm(), atol=1e-6, rtol=0)

def test_inverse():
    q = torch.tensor([0., 0., math.sin(math.pi/4), math.cos(math.pi/4)])
    v = torch.tensor([1., 2., 3.])
    assert torch.allclose(quat_rotate(quat_inv(q), quat_rotate(q, v)), v, atol=1e-6)

def test_rotate_x_axis_90_around_x():
    # Rotating x-axis around x-axis should leave it unchanged
    q = torch.tensor([math.sin(math.pi/4), 0., 0., math.cos(math.pi/4)])
    v = torch.tensor([1., 0., 0.])
    torch.testing.assert_close(quat_rotate(q, v), v, atol=1e-6, rtol=0)

def test_rotate_y_axis_90_around_z():
    q = torch.tensor([0., 0., math.sin(math.pi/4), math.cos(math.pi/4)])
    v = torch.tensor([0., 1., 0.])
    torch.testing.assert_close(quat_rotate(q, v), torch.tensor([-1., 0., 0.]), atol=1e-6, rtol=0)


# --- quat_to_euler ---

def test_quat_to_euler_identity():
    q = quat_identity()
    e = quat_to_euler(q)
    torch.testing.assert_close(e, torch.zeros(3))

def test_quat_to_euler_pure_yaw():
    angle = math.pi / 3
    q = torch.tensor([0., 0., math.sin(angle/2), math.cos(angle/2)])
    e = quat_to_euler(q)
    torch.testing.assert_close(e, torch.tensor([0., 0., angle]), atol=1e-6, rtol=0)

def test_quat_to_euler_pure_pitch():
    angle = math.pi / 6
    q = torch.tensor([0., math.sin(angle/2), 0., math.cos(angle/2)])
    e = quat_to_euler(q)
    torch.testing.assert_close(e, torch.tensor([0., angle, 0.]), atol=1e-6, rtol=0)

def test_quat_to_euler_pure_roll():
    angle = math.pi / 4
    q = torch.tensor([math.sin(angle/2), 0., 0., math.cos(angle/2)])
    e = quat_to_euler(q)
    torch.testing.assert_close(e, torch.tensor([angle, 0., 0.]), atol=1e-6, rtol=0)

def test_quat_to_euler_shape():
    q = quat_identity()
    e = quat_to_euler(q)
    assert e.shape == (3,)


# --- quat_deriv ---

def test_deriv_zero_omega():
    q = quat_identity()
    omega = torch.zeros(3)
    dq = quat_deriv(q, omega)
    torch.testing.assert_close(dq, torch.zeros(4))

def test_deriv_output_shape():
    q = quat_identity()
    omega = torch.tensor([0.1, 0.2, 0.3])
    dq = quat_deriv(q, omega)
    assert dq.shape == (4,)

def test_deriv_identity_spin_z():
    # Spinning around Z from identity: dw should be 0, dz = 0.5 * omega_z
    q = quat_identity()
    omega = torch.tensor([0., 0., 1.])
    dq = quat_deriv(q, omega)
    torch.testing.assert_close(dq, torch.tensor([0., 0., 0.5, 0.]), atol=1e-6, rtol=0)

def test_deriv_norm_rate():
    # For a unit quaternion, d/dt(|q|^2) = 2 q·q̇ = 0
    q = torch.tensor([0., 0., math.sin(math.pi/4), math.cos(math.pi/4)])
    omega = torch.tensor([0.3, 0.7, 1.2])
    dq = quat_deriv(q, omega)
    assert abs(torch.dot(q, dq).item()) < 1e-6

def normalize_quat(q):
    q_new = q.clone()
    q_new = q_new / q_new.norm(dim=-1, keepdim=True)
    return q_new