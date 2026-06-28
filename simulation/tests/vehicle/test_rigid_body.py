"""Unit tests for the rocket-agnostic 6-DOF core.

RigidBody owns only the Newton-Euler equations of motion: given net force,
net torque, and mass properties it returns the kinematic derivatives. It knows
nothing about motors, aero, or the state layout, so it can be tested in complete
isolation with synthetic inputs -- no vehicle required.
"""

import math
import torch

from vehicle.rigid_body import RigidBody
from utils.math import quat_identity


def _diag_inertia(ix, iy, iz):
    return torch.diag(torch.tensor([ix, iy, iz])).unsqueeze(0)  # (1, 3, 3)


# --- linear: a = F / m ---

def test_acceleration_is_force_over_mass():
    mass = torch.tensor([[2.0]])
    force = torch.tensor([[4.0, -6.0, 10.0]])
    d = RigidBody().derivative(
        orientation_quat=quat_identity().unsqueeze(0),
        velocity=torch.zeros(1, 3),
        angular_velocity=torch.zeros(1, 3),
        force_world=force,
        torque_body=torch.zeros(1, 3),
        mass=mass,
        mmoi=_diag_inertia(1.0, 1.0, 1.0),
    )
    torch.testing.assert_close(d.acceleration, torch.tensor([[2.0, -3.0, 5.0]]))


def test_velocity_passes_through_as_position_derivative():
    v = torch.tensor([[1.0, 2.0, 3.0]])
    d = RigidBody().derivative(
        orientation_quat=quat_identity().unsqueeze(0),
        velocity=v,
        angular_velocity=torch.zeros(1, 3),
        force_world=torch.zeros(1, 3),
        torque_body=torch.zeros(1, 3),
        mass=torch.tensor([[1.0]]),
        mmoi=_diag_inertia(1.0, 1.0, 1.0),
    )
    torch.testing.assert_close(d.velocity, v)


# --- rotational: Euler's equation, I*alpha = tau - omega x (I omega) ---

def test_angular_accel_from_torque_no_spin():
    # With zero angular velocity the gyroscopic term vanishes: alpha = I^-1 tau.
    torque = torch.tensor([[2.0, 0.0, 0.0]])
    d = RigidBody().derivative(
        orientation_quat=quat_identity().unsqueeze(0),
        velocity=torch.zeros(1, 3),
        angular_velocity=torch.zeros(1, 3),
        force_world=torch.zeros(1, 3),
        torque_body=torque,
        mass=torch.tensor([[1.0]]),
        mmoi=_diag_inertia(1.0, 2.0, 3.0),
    )
    torch.testing.assert_close(d.angular_acceleration, torch.tensor([[2.0, 0.0, 0.0]]))


def test_gyroscopic_term_with_zero_torque():
    # Tumbling about a non-principal axis with no applied torque: the only
    # angular acceleration is the gyroscopic coupling -I^-1 (omega x I omega).
    # For I=diag(1,2,3), omega=(1,1,0): I omega=(1,2,0), omega x I omega=(0,0,1),
    # so alpha = I^-1 (0 - (0,0,1)) = (0, 0, -1/3).
    d = RigidBody().derivative(
        orientation_quat=quat_identity().unsqueeze(0),
        velocity=torch.zeros(1, 3),
        angular_velocity=torch.tensor([[1.0, 1.0, 0.0]]),
        force_world=torch.zeros(1, 3),
        torque_body=torch.zeros(1, 3),
        mass=torch.tensor([[1.0]]),
        mmoi=_diag_inertia(1.0, 2.0, 3.0),
    )
    torch.testing.assert_close(d.angular_acceleration, torch.tensor([[0.0, 0.0, -1.0 / 3.0]]))


def test_spin_about_principal_axis_is_torque_free():
    # Spinning purely about a principal axis, omega is parallel to I omega, so
    # the cross product is zero and a torque-free body has zero angular accel.
    d = RigidBody().derivative(
        orientation_quat=quat_identity().unsqueeze(0),
        velocity=torch.zeros(1, 3),
        angular_velocity=torch.tensor([[0.0, 0.0, 5.0]]),
        force_world=torch.zeros(1, 3),
        torque_body=torch.zeros(1, 3),
        mass=torch.tensor([[1.0]]),
        mmoi=_diag_inertia(1.0, 2.0, 3.0),
    )
    torch.testing.assert_close(d.angular_acceleration, torch.zeros(1, 3), atol=1e-6, rtol=0)


# --- quaternion kinematics: q_dot = 0.5 q (x) [omega, 0] ---

def test_quat_derivative_identity_spin_z():
    d = RigidBody().derivative(
        orientation_quat=quat_identity().unsqueeze(0),
        velocity=torch.zeros(1, 3),
        angular_velocity=torch.tensor([[0.0, 0.0, 1.0]]),
        force_world=torch.zeros(1, 3),
        torque_body=torch.zeros(1, 3),
        mass=torch.tensor([[1.0]]),
        mmoi=_diag_inertia(1.0, 1.0, 1.0),
    )
    torch.testing.assert_close(d.quat_deriv, torch.tensor([[0.0, 0.0, 0.5, 0.0]]))


# --- batching ---

def test_preserves_batch_dimension():
    B = 4
    quat = quat_identity().expand(B, 4)
    d = RigidBody().derivative(
        orientation_quat=quat,
        velocity=torch.randn(B, 3),
        angular_velocity=torch.randn(B, 3),
        force_world=torch.randn(B, 3),
        torque_body=torch.randn(B, 3),
        mass=torch.rand(B, 1) + 1.0,
        mmoi=_diag_inertia(1.0, 2.0, 3.0).expand(B, 3, 3),
    )
    assert d.velocity.shape == (B, 3)
    assert d.quat_deriv.shape == (B, 4)
    assert d.acceleration.shape == (B, 3)
    assert d.angular_acceleration.shape == (B, 3)


# --- differentiability: gradients must flow (this is a diffsim) ---

def test_gradients_flow_through_derivative():
    force = torch.zeros(1, 3, requires_grad=True)
    torque = torch.zeros(1, 3, requires_grad=True)
    d = RigidBody().derivative(
        orientation_quat=quat_identity().unsqueeze(0),
        velocity=torch.zeros(1, 3),
        angular_velocity=torch.tensor([[0.2, 0.1, 0.0]]),
        force_world=force,
        torque_body=torque,
        mass=torch.tensor([[1.5]]),
        mmoi=_diag_inertia(1.0, 2.0, 3.0),
    )
    (d.acceleration.sum() + d.angular_acceleration.sum()).backward()
    assert force.grad is not None and torch.isfinite(force.grad).all()
    assert torque.grad is not None and torch.isfinite(torque.grad).all()
