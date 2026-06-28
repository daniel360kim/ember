from dataclasses import dataclass
import torch

from utils.math import quat_deriv

@dataclass
class Derivative:
    velocity: torch.Tensor
    quat_deriv: torch.Tensor
    acceleration: torch.Tensor
    angular_acceleration: torch.Tensor

    
    
class RigidBody:
    def derivative(
        self, *,
        orientation_quat: torch.Tensor,
        velocity: torch.Tensor, 
        angular_velocity: torch.Tensor,
        force_world: torch.Tensor,
        torque_body: torch.Tensor,
        mass: torch.Tensor,
        mmoi: torch.Tensor,
    ) -> Derivative:
        """
        Uses states collected by the dynamics and computes the derivatives needed by the integrator
        
        Args:
            orientation_quat (tensor): current orientation quaternion (B, 4)
            velocity (tensor): world frame velocity of the vehicle (B, 3)
            angular_velocity (tensor): angular velocity of the vehicle (B, 3)
            force_world (tensor): net force applied to vehicle in world frame (B, 3)
            torque_body (tensor); net torque applied to vehicle in body frame (B, 3)
            mass: mass of the vehicle (B, 1)
            mmoi: moment of inertia matrix of the vehicle (B, 3, 3)
            
        Returns: a derivative object holding the derivatives of the states
        """
        
        orientation_deriv = quat_deriv(orientation_quat, angular_velocity)
        acceleration = force_world / mass
        angular_acceleration = self._get_angular_accel(angular_velocity, torque_body, mmoi)
        
        return Derivative(
            velocity=velocity,
            quat_deriv=orientation_deriv,
            acceleration=acceleration,
            angular_acceleration=angular_acceleration,
        )
        
    def _get_angular_accel(self, angular_velocity: torch.Tensor, torque_body: torch.Tensor, mmoi: torch.Tensor) -> torch.Tensor:
        """
        Gets the angular acceleration from a torque and mmoi
        
        Args:
            angular_vel (tensor): the angular velocity of the vehicle (B, 3)
            torque_body (tensor): the torque applied to the vehicle (B, 3)
            mmoi (tensor): mmoi matrix (B, 3, 3)
            
        Returns the angular acceleration vector (B, 3)
        """
        # Add vector dim for linalg operations
        angular_vel = angular_velocity.unsqueeze(-2) # (B, 1, 3)
        torque_body = torque_body.unsqueeze(-2) # (B, 1, 3)
        
        I_omega = torch.matmul(angular_vel, mmoi.transpose(-1, -2))
        gyro = torch.cross(angular_vel, I_omega, dim=-1)
        
        torque_body = torque_body - gyro
        return torch.matmul(torque_body, torch.linalg.inv(mmoi).transpose(-1, -2)).squeeze(-2)
    
        
        
        
        
        
        