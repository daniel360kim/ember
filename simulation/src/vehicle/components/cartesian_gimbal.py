import torch
import numpy as np

from vehicle.components.base_component import Component, Wrench
from vehicle.components.motor import Motor
from vehicle.config import LocationConfig, GimbalConfig
from utils.math import quat_rotate, quat_inv

class CartesianGimbal(Component):
    def __init__(self, gimbal_config: GimbalConfig, motor: Motor):
        ## TODO: in config add max_deflection, initial angles
        self.tau = gimbal_config.tau
        self.max_deflection = np.deg2rad(gimbal_config.angle_limit_deg)
        gimbal_location = gimbal_config.gimbal_location
        self.motor = motor
        # TODO: wire into location config
        self.gimbal_location = torch.tensor([gimbal_location.x, gimbal_location.y, gimbal_location.z])
        
    def get_wrench(self, X: torch.tensor, t: torch.tensor) -> Wrench:
        orientation_quat = X[..., 3:7]
        gimbal_state = X[..., 13:15]
        thrust_body = self.get_thrust_body(gimbal_state, t)
        thrust_world = quat_rotate(orientation_quat, thrust_body) 
        
        return Wrench(
            force_world=thrust_world,
            application_point_body=self.gimbal_location.expand(*t.shape[:-1], 3), # (B, 3)
            moment_body=None,
        )
        
    def dynamics(self, gimbal_state: torch.tensor, gimbal_cmd: torch.tensor):
        """
        Gets the change in angle which is integrated with the integrator with the other rocket dynamics
        
        Args:
            gimbal_state (tensor): current angle of the gimbal (B, 2)
            gimbal_cmd (tensor): the commanded angle of the gimbal (B, 2).
        
        Returns: a tensor representing the change in angle with time constant adjustment
        """
        gimbal_cmd = torch.clamp(gimbal_cmd, -self.max_deflection, self.max_deflection)
        return (gimbal_cmd - gimbal_state) / self.tau
        
    def get_thrust_body(self, gimbal_state: torch.tensor, t: torch.tensor, ):
        thrust_magnitude = self.motor.get_thrust(t) # (B, 1)
        x_angle = gimbal_state[..., 0]
        y_angle = gimbal_state[..., 1]
        
        thrust_body = torch.stack([
            torch.sin(x_angle),
            torch.sin(y_angle),
            torch.cos(x_angle) * torch.cos(y_angle),
        ], dim=-1)
        thrust_body = thrust_body / torch.norm(thrust_body, dim=-1, keepdim=True) # (B, 3)
        
        thrust_body = thrust_magnitude * thrust_body
        
        return thrust_body
        
        