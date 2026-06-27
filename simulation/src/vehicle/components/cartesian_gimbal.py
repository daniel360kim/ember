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
        
    def dynamics(self, gimbal_state, torque_cmd, cg, t):
        moment_arm = torch.abs(self.gimbal_location[2] - cg[..., 2:3])  # L, (B,1)
        inv = 1.0 / (self.motor.get_thrust(t) * moment_arm + 1e-8)      # guard burnout T→0
        gimbal_cmd = torch.stack([
            -torque_cmd[..., 1] * inv[..., 0],   # x_angle (δx) from -τy
            torque_cmd[..., 0] * inv[..., 0],   # y_angle (δy) from  τx
        ], dim=-1)
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
        
        