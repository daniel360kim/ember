import torch
import numpy as np
from vehicle.config import NoseConeConfig
from vehicle.config import BodyTubeConfig

from utils.math import quat_rotate

class Aero:
    def __init__(self, drag_coeff: float, air_density: float, nose_cone_config: NoseConeConfig, body_tube_config: BodyTubeConfig):
        self.drag_coeff = drag_coeff
        self.air_density = air_density
        self.nose_cone_config = nose_cone_config
        self.body_tube_config = body_tube_config
    
    def get_drag_force(self, X: torch.tensor, t: torch.tensor) -> torch.tensor:
        """
        Gets the drag force based on the rocket state
        
        Args:
            X: tensor holding the vehicle dynamics
            t: tensor holding time in seconds
            
        Returns the drag force in Newtons
        """
        
        orientation_quat = X[..., 3:7]
        velocity = X[..., 7:10]
        
        speed = torch.norm(velocity, dim=-1, keepdim=True)
        v_hat = velocity / (speed + 1e-8)
        
        area = self._get_cross_sectional_area(orientation_quat, velocity)
        return -0.5 * self.air_density * self.drag_coeff * speed**2 * area * v_hat
        
    
    def _get_attack_angle(self, orientation_quat: torch.tensor, velocity: torch.tensor) -> torch.tensor:
        """
        Get the angle of attack in radians used to calculate drag
        
        Args:
            orientation_quat: tensor quaternion representing the orientation of the rocket to world frame
            velocity: a tensor representing current velocity vector (unnormalized) of the rocket
            
        Returns: the angle of attack in radians
        """
        
        up = torch.tensor([0., 0, 1.]).expand(*orientation_quat.shape[:-1], 3)
        body_axis = quat_rotate(orientation_quat, up)

        v_hat = velocity / (torch.norm(velocity, dim=-1, keepdim=True) + 1e-8)
        
        return torch.arccos(torch.clamp(torch.sum(body_axis * v_hat, dim=-1, keepdim=True), min=-1.0, max=1.0))
    
    def _get_cross_sectional_area(self, orientation_quat: torch.tensor, velocity: torch.tensor) -> torch.tensor:
        """
        Gets the cross sectional area of the rocket.
        
        Args:
            orientation_quat: tensor quaternion representing the orientation of the rocket to world frame
            velocity: a tensor representing current velocity vector (unnormalized) of the rocket
            
        Returns: the cross sectional area of the rocket for a given velocity and orientation
        """
        
        angle_of_attack = self._get_attack_angle(orientation_quat, velocity)
        
        body_tube_projection = 2.0 * self.body_tube_config.radius * self.body_tube_config.length * torch.sin(angle_of_attack)
        
        end_cap_projection_low = np.pi * self.nose_cone_config.radius**2 * torch.cos(angle_of_attack)
        end_cap_projection_high = self.nose_cone_config.radius * self.nose_cone_config.length * torch.sin(angle_of_attack)
        
        threshold = np.atan(np.pi * self.nose_cone_config.radius / self.nose_cone_config.length)
        
        end_cap_projection = torch.where(angle_of_attack <= threshold, end_cap_projection_low, end_cap_projection_high)
        
        return body_tube_projection + end_cap_projection
        
        
        
    
    