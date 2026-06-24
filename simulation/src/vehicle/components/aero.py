import torch
import numpy as np
from vehicle.config import NoseConeConfig, BodyTubeConfig, AeroConfig, LocationConfig, MomentInertiaConfig

from utils.math import quat_rotate, quat_inv

class Aero:
    def __init__(self, 
                 aero: AeroConfig,
                 nose_cone_config: NoseConeConfig, body_tube_config: BodyTubeConfig,
                 cp: LocationConfig,
                 mmoi: MomentInertiaConfig
                 ):
        self.aero_config = aero
        self.nose_cone_config = nose_cone_config
        self.body_tube_config = body_tube_config
        
        self.cp = torch.tensor([cp.x, cp.y, cp.z])
        
        self.I = torch.diag(torch.tensor([mmoi.Ixx, mmoi.Iyy, mmoi.Izz]))
        
        self.ref_area = np.pi * nose_cone_config.radius ** 2 # for normal force calculation
        
        self.aoa_warned = False # printed a warning if aoa is greater than approximation threshold
        
    
    def get_dynamics(self, X: torch.tensor, t: torch.tensor, cg: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        """
        Gets the forces and torques acting on the rocket in world frame.
        
        Args:
            X (tensor): the current rocket dynamics (B, 13)
            t (tensor): the current time (B, 1)
            cg (tensor): center of gravity location in the rocket
            
        Returns:
            drag_force (tensor): drag force acting on the rocket in world frame in N (B, 3)
            torque (tensor): torque force applied to rocket in body frame in Nm (B, 3)
            
        """
        
        orientation_quat = X[..., 3:7]
        velocity = X[..., 7:10]
        
        # Get unit body axis vector
        up = torch.tensor([0., 0, 1.]).expand(*orientation_quat.shape[:-1], 3)
        b_hat = quat_rotate(orientation_quat, up)
        
        # Get speed and unit velocity vector
        speed = torch.norm(velocity, dim=-1, keepdim=True)
        v_hat = velocity / (speed + 1e-8)
        
        drag_force = self._get_drag_force(b_hat, v_hat, speed, t)
        normal_force = self._get_normal_force(b_hat, v_hat, speed, t)
        net_force = drag_force + normal_force
        torque = self._get_torque(X, t, net_force, cg, speed)
        
        return net_force, torque
    
    def _get_drag_force(self, b_hat: torch.tensor, v_hat: torch.tensor, speed: torch.tensor, t: torch.tensor) -> torch.tensor:
        """
        Gets the drag force based on the rocket state in world frame
        
        Args:
            b_hat (tensor): Unit body axis vector in world frame (B, 3)
            v_hat (tensor): Unit velocity vector in world frame (B, 3)
            speed (tensor): Magnitude of velocity (B, 1)
            t: tensor holding time in seconds (B, 1)
            
        Returns the drag force in Newtons
        """

        
        area = self._get_cross_sectional_area(b_hat, v_hat)
        return -0.5 * self.aero_config.air_density * self.aero_config.drag_coeff * speed**2 * area * v_hat
    
    def _get_normal_force(self, b_hat: torch.tensor, v_hat: torch.tensor, speed: torch.tensor, t: torch.tensor) -> torch.tensor:
        """
        Gets the normal force from aerodynamics based on the rocket state in world frame
        
        Args:
            b_hat (tensor): Unit body axis vector in world frame (B, 3)
            v_hat (tensor): Unit velocity vector in world frame (B, 3)
            speed (tensor): Magnitude of velocity (B, 1)
            t (tensor): tensor holding time in seconds (B, 1)
            
        Returns the normal force in Newtons

        """
        normal = b_hat - torch.sum(b_hat * v_hat, dim=-1, keepdim=True) * v_hat
        angle_of_attack = self._get_angle_of_attack(b_hat, v_hat)
        
        out_of_range = angle_of_attack > np.deg2rad(15.0)
        if torch.any(out_of_range) and not self.aoa_warned:
            print("Warning: angle of attack greater than 15 violates Barrowman approximation")
            self.aoa_warned = True
        
        return 0.5 * self.aero_config.air_density * speed**2 * self.ref_area * self.aero_config.normal_force_coeff * normal
        
        
    def _get_torque(self, X: torch.tensor, t: torch.tensor, net_force: torch.tensor, cg: torch.tensor, speed: torch.tensor) -> torch.tensor:
        """
        Gets the torque applied to the rocket in body frame
        
        Args:
            X (tensor): the current rocket dynamics (B, 13)
            t: (tensor): the current time (B, 1)
            net_force (tensor): the net force from the aerodynamic forces applied to CP (B, 3)
            cg (tensor): the location of the center of gravity (B, 3)
            speed (tensor): Current speed of rocket (B, 1)
        """
        
        lever_arm = self._get_lever_arm(cg) # (B, 3)
        world_to_body = quat_inv(X[..., 3:7]) # world to body quaternion
        net_force_body = quat_rotate(world_to_body, net_force)
        
        aero_force_torque = torch.cross(lever_arm, net_force_body, dim=-1)
        damping_torque = self._get_damping_torque(X[..., 10:13], speed)
        
        return aero_force_torque + damping_torque
    
    def _get_damping_torque(self, angular_vel: torch.tensor, speed: torch.tensor):
        """
        Get 
        """
        
        ang_vel_perp = angular_vel * torch.tensor([1., 1., 0.], device=angular_vel.device)
        ref_length = (2 * self.body_tube_config.radius)**2 # diameter^2
        return -0.25 * self.aero_config.air_density * speed * self.ref_area * ref_length * self.aero_config.pitch_damping_coeff * ang_vel_perp
        
    def _get_angle_of_attack(self, b_hat: torch.tensor, v_hat: torch.tensor) -> torch.tensor:
        """
        Get the angle of attack of the rocket in radians 
        Args:
            b_hat (tensor): Unit body axis vector in world frame (B, 3)
            v_hat (tensor): Unit velocity vector in world frame (B, 3)
            
        Returns the angle of attack in radians
        """
        # Prevent gradient explosion at end points of arccos 
        return torch.arccos(torch.clamp(torch.sum(b_hat * v_hat, dim=-1, keepdim=True), min=-1.0 + 1e-6, max=1.0 - 1e-6))
    
    def _get_cross_sectional_area(self, b_hat: torch.tensor, v_hat: torch.tensor) -> torch.tensor:
        """
        Gets the cross sectional area of the rocket.
        
        Args:
            b_hat (tensor): Unit body axis vector in world frame (B, 3)
            v_hat (tensor): Unit velocity vector in world frame (B, 3)
            
        Returns: the cross sectional area of the rocket for a given velocity and orientation
        """
        # Prevent gradient explosion at end points of arccos 
        angle_of_attack = self._get_angle_of_attack(b_hat, v_hat)
        
        body_tube_projection = 2.0 * self.body_tube_config.radius * self.body_tube_config.length * torch.sin(angle_of_attack)
        
        end_cap_projection_low = np.pi * self.nose_cone_config.radius**2 * torch.cos(angle_of_attack)
        end_cap_projection_high = self.nose_cone_config.radius * self.nose_cone_config.length * torch.sin(angle_of_attack)
        
        threshold = np.atan(np.pi * self.nose_cone_config.radius / self.nose_cone_config.length)
        
        end_cap_projection = torch.where(angle_of_attack <= threshold, end_cap_projection_low, end_cap_projection_high)
        
        return body_tube_projection + end_cap_projection
    
    def _get_lever_arm(self, cg: torch.tensor) -> torch.tensor:
        return self.cp - cg[...,:]
    

        
    
    