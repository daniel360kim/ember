import torch
import numpy as np

from vehicle.base_vehicle import BaseVehicle
from vehicle.components.motor import Motor
from vehicle.components.aero import Aero
from vehicle.config import VehicleConfig, MomentInertiaConfig, LocationConfig
from utils.math import quat_deriv, quat_rotate

G = 9.80665 # todo, move to config file 

class Atlas(BaseVehicle):
    def __init__(self, vehicle_mass: float, motor: Motor, aero: Aero, mmoi: MomentInertiaConfig, cg: LocationConfig):
        super().__init__()
        
        self.vehicle_mass = vehicle_mass
        self.motor = motor
        self.aero = aero
        
        self.I = torch.diag(torch.tensor([mmoi.Ixx, mmoi.Iyy, mmoi.Izz]))
        self.I_inv = torch.linalg.inv(self.I)
        
        self.cg = torch.tensor([cg.x, cg.y, cg.z])
        
        
    def dynamics(self, X: torch.tensor, U: torch.tensor, t: torch.tensor):
        thrust = self.motor.get_thrust(t)
        motor_mass = self.motor.get_mass(t)
        
        total_mass = motor_mass + self.vehicle_mass        
        position = X[..., 0:3]
        velocity = X[..., 7:10]
        orientation_quat = X[..., 3:7]
        ang_vel = X[..., 10:13]
        
        F_drag, torque_drag = self.aero.get_dynamics(X, t, self.cg.expand(*X.shape[:-1], 3))
        
        thrust_body = torch.cat((torch.zeros_like(thrust), torch.zeros_like(thrust), thrust), dim=-1)
        F_thrust = quat_rotate(orientation_quat, thrust_body)
       
        F_grav = torch.cat([torch.zeros_like(total_mass), torch.zeros_like(total_mass), -total_mass * G], dim=-1)

        accel = (F_thrust + F_grav + F_drag) / total_mass
        
        ang_accel = self._get_ang_accel(angular_vel=ang_vel, net_torque=torque_drag)
        
        # If rocket is sitting on the ground
        accel_clamped = torch.clamp(accel[..., 2], min=0.0)
        on_ground = (position[..., 2] <= 0) & (velocity[..., 2] <= 0)
        accel_z_clamped = torch.where(on_ground, accel_clamped, accel[..., 2])
        
        accel[..., 2] = accel_z_clamped
        
        q_deriv = quat_deriv(orientation_quat, X[..., 10:13])
        
        return torch.cat((velocity, q_deriv, accel, ang_accel), dim=-1)
    
    def _get_ang_accel(self, angular_vel: torch.tensor, net_torque: torch.tensor) -> torch.tensor:
        I_omega = angular_vel @ self.I.T
        gyro = torch.cross(angular_vel, I_omega, dim=-1)
        net_torque = net_torque - gyro
        return net_torque @ self.I_inv.T
        
    def get_extras(self, t):
        motor_mass = self.motor.get_mass(t)
        total_mass = motor_mass + self.vehicle_mass
        
        thrust = self.motor.get_thrust(t)
        
        return {
            "total_mass": total_mass,
            "thrust": thrust
        }
        
def build_vehicle(config: VehicleConfig) -> Atlas:
    motor = Motor() # TODO allow for different motors from a registry
    aero = Aero(drag_coeff=config.drag_coeff, 
                air_density = config.air_density, 
                nose_cone_config=config.nose_cone, body_tube_config=config.body_tube, 
                cp=config.cp, mmoi=config.mmoi)
    
    return Atlas(config.vehicle_mass, motor=motor, aero=aero, mmoi=config.mmoi, cg=config.cg)
    
    