import torch
import numpy as np

from vehicle.base_vehicle import BaseVehicle
from vehicle.components.motor import Motor
from vehicle.components.aero import Aero
from vehicle.config import VehicleConfig
from utils.math import quat_deriv, quat_rotate

G = 9.80665 # todo, move to config file 

class Atlas(BaseVehicle):
    def __init__(self, vehicle_mass: float, motor: Motor, aero: Aero):
        super().__init__()
        
        self.vehicle_mass = vehicle_mass
        self.motor = motor
        self.aero = aero
        
        
    def dynamics(self, X: torch.tensor, U: torch.tensor, t: torch.tensor):
        thrust = self.motor.get_thrust(t)
        motor_mass = self.motor.get_mass(t)
        
        total_mass = motor_mass + self.vehicle_mass        
        position = X[..., 0:3]
        velocity = X[..., 7:10]
        orientation_quat = X[..., 3:7]
        ang_vel = X[..., 10:13]
        
        F_drag = self.aero.get_drag_force(X, t)
        
        thrust_body = torch.cat((torch.zeros_like(thrust), torch.zeros_like(thrust), thrust), dim=-1)
        
        F_thrust = quat_rotate(orientation_quat, thrust_body)
       
        F_grav = torch.cat([torch.zeros_like(total_mass), torch.zeros_like(total_mass), -total_mass * G], dim=-1)

        
        accel = (F_thrust + F_grav + F_drag) / total_mass
        
        # If rocket is sitting on the ground
        accel_clamped = torch.clamp(accel[..., 2], min=0.0)
        on_ground = (position[..., 2] <= 0) & (velocity[..., 2] <= 0)
        accel_z_clamped = torch.where(on_ground, accel_clamped, accel[..., 2])
        
        accel[..., 2] = accel_z_clamped
        
        q_deriv = quat_deriv(orientation_quat, X[..., 10:13])
        ang_accel = torch.zeros_like(ang_vel)
        
        return torch.cat((velocity, q_deriv, accel, ang_accel), dim=-1)
    
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
    aero = Aero(drag_coeff=config.drag_coeff, air_density = config.air_density, nose_cone_config=config.nose_cone, body_tube_config=config.body_tube)
    
    return Atlas(config.vehicle_mass, motor=motor, aero=aero)
    
    