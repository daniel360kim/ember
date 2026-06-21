import torch
import numpy as np

from vehicle.base_vehicle import BaseVehicle
from vehicle.components.motor import Motor
from vehicle.components.aero import Aero
from vehicle.config import VehicleConfig
from utils.math import quat_deriv

G = 9.80665 # todo, move to config file 

class Atlas(BaseVehicle):
    def __init__(self, vehicle_mass: float, motor: Motor, aero: Aero):
        super().__init__()
        
        self.vehicle_mass = vehicle_mass
        self.motor = motor
        self.aero = aero
        
        
    def dynamics(self, X: torch.tensor, U: torch.tensor, t: torch.tensor):
        thrust = self.motor.get_thrust(t) # scalar
        motor_mass = self.motor.get_mass(t)
        
        total_mass = motor_mass + self.vehicle_mass        
        position = X[0:3]
        velocity = X[7:10]
        
        F_drag = self.aero.get_drag_force(X, t)
        
        zero = torch.zeros(1)
        F_thrust = torch.cat((zero, zero, thrust.unsqueeze(0)))
        F_grav = torch.tensor([0, 0, -total_mass * G])
        
        accel = (F_thrust + F_grav + F_drag) / total_mass
        
        # If rocket is sitting on the ground
        accel_clamped = torch.clamp(accel, min=0.0)
        on_ground = (position[2] <= 0) & (velocity[2] <= 0)
        accel = torch.where(on_ground, accel_clamped, accel)
        
        q_deriv = quat_deriv(X[3:7], X[10:13])
        ang_accel = torch.zeros(3)
        
        return torch.cat((velocity, q_deriv, accel, ang_accel))
    
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
    
    