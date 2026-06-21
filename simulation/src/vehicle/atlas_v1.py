import torch
import numpy as np

from vehicle.base_vehicle import BaseVehicle
from vehicle.components.motor import Motor
from utils.math import quat_deriv

G = 9.80665 # todo, move to config file 

class Atlas(BaseVehicle):
    def __init__(self, vehicle_mass: float = 1.0, vehicle_radius: float = 0.09, drag_coeff: float = 0.35, air_density: float = 1.225):
        super().__init__()
        
        self.vehicle_mass = vehicle_mass
        self.vehicle_radius = vehicle_radius
        self.drag_coeff = drag_coeff
        self.air_density = air_density
        
        self.motor = Motor()
        
    def dynamics(self, X: torch.tensor, U: torch.tensor, t: torch.tensor):
        thrust = self.motor.get_thrust(t) # scalar
        motor_mass = self.motor.get_mass(t)
        
        total_mass = motor_mass + self.vehicle_mass        
        position = X[0:3]
        velocity = X[7:10]
        
        F_drag = self.get_drag(velocity)
        
        zero = torch.zeros(1)
        F_thrust = torch.cat((zero, zero, thrust.unsqueeze(0)))
        F_grav = torch.tensor([0, 0, -total_mass * G])
        
        
        accel = (F_thrust + F_grav + F_drag) / total_mass
        
        # If rocket is sitting on the ground
        if torch.all(position <= torch.zeros(3)) and torch.all(velocity <= torch.zeros(3)):
            accel = torch.clamp(accel, min=0.0)
        
        q_deriv = quat_deriv(X[3:7], X[10:13])
        ang_accel = torch.zeros(3)
        
        return torch.cat((velocity, q_deriv, accel, ang_accel))
    
    # For plotting for now
    def get_current_mass(self, t):
        motor_mass = self.motor.get_mass(t)
        return motor_mass + self.vehicle_mass
        
    def get_drag(self, velocity: torch.tensor) -> torch.tensor:
        speed = torch.norm(velocity)
        unit_vel = velocity / (speed + 1e-8)
        
        area = np.pi * self.vehicle_radius**2
        
        return -0.5 * self.air_density * speed**2 * self.drag_coeff * area * unit_vel
        