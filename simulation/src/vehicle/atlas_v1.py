import torch
import numpy as np

from vehicle.base_vehicle import BaseVehicle
from vehicle.components.cartesian_gimbal import CartesianGimbal
from vehicle.components.motor import Motor
from vehicle.components.aero import Aero
from vehicle.config import VehicleConfig, MomentInertiaConfig, LocationConfig
from utils.math import quat_deriv, quat_rotate, quat_inv

G = 9.80665 # todo, move to config file 

class Atlas(BaseVehicle):
    def __init__(self, vehicle_mass: float, gimbal: CartesianGimbal, aero: Aero, mmoi: MomentInertiaConfig, cg_wet: LocationConfig, cg_dry: LocationConfig):
        super().__init__()
        
        self.vehicle_mass = vehicle_mass
        self.gimbal = gimbal
        self.aero = aero
        
        self.I = torch.diag(torch.tensor([mmoi.Ixx, mmoi.Iyy, mmoi.Izz]))
        self.I_inv = torch.linalg.inv(self.I)
        
        self.cg = torch.tensor([cg_wet.x, cg_wet.y, cg_wet.z])
        
        
    def dynamics(self, X: torch.tensor, U: torch.tensor, t: torch.tensor):
        motor_mass = self.gimbal.motor.get_mass(t)
        
        total_mass = motor_mass + self.vehicle_mass        
        position = X[..., 0:3]
        velocity = X[..., 7:10]
        orientation_quat = X[..., 3:7]
        ang_vel = X[..., 10:13]
        gimbal_angle = X[..., 13:15]
        
        aero_wrench = self.aero.get_wrench(X, t)
        gimbal_wrench = self.gimbal.get_wrench(X, t)
        
        wrenches = [aero_wrench, gimbal_wrench]
        
        # Net force
        F_grav = torch.cat([torch.zeros_like(total_mass), torch.zeros_like(total_mass), -total_mass * G], dim=-1)
        F_net = F_grav + sum(wrench.force_world for wrench in wrenches)
        
        # Net torque
        torque = sum(
            torch.cross(wrench.application_point_body - self.cg,
                        quat_rotate(quat_inv(orientation_quat), wrench.force_world),
                        dim=-1)
            for wrench in wrenches if wrench.application_point_body is not None
        ) + sum(w.moment_body for w in wrenches if w.moment_body is not None)

        accel = F_net / total_mass
        
        ang_accel = self._get_ang_accel(angular_vel=ang_vel, net_torque=torque)
        
        # If rocket is sitting on the ground
        accel_clamped = torch.clamp(accel[..., 2], min=0.0)
        on_ground = (position[..., 2] <= 0) & (velocity[..., 2] <= 0)
        accel_z_clamped = torch.where(on_ground, accel_clamped, accel[..., 2])
        
        accel[..., 2] = accel_z_clamped
        
        q_deriv = quat_deriv(orientation_quat, X[..., 10:13])
        
        # Gimbal dynamics
        gimbal_delta = self.gimbal.dynamics(gimbal_state=gimbal_angle, gimbal_cmd=gimbal_angle)
        
        return torch.cat((velocity, q_deriv, accel, ang_accel, gimbal_delta), dim=-1)
    
    def _get_ang_accel(self, angular_vel: torch.tensor, net_torque: torch.tensor) -> torch.tensor:
        I_omega = angular_vel @ self.I.T
        gyro = torch.cross(angular_vel, I_omega, dim=-1)
        net_torque = net_torque - gyro
        return net_torque @ self.I_inv.T

    def get_extras(self, t):
        motor_mass = self.gimbal.motor.get_mass(t)
        total_mass = motor_mass + self.vehicle_mass
        
        thrust = self.gimbal.motor.get_thrust(t)
        
        return {
            "total_mass": total_mass,
            "thrust": thrust
        }
        
def build_vehicle(config: VehicleConfig) -> Atlas:
    motor = Motor(name="F15", total_impulse=config.motor.total_impulse, total_mass=config.motor.total_mass, prop_mass=config.motor.propellant_mass) # TODO allow for different motors from a registry
    gimbal = CartesianGimbal(gimbal_config=config.gimbal_config, motor=motor)
    aero = Aero(aero=config.aero,
                nose_cone_config=config.nose_cone, body_tube_config=config.body_tube, 
                cp=config.cp, mmoi=config.mmoi)
    
    return Atlas(config.vehicle_mass, gimbal=gimbal, aero=aero, mmoi=config.mmoi, cg_wet=config.cg_wet, cg_dry=config.cg_dry)
    
    