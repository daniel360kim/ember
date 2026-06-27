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
    def __init__(self, mass_airframe: float, gimbal: CartesianGimbal, aero: Aero, mmoi: MomentInertiaConfig, cg_wet: LocationConfig, cg_dry: LocationConfig):
        super().__init__()
        
        self.mass_airframe = mass_airframe
        self.gimbal = gimbal
        self.aero = aero
        
        self.I = torch.diag(torch.tensor([mmoi.Ixx, mmoi.Iyy, mmoi.Izz]))
        self.I_inv = torch.linalg.inv(self.I)
        
        self.cg_wet = torch.tensor([cg_wet.x, cg_wet.y, cg_wet.z])
        self.cg_dry = torch.tensor([cg_dry.x, cg_dry.y, cg_dry.z])
        
        mass_motor_dry =  self.gimbal.motor.total_mass - self.gimbal.motor.prop_mass
        self.mass_dry_total = mass_airframe + mass_motor_dry # Vehicle + motor casing (no prop)
        self.mass_wet_total = self.mass_dry_total + self.gimbal.motor.prop_mass # Everyhting including prop
        
        # Center of gravity of the propellant
        self.cg_prop = (self.mass_wet_total * self.cg_wet - self.mass_dry_total * self.cg_dry) / self.gimbal.motor.prop_mass # (3)
        
    def dynamics(self, X: torch.tensor, U: torch.tensor, t: torch.tensor):
        mass_motor_current = self.gimbal.motor.get_mass(t)
        
        mass_total_current = mass_motor_current + self.mass_airframe  
        position = X[..., 0:3]
        velocity = X[..., 7:10]
        orientation_quat = X[..., 3:7]
        ang_vel = X[..., 10:13]
        gimbal_angle = X[..., 13:15]
        
        cg = self.get_cg(mass_motor_current, t)
        aero_wrench = self.aero.get_wrench(X, t)
        gimbal_wrench = self.gimbal.get_wrench(X, t)
        
        wrenches = [aero_wrench, gimbal_wrench]
        
        # Net force
        F_grav = torch.cat([torch.zeros_like(mass_total_current), torch.zeros_like(mass_total_current), -mass_total_current * G], dim=-1)
        F_net = F_grav + sum(wrench.force_world for wrench in wrenches)
        
        # Net torque
        torque = sum(
            torch.cross(wrench.application_point_body - cg,
                        quat_rotate(quat_inv(orientation_quat), wrench.force_world),
                        dim=-1)
            for wrench in wrenches if wrench.application_point_body is not None
        ) + sum(w.moment_body for w in wrenches if w.moment_body is not None)

        accel = F_net / mass_total_current
        
        ang_accel = self._get_ang_accel(angular_vel=ang_vel, net_torque=torque)
        
        # If rocket is sitting on the ground
        accel_clamped = torch.clamp(accel[..., 2], min=0.0)
        on_ground = (position[..., 2] <= 0) & (velocity[..., 2] <= 0)
        accel_z_clamped = torch.where(on_ground, accel_clamped, accel[..., 2])
        
        accel[..., 2] = accel_z_clamped
        
        q_deriv = quat_deriv(orientation_quat, X[..., 10:13])
        
        # Gimbal dynamics
        gimbal_delta = self.gimbal.dynamics(gimbal_state=gimbal_angle, torque_cmd=U, cg=cg, t=t)
        
        return torch.cat((velocity, q_deriv, accel, ang_accel, gimbal_delta), dim=-1)
    
    def _get_ang_accel(self, angular_vel: torch.tensor, net_torque: torch.tensor) -> torch.tensor:
        I_omega = angular_vel @ self.I.T
        gyro = torch.cross(angular_vel, I_omega, dim=-1)
        net_torque = net_torque - gyro
        return net_torque @ self.I_inv.T
    
    def get_cg(self, mass_motor_current: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cg_dry = self.cg_dry.expand(*t.shape[:-1], 3)
        
        mass_prop_current = self.gimbal.motor.get_prop_mass(t)
        
        mass_total_current = mass_motor_current + self.mass_airframe
        return (self.mass_dry_total * cg_dry + mass_prop_current * self.cg_prop) / mass_total_current

    def get_extras(self, t):
        mass_motor_current = self.gimbal.motor.get_mass(t)
        mass_total_current = mass_motor_current + self.mass_airframe
        
        thrust = self.gimbal.motor.get_thrust(t)
        
        cg = self.get_cg(mass_motor_current, t)
        
        return {
            "total_mass": mass_total_current,
            "thrust": thrust,
            "cg": cg,
        }
        
def build_vehicle(config: VehicleConfig) -> Atlas:
    motor = Motor(name="F15", total_impulse=config.motor.total_impulse, total_mass=config.motor.total_mass, prop_mass=config.motor.propellant_mass) # TODO allow for different motors from a registry
    gimbal = CartesianGimbal(gimbal_config=config.gimbal_config, motor=motor)
    aero = Aero(aero=config.aero,
                nose_cone_config=config.nose_cone, body_tube_config=config.body_tube, 
                cp=config.cp, mmoi=config.mmoi)
    
    return Atlas(config.vehicle_mass, gimbal=gimbal, aero=aero, mmoi=config.mmoi, cg_wet=config.cg_wet, cg_dry=config.cg_dry)
    
    