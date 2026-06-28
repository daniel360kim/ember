import torch
import numpy as np

from vehicle.base_vehicle import BaseVehicle
from vehicle.components.cartesian_gimbal import CartesianGimbal
from vehicle.components.motor import Motor
from vehicle.components.aero import Aero
from vehicle.config import VehicleConfig
from vehicle.state import S
from vehicle.mass_properties import MassProperties
from utils.math import quat_deriv, quat_rotate, quat_inv



class Atlas(BaseVehicle):
    def __init__(self, mass_properties: MassProperties, gimbal: CartesianGimbal, aero: Aero):
        super().__init__()
        
        self.gimbal = gimbal
        self.aero = aero
        
        self.mass_properties = mass_properties
        
    def dynamics(self, X: torch.tensor, U: torch.tensor, t: torch.tensor):
        position = X[..., S.POS]
        velocity = X[..., S.VEL]
        orientation_quat = X[..., S.ORI]
        ang_vel = X[..., S.ANG_VEL]
        gimbal_angle = X[..., S.GIMBAL_ANGLE]
        
        aero_wrench = self.aero.get_wrench(X, t)
        gimbal_wrench = self.gimbal.get_wrench(X, t)
        
        wrenches = [aero_wrench, gimbal_wrench]
        force_gravity, mass, cg, mmoi = self.mass_properties.get_dynamics(t)
        # Net force
        F_net = force_gravity + sum(wrench.force_world for wrench in wrenches)
        
        # Net torque
        torque = sum(
            torch.cross(wrench.application_point_body - cg,
                        quat_rotate(quat_inv(orientation_quat), wrench.force_world),
                        dim=-1)
            for wrench in wrenches if wrench.application_point_body is not None
        ) + sum(w.moment_body for w in wrenches if w.moment_body is not None)

        accel = F_net / mass
        
        ang_accel = self._get_ang_accel(angular_vel=ang_vel, net_torque=torque, mmoi=mmoi)
        
        # If rocket is sitting on the ground
        accel_clamped = torch.clamp(accel[..., 2], min=0.0)
        on_ground = (position[..., 2] <= 0) & (velocity[..., 2] <= 0)
        accel_z_clamped = torch.where(on_ground, accel_clamped, accel[..., 2])
        
        accel[..., 2] = accel_z_clamped
        
        q_deriv = quat_deriv(orientation_quat, X[..., 10:13])
        
        # Gimbal dynamics
        gimbal_delta = self.gimbal.dynamics(gimbal_state=gimbal_angle, torque_cmd=U, cg=cg, t=t)
        
        return torch.cat((velocity, q_deriv, accel, ang_accel, gimbal_delta), dim=-1)
    
    def _get_ang_accel(self, angular_vel: torch.tensor, net_torque: torch.tensor, mmoi: torch.Tensor) -> torch.tensor:
        angular_vel = angular_vel.unsqueeze(-2) # (B, 1, 3)
        net_torque = net_torque.unsqueeze(-2) # (B, 1, 3)
        I_omega = torch.matmul(angular_vel, mmoi.transpose(-1, -2))
        gyro = torch.cross(angular_vel, I_omega, dim=-1)
        net_torque = net_torque - gyro
        return torch.matmul(net_torque, torch.linalg.inv(mmoi).transpose(-1, -2)).squeeze(-2)
    
    def get_extras(self, t):
        thrust = self.gimbal.motor.get_thrust(t)
        _, mass, cg, mmoi = self.mass_properties.get_dynamics(t)
        return {
            "total_mass": mass,
            "thrust": thrust,
            "cg": cg,
            "mmoi": mmoi,
        }
        
def build_vehicle(config: VehicleConfig) -> Atlas:
    motor = Motor(config.motor)
    gimbal = CartesianGimbal(gimbal_config=config.gimbal_config, motor=motor)
    aero = Aero(aero=config.aero,
                nose_cone_config=config.nose_cone, body_tube_config=config.body_tube, 
                cp=config.cp)
    mass_properties = MassProperties(mass_airframe=config.vehicle_mass, motor=motor, cg_airframe=config.cg_airframe, mmoi_airframe=config.mmoi_airframe)
    return Atlas(mass_properties=mass_properties, gimbal=gimbal, aero=aero)
    
    