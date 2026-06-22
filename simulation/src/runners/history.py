import numpy as np
import torch

from vehicle.state import State
from utils.math import quat_to_euler, quat_rotate

class SimulationHistory:
    def __init__(self):
        self.states = []
        
    def add(self, state: State):
        self.states.append(state)
        
    def get_position_history(self):
        return np.stack([state.position.detach().cpu().numpy() for state in self.states])

    def get_velocity_history(self):
        return np.stack([state.velocity.detach().cpu().numpy() for state in self.states])
    
    def get_quaternion_history(self):
        return np.stack([state.orientation_quat.detach().cpu().numpy() for state in self.states])
    
    def get_orientation_euler_history(self):
        orientation_history = torch.stack([np.rad2deg(quat_to_euler(state.orientation_quat.detach().cpu())) for state in self.states])
        return orientation_history.numpy()

    def get_tilt_history(self):
        """Attitude as tilt-from-vertical (zenith) and heading, both in degrees.

        Unlike roll/pitch/yaw, this is free of the Euler gimbal-lock singularity
        at vertical, so it stays continuous as the rocket noses over through 90 deg.
        Returns an array of shape (..., 2): [zenith, heading].
        """
        quats = torch.stack([state.orientation_quat.detach().cpu() for state in self.states])
        up = torch.tensor([0., 0., 1.]).expand(*quats.shape[:-1], 3)
        nose = quat_rotate(quats, up)  # body axis expressed in the world frame

        zenith = torch.rad2deg(torch.arccos(nose[..., 2].clamp(-1.0, 1.0)))
        heading = torch.rad2deg(torch.atan2(nose[..., 1], nose[..., 0]))
        return torch.stack([zenith, heading], dim=-1).numpy()
    
    def get_angular_velocity_history(self):
        return np.stack([np.rad2deg(state.angular_velocity.detach().cpu().numpy()) for state in self.states])
    
    def get_extra_history(self, key: str):
        return np.stack([state.extras[key].detach().cpu().numpy() for state in self.states])

        
        
        
    