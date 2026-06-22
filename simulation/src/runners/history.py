import numpy as np
import torch

from vehicle.state import State
from utils.math import quat_to_euler

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
    
    def get_angular_velocity_history(self):
        return np.stack([np.rad2deg(state.angular_velocity.detach().cpu().numpy()) for state in self.states])
    
    def get_extra_history(self, key: str):
        return np.stack([state.extras[key].detach().cpu().numpy() for state in self.states])

        
        
        
    