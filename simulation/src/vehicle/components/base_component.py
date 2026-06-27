from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class Wrench:
    """
    Each component returns a wrench that describes how forces/torques are applied to the rocket
    """
    
    force_world: torch.Tensor # (B, 3)
    # To calculate torque, or None if no torque applied by the force
    application_point_body: Optional[torch.Tensor] # (B,3) body-fixed offset (for lever arm)
    moment_body: Optional[torch.Tensor] # Direct moment (damping, gyroscopic) - no lever arm
    
    

class Component(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def get_wrench(self, X: torch.tensor, t: torch.tensor) -> Wrench:
        pass
    
    