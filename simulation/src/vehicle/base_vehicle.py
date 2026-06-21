from abc import ABC, abstractmethod
import torch

from vehicle.state import State

class BaseVehicle(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def dynamics(self, X: torch.tensor, U, t: float) -> torch.tensor:
        pass
    
    @abstractmethod
    def get_mass(self, t: torch.tensor) -> torch.tensor:
        pass
    
    def get_state(self, X: torch.tensor, t: torch.tensor):
        mass = self.get_mass(t)
        return State.from_tensor(X, mass)
        
        
    
    
    