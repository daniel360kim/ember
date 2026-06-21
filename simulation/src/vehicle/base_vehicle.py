from abc import ABC, abstractmethod
import torch

from vehicle.state import State

class BaseVehicle(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def dynamics(self, X: torch.tensor, U, t: float) -> torch.tensor:
        pass
    
    # Extra parameters we want to report back (mass, thrust, etc.)
    def get_extras(self, t: torch.tensor) -> dict[str, torch.tensor]:
        return {}
    
    def get_state(self, X: torch.tensor, t: torch.tensor):
        extras = self.get_extras(t)
        return State.from_tensor(X, extras=extras)
        
        
    
    
    