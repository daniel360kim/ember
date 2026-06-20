from abc import ABC, abstractmethod
import torch

class BaseVehicle(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def dynamics(X: torch.tensor, U, t: float):
        pass
    
    
    