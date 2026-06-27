import torch

from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def forward(self, X: torch.tensor, setpoint: torch.tensor) -> torch.tensor:
        pass