from base_vehicle import BaseVehicle
import torch
class Apex(BaseVehicle):
    def __init__(self):
        super().__init__()
        
    def dynamics(X: torch.tensor, U, t: float):
        pass