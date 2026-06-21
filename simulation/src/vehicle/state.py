from dataclasses import dataclass
import torch

@dataclass
class State:
    position: torch.tensor # (3,)
    orientation_quat: torch.tensor # (4, )
    velocity: torch.tensor # (3, )
    angular_velocity: torch.tensor # (3, )
    
    extras: dict[str, torch.tensor]
    
    def to_tensor(self) -> torch.tensor:
        return torch.stack([self.position, self.orientation_quat, self.velocity, self.angular_velocity], dim=-1)
    
    @classmethod
    def from_tensor(cls, X: torch.tensor, extras: dict[str, torch.tensor]):
        return cls(
            position = X[0:3],
            orientation_quat = X[3:7],
            velocity = X[7:10],
            angular_velocity = X[10:13],
            extras = extras,
        )