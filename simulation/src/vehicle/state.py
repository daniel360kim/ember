import torch

from dataclasses import dataclass

class S:
    POS = slice(0, 3)
    ORI = slice(3, 7)
    VEL = slice(7, 10)
    ANG_VEL = slice(10, 13)
    GIMBAL_ANGLE = slice(13, 15)
@dataclass
class State:
    position: torch.tensor # (3,)
    orientation_quat: torch.tensor # (4, )
    velocity: torch.tensor # (3, )
    angular_velocity: torch.tensor # (3, )
    gimbal_angle: torch.tensor # (2, )
    
    extras: dict[str, torch.tensor]
    
    def to_tensor(self) -> torch.tensor:
        return torch.stack([self.position, self.orientation_quat, self.velocity, self.angular_velocity, self.gimbal_angle], dim=-1).unsqueeze(0)
    
    @classmethod
    def from_tensor(cls, X: torch.tensor, extras: dict[str, torch.tensor]):
        return cls(
            position = X[..., 0:3],
            orientation_quat = X[..., 3:7],
            velocity = X[..., 7:10],
            angular_velocity = X[..., 10:13],
            gimbal_angle = X[..., 13:15],
            extras = extras,
        )