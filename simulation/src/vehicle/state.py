import torch

from dataclasses import dataclass

class S:
    POS = slice(0, 3)
    ORI = slice(3, 7)
    VEL = slice(7, 10)
    ANG_VEL = slice(10, 13)
    GIMBAL_ANGLE = slice(13, 15)
    DIM = 15  # full state-vector width
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
            position = X[..., S.POS],
            orientation_quat = X[..., S.ORI],
            velocity = X[..., S.VEL],
            angular_velocity = X[..., S.ANG_VEL],
            gimbal_angle = X[..., S.GIMBAL_ANGLE],
            extras = extras,
        )