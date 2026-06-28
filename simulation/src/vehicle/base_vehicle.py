from abc import ABC, abstractmethod
import torch

from vehicle.state import State, S
from utils.math import normalize_quat


class BaseVehicle(ABC):
    @abstractmethod
    def dynamics(self, X: torch.Tensor, U: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """Project an integrated state back onto its constraint manifold.

        The integrator only does arithmetic; renormalizing the orientation
        quaternion is vehicle (layout) knowledge, so it lives here. Returns a
        new tensor (no in-place write) so it is safe to backprop through.
        """
        normalized_quat = normalize_quat(X[..., S.ORI])
        return torch.cat([X[..., :S.ORI.start], normalized_quat, X[..., S.ORI.stop:]], dim=-1)

    # Extra parameters we want to report back (mass, thrust, etc.)
    def get_extras(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        return {}

    def get_state(self, X: torch.Tensor, t: torch.Tensor):
        extras = self.get_extras(t)
        return State.from_tensor(X, extras=extras)
        
        
    
    
    