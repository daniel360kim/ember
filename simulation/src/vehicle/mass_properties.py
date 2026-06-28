import torch

from vehicle.components.motor import Motor
from vehicle.config import LocationConfig, MomentInertiaConfig

G = 9.80665 # todo, move to config file

class MassProperties:
    """Combines two rigid bodies into the vehicle's time-varying mass properties:

      - the airframe (everything except the motor): constant mass, CG and inertia,
        measured directly with the motor removed.
      - the motor (casing + propellant): a time-varying body synthesized from the
        motor geometry (see Motor.get_cg / Motor.get_mmoi).

    Each body's inertia is given about its own CG and shifted to the combined CG
    with the parallel axis theorem every timestep.
    """

    def __init__(self, mass_airframe: float, motor: Motor, cg_airframe: LocationConfig, mmoi_airframe: MomentInertiaConfig):
        self.mass_airframe = mass_airframe
        self.motor = motor

        # Airframe inertia about the airframe CG (motor removed)
        self.mmoi_airframe = torch.diag(torch.tensor([mmoi_airframe.Ixx, mmoi_airframe.Iyy, mmoi_airframe.Izz]))
        self.cg_airframe = torch.tensor([cg_airframe.x, cg_airframe.y, cg_airframe.z])

    def get_dynamics(self, t: torch.Tensor):
        mass_motor_current = self.motor.get_mass(t)
        mass_total_current = self.mass_airframe + mass_motor_current
        force_gravity = torch.cat([torch.zeros_like(mass_total_current), torch.zeros_like(mass_total_current), -mass_total_current * G], dim=-1)

        cg = self.get_cg(t)
        mmoi = self.get_mmoi(cg, t)
        return force_gravity, mass_total_current, cg, mmoi

    def get_cg(self, t: torch.Tensor) -> torch.Tensor:
        cg_airframe = self.cg_airframe.expand(*t.shape[:-1], 3)

        mass_motor_current = self.motor.get_mass(t)            # (B, 1)
        cg_motor = self.motor.get_cg(t)                        # (B, 3)
        mass_total_current = self.mass_airframe + mass_motor_current
        return (self.mass_airframe * cg_airframe + mass_motor_current * cg_motor) / mass_total_current

    def get_mmoi(self, cg: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r_airframe = self.cg_airframe - cg                     # (B, 3)
        mmoi_airframe = self.mmoi_airframe + self.mass_airframe * self._get_parallel_shift_axis(r_airframe)

        r_motor = self.motor.get_cg(t) - cg                    # (B, 3)
        mass_motor_current = self.motor.get_mass(t)[..., None] # (B, 1, 1) to broadcast over (B, 3, 3)
        mmoi_motor = self.motor.get_mmoi(t) + mass_motor_current * self._get_parallel_shift_axis(r_motor)

        return mmoi_airframe + mmoi_motor

    def _get_parallel_shift_axis(self, r: torch.Tensor) -> torch.Tensor:
        """
        Parallel axis theorem shift tensor for a lever arm r: |r|^2 * E - r (outer) r

        Args:
            r (tensor): lever arm from a body's own CG to the combined CG (B, 3)

        Returns the shift tensor (B, 3, 3)
        """
        E = torch.eye(3)
        r2 = r.pow(2).sum(-1)[..., None, None] # (B, 1, 1)
        return r2 * E - torch.einsum('bi,bj->bij', r, r)
