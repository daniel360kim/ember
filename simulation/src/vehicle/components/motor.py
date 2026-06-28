import torch

from vehicle.config import MotorConfig

class Motor:
    def __init__(self, config: MotorConfig):
        self.config = config
        
        # Get mmoi of the casing itself
        r_inner = config.propellant_radius
        r_outer = config.total_radius
        self.casing_mass = config.total_mass - config.propellant_mass
        Izz = 0.5 * self.casing_mass * (r_inner**2 + r_outer**2)
        Ixx = 0.5 * Izz + (1.0 / 12.0) * self.casing_mass * config.length**2
        
        self.casing_mmoi = torch.diag(torch.tensor([Ixx, Ixx, Izz])) # (3,3)

        # Casing and propellant are concentric and share the motor length, so both
        # are centered at the motor location. The motor location is therefore the
        # CG of the whole motor (casing + propellant) and stays fixed as it burns.
        self.location = torch.tensor([config.motor_location.x, config.motor_location.y, config.motor_location.z])


    def get_thrust(self, t: torch.tensor) -> torch.tensor:
        """
        Return the thrust in Newtons for a given timepoint. Uses interpolation from:
        https://onedrive.live.com/:x:/g/personal/445af9a6dabdb301/IQAtde1O-7DkSL_Rftf_KEzSAcRCM8Dv1sPfJ4oeYJBMV3Y?rtime=J2_XV_7O3kg&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3gvYy80NDVhZjlhNmRhYmRiMzAxL0lRQXRkZTFPLTdEa1NMX1JmdGZfS0V6U0FjUkNNOER2MXNQZko0b2VZSkJNVjNZP2U9c2Jib3ox
        
        Args:
            t: float the time in seconds
            
        Output: the thrust of the rocket motor in seconds
        """
        thrust = torch.zeros_like(t)
        
        c1 = (t >= 0.063) & (t < 0.386)
        c2 = (t >= 0.386) & (t < 3.35)
        c3 = (t >= 3.35) & (t <= 3.4)
        
        p1 =  -(44919 * t**5) + (60753 * t**4) - (30010 * t**3) + (6689.8 * t**2) - (588.2 * t) + 19.224
        p2 = (1.203 * t**6) - (15.6 * t**5) + (81.307 * t**4) - (216.29 * t**3) + (306.73 * t**2) - (218.82 * t) + 76.536
        p3 = -(11750 * t**2) + (79053 * t) - 132949
        
        thrust = torch.where(c1, p1, thrust)
        thrust = torch.where(c2, p2, thrust)
        thrust = torch.where(c3, p3, thrust)
        
        return thrust
    
    def get_cumulative_impulse(self, t: torch.tensor) -> torch.tensor:
        """
        Analytically integrate the piecewise thrust curve to get cumulative impulse(t).
        Each segment of get_thrust is a polynomial in t, so its antiderivative is exact
        and evaluated directly rather than integrated numerically.

        Args:
            t: float the time in seconds

        Output: the cumulative impulse delivered by the motor up to time t, in N*s
        """
        # Segment 3's antiderivative (~ -149040 at t=3.4) is added to offset3 (~ +149090)
        # to land on ~48 N*s. In float32 (~7 significant digits) that subtraction loses
        # all precision and makes the curve wobble, so do the algebra in float64 and cast
        # back to the caller's dtype at the end.
        orig_dtype = t.dtype
        t = t.to(torch.float64)

        # Antiderivatives of the thrust polynomials from get_thrust, each referenced to t=0
        P1 = -7486.5 * t**6 + 12150.6 * t**5 - 7502.5 * t**4 + 2229.93333333333 * t**3 - 294.1 * t**2 + 19.224 * t
        P2 = 0.171857142857143 * t**7 - 2.6 * t**6 + 16.2614 * t**5 - 54.0725 * t**4 + 102.243333333333 * t**3 - 109.41 * t**2 + 76.536 * t
        P3 = -11750 * t**3 / 3 + 79053 * t**2 / 2 - 132949 * t

        # The thrust polynomial p1 is nonzero at t=0, but the motor does not ignite
        # until t=0.063. Anchor segment 1 to zero impulse at ignition by removing the
        # spurious area P1 accrues over [0, 0.063].
        offset1 = -0.49482126755758693

        # Offsets so each segment's antiderivative is continuous with the cumulative
        # impulse accrued by the end of the previous segment
        offset2 = -13.89421761117859
        offset3 = 149089.7194417575

        c1 = (t >= 0.063) & (t < 0.386)
        c2 = (t >= 0.386) & (t < 3.35)
        c3 = (t >= 3.35) & (t <= 3.4)

        impulse_cum = torch.zeros_like(t)
        impulse_cum = torch.where(c1, P1 + offset1, impulse_cum)
        impulse_cum = torch.where(c2, P2 + offset2, impulse_cum)
        impulse_cum = torch.where(c3, P3 + offset3, impulse_cum)
        impulse_cum = torch.where(t >= 3.4, torch.full_like(t, self.config.total_impulse), impulse_cum)

        return impulse_cum.to(orig_dtype)

    def get_fraction_burned(self, t: torch.tensor):
        impulse_cum = self.get_cumulative_impulse(t)
        return torch.clamp(input=impulse_cum / self.config.total_impulse, min=0.0, max=1.0)
    
    def get_prop_mass(self, t: torch.tensor) -> torch.Tensor:
        fraction_burned = self.get_fraction_burned(t)
        return self.config.propellant_mass * (1 - fraction_burned)
    
    def get_motor_mass(self, t: torch.Tensor) -> torch.Tensor:
        prop_mass = self.get_prop_mass(t)
        return prop_mass + self.casing_mass
    
    def _get_prop_inner_radius(self, t: torch.Tensor) -> torch.Tensor:
        """
        Propellant burns from the inside out. Get's how much the radius from inside to the outside of 
        propellant has burned at time t
        
        Args:
            t (tensor): the current time (B, 1)
            
        Returns: a tensor representing the radius of the propellant at time t in meters (B, 1)
        """
        fraction_burned = self.get_fraction_burned(t)
        # Volume burned is prop to fraction burned
        return torch.sqrt(fraction_burned) * self.config.propellant_radius
    
    def get_propellant_mmoi(self, t: torch.Tensor) -> torch.Tensor:
        prop_mass = self.get_prop_mass(t) # (B, 1)
        inner_radius = self._get_prop_inner_radius(t) # (B, 1)
        
        
        # Izz = 1/2m(t)(r_inner(t)^2 + r_outer^2)
        Izz = 0.5 * prop_mass * (inner_radius**2 + self.config.propellant_radius**2)
        
        Ixx = 0.5 * Izz + (1.0 / 12.0) * prop_mass * self.config.length**2 # (B, 1)
        
        return torch.diag_embed(torch.cat([Ixx, Ixx, Izz], dim=-1))

    def get_cg(self, t: torch.Tensor) -> torch.Tensor:
        """CG of the whole motor (casing + propellant) in body coordinates, (B, 3).

        Casing and propellant are concentric and co-length, both centered at the
        motor location, so the motor CG is fixed at that location as it burns.
        """
        return self.location.expand(*t.shape[:-1], 3)

    def get_mmoi(self, t: torch.Tensor) -> torch.Tensor:
        """Inertia tensor of the whole motor about its own CG, (B, 3, 3).

        Casing and propellant share the motor CG, so each component's inertia is
        already referenced to that point and they sum directly (no internal shift).
        """
        return self.casing_mmoi + self.get_propellant_mmoi(t)


    def get_mass(self, t: torch.tensor) -> torch.tensor:
        """
        Calculate the current mass of the rocket motor at time t.

        Args:
            t: float the time in seconds

        Output: the current rocket motor mass
        """
        m_prop_remaining = self.get_prop_mass(t)

        return (self.config.total_mass - self.config.propellant_mass) + m_prop_remaining
    
    
        
        



        
    