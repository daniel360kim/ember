import torch

class Motor:
    def __init__(self, name: str = 'F15', total_impulse: float = 48.792775090871146, total_mass: float = 0.103, prop_mass: float = 0.060):
        self.name = name
        self.total_impulse = total_impulse
        self.total_mass = total_mass
        self.prop_mass = prop_mass
        
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
        impulse_cum = torch.where(t >= 3.4, torch.full_like(t, self.total_impulse), impulse_cum)

        return impulse_cum.to(orig_dtype)

    def get_mass(self, t: torch.tensor) -> torch.tensor:
        """
        Calculate the current mass of the rocket motor at time t.

        Args:
            t: float the time in seconds

        Output: the current rocket motor mass
        """
        impulse_cum = self.get_cumulative_impulse(t)
        fraction_burned = torch.clamp(input=impulse_cum / self.total_impulse, min=0.0, max=1.0)
        m_prop_remaining = self.prop_mass * (1 - fraction_burned)

        return (self.total_mass - self.prop_mass) + m_prop_remaining
    
        
        



        
    