import torch

class Motor:
    def __init__(self, name: str = 'F15', total_impulse: float = 49.6, total_mass: float = 0.103, prop_mass: float = 0.060):
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
        
        c1 = (t >= 0) & (t < 0.386)
        c2 = (t >= 0.386) & (t < 3.25)
        c3 = (t >= 3.2) & (t <= 3.4)
        
        p1 =  -(44919 * t**5) + (60753 * t**4) - (30010 * t**3) + (6689.8 * t**2) - (588.2 * t) + 19.224
        p2 = (1.203 * t**6) - (15.6 * t**5) + (81.307 * t**4) - (216.29 * t**3) + (306.73 * t**2) - (218.82 * t) + 76.536
        p3 = -(11750 * t**2) + (79053 * t) - 132949
        
        thrust = torch.where(c1, p1, thrust)
        thrust = torch.where(c2, p2, thrust)
        thrust = torch.where(c3, p3, thrust)
        
        return thrust
    
    def get_mass(self, impulse_cum: torch.tensor) -> torch.tensor:
        """
        Calculate the current mass of the rocket given the current cum. impulse(t)
        
        Args:
            impulse_cum: the cumulative impulse at time t
        
        Output: the current rocket motor mass
        """
        fraction_burned = torch.clamp(input=impulse_cum, min=0.0, max=1.0)
        m_prop_remaining = self.prop_mass * (1 - fraction_burned)
        
        return (self.total_mass - self.prop_mass)  + m_prop_remaining
    
        
        



        
    