
class Motor:
    def __init__(self, name: str = 'F15'):
        self.name = name
        
    def get_thrust(self, t: float) -> float:
        """
        Return the thrust in Newtons for a given timepoint. Uses interpolation from:
        https://onedrive.live.com/:x:/g/personal/445af9a6dabdb301/IQAtde1O-7DkSL_Rftf_KEzSAcRCM8Dv1sPfJ4oeYJBMV3Y?rtime=J2_XV_7O3kg&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3gvYy80NDVhZjlhNmRhYmRiMzAxL0lRQXRkZTFPLTdEa1NMX1JmdGZfS0V6U0FjUkNNOER2MXNQZko0b2VZSkJNVjNZP2U9c2Jib3ox
        
        Args:
            t: float the time in seconds
            
        Output: the thrust of the rocket motor in seconds
        """
        if t >= 0 and t < 0.425:
            return -(44919 * t**5) + (60753 * t**4) - (30010 * t**3) + (6689.8 * t**2) - (588.2 * t) + 19.224
        elif t >= 0.425 and t < 3.25:
            return (1.203 * t**6) - (15.6 * t**5) + (81.307 * t**4) - (216.29 * t**3) + (306.73 * t**2) - (218.82 * t) + 76.536
        elif t >= 3.25 and t <= 3.4:
            return -(11750 * t**2) + (79053 * t) - 132949
        
        return 0.0
        



        
    