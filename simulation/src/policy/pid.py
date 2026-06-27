import torch

from policy.base_policy import Policy
from utils.math import quat_mul, quat_inv, normalize_quat

class PID(Policy):
    def __init__(self, Kp=1.8, Ki=0.01, Kd=0.3, dt=1.0/30.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.dt = dt
        
        self.i_limit = 30
        
        # Need batch dim to initialize
        self.i_accum = None
        self.i_accum_initialized = False
    
    def reset(self):
        self.i_accum = None
        self.i_accum_initialized = False
    
    def forward(self, X, setpoint):
        q  = normalize_quat(X[..., 3:7])
        w_b = X[..., 10:13]
        setpoint = normalize_quat(setpoint.expand(*q.shape[:-1], 4))
        

        q_error = quat_mul(quat_inv(q), setpoint)                       # body-frame error  (#2)
        q_error = q_error * torch.where(q_error[..., 3:4] < 0, -1., 1.) # shortest path     (#1)
        error   = q_error[..., :2]

        if not self.i_accum_initialized:
            self.i_accum = torch.zeros_like(error)
            self.i_accum_initialized = True

        self.i_accum = torch.clamp(self.i_accum + error.detach() * self.dt,
                                -self.i_limit, self.i_limit)         # out-of-place + windup (#3,#4)

        return self.Kp * error + self.Ki * self.i_accum - self.Kd * w_b[..., :2]

        
            