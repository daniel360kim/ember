import torch

def quat_identity():
    return torch.tensor([0., 0., 0., 1.])

def quat_inv(q):
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)

def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    
    x_prod = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_prod = w1 * y2  - x1 * z2 + y1 * w2 + z1 * x2
    z_prod = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w_prod = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    
    return torch.stack([x_prod, y_prod, z_prod, w_prod], dim=-1)
    
def quat_rotate(q, v):
    q_vec = q[..., :3]
    qw = q[..., 3:]
    return v + 2.0 * qw * torch.cross(q_vec, v, dim=-1) + 2.0 * torch.cross(q_vec, torch.cross(q_vec, v, dim=-1), dim=-1)

def quat_to_euler(q):
    """Convert quaternion to roll/pitch/yaw in radians"""
    x, y, z, w = q.unbind(-1)

    roll  = torch.atan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = torch.asin( 2*(w*y - z*x).clamp(-1, 1))
    yaw   = torch.atan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))

    return torch.stack([roll, pitch, yaw], dim=-1)

    
def quat_deriv(q, omega):
    """Assumes omega in body frame"""
    
    # q: [x, y, z, w], omega: body-frame angular velocity [3]
    omega_pure = torch.cat([omega, torch.zeros(1, dtype=omega.dtype)])
    return 0.5 * quat_mul(q, omega_pure)

