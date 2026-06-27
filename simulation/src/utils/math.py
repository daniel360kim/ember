import torch

def quat_identity():
    return torch.tensor([0., 0., 0., 1.])

def quat_inv(q):
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)

def normalize_quat(q):
    return q / q.norm(dim=-1, keepdim=True)

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
    """Convert quaternion to Euler angles (rotation about body X, Y, Z, in radians).

    Z is the rocket's long axis (points up through the nose cone). Singular
    when the Y-axis angle approaches +/-90 deg (see SimulationHistory.get_tilt_history
    for a singularity-free alternative).
    """
    x, y, z, w = q.unbind(-1)

    x_rot = torch.atan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    y_rot = torch.asin( 2*(w*y - z*x).clamp(-1, 1))
    z_rot = torch.atan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))

    return torch.stack([x_rot, y_rot, z_rot], dim=-1)


def euler_to_quat(euler):
    """Convert Euler angles (rotation about body X, Y, Z, in radians) to quaternion [x, y, z, w].

    Z is the rocket's long axis (points up through the nose cone).
    """
    x_rot, y_rot, z_rot = euler.unbind(-1)

    cr, sr = torch.cos(x_rot * 0.5), torch.sin(x_rot * 0.5)
    cp, sp = torch.cos(y_rot * 0.5), torch.sin(y_rot * 0.5)
    cy, sy = torch.cos(z_rot * 0.5), torch.sin(z_rot * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    return torch.stack([x, y, z, w], dim=-1)


def quat_deriv(q, omega):
    """Assumes omega in body frame"""
    
    # q: [x, y, z, w], omega: body-frame angular velocity [3]
    omega_pure = torch.cat([omega, torch.zeros(*omega.shape[:-1], 1, dtype=omega.dtype)], dim=-1)
    return 0.5 * quat_mul(q, omega_pure)

