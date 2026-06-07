import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def simulate(thrust_sequence, device, drag_coeff=0.01, m=0.5, v_0=0.0, delta_t=0.05, N=100):
    g = 9.81
    v = torch.tensor(v_0, device=device)
    x = torch.tensor(0.0, device=device)

    thrust_clamped = torch.clamp(thrust_sequence, min=0)

    for n in range(N):
        a = (thrust_clamped[n] / m) - g - (drag_coeff / m) * v * torch.abs(v)
        v = v + a * delta_t
        x = x + v * delta_t

    return x

device = "mps"
TARGET_FINAL_X = torch.tensor(100.0, device=device)
ITERATIONS = 1000

thrust = torch.full((100,), 4.905, device=device, requires_grad=True)
optimizer = torch.optim.Adam([thrust], lr=0.1)
'''
for iteration in range(ITERATIONS):
    optimizer.zero_grad()
    final_x = simulate(thrust, device)
    loss = nn.functional.mse_loss(final_x, TARGET_FINAL_X) + 0.01 * torch.sum(thrust ** 2)
    loss.backward()
    optimizer.step()

    if iteration % 100 == 0:
        print(f"iter {iteration:4d} | loss={loss.detach().cpu().item():.4f} | "
              f"final_x={final_x.detach().cpu().item():.2f}m")
'''

eps = 1e-4

thrust_plus  = thrust.detach().clone()
thrust_minus = thrust.detach().clone()

thrust_plus[50]  = thrust_plus[50]  + eps
thrust_minus[50] = thrust_minus[50] - eps

with torch.no_grad():
    loss_plus  = nn.functional.mse_loss(simulate(thrust_plus,  device), TARGET_FINAL_X) \
                 + 0.01 * torch.sum(thrust_plus  ** 2)
    loss_minus = nn.functional.mse_loss(simulate(thrust_minus, device), TARGET_FINAL_X) \
                 + 0.01 * torch.sum(thrust_minus ** 2)

fd_grad = (loss_plus - loss_minus) / (2 * eps)

# autograd gradient at same point
thrust_check = thrust.detach().clone().requires_grad_(True)
loss_check = nn.functional.mse_loss(simulate(thrust_check, device), TARGET_FINAL_X) \
             + 0.01 * torch.sum(thrust_check ** 2)
loss_check.backward()

print(f"Finite difference: {fd_grad.item():.6f}")
print(f"Autograd:          {thrust_check.grad[50].item():.6f}")