import torch


def simulate(drag_coeff, m=0.5, v_0 = 500.0, delta_t = 0.001, T = 200):
    v = torch.tensor(v_0)
    x = torch.zeros(1)
    g = 9.81
    N = int(T / delta_t)
    for n in range(N):
        a = -g - (drag_coeff / m) * v * torch.abs(v)
        v = v + a * delta_t
        x = x + v * delta_t
    return x



SIM_TIME = 3

with torch.no_grad():
    TRUE_DRAG = torch.tensor(0.1)
    target_pos = simulate(TRUE_DRAG, T=SIM_TIME)
    print(f"Target Apogee: {target_pos}")
    
    
ITERATIONS = 500
drag = torch.tensor(0.5, requires_grad=True)
optimizer = torch.optim.Adam([drag], lr = 0.01)
for iteration in range(ITERATIONS):
    
    optimizer.zero_grad()
    pos = simulate(drag_coeff=drag, T = SIM_TIME)
    loss = (pos - target_pos) ** 2
    loss.backward()
    
    optimizer.step()
    
    if iteration % 10 == 0:
        print(f"Loss: {loss.item()}")
    

    


