import torch
import numpy as np
import matplotlib.pyplot as plt

from vehicle.atlas_v1 import Atlas
from vehicle.integrators import euler_step, rk4_step
class TestRunner:
    def __init__(self, duration: float, dt: float):
        self.duration = duration
        self.dt = dt
        
        self.current_time = 0.0
        
        
    def run(self):
        X_current = torch.zeros(13)
        X_current[6] = 1
        vehicle = Atlas()
        solver = rk4_step
        positions = []
        velocities = []
        while self.current_time <= self.duration:
            X_current = solver(vehicle.dynamics, X_current, None, torch.tensor(self.current_time), self.dt)
            positions.append(X_current[0:3].detach().numpy())
            velocities.append(X_current[7:10].detach().numpy())
            self.current_time += self.dt
        
        return positions, velocities
            

if __name__ == "__main__":
    duration = 5
    runner = TestRunner(duration, dt=0.001)

    positions, velocities = runner.run()
    positions = np.array(positions)
    velocities = np.array(velocities)

    apogee = np.max(positions[:,2])

    print(f"Rocket reached apogee of {apogee} m")
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    ax1.plot(np.linspace(0, duration, len(positions[:,2])), positions[:,2])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (m)')
    
    ax2.plot(np.linspace(0, duration, len(velocities[:,2])), velocities[:,2])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Z velocity (m/s)')
    
    plt.tight_layout()
    plt.show()


