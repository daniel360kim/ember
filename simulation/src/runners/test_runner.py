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
        masses = []
        while self.current_time <= self.duration:
            X_current = solver(vehicle.dynamics, X_current, None, torch.tensor(self.current_time), self.dt)
            vehicle_mass = vehicle.get_current_mass(torch.tensor(self.current_time))
            
            positions.append(X_current[0:3].detach().numpy())
            velocities.append(X_current[7:10].detach().numpy())
            masses.append(vehicle_mass)
            
            self.current_time += self.dt
        
        return positions, velocities, masses
            

if __name__ == "__main__":
    duration = 5
    runner = TestRunner(duration, dt=0.001)

    positions, velocities, masses = runner.run()
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    masses = np.array(masses)

    apogee = np.max(positions[:,2])

    print(f"Rocket reached apogee of {apogee} m")
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)

    ax1.plot(np.linspace(0, duration, len(positions[:,2])), positions[:,2])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (m)')
    
    ax2.plot(np.linspace(0, duration, len(velocities[:,2])), velocities[:,2])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Z velocity (m/s)')
    
    ax3.plot(np.linspace(0, duration, len(masses)), masses)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Vehicle mass (kg)')
    
    plt.tight_layout()
    plt.show()


