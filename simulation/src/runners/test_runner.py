import torch
import numpy as np
import matplotlib.pyplot as plt

from vehicle.atlas_v1 import Atlas, build_vehicle
from vehicle.config import VehicleConfig
from vehicle.integrators import euler_step, rk4_step
from runners.history import SimulationHistory

class TestRunner:
    def __init__(self, duration: float, dt: float, config_path: str):
        self.duration = duration
        self.dt = dt
        self.current_time = torch.full((100, 1), 0.0)
        
        vehicle_config = VehicleConfig.from_yaml(config_path)
        self.vehicle = build_vehicle(vehicle_config)
        
        
        
    def run(self):
        X_current = torch.zeros(100, 13)
        X_current[..., 6] = 1
        solver = rk4_step
        history = SimulationHistory()
        
        while torch.all(self.current_time <= self.duration):

            state = self.vehicle.get_state(X_current, self.current_time)
            X_current = solver(self.vehicle.dynamics, X_current, None, self.current_time, self.dt)
            history.add(state)
            self.current_time += self.dt
        
        return history
            

if __name__ == "__main__":
    duration = 7
    runner = TestRunner(duration, dt=0.001, config_path="configs/vehicles/atlas.yaml")

    history = runner.run()
    
    positions = history.get_position_history()[:,0]
    velocities = history.get_velocity_history()[:,0]
    masses = history.get_extra_history(key="total_mass")[:,0]
    thrusts = history.get_extra_history(key="thrust")[:,0]
    apogee = np.max(positions[:,2])
    

    print(f"Rocket reached apogee of {apogee} m")
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)

    ax1.plot(np.linspace(0, duration, len(positions[:,2])), positions[:,2])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (m)')
    
    ax2.plot(np.linspace(0, duration, len(velocities[:,2])), velocities[:,2])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Z velocity (m/s)')
    
    ax3.plot(np.linspace(0, duration, len(masses)), masses)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Vehicle mass (kg)')
    
    ax4.plot(np.linspace(0, duration, len(thrusts)), thrusts)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Thrust (N)')
    
    plt.tight_layout()
    plt.show()


