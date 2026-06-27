import torch
import numpy as np
import matplotlib.pyplot as plt

from vehicle.atlas_v1 import Atlas, build_vehicle
from vehicle.config import VehicleConfig
from vehicle.integrators import euler_step, rk4_step
from runners.history import SimulationHistory

from utils.math import euler_to_quat

class TestRunner:
    def __init__(self, duration: float, dt: float, config_path: str):
        self.duration = duration
        self.dt = dt
        self.current_time = torch.full((1, 1), 0.0)
        
        vehicle_config = VehicleConfig.from_yaml(config_path)
        self.vehicle = build_vehicle(vehicle_config)
        
        
    def run(self):
        start_orientation = euler_to_quat(torch.tensor([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]))
        X_current = torch.zeros(1, 15)
        X_current[..., 3:7] = start_orientation
        
        X_current[..., 13] = np.deg2rad(0.1) # x axis gimbal tilt
        
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
    orientations = history.get_orientation_euler_history()[:,0]
    angular_velocities = history.get_angular_velocity_history()[:,0]
    masses = history.get_extra_history(key="total_mass")[:,0]
    thrusts = history.get_extra_history(key="thrust")[:,0]
    apogee = np.max(positions[:,2])
    

    print(f"Rocket reached apogee of {apogee} m")
    
    fig, ax = plt.subplots(nrows=3, ncols=2)

    ax[0, 0].plot(np.linspace(0, duration, len(positions[:,0])), positions[:,0], label="X")
    ax[0, 0].plot(np.linspace(0, duration, len(positions[:,1])), positions[:,1], label="Y")
    ax[0, 0].plot(np.linspace(0, duration, len(positions[:,2])), positions[:,2], label="Z")
    ax[0, 0].legend()
    ax[0, 0].set_title('Position')
    ax[0, 0].set_xlabel('Time (s)')
    ax[0, 0].set_ylabel('Position (m)')
    
    ax[1, 0].plot(np.linspace(0, duration, len(orientations[:,0])), orientations[:,0], label="X")
    ax[1, 0].plot(np.linspace(0, duration, len(orientations[:,1])), orientations[:,1], label="Y")
    ax[1, 0].plot(np.linspace(0, duration, len(orientations[:,2])), orientations[:,2], label="Z")
    ax[1, 0].legend()
    ax[1, 0].set_title('Orientation')
    ax[1, 0].set_xlabel('Time (s)')
    ax[1, 0].set_ylabel('Orientation (deg)')
    
    ax[1, 1].plot(np.linspace(0, duration, len(angular_velocities[:,0])), angular_velocities[:,0], label="X")
    ax[1, 1].plot(np.linspace(0, duration, len(angular_velocities[:,0])), angular_velocities[:,1], label="Y")
    ax[1, 1].plot(np.linspace(0, duration, len(angular_velocities[:,0])), angular_velocities[:,2], label="Z")
    ax[1, 1].legend()
    ax[1, 1].set_title('Ang. Velocity')
    ax[1, 1].set_xlabel('Time (s)')
    ax[1, 1].set_ylabel('Ang. Velocity (deg/s)')

    ax[0, 1].plot(np.linspace(0, duration, len(velocities[:,0])), velocities[:,0], label="X")
    ax[0, 1].plot(np.linspace(0, duration, len(velocities[:,1])), velocities[:,1], label="Y")
    ax[0, 1].plot(np.linspace(0, duration, len(velocities[:,2])), velocities[:,2], label="Z")
    ax[0, 1].set_title('Velocity')
    ax[0, 1].set_xlabel('Time (s)')
    ax[0, 1].set_ylabel('Velocity (m/s)')
    
    
    ax[2, 0].plot(np.linspace(0, duration, len(masses)), masses)
    ax[2, 0].set_title('Vehicle mass')
    ax[2, 0].set_xlabel('Time (s)')
    ax[2, 0].set_ylabel('Vehicle mass (kg)')

    ax[2, 1].plot(np.linspace(0, duration, len(thrusts)), thrusts)
    ax[2, 1].set_title('Motor thrust')
    ax[2, 1].set_xlabel('Time (s)')
    ax[2, 1].set_ylabel('Thrust (N)')
    
    plt.tight_layout()
    plt.show()


