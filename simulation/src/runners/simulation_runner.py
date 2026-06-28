import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from vehicle.atlas_v1 import Atlas, build_vehicle
from vehicle.config import VehicleConfig
from vehicle.integrators import euler_step, rk4_step
from vehicle.state import S

from policy.base_policy import Policy
from policy.pid import PID
from runners.history import SimulationHistory

from utils.math import euler_to_quat, quat_identity

class SimulationRunner:
    def __init__(self, duration: float, dt: float, config_path: str, policy: Policy, policy_dt: float):
        self.duration = duration
        self.dt = dt
        self.current_time = torch.full((1, 1), 0.0)
        
        vehicle_config = VehicleConfig.from_yaml(config_path)
        self.vehicle = build_vehicle(vehicle_config)
        
        self.policy = policy
        
        if policy_dt < dt:
            raise ValueError(f"policy_dt ({policy_dt}) must be >= sim dt ({dt})")
        ratio = policy_dt / dt
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(f"policy_dt ({policy_dt}) must be an integer multiple of dt ({dt})")
        self.steps_per_policy = round(ratio)
        self.current_steps_taken = 0
        
        
    def run(self):
        start_orientation = euler_to_quat(torch.tensor([np.deg2rad(3), np.deg2rad(5), np.deg2rad(0)]))
        X_current = torch.zeros(1, S.DIM)
        X_current[..., S.ORI] = start_orientation
    
        solver = rk4_step
        history = SimulationHistory()
        
        U = torch.zeros(1, 2) # zero commanded
        setpoint = quat_identity().expand(1, 4) # upright
        total_steps = int(round(self.duration / self.dt)) + 1
        with tqdm(total=total_steps, desc="Simulating", unit="step") as pbar:
            while torch.all(self.current_time <= self.duration):

                state = self.vehicle.get_state(X_current, self.current_time)
                if self.current_steps_taken % self.steps_per_policy == 0:
                    U = self.policy.forward(X_current, setpoint)
                
                X_current = solver(self.vehicle.dynamics, X_current, U, self.current_time, self.dt, project=self.vehicle.project)


                history.add(state)
                self.current_time += self.dt
                self.current_steps_taken += 1
                pbar.update(1)

        return history
            

if __name__ == "__main__":
    duration = 3.4
    policy = PID(dt=0.01)
    runner = SimulationRunner(duration, dt=0.001, config_path="configs/vehicles/atlas.yaml", policy=policy, policy_dt=0.01)

    history = runner.run()
    
    positions = history.get_position_history()[:,0]
    velocities = history.get_velocity_history()[:,0]
    orientations = history.get_orientation_euler_history()[:,0]
    angular_velocities = history.get_angular_velocity_history()[:,0]
    gimbal_angles = history.get_gimbal_angle_history()[:,0]
    masses = history.get_extra_history(key="total_mass")[:,0]
    thrusts = history.get_extra_history(key="thrust")[:,0]
    cgs = history.get_extra_history(key="cg")[:,0]
    mmois = np.diagonal(history.get_extra_history(key="mmoi")[:,0], axis1=1, axis2=2) # (T, 3): [Ixx, Iyy, Izz]
    apogee = np.max(positions[:,2])
    

    print(f"Rocket reached apogee of {apogee} m")
    
    fig, ax = plt.subplots(nrows=5, ncols=2)

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
    
    ax[3, 0].plot(np.linspace(0, duration, len(gimbal_angles[:,0])), gimbal_angles[:,0], label="X")
    ax[3, 0].plot(np.linspace(0, duration, len(gimbal_angles[:,1])), gimbal_angles[:,1], label="Y")
    ax[3, 0].legend()
    ax[3, 0].set_title('Gimbal Angle')
    ax[3, 0].set_xlabel('Time (s)')
    ax[3, 0].set_ylabel('Angle (deg)')
    
    ax[4, 0].plot(np.linspace(0, duration, len(cgs[:,2])), cgs[:,2])
    ax[4, 0].set_title('Center of gravity (z)')
    ax[4, 0].set_xlabel('Time (s)')
    ax[4, 0].set_ylabel('Cg (z) (m)')
    
    
    t_mmoi = np.linspace(0, duration, len(mmois))
    ax[4, 1].plot(t_mmoi, mmois[:,0], label="Ixx")
    ax[4, 1].plot(t_mmoi, mmois[:,1], label="Iyy")
    ax[4, 1].plot(t_mmoi, mmois[:,2], label="Izz")
    ax[4, 1].legend()
    ax[4, 1].set_title('MMOI (diagonal)')
    ax[4, 1].set_xlabel('Time (s)')
    ax[4, 1].set_ylabel('Inertia (kg·m²)')
    plt.show()


