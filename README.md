# Ember

**A differentiable 6-DOF rocket simulator that trains a neural flight controller by backpropagating through physics.**

Ember is an end-to-end platform for developing thrust-vector-controlled (TVC) model rockets: a fully differentiable PyTorch flight dynamics engine, a neural attitude-control policy trained directly through the simulated physics, and a custom flight computer to fly the learned controller on real hardware.

The core idea is **differentiable simulation** (a.k.a. "diffsim"). Instead of treating the physics engine as an opaque, non-differentiable black box and learning a controller with reinforcement learning, Ember implements the entire 6-DOF rigid-body simulation in PyTorch so that gradients flow *through* every integration step, every aerodynamic force, and every actuator model. The attitude error at the end of a flight can be differentiated all the way back to the neural network's weights, letting the controller be optimized with plain gradient descent — the same way you would train any other neural network.

---

## Project status

Ember is an active work in progress. The current state:

**Working today**

- Complete, batched, GPU-ready 6-DOF differentiable physics (motor, TVC gimbal, aerodynamics, time-varying mass), integrated with RK4 and quaternion attitude kinematics.
- Batched rollout stack (`RocketEnv` + `EpisodeManager`) with flight-phase termination (burnout, ground impact, excessive tilt).
- A neural MLP attitude controller trained end-to-end through the physics via autograd, plus a PID baseline.
- Declarative YAML vehicle configuration.
- An extensive test suite (~40+ tests) including explicit gradient-flow verification.
- A custom Teensy-based flight computer designed in KiCad (`hardware/`).

**On the roadmap**

- Fully config-driven training/eval runners (task and hyperparameter YAML layers).
- Domain randomization over motor thrust, mass, servo lag, and drag for robust sim-to-real transfer.
- Trajectory-length masking of the loss past episode termination.
- Experiment logging, checkpointing, and a dedicated evaluation runner.
- Export of the trained policy to run in real time on the flight computer, closing the sim-to-hardware loop.

---

## Why differentiable simulation?

Learning a controller for an unstable, fast, safety-critical system like a TVC rocket is hard. The two conventional options each have a major drawback:

- **Reinforcement learning (RL)** treats the simulator as a black box and estimates gradients from scalar rewards. This works, but it is notoriously *sample-inefficient* — it throws away all the structure of the physics and needs enormous numbers of rollouts to converge.
- **Classical control (PID, LQR)** is sample-free but requires hand-tuning and a linearized model, and it struggles with the strongly time-varying, nonlinear dynamics of a rocket whose mass, center of gravity, and inertia change every millisecond as propellant burns.

Differentiable simulation gets the best of both. Because the physics engine is written in an autodiff framework, the *analytic gradient* of the flight outcome with respect to the controller parameters is available for free. Training becomes a direct optimization problem:

> "Given this rocket, what network weights minimize the attitude error over the boost phase?"

This is dramatically more sample-efficient than RL and far more flexible than classical control, because the exact gradient — not a noisy estimate — points the optimizer in the right direction on every single rollout.

```
                            ┌─────────────────────────────────────────────┐
   setpoint (upright) ─────►│  Neural Policy (MLP)                         │
                            │  attitude error + body rates → gimbal command│
                            └───────────────────┬─────────────────────────┘
                                                │  U (2-axis TVC torque cmd)
                                                ▼
        ┌───────────────────────────────────────────────────────────────┐
        │  Differentiable 6-DOF Physics (PyTorch)                        │
        │  motor thrust · TVC gimbal · aerodynamics · time-varying mass  │
        │  RK4 integration · quaternion attitude kinematics              │
        └───────────────────────────┬───────────────────────────────────┘
                                     │  trajectory X(t)   (fully differentiable)
                                     ▼
                            ┌────────────────────┐
                            │  Attitude-error    │
                            │  loss              │
                            └─────────┬──────────┘
                                      │
     ◄────────────────────────────────┘
        loss.backward(): gradient flows through the ENTIRE rollout,
        through every RK4 step and physics term, back to the MLP weights.
```

---

## How the differentiable simulation works

Everything in the physics path is a `torch.Tensor` operation, so the whole rollout is a single differentiable computation graph. A few deliberate engineering choices keep that graph intact and its gradients well-behaved:

- **Autograd through the integrator.** The RK4 step chains four evaluations of the vehicle dynamics per timestep, and every one is differentiable — the full trajectory is one continuous graph from initial condition to final state.

```21:27:simulation/src/vehicle/integrators.py
def rk4_step(...):
    k1 = dynamics_fn(X, U, t)
    k2 = dynamics_fn(X + dt / 2 * k1, U, t + dt / 2)
    k3 = dynamics_fn(X + dt / 2 * k2, U, t + dt / 2)
    k4 = dynamics_fn(X + dt * k3, U, t + dt)
    X_new = X + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
```

- **Out-of-place quaternion normalization.** After each step the attitude quaternion is renormalized by constructing a *new* tensor rather than mutating in place, so backpropagation through the projection is safe (in-place writes are a classic way to silently break an autograd graph).

- **Branch-free physics.** Discontinuous logic — like the piecewise F15 thrust curve — is written with `torch.where` instead of Python `if` statements, so gradients propagate across every phase of the burn.

- **Gradient-safe numerics.** Operations that can produce infinite gradients (e.g. `arccos` at `±1` when computing angle of attack) clamp their inputs away from the singularity, keeping training stable.

Because the physics is written this way, `loss.backward()` on the training loss populates gradients on the neural network's parameters through thousands of chained physics operations, and a standard `Adam` optimizer step improves the controller. There is no reward function, no value network, and no policy-gradient estimator — just supervised optimization of a trajectory-level loss.

```53:69:simulation/src/runners/train_runner.py
        for i in range(N_EPOCHS):
            initial_state = torch.zeros(BATCH_DIM, S.DIM)
            initial_state[..., S.ORI] = start_orientation

            history = episode_runner.run(
                initial_state=initial_state,
                setpoint=setpoint,
                policy=policy,
                vehicle=self.vehicle,
            )

            loss = self.calculate_loss(history.to_episode_result(), setpoint=setpoint)

            print(f"Epoch {i + 1}/{N_EPOCHS} — loss: {loss.item():.6f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## The physics model

Ember simulates a full six-degree-of-freedom rigid body with a compact, batched 15-dimensional state vector (`simulation/src/vehicle/state.py`):

| State           | Indices | Description                                          |
| --------------- | :-----: | ---------------------------------------------------- |
| Position        | `0:3`   | World-frame position (Z up)                          |
| Orientation     | `3:7`   | Attitude quaternion, scalar-last `[x, y, z, w]`      |
| Velocity        | `7:10`  | World-frame linear velocity                          |
| Angular velocity| `10:13` | Body-frame angular rates                             |
| Gimbal angle    | `13:15` | 2-axis TVC deflections `(δx, δy)`                    |

Each timestep, the vehicle sums the physical forces and torques and integrates the rigid-body equations of motion. The physics is decomposed into independently-testable, composable components:

- **Motor** — a piecewise-polynomial F15 thrust curve with analytic total impulse and a time-varying propellant mass / moment-of-inertia model (inside-out burn).
- **Cartesian TVC gimbal** — a 2-axis thrust-vectoring actuator that tilts the thrust vector, with a first-order servo lag between commanded and actual deflection.
- **Aerodynamics** — quadratic drag, Barrowman-style normal force, and a pitch-damping moment.
- **Mass properties** — combines the dry airframe and burning motor, shifting the center of gravity and inertia tensor (via the parallel-axis theorem) as propellant is consumed.
- **Rigid-body core** — Newton–Euler dynamics including the gyroscopic term `ω × (Iω)`, integrated with swappable explicit-Euler or RK4 schemes.

The entire vehicle is specified declaratively in YAML (`simulation/configs/vehicles/atlas.yaml`) — mass, motor, geometry, aero coefficients, and gimbal limits — so a rocket is fully described by data rather than code.

---

## The neural controller

The policy is a small multilayer perceptron (`simulation/src/policy/mlp.py`) — deliberately kept tiny so it can eventually be exported and run in real time on a microcontroller. It observes the *shortest-path* body-frame attitude error (the x/y components of the error quaternion) together with the body angular rates, and outputs a 2-axis gimbal torque command squashed through `tanh` to respect the actuator's physical limits.

```26:38:simulation/src/policy/mlp.py
    def forward(self, X, setpoint):
        q  = normalize_quat(X[..., S.ORI])
        w_b = X[..., S.ANG_VEL]
        setpoint = normalize_quat(setpoint.expand(*q.shape[:-1], 4))

        q_error = quat_mul(quat_inv(q), setpoint)
        q_error = q_error * torch.where(q_error[..., 3:4] < 0, -1., 1.)
        error   = q_error[..., :2]

        input = torch.cat([error, X[..., S.ANG_VEL]], dim=-1)
        output = self.network(input)
        return self.max_output * torch.tanh(output)
```

A conventional **PID controller** (`simulation/src/policy/pid.py`) is included as a baseline for comparison against the learned policy.

---

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/daniel360kim/ember.git
cd ember/simulation
pip install -e .
```

This installs `torch`, `numpy`, `matplotlib`, and `pyyaml`, and puts `src/` on the import path so `from vehicle.atlas_v1 import Atlas`-style imports resolve.

### Train the neural controller through the physics

```bash
cd ember/simulation
PYTHONPATH=src python src/runners/train_runner.py
```

This builds the `Atlas` vehicle, rolls it out under the MLP policy from a slightly tilted initial attitude, computes the attitude-error loss over the boost, and backpropagates through the entire flight to update the network — for 100 epochs. It plots the orientation and gimbal histories at the first, middle, and last epoch so you can watch the controller learn to stabilize the rocket.

### Run the test suite

The physics engine is covered by an extensive suite of unit and integration tests, including explicit checks that gradients flow correctly through the integrators, the rigid-body core, and the full vehicle dynamics.

```bash
cd ember/simulation
pytest                 # everything
pytest tests/vehicle/  # physics component & dynamics tests
pytest tests/runners/  # environment & rollout integration tests
```

---

## Repository layout

```
ember/
├── simulation/                     # Differentiable PyTorch simulator (main project)
│   ├── src/
│   │   ├── vehicle/                # 6-DOF physics
│   │   │   ├── atlas_v1.py         # Concrete rocket: assembles the full dynamics
│   │   │   ├── base_vehicle.py     # Vehicle interface + quaternion projection
│   │   │   ├── rigid_body.py       # Newton–Euler derivative core
│   │   │   ├── mass_properties.py  # Time-varying mass / CG / inertia
│   │   │   ├── integrators.py      # euler_step, rk4_step (differentiable)
│   │   │   ├── state.py            # 15-D state-vector layout
│   │   │   └── components/         # motor, cartesian_gimbal (TVC), aero
│   │   ├── policy/
│   │   │   ├── mlp.py              # Neural attitude controller
│   │   │   └── pid.py              # Classical baseline
│   │   ├── runners/
│   │   │   ├── rocket_env.py       # Batched, Gym-like physics environment
│   │   │   ├── episode_manager.py  # Policy-in-the-loop rollout engine
│   │   │   └── train_runner.py     # Differentiable training entry point
│   │   └── utils/math.py           # Quaternion algebra & kinematics
│   ├── configs/vehicles/atlas.yaml # Declarative vehicle definition
│   └── tests/                      # Physics unit tests + gradient-flow checks
│
├── hardware/                       # Custom flight computer (KiCad)
│   └── pcb/flight_computer/        # Teensy-based board: IMU, barometer,
│                                   # servo drivers, pyro channels, power
└── background/                     # Early diffsim prototypes & experiments
```

---

## Hardware

The `hardware/` directory contains the KiCad design for a custom rocket flight computer — a Teensy-class board with an IMU, barometer, servo drivers for the TVC gimbal, pyro channels for recovery, and power management. The long-term goal of Ember is **zero-shot sim-to-real transfer**: train the attitude controller entirely in the differentiable simulator, then deploy the exact learned weights to this board to fly a physical rocket.
