# Ember

Differentiable PyTorch rocket simulator. Trains a neural attitude control
policy end-to-end via gradient descent through physics (not RL), targeting
zero-shot transfer to a Teensy 4.1 flight computer driving a 2-axis TVC
gimbal on a real model rocket.

See `CLAUDE.md` for the state vector convention, design rules, and learning
roadmap. This file documents the project layout.

## Project Structure

```
ember/
├── configs/                    # all numbers live here, not in code
│   ├── vehicles/
│   │   └── apex_v1.yaml        # physical parameters
│   ├── env/
│   │   └── stabilization.yaml  # task settings
│   └── training/
│       └── diffsim.yaml        # training hyperparameters
│
├── ember/                      # main package
│   ├── vehicle/                # physics — rocket-specific
│   │   ├── base_vehicle.py     # abstract interface
│   │   ├── apex_v1.py          # your first rocket
│   │   ├── components/         # modular physics pieces
│   │   │   ├── motor.py        # thrust curve
│   │   │   ├── mass_model.py   # CoM/MoI as propellant burns
│   │   │   └── gimbal.py       # actuator with lag
│   │   └── integrators.py      # euler, rk4 — swappable
│   │
│   ├── policy/
│   │   └── mlp.py              # your Policy class
│   │
│   ├── training/
│   │   ├── diffsim_trainer.py  # training loop
│   │   └── randomizer.py       # domain randomization
│   │
│   └── utils/
│       └── math.py             # quaternions, rotations
│
├── scripts/
│   └── train.py                # entry point
│
├── tests/
│   └── test_gradient_flow.py   # finite difference checks
│
├── pyproject.toml
└── NOTES.md
```

## `configs/`

Every physical constant and hyperparameter lives in YAML, never hardcoded
in source. A vehicle, task, or training run is fully specified by which
three config files get loaded.

- **`vehicles/apex_v1.yaml`** — motor specs (thrust curve reference, total
  impulse), dry/wet mass, geometry, aero coefficients, gimbal angle limits
  and servo time constant, IMU/baro noise parameters. Anything that
  describes *the rocket itself*.
- **`env/stabilization.yaml`** — task definition: phase thresholds
  (ON_RAIL → BOOST → COAST → APOGEE), initial-condition distributions for
  randomized rollouts, target attitude/setpoint, episode length, dt.
- **`training/diffsim.yaml`** — optimizer, learning rate, rollout batch
  size, loss term weights, domain randomization ranges (thrust/mass/lag/Cd
  perturbation bounds).

## `ember/` (main package)

### `vehicle/`

Physics, rocket-specific.

- **`base_vehicle.py`** — abstract `BaseVehicle`. Defines the one contract
  every vehicle must satisfy: `dynamics(X, U, t) -> X_dot`. The integrator
  owns stepping; a vehicle only ever evaluates the derivative.
- **`apex_v1.py`** — the concrete rocket. Wires `components/` together:
  sums thrust/drag/gravity into translational `X_dot`, sums TVC + aero
  torque into rotational `X_dot`, assembles the full 13D derivative.
- **`components/`** — modular physics pieces, each independently testable,
  composed by a vehicle rather than baked into it.
  - **`motor.py`** — thrust curve lookup (F15), differentiable
    interpolation over burn time.
  - **`mass_model.py`** — time-varying center of mass and moment of
    inertia tensor as propellant burns off.
  - **`gimbal.py`** — actuator model: first-order lag toward a commanded
    deflection, clamped to mechanical angle limits.
- **`integrators.py`** — `euler_step`, `rk4_step`. Pure functions of
  `(dynamics_fn, X, U, dt) -> X_new`. Swappable independent of physics
  because they only ever consume `X_dot`, never compute it.

### `policy/`

- **`mlp.py`** — the `Policy` class. Small MLP, kept exportable to TFLite
  Micro so the exact weights trained here run on the Teensy.

### `training/`

- **`diffsim_trainer.py`** — the differentiable rollout loop: step the
  vehicle forward under the policy's commands, accumulate a BOOST-phase
  attitude-error loss, backprop through every integration step back to
  policy weights.
- **`randomizer.py`** — applies domain randomization (motor thrust, mass,
  servo lag, Cd) per rollout, sampled from `training/diffsim.yaml`'s
  ranges. Still pure diffsim — randomization perturbs parameters, it
  doesn't introduce RL.

### `utils/`

- **`math.py`** — quaternion algebra used everywhere: identity, inverse,
  Hamilton product, vector rotation, quaternion→Euler, and the kinematic
  derivative `q_dot = 0.5 * q ⊗ [omega, 0]` used inside every vehicle's
  `dynamics()`. Scalar-last `[x, y, z, w]` convention throughout — matches
  the state vector's `quat = X[3:7]` slice.

## `scripts/train.py`

Entry point. Loads the three configs, builds the vehicle + policy +
trainer, runs training, logs to wandb.

## `tests/test_gradient_flow.py`

Finite-difference checks that gradients actually reach policy weights
through the full rollout. This is the test that catches a stray in-place
tensor write or pre-allocated assignment before it silently zeroes out
backprop on a real training run — run it after touching anything in
`vehicle/` or `training/`.
