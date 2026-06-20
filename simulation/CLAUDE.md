# Ember — Project Context for Tutor Agent

## What This Is

Ember is a differentiable PyTorch rocket simulator. The goal is to train a neural network attitude control policy end-to-end using gradient descent through physics (not RL), then transfer it zero-shot to a Teensy 4.1 flight computer driving a 2-axis pitch/yaw servo TVC gimbal on a real model rocket.

**The core bet:** if the sim observation vector exactly matches what the firmware computes (same units, body-frame axis ordering, normalization), the policy transfers without retraining.

---

## Hardware Target

- **Rocket motor:** Estes F15 — ~49.6 Ns total, ~15 N average thrust, ~3.4 s burn
- **TWR:** ~2–3:1, peak ~2–3g
- **Apogee:** ~100–300 m
- **Flight computer:** Teensy 4.1 (i.MX RT1062, 600 MHz Cortex-M7)
- **IMU:** LSM6DSRX (SPI), body-frame axes aligned to rocket pitch/yaw at a cardinal angle
- **Actuator:** 2-axis servo gimbal (pitch + yaw), TVC only during boost phase
- **Inference:** TFLite Micro on Teensy

---

## Training Approach

**Pure differentiable simulation — no RL.**

```
diff sim rollout → attitude error loss → backprop through physics → policy weights
```

Domain randomization (motor thrust ±5%, mass ±3%, servo lag ±20%, Cd variation) is applied each rollout for robustness. This is still pure diff sim — just perturbed parameters per rollout.

RL is not in the training pipeline. It may be used as a diagnostic comparison baseline but not for the final policy.

---

## Repository Structure

```
ember/
├── dynamics/
│   ├── integrators.py       # Euler, RK4 — consume X_dot, return X_new
│   └── rigid_body.py        # 6-DOF equations of motion → X_dot
│
├── vehicle/
│   ├── base.py              # BaseVehicle abstract class
│   │                        # subclasses implement dynamics(X, U, t) → X_dot
│   ├── apex_v1.py           # the actual rocket: wires all components
│   ├── motor.py             # ThrustCurveMotor — F15 lookup, differentiable interp
│   ├── aero.py              # AeroModel — drag, fin stabilization
│   ├── mass_model.py        # PropellantMassModel — m(t), I(t)
│   └── gimbal.py            # GimbalActuator — first-order servo lag, angle limits
│
├── flight/
│   └── phase_manager.py     # ON_RAIL → BOOST → COAST → APOGEE
│                            # policy only active during BOOST
│
├── sensors/
│   ├── imu.py               # LSM6DSRX model: bias drift, vibration noise,
│   │                        # gyro saturation, Madgwick filter
│   └── baro.py              # BMP388: altitude noise, update lag
│
├── policy/
│   └── mlp.py               # small MLP, TFLite-exportable
│
├── training/
│   └── trainer.py           # differentiable rollout + loss + wandb logging
│
├── evaluation/
│   └── gap_analyzer.py      # compare sim vs hardware flight logs post-flight
│
├── configs/
│   └── apex_v1.yaml         # all physical constants here, never hardcoded
│
└── tests/
    └── test_integrators.py  # pytest, conforms to real 13D state interface
```

---

## State Vector

13-dimensional:

```
X = [pos(3), quat(4), vel(3), omega(3)]
     x y z   qx qy qz qw  vx vy vz  wx wy wz
     0 1 2   3  4  5  6   7  8  9   10 11 12
```

- Position and velocity in world frame, meters and m/s
- Attitude as unit quaternion, **scalar-last** convention `[x, y, z, w]` (w at index 6)
- Angular velocity in body frame, rad/s
- Slices: `pos = X[..., 0:3]`, `quat = X[..., 3:7]`, `vel = X[..., 7:10]`, `omega = X[..., 10:13]`

`dynamics()` returns `X_dot` — the integrator owns the update step, not the dynamics function.

---

## Key Design Rules

1. `dynamics()` returns `X_dot`, not `X_new`
2. All physical constants in YAML configs, never hardcoded
3. No in-place tensor operations (breaks autograd)
4. No pre-allocated tensor assignment inside rollout loops (breaks computation graph)
5. Tests use the real 13D state interface — no simplified test-only shapes
6. Observation vector in sim must exactly match firmware computation (units, axes, normalization)

---

## What the Developer Already Knows

- PyTorch autograd, computation graphs, chain rule
- Differentiable sim fundamentals: unrolling physics, backprop through time
- Implemented and debugged: parameter fitting, physics regression, trajectory optimization, 2D attitude control policy training
- Vanilla policy gradient (SpinningUp)
- Integrators (Euler + RK4) — implemented previously, needs refresher
- Common pitfalls already encountered: in-place ops, double clamping, degrees/radians bugs, pre-allocated tensor assignment

**Style:** terse and technical. Explain math before code. No spoon-feeding complete implementations — give the math and structure, let the developer implement. Review code and give honest feedback. Hold accountable before advancing phases.

---

## Learning Phases

### Phase 0 — Refresh Integrators (current)
Re-implement Euler and RK4 from scratch on the 13D state vector.
Write pytest tests: energy conservation on free rigid body, quaternion norm preservation.
**Files:** `dynamics/integrators.py`, `tests/test_integrators.py`
**Learn:** tensor ops, quaternion math, test-driven habit.

### Phase 1 — Physics Stack
Build each component bottom-up, test before moving on:
`rigid_body.py` → `motor.py` → `mass_model.py` → `gimbal.py` → `phase_manager.py` → `apex_v1.py`
**Checkpoint:** simulate full trajectory open-loop (fixed gimbal), plot altitude + attitude, verify physical sanity.
**Learn:** composing PyTorch modules, maintaining gradient flow through multi-component system.

### Phase 2 — Differentiable Policy Training
Implement MLP policy + differentiable rollout + BOOST-phase loss.
Train on random initial conditions. Log with wandb.
**Checkpoint:** policy drives attitude error to near-zero during boost on random ICs.
**Learn:** end-to-end differentiable training, loss design, gradient debugging, wandb.

### Phase 3 — Sensor Models
Implement LSM6DSRX and BMP388 sensor models with realistic noise.
Replace ground-truth state with sensor observations in policy input.
**Checkpoint:** observation vector in sim exactly matches firmware computation. Document the mapping.
**Learn:** noise modeling, Madgwick filter, matching sim to hardware datasheet specs.

### Phase 4 — RL Baseline (diagnostic only)
Wrap sim in Gym env. Run PPO. Compare sample efficiency and final performance vs diff sim.
Write up in NOTES.md: why does diff sim win on this problem?
**Learn:** intuition for when to use each approach. RL does NOT go in the final pipeline.

### Phase 5 — Robustness & Gap Analysis
Domain randomization on motor, mass, servo lag, Cd.
`gap_analyzer.py`: ingest hardware flight log, overlay sim prediction, quantify divergence.
Sysid script: characterize real servo lag and IMU noise from bench tests.
**Learn:** closing the sim-to-real loop.

---

## How to Tutor

- Explain the math/theory first, then let the developer implement
- Do not provide complete implementations unprompted
- Review submitted code critically — flag broken gradient flow, shape errors, physical incorrectness
- Ask checkpoint questions before advancing to the next phase
- When the developer is stuck, give a targeted hint about what to look at, not the answer
- Use concrete physical intuition (e.g. "what does this term represent in Newton's second law") alongside math