# asteroid-flyby-syssim

Syssim-based asteroid flyby simulation with translational dynamics, rigid-body attitude dynamics, and reaction-wheel pointing control.

## What It Simulates

- Spacecraft hyperbolic flyby around an asteroid in an asteroid-centered frame.
- Asteroid gravity using spherical harmonics via `NodeAsteroidGravity`.
- Camera center-pointing guidance (asteroid center is assumed known with no estimation).
- Rigid-body attitude control through an extended reaction-wheel model with:
  - torque saturation
  - wheel speed saturation
  - momentum saturation
  - first-order motor lag
  - viscous and Coulomb friction
  - optional imbalance/jitter disturbance torque
- Optional rendered camera video using Mitsuba.
- Trajectory plus look-vector animation and CSV/NPZ logs.

## Run

```bash
python -m asteroid_flyby_syssim --help
python -m asteroid_flyby_syssim
```

### Useful CLI Overrides

```bash
python -m asteroid_flyby_syssim \
  --run.output.render-video False \
  --run.sim.duration-s 1200 \
  --run.flyby.periapsis-radius-m 700000 \
  --run.flyby.v-infinity-mps 850 \
  --run.output.run-name ceres_flyby_demo
```

## Outputs

Created under `--run.output.output-dir/--run.output.run-name`:

- `trajectory_look.gif`: trajectory and look-vector animation
- `timeseries.csv`: scalar timeseries log
- `timeseries.npz`: same data in numpy archive
- `camera_render.gif`: rendered camera video (if rendering enabled)

## Frame Convention

- Simulation state is asteroid-centered Cartesian coordinates.
- Attitude quaternions map spacecraft body frame to asteroid-centered inertial frame.
- Camera boresight defaults to body `+X` axis.
