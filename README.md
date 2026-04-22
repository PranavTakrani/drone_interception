# Drone Interception — Design Decisions

## intercept_controller_occluded

Extension of `intercept_controller.py` that handles occluded / unreliable target position data.

---

### Occlusion Model

`P(occluded | t) = 0.6 * exp(-0.08 * t) + 0.05`

- Starts at ~65% occluded, decays exponentially to a 5% persistent floor
- Per-step ±5% jitter makes the probability itself stochastic rather than deterministic
- When visible: Gaussian noise (σ = 0.3 m) on the observed position
- 5% chance of a 3× outlier spike on any visible observation

---

### Kalman Filter

6-state constant-velocity model: `[x, y, z, vx, vy, vz]`

- **Predict-only** on occluded steps (dead-reckoning via constant velocity)
- **Predict + update** on observed steps
- Process noise Q tuned for ~3 m/s² acceleration uncertainty — matches `RandomEvasiveTarget.max_accel`
- Outliers are naturally down-weighted by the innovation covariance; no explicit rejection needed

**Tradeoff:** A full IMM (Interacting Multiple Models) would handle sharp jinks better but adds significant complexity. The single CV model with generous Q is a good balance for this target behaviour.

---

### Weights File

`best_weights_mppi_occluded.json` is seeded from `best_weights_mppi.json` so occluded-specific tuning doesn't affect the clean-data controller.

---

### Graph Output

Run with `--graph-out <path.png>` to save, or omit for interactive display.

- **Top panel:** XY actual vs estimated trajectory + raw observations
- **Bottom panel:** Estimation error over time with occluded timesteps shaded

---

### Usage

```bash
# Headless, save graph
python skydio_x2/intercept_controller_occluded.py --mode evasive --headless --graph-out traj.png

# Interactive with per-step debug
python skydio_x2/intercept_controller_occluded.py --mode evasive --debug

# Batch test with pass/fail summary
python skydio_x2/intercept_controller_occluded.py --mode evasive --headless --iters 20
```
