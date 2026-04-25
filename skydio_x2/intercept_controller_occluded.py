"""
intercept_controller_occluded.py — Intercept a moving target with occluded/noisy position data.

Extends intercept_controller.py with:
  - Stochastic occlusion model (time-varying probability)
  - Gaussian noise + occasional outliers on observations
  - Kalman filter for target state estimation
  - Post-run graph: estimated vs actual target trajectory

Usage:
    python skydio_x2/intercept_controller_occluded.py --mode evasive --headless
    python skydio_x2/intercept_controller_occluded.py --mode evasive --headless --iters 10
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="Could not import matplotlib")

import numpy as np
import mujoco
import mujoco.viewer
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skydio_x2.mppi_controller import MPPIController
from skydio_x2.skydio_x2_movement import apply_motor_mixing
from skydio_x2.mpc_control_config import DEFAULT_MPPI_CONFIG
from skydio_x2.mpc_control_config import MPPIControlConfig
from skydio_x2.intercept_controller import (
    build_intercept_xml,
    compute_lead_point,
    ScriptedTargetPath,
    RandomEvasiveTarget,
)


# ======================================================================
# Occlusion / Sensor Model
#
# Occlusion has two independent parts:
#
#   1. NO-DATA  — sensor returns nothing (hard dropout)
#   2. NOISY    — sensor returns a corrupted position (Gaussian + outliers)
#
# Both probabilities share the same distance-driven exponential decay:
#
#   P(event | d) = initial * exp(-decay * (1/d + jitter)) + floor
#
# where d = distance between drone and target.
# Closer drone → smaller d → faster decay → lower occlusion probability.
# Jitter is per-step Gaussian noise on the exponent, making the decay noisy.
#
# Occlusion rate reported = % of steps that were either no-data OR noisy.
# ======================================================================

# --- No-data dropout parameters ---
NO_DATA_INITIAL = 0.35   # starting probability of total signal loss
NO_DATA_DECAY   = 0.08   # how fast dropout falls with proximity
NO_DATA_FLOOR   = 0.03   # minimum dropout probability (never fully reliable)

# --- Noisy / outlier observation parameters ---
NOISY_INITIAL   = 0.45   # starting probability of receiving a corrupted obs
NOISY_DECAY     = 0.05   # how fast noise probability falls with proximity
NOISY_FLOOR     = 0.02   # minimum noisy-obs probability
OBS_NOISE_STD   = 0.3    # metres — Gaussian noise std on normal observations
OUTLIER_PROB    = 0.15   # fraction of noisy obs that are large outlier spikes
OUTLIER_SCALE   = 3.0    # outlier spike magnitude (multiplier on OBS_NOISE_STD)

# --- Jitter on the decay exponent (makes rate itself stochastic) ---
DECAY_JITTER_STD = 0.02


def _decayed_prob(initial: float, decay: float, floor: float,
                  dist: float, rng: np.random.RandomState) -> float:
    """
    Distance-driven exponential decay with per-step jitter.
    Uses 1/dist so probability drops as drone closes in.
    """
    d = max(dist, 0.1)  # avoid division by zero
    jitter = rng.normal(0.0, DECAY_JITTER_STD)
    p = initial * np.exp(-decay * (1.0 / d + jitter)) + floor
    return float(np.clip(p, 0.0, 1.0))


def observe_target(true_pos: np.ndarray, drone_pos: np.ndarray,
                   rng: np.random.RandomState):
    """
    Returns (observation, obs_type):
      observation — noisy 3-vector, or None if no-data
      obs_type    — "clean", "noisy", "outlier", or "no_data"
    """
    dist = np.linalg.norm(true_pos - drone_pos)

    # Part 1: hard dropout — no data at all
    if rng.random() < _decayed_prob(NO_DATA_INITIAL, NO_DATA_DECAY, NO_DATA_FLOOR, dist, rng):
        return None, "no_data"

    # Part 2: noisy observation
    if rng.random() < _decayed_prob(NOISY_INITIAL, NOISY_DECAY, NOISY_FLOOR, dist, rng):
        noise_std = OBS_NOISE_STD * (OUTLIER_SCALE if rng.random() < OUTLIER_PROB else 1.0)
        obs_type  = "outlier" if noise_std > OBS_NOISE_STD else "noisy"
        return true_pos + rng.normal(0.0, noise_std, size=3), obs_type

    # Clean observation (still has small sensor noise)
    return true_pos + rng.normal(0.0, OBS_NOISE_STD * 0.3, size=3), "clean"


# ======================================================================
# Kalman Filter — constant-velocity model, 6-state [x,y,z,vx,vy,vz]
#
# State transition: x_{k+1} = F x_k  (constant velocity)
# Observation:      z_k = H x_k + noise
#
# Process noise Q is tuned to allow fast target manoeuvres (~3 m/s²).
# Measurement noise R matches OBS_NOISE_STD.
# ======================================================================

class TargetKalmanFilter:
    def __init__(self, init_pos: np.ndarray, dt: float):
        self.dt = dt
        n = 6

        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros(n)
        self.x[:3] = init_pos

        # State transition matrix (constant velocity)
        self.F = np.eye(n)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Observation matrix (observe position only)
        self.H = np.zeros((3, n))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Process noise — allows ~3 m/s² acceleration uncertainty
        q_pos = 0.5 * dt**2 * 3.0
        q_vel = dt * 3.0
        self.Q = np.diag([q_pos**2]*3 + [q_vel**2]*3)

        # Measurement noise
        self.R = np.eye(3) * OBS_NOISE_STD**2

        # Covariance — start uncertain about velocity
        self.P = np.diag([1.0, 1.0, 1.0, 4.0, 4.0, 4.0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        """Fuse a position observation z (3-vector)."""
        y = z - self.H @ self.x                          # innovation
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def step(self, obs):
        """Predict then optionally update. Returns (est_pos, est_vel)."""
        self.predict()
        if obs is not None:
            self.update(obs)
        return self.x[:3].copy(), self.x[3:6].copy()


# ======================================================================
# Graph — estimated vs actual target trajectory
# ======================================================================

def plot_trajectory(log: dict, save_path: str = None):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("  matplotlib not available — skipping graph.")
        return

    times     = np.array(log["t"])
    actual    = np.array(log["actual"])
    estimated = np.array(log["estimated"])
    obs_pos   = np.array(log["obs_pos"]) if log["obs_pos"] else np.empty((0, 3))
    obs_types = np.array(log["obs_type"]) if log["obs_type"] else np.array([])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    # --- Top: XY trajectories ---
    ax = axes[0]
    ax.plot(actual[:, 0],    actual[:, 1],    "b-",  lw=1.5, label="Actual target")
    ax.plot(estimated[:, 0], estimated[:, 1], "r--", lw=1.5, label="KF estimate")
    if len(obs_pos):
        for otype, color, label in [("clean",   "green",  "Clean obs"),
                                     ("noisy",   "orange", "Noisy obs"),
                                     ("outlier", "red",    "Outlier obs")]:
            mask = obs_types == otype
            if mask.any():
                ax.scatter(obs_pos[mask, 0], obs_pos[mask, 1],
                           s=8, c=color, alpha=0.5, label=label)
    ax.set_ylabel("Y position (m)")
    ax.set_xlabel("X position (m)")
    ax.set_title("Target trajectory — actual vs KF estimate (XY plane)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Bottom: estimation error over time ---
    ax2 = axes[1]
    err = np.linalg.norm(actual - estimated, axis=1)
    ax2.plot(times, err, "k-", lw=1.2)
    for t in log["no_data_t"]:
        ax2.axvline(t, color="red",    alpha=0.08, lw=0.8)
    for t in log["noisy_t"]:
        ax2.axvline(t, color="orange", alpha=0.12, lw=0.8)

    ax2.legend(handles=[
        Line2D([0], [0], color="k",      lw=1.5,           label="Estimation error (m)"),
        Line2D([0], [0], color="red",    lw=4, alpha=0.4,  label="No-data steps"),
        Line2D([0], [0], color="orange", lw=4, alpha=0.4,  label="Noisy/outlier steps"),
    ])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position error (m)")
    ax2.set_title("KF estimation error — shaded by occlusion type")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"  Graph saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ======================================================================
# Runner
# ======================================================================

def run_intercept_occluded(
    target_waypoints=None,
    target_speed=2,
    interceptor_start="0 0 1.5",
    target_start=None,
    intercept_radius=1.0,
    max_duration=30.0,
    record=False,
    video_out="intercept_occluded.mp4",
    target_path_override=None,
    debug=False,
    headless=False,
    graph_out=None,
    no_graph=False,
    seed=None,
):
    rng = np.random.RandomState(seed)

    drone_xml_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__))
        if os.path.isdir(os.path.join(os.path.dirname(__file__), "assets"))
        else os.path.join("skydio_x2")
    )

    if target_start is None:
        if target_path_override is not None:
            p = target_path_override.pos
            target_start = f"{p[0]} {p[1]} {p[2]}"
        elif target_waypoints is not None:
            wp0 = target_waypoints[0]
            target_start = f"{wp0[0]} {wp0[1]} {wp0[2]}"
        else:
            target_start = "8 8 2"

    xml   = build_intercept_xml(drone_xml_dir, interceptor_start, target_start)
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    target_qpos_start = 7
    fpv_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv")
    renderer = mujoco.Renderer(model, height=480, width=640)

    steps_per_frame = 4
    dt_ctrl = model.opt.timestep * steps_per_frame

    cfg = MPPIControlConfig.from_json("skydio_x2/best_weights_mppi_occluded.json")
    controller = MPPIController(horizon=cfg.horizon, n_samples=cfg.n_samples, dt=dt_ctrl)
    cfg.apply_to(controller)

    target_path = target_path_override or ScriptedTargetPath(target_waypoints, speed=target_speed)
    hover_thrust = 3.2495625
    dt_render    = 1.0 / 60.0

    # Initialise Kalman filter at target start position
    ts = np.array([float(v) for v in target_start.split()])
    kf = TargetKalmanFilter(ts, dt_ctrl)

    # Trajectory log for graphing
    log = {"t": [], "actual": [], "estimated": [],
           "obs_t": [], "obs_pos": [], "obs_type": [],
           "no_data_t": [], "noisy_t": []}

    intercepted   = False
    min_dist_seen = float("inf")
    phase         = "cruise"
    prev_target_pos = None

    print(f"\n{'='*60}")
    print(f"  DRONE INTERCEPT (OCCLUDED) — MPPI + Kalman Filter")
    print(f"{'='*60}")
    print(f"  Interceptor start : {interceptor_start}")
    print(f"  Target start      : {target_start}")
    print(f"  No-data model     : initial={NO_DATA_INITIAL} decay={NO_DATA_DECAY} floor={NO_DATA_FLOOR}")
    print(f"  Noisy model       : initial={NOISY_INITIAL} decay={NOISY_DECAY} floor={NOISY_FLOOR}")
    print(f"  Obs noise std     : {OBS_NOISE_STD} m  outlier_prob={OUTLIER_PROB}  outlier_scale={OUTLIER_SCALE}x")
    print(f"{'='*60}\n")

    def _run_loop(viewer):
        nonlocal intercepted, min_dist_seen, phase, prev_target_pos
        sim_time = 0.0

        try:
            while viewer is None or viewer.is_running():
                step_start = time.time()

                if sim_time > max_duration:
                    print(f"\n  TIMEOUT at t={sim_time:.1f}s  (closest: {min_dist_seen:.2f}m)")
                    break

                # --- True target state ---
                true_pos, true_vel = target_path.get_state(dt_ctrl)

                # Pin target in sim
                data.qpos[target_qpos_start:target_qpos_start + 3] = true_pos
                data.qpos[target_qpos_start + 3:target_qpos_start + 7] = [1, 0, 0, 0]
                data.qvel[6:9]  = true_vel
                data.qvel[9:12] = [0, 0, 0]

                # --- Observation model ---
                drone_pos  = data.qpos[:3].copy()
                obs, obs_type = observe_target(true_pos, drone_pos, rng)

                # --- Kalman filter update ---
                est_pos, est_vel = kf.step(obs)

                # --- Log ---
                log["t"].append(sim_time)
                log["actual"].append(true_pos.copy())
                log["estimated"].append(est_pos.copy())
                if obs_type == "no_data":
                    log["no_data_t"].append(sim_time)
                else:
                    log["obs_t"].append(sim_time)
                    log["obs_pos"].append(obs.copy())
                    log["obs_type"].append(obs_type)
                    if obs_type in ("noisy", "outlier"):
                        log["noisy_t"].append(sim_time)

                # --- Drone state ---
                drone_vel  = data.qvel[:3].copy()
                drone_quat = data.qpos[3:7].copy()
                tilt_deg   = np.degrees(2.0 * np.arcsin(np.clip(
                    np.sqrt(drone_quat[1]**2 + drone_quat[2]**2), 0, 1)))
                dist_to_target = np.linalg.norm(drone_pos - true_pos)
                min_dist_seen  = min(min_dist_seen, dist_to_target)

                # --- Intercept check (uses true pos) ---
                if dist_to_target < intercept_radius:
                    print(f"\n{'*'*60}")
                    print(f"  TARGET INTERCEPTED at t={sim_time:.2f}s")
                    print(f"  Distance: {dist_to_target:.4f} m")
                    print(f"{'*'*60}\n")
                    intercepted = True
                    break

                # --- Phase transition ---
                drone_speed = np.linalg.norm(drone_vel)
                if (phase == "cruise"
                        and np.linalg.norm(drone_pos - est_pos) <= cfg.strike_range
                        and 2.0 <= drone_speed <= 8.0
                        and tilt_deg < 45.0):
                    phase = "strike"
                    cfg.apply_strike(controller)
                    controller.reset()
                    controller._disturbance_force  = np.zeros(3)
                    controller._predicted_vel      = None
                    print(f"  t={sim_time:.2f}s  STRIKE PHASE  dist={dist_to_target:.2f}m")

                # --- Lead point from KF estimate ---
                if phase == "strike":
                    lead = est_pos + est_vel * dt_ctrl * 3
                else:
                    lead = compute_lead_point(
                        drone_pos, drone_vel, est_pos, est_vel,
                        drone_speed=drone_speed,
                    )

                ctrl = controller.solve(data, lead,
                                        target_vel=est_vel if phase == "strike" else None)
                thrust_offset, pitch_cmd, roll_cmd, yaw_cmd = ctrl
                base = hover_thrust + thrust_offset
                apply_motor_mixing(data, pitch_cmd, roll_cmd, yaw_cmd, base)

                # Re-pin target, step physics
                data.qpos[target_qpos_start:target_qpos_start + 3] = true_pos
                data.qpos[target_qpos_start + 3:target_qpos_start + 7] = [1, 0, 0, 0]
                data.qvel[6:9]  = true_vel
                data.qvel[9:12] = [0, 0, 0]
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)
                sim_time += dt_ctrl

                data.qpos[target_qpos_start:target_qpos_start + 3] = true_pos
                data.qpos[target_qpos_start + 3:target_qpos_start + 7] = [1, 0, 0, 0]
                data.qvel[6:9]  = true_vel
                data.qvel[9:12] = [0, 0, 0]

                if debug:
                    print(
                        f"  t={sim_time:.2f}s  occluded={occluded}  "
                        f"est_err={np.linalg.norm(est_pos - true_pos):.2f}m  "
                        f"dist={dist_to_target:.2f}m  phase={phase}"
                    )

                if not headless:
                    viewer.sync()
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=fpv_cam_id)
                    fpv_img = renderer.render()
                    fpv_bgr = cv2.cvtColor(fpv_img, cv2.COLOR_RGB2BGR)
                    fpv_bgr = cv2.flip(fpv_bgr, 0)
                    occ_color = (0, 0, 255) if obs_type == "no_data" else (0, 165, 255) if obs_type in ("noisy", "outlier") else (0, 255, 0)
                    cv2.putText(fpv_bgr,
                        f"{obs_type.upper()}  t={sim_time:.1f}s  "
                        f"dist={dist_to_target:.2f}m  est_err={np.linalg.norm(est_pos-true_pos):.2f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, occ_color, 2)
                    cv2.imshow("Interceptor FPV (Occluded)", fpv_bgr)
                    cv2.waitKey(1)
                    frame_elapsed = time.time() - step_start
                    if frame_elapsed < dt_render:
                        time.sleep(dt_render - frame_elapsed)

                # Periodic status
                if int(sim_time * 10) % 20 == 0 and int(sim_time * 10) > 0:
                    n = max(len(log["t"]), 1)
                    nd_pct = 100 * len(log["no_data_t"]) / n
                    ny_pct = 100 * len(log["noisy_t"]) / n
                    print(
                        f"  t={sim_time:.1f}s  dist={dist_to_target:.2f}m  "
                        f"est_err={np.linalg.norm(est_pos-true_pos):.2f}m  "
                        f"no_data={nd_pct:.0f}%  noisy={ny_pct:.0f}%  phase={phase}"
                    )

        finally:
            if not headless:
                cv2.destroyAllWindows()
                renderer.close()

    if headless:
        _run_loop(None)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth  = -60
            viewer.cam.elevation = -30
            viewer.cam.distance  = 10.0
            viewer.cam.lookat    = [2.5, 2.5, 2.0]
            _run_loop(viewer)

    n = max(len(log["t"]), 1)
    nd_pct = 100 * len(log["no_data_t"]) / n
    ny_pct = 100 * len(log["noisy_t"]) / n
    print(f"Intercept scenario complete — {'HIT' if intercepted else 'MISS'}  "
          f"(no-data: {nd_pct:.1f}%  noisy/outlier: {ny_pct:.1f}%  "
          f"total occlusion: {nd_pct + ny_pct:.1f}%)")

    # Generate graph
    if not no_graph:
        plot_trajectory(log, save_path=graph_out)

    return intercepted


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Occluded drone intercept simulation")
    parser.add_argument("--mode", choices=["scripted", "evasive"], default="scripted")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target-speed", type=float, default=4.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--video-out", default="intercept_occluded.mp4")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--graph-out", default=None,
                        help="Path to save trajectory graph (e.g. traj.png). "
                             "If omitted, graph is shown interactively.")
    parser.add_argument("--no-graph", action="store_true",
                        help="Disable graph output entirely.")
    args = parser.parse_args()

    failed_seeds = []

    for iteration in range(args.iters):
        if args.iters > 1:
            print(f"\n{'='*60}\n  ITERATION {iteration + 1}/{args.iters}\n{'='*60}")

        if args.mode == "evasive":
            seed = args.seed if args.seed is not None else int(np.random.randint(0, 100000))
            print(f"  Seed: {seed}")
            target = RandomEvasiveTarget(
                start_pos=(8, 8, 6.0),
                max_speed=args.target_speed,
                max_accel=3.0,
                bias_pos=(-8, 0, 5.0),
                bias_strength=2.5,
                altitude_range=(4.0, 8.0),
                bounds=((-10, 10), (-10, 10)),
                seed=seed,
            )
            hit = run_intercept_occluded(
                interceptor_start="0 0 5.0",
                intercept_radius=1.0,
                max_duration=45.0,
                record=args.record,
                video_out=args.video_out,
                target_path_override=target,
                debug=args.debug,
                headless=args.headless,
                graph_out=args.graph_out,
                no_graph=args.no_graph,
                seed=seed,
            )
        else:
            seed = None
            TARGET_PATH = [
                [10, 10, 6.0], [8, 6, 6.5], [6, 8, 6.0], [4, 4, 6.5],
                [2, 6, 6.0],   [0, 2, 6.5], [-2, 4, 6.0], [-4, 0, 6.5],
                [-6, 2, 6.0],  [-8, -2, 6.5], [-6, -4, 6.0], [-4, -2, 6.5],
                [-2, -4, 6.0], [0, 0, 6.5],
            ]
            hit = run_intercept_occluded(
                target_waypoints=TARGET_PATH,
                target_speed=args.target_speed,
                interceptor_start="0 0 5.0",
                intercept_radius=1.0,
                max_duration=45.0,
                record=args.record,
                video_out=args.video_out,
                debug=args.debug,
                headless=args.headless,
                graph_out=args.graph_out,
                no_graph=args.no_graph,
            )

        if not hit:
            print(f"\n  FAILED on iteration {iteration + 1}" +
                  (f"  seed={seed}" if seed is not None else ""))
            if seed is not None:
                failed_seeds.append(seed)
        elif args.iters > 1:
            print(f"  HIT  iteration {iteration + 1}/{args.iters}")

    if args.iters > 1:
        passed = args.iters - len(failed_seeds)
        print(f"\nPassed: {passed}/{args.iters} ({100 * passed / args.iters:.1f}%)")
        if failed_seeds:
            print(f"Failed seeds ({len(failed_seeds)}): {failed_seeds}")
