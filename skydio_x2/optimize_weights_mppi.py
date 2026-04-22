"""
optimize_weights_mppi.py -- Weight optimizer for the MPPI intercept controller.

Usage:
    python skydio_x2/optimize_weights_mppi.py
    python skydio_x2/optimize_weights_mppi.py --trials 500 --seeds-per-trial 10
    python skydio_x2/optimize_weights_mppi.py --resume skydio_x2/best_weights_mppi.json
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import mujoco
import cma

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skydio_x2.mppi_controller import MPPIController
from skydio_x2.mpc_control_config import MPPIControlConfig, DEFAULT_MPPI_CONFIG
from skydio_x2.intercept_controller import (
    RandomEvasiveTarget,
    build_intercept_xml,
    compute_lead_point,
)
from skydio_x2.skydio_x2_movement import apply_motor_mixing


# ======================================================================
# Search space — (min, max, log_scale)
# ======================================================================
SEARCH_SPACE = {
    # MPPI algorithm
    "lam":                   (0.01,  0.5,   True),
    # cruise
    "cruise_w_pos":          (5.0,   50.0,  True),
    "cruise_w_vel":          (0.0,   10.0,  False),
    "cruise_w_pos_running":  (0.5,   20.0,  True),
    "cruise_w_tilt":         (5.0,   60.0,  True),
    "cruise_w_tilt_running": (5.0,   40.0,  True),
    "cruise_w_ctrl_rate":    (0.1,    3.0,  True),
    "cruise_att_max":        (0.5,    1.4,  False),
    "cruise_thrust_max":     (4.0,   12.0,  False),
    # strike
    "strike_w_pos":          (20.0, 120.0,  True),
    "strike_w_vel":          (0.0,   10.0,  False),
    "strike_w_pos_running":  (0.0,   20.0,  False),
    "strike_w_tilt":         (0.0,    5.0,  False),
    "strike_w_closing":      (0.0,   10.0,  False),
    "strike_att_max":        (0.6,    1.4,  False),
    "strike_thrust_max":     (6.0,   13.0,  False),
    # phase transition
    "strike_range":          (3.0,   10.0,  False),
}


PARAM_KEYS = list(SEARCH_SPACE.keys())


def vec_to_config(x: np.ndarray) -> MPPIControlConfig:
    """Map CMA-ES vector (unbounded reals) → MPPIControlConfig via per-param transform."""
    cfg = copy.deepcopy(DEFAULT_MPPI_CONFIG)
    for i, key in enumerate(PARAM_KEYS):
        lo, hi, log = SEARCH_SPACE[key]
        if log:
            val = np.exp(x[i])
            val = float(np.clip(val, lo, hi))
        else:
            # sigmoid squash into [lo, hi]
            val = float(lo + (hi - lo) * (1 / (1 + np.exp(-x[i]))))
        setattr(cfg, key, val)
    return cfg


def config_to_vec(cfg: MPPIControlConfig) -> np.ndarray:
    """Map MPPIControlConfig → CMA-ES vector."""
    x = np.zeros(len(PARAM_KEYS))
    for i, key in enumerate(PARAM_KEYS):
        lo, hi, log = SEARCH_SPACE[key]
        val = getattr(cfg, key)
        if log:
            x[i] = np.log(max(val, 1e-9))
        else:
            p = np.clip((val - lo) / (hi - lo), 1e-6, 1 - 1e-6)
            x[i] = np.log(p / (1 - p))  # logit
    return x


# ======================================================================
# Single trial runner
# ======================================================================

def run_trial(cfg: MPPIControlConfig, seed: int, max_time: float = 30.0,
              intercept_radius: float = 1.0) -> dict:
    target = RandomEvasiveTarget(
        start_pos=(8, 8, 6.0),
        max_speed=4.0,
        max_accel=3.0,
        bias_pos=(-8, 0, 5.0),
        bias_strength=2.5,
        altitude_range=(4.0, 8.0),
        bounds=((-10, 10), (-10, 10)),
        seed=seed,
    )

    drone_xml_dir = os.path.abspath(os.path.dirname(__file__))
    xml = build_intercept_xml(drone_xml_dir, "0 0 5.0",
                              f"{target.pos[0]} {target.pos[1]} {target.pos[2]}")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    steps_per_frame = 4
    dt_ctrl = model.opt.timestep * steps_per_frame
    hover_thrust = 3.2495625
    target_qpos_start = 7

    controller = MPPIController(horizon=cfg.horizon, n_samples=cfg.n_samples, dt=dt_ctrl)
    cfg.apply_to(controller)

    max_steps = int(max_time / dt_ctrl)
    min_dist = float("inf")
    phase = "cruise"
    prev_target_pos = None

    for _ in range(max_steps):
        target_pos, target_vel = target.get_state(dt_ctrl)

        if prev_target_pos is not None:
            accel_est = np.linalg.norm(target_vel - data.qvel[6:9]) / dt_ctrl
            if accel_est > 8.0:
                controller._mean = None
        prev_target_pos = target_pos.copy()

        data.qpos[target_qpos_start:target_qpos_start + 3] = target_pos
        data.qpos[target_qpos_start + 3:target_qpos_start + 7] = [1, 0, 0, 0]
        data.qvel[6:9] = target_vel
        data.qvel[9:12] = [0, 0, 0]

        drone_pos = data.qpos[:3].copy()
        drone_vel = data.qvel[:3].copy()
        drone_quat = data.qpos[3:7].copy()
        tilt_deg = np.degrees(2.0 * np.arcsin(np.clip(
            np.sqrt(drone_quat[1]**2 + drone_quat[2]**2), 0, 1)))
        dist = np.linalg.norm(drone_pos - target_pos)
        min_dist = min(min_dist, dist)

        if dist < intercept_radius:
            return {"hit": True, "min_dist": min_dist, "time": _ * dt_ctrl}

        drone_speed = np.linalg.norm(drone_vel)
        if phase == "cruise" and dist <= cfg.strike_range and 2.0 <= drone_speed <= 8.0 and tilt_deg < 45.0:
            phase = "strike"
            cfg.apply_strike(controller)
            controller._disturbance_force = np.zeros(3)
            controller._predicted_vel = None
            controller.reset()

        if phase == "strike":
            lead = target_pos + target_vel * dt_ctrl * 3
        else:
            lead = compute_lead_point(drone_pos, drone_vel, target_pos, target_vel,
                                      drone_speed=drone_speed)

        ctrl = controller.solve(data, lead,
                                target_vel=target_vel if phase == "strike" else None)
        t_off, p_cmd, r_cmd, y_cmd = ctrl
        apply_motor_mixing(data, p_cmd, r_cmd, y_cmd, hover_thrust + t_off)

        data.qpos[target_qpos_start:target_qpos_start + 3] = target_pos
        data.qpos[target_qpos_start + 3:target_qpos_start + 7] = [1, 0, 0, 0]
        data.qvel[6:9] = target_vel
        data.qvel[9:12] = [0, 0, 0]
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        data.qpos[target_qpos_start:target_qpos_start + 3] = target_pos
        data.qpos[target_qpos_start + 3:target_qpos_start + 7] = [1, 0, 0, 0]
        data.qvel[6:9] = target_vel
        data.qvel[9:12] = [0, 0, 0]

        if phase == "strike" and dist > cfg.strike_range * 2:
            phase = "cruise"
            cfg.apply_cruise(controller)

    return {"hit": False, "min_dist": min_dist, "time": None}


# ======================================================================
# Score function
# ======================================================================

def score_config(cfg: MPPIControlConfig, seeds: list, max_time: float) -> dict:
    hits, min_dists, times = 0, [], []
    for seed in seeds:
        try:
            r = run_trial(cfg, seed, max_time=max_time)
        except Exception:
            r = {"hit": False, "min_dist": 99.0, "time": None}
        if r["hit"]:
            hits += 1
            times.append(r["time"])
        min_dists.append(r["min_dist"])

    hit_rate = hits / len(seeds)
    avg_min_dist = float(np.mean(min_dists))
    avg_time = float(np.mean(times)) if times else max_time
    score = hit_rate * 1000.0 - avg_time * 0.5 - avg_min_dist * 2.0
    return {"score": score, "hit_rate": hit_rate, "avg_min_dist": avg_min_dist,
            "avg_time": avg_time, "hits": hits, "n": len(seeds)}


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200,
                        help="CMA-ES function evaluations (default: 200)")
    parser.add_argument("--seeds-per-trial", type=int, default=8)
    parser.add_argument("--max-time", type=float, default=30.0)
    parser.add_argument("--out", default="skydio_x2/best_weights_mppi.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--sigma0", type=float, default=0.5,
                        help="Initial CMA-ES step size (default: 0.5)")
    args = parser.parse_args()

    eval_seeds = list(range(args.seeds_per_trial))

    # Warm-start from existing best or defaults
    if args.resume and os.path.exists(args.resume):
        best_cfg = MPPIControlConfig.from_json(args.resume)
        print(f"Resumed from {args.resume}")
    else:
        best_cfg = copy.deepcopy(DEFAULT_MPPI_CONFIG)

    print(f"\nScoring baseline on {args.seeds_per_trial} seeds...")
    best_result = score_config(best_cfg, eval_seeds, args.max_time)
    best_score = best_result["score"]
    print(f"  Baseline: score={best_score:.2f}  hit={best_result['hit_rate']:.0%}  "
          f"min_d={best_result['avg_min_dist']:.3f}m")

    x0 = config_to_vec(best_cfg)
    es = cma.CMAEvolutionStrategy(x0, args.sigma0, {
        "maxfevals": args.trials,
        "verbose": -9,   # suppress CMA's own output
        "tolx": 1e-4,
        "tolfun": 1e-3,
    })

    print(f"\nCMA-ES: {args.trials} evals, popsize={es.popsize}, "
          f"{args.seeds_per_trial} seeds/eval\n")
    t_start = time.time()
    eval_count = 0

    while not es.stop():
        solutions = es.ask()
        fitnesses = []

        for x in solutions:
            cfg = vec_to_config(x)
            result = score_config(cfg, eval_seeds, args.max_time)
            fitnesses.append(-result["score"])  # CMA-ES minimises
            eval_count += 1

            elapsed = time.time() - t_start
            eta = elapsed / eval_count * (args.trials - eval_count) if eval_count < args.trials else 0
            bar = "█" * int(30 * eval_count / args.trials) + "░" * (30 - int(30 * eval_count / args.trials))
            print(f"\r  [{bar}] {eval_count/args.trials*100:5.1f}%  "
                  f"score={result['score']:7.2f}  hit={result['hit_rate']:.0%}  "
                  f"best={best_score:.2f}  eta={eta/60:.1f}min",
                  end="", flush=True)

            if result["score"] > best_score:
                best_score = result["score"]
                best_cfg = cfg
                best_result = result
                best_cfg.to_json(args.out)
                print(f"\n  *** New best: score={best_score:.2f}  hit={best_result['hit_rate']:.0%}  "
                      f"min_d={best_result['avg_min_dist']:.3f}m  -> {args.out}")

        es.tell(solutions, fitnesses)

    print(f"\n{'='*60}")
    print(f"  DONE  score={best_score:.2f}  hit={best_result['hit_rate']:.0%}  "
          f"({best_result['hits']}/{best_result['n']})  saved to {args.out}")
    print(f"{'='*60}\n")
    for key in PARAM_KEYS:
        print(f"  {key} = {getattr(best_cfg, key):.4f}")


if __name__ == "__main__":
    main()
