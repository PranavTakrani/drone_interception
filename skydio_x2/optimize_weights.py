"""
optimize_weights.py -- Overnight weight optimizer for the intercept MPC.

Runs random search over the weight space, evaluating each candidate on a
batch of random seeds. Saves the best config found to a JSON file.

Usage:
    python skydio_x2/optimize_weights.py
    python skydio_x2/optimize_weights.py --trials 500 --seeds-per-trial 10
    python skydio_x2/optimize_weights.py --out best_weights.json
    python skydio_x2/optimize_weights.py --resume best_weights.json  # warm-start
"""

import argparse
import json
import os
import sys
import time
import copy

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skydio_x2.mpc_controller import MPCController
from skydio_x2.mpc_control_config import MPCControlConfig, DEFAULT_MPC_CONFIG
from skydio_x2.intercept_controller import (
    RandomEvasiveTarget,
    ScriptedTargetPath,
    build_intercept_xml,
    predict_target_pos,
)
from skydio_x2.skydio_x2_movement import apply_motor_mixing


# ======================================================================
# Search space — (min, max, log_scale)
# ======================================================================
SEARCH_SPACE = {
    # cruise
    "cruise_w_pos":          (5.0,  40.0,  True),
    "cruise_w_vel":          (0.0,  10.0,  False),
    "cruise_w_pos_running":  (1.0,  20.0,  True),
    "cruise_w_tilt":         (0.5,  10.0,  True),
    "cruise_w_tilt_running": (0.5,  10.0,  True),
    "cruise_att_max":        (0.5,   1.4,  False),
    "cruise_thrust_max":     (4.0,  12.0,  False),
    # strike
    "strike_w_pos":          (20.0, 100.0, True),
    "strike_w_vel":          (0.0,  10.0,  False),
    "strike_w_pos_running":  (0.0,  20.0,  False),
    "strike_w_tilt":         (0.0,   5.0,  False),
    "strike_att_max":        (0.6,   1.4,  False),
    "strike_thrust_max":     (6.0,  13.0,  False),
    # phase transition
    "strike_range":          (2.0,   7.0,  False),
}


def sample_config(rng, base: MPCControlConfig = None, noise_scale: float = 1.0) -> MPCControlConfig:
    """Sample a random config, optionally perturbing around a base config."""
    cfg = copy.deepcopy(DEFAULT_MPC_CONFIG)
    for key, (lo, hi, log) in SEARCH_SPACE.items():
        if base is not None:
            # Perturb around base value
            base_val = getattr(base, key)
            if log and base_val > 0:
                log_lo, log_hi = np.log(lo), np.log(hi)
                log_base = np.log(base_val)
                span = (log_hi - log_lo) * noise_scale * 0.3
                val = np.exp(np.clip(rng.normal(log_base, span), log_lo, log_hi))
            else:
                span = (hi - lo) * noise_scale * 0.3
                val = np.clip(rng.normal(base_val, span), lo, hi)
        else:
            if log:
                val = np.exp(rng.uniform(np.log(lo), np.log(hi)))
            else:
                val = rng.uniform(lo, hi)
        setattr(cfg, key, float(val))
    return cfg


# ======================================================================
# Single trial runner (headless)
# ======================================================================

def run_trial(cfg: MPCControlConfig, seed: int, max_time: float = 30.0,
              intercept_radius: float = 1.0) -> dict:
    """Run one headless intercept trial. Returns dict with hit, min_dist, time."""
    rng = np.random.RandomState(seed)

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

    controller = MPCController(
        horizon=cfg.horizon,
        n_samples=cfg.n_samples,
        n_elite=cfg.n_elite,
        n_iterations=cfg.n_iterations,
        dt=dt_ctrl,
    )
    cfg.apply_to(controller)

    max_steps = int(max_time / dt_ctrl)
    min_dist = float("inf")
    phase = "cruise"
    prev_target_pos = None

    for step in range(max_steps):
        target_pos, target_vel = target.get_state(dt_ctrl)

        # Jink detection
        if prev_target_pos is not None:
            accel_est = np.linalg.norm(target_vel - data.qvel[6:9]) / dt_ctrl
            if accel_est > 8.0:
                controller._mean = None
        prev_target_pos = target_pos.copy()

        # Pin target
        data.qpos[target_qpos_start:target_qpos_start + 3] = target_pos
        data.qpos[target_qpos_start + 3:target_qpos_start + 7] = [1, 0, 0, 0]
        data.qvel[6:9] = target_vel
        data.qvel[9:12] = [0, 0, 0]

        drone_pos = data.qpos[:3].copy()
        drone_vel = data.qvel[:3].copy()
        dist = np.linalg.norm(drone_pos - target_pos)
        min_dist = min(min_dist, dist)

        if dist < intercept_radius:
            return {"hit": True, "min_dist": min_dist, "time": step * dt_ctrl}

        # Phase transition using predicted pos
        pred_target = predict_target_pos(target_pos, target_vel,
                                         controller.horizon * 2, dt_ctrl)
        dist_to_pred = np.linalg.norm(drone_pos - pred_target)
        if phase == "cruise" and dist_to_pred <= cfg.strike_range:
            phase = "strike"
            cfg.apply_strike(controller)
            controller._disturbance_force = np.zeros(3)
            controller._predicted_vel = None
            controller.reset()

        # Lead point
        lead = predict_target_pos(target_pos, target_vel,
                                   controller.horizon * 2, dt_ctrl)
        lead_offset = lead - target_pos
        lead_dist = np.linalg.norm(lead_offset)
        if lead_dist > 5.0:
            lead = target_pos + lead_offset * (5.0 / lead_dist)

        if phase == "strike":
            lead = predict_target_pos(target_pos, target_vel,
                                       controller.horizon, dt_ctrl)
            blend = np.clip((dist - 1.5) / 2.5, 0.0, 1.0)
            lead = blend * lead + (1.0 - blend) * target_pos

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

        # Reset phase for next trial if needed
        if phase == "strike" and dist > cfg.strike_range * 2:
            phase = "cruise"
            cfg.apply_cruise(controller)

    return {"hit": False, "min_dist": min_dist, "time": None}


# ======================================================================
# Score function
# ======================================================================

def score_config(cfg: MPCControlConfig, seeds: list, max_time: float) -> dict:
    """Evaluate config on multiple seeds. Returns score dict."""
    hits = 0
    min_dists = []
    times = []

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

    # Score: maximize hit rate, then minimize time, then minimize miss distance
    score = hit_rate * 1000.0 - avg_time * 0.5 - avg_min_dist * 2.0

    return {
        "score": score,
        "hit_rate": hit_rate,
        "avg_min_dist": avg_min_dist,
        "avg_time": avg_time,
        "hits": hits,
        "n": len(seeds),
    }


# ======================================================================
# Main optimizer
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="MPC weight optimizer")
    parser.add_argument("--trials", type=int, default=200,
                        help="Number of random configs to try (default: 200)")
    parser.add_argument("--seeds-per-trial", type=int, default=8,
                        help="Seeds evaluated per config (default: 8)")
    parser.add_argument("--max-time", type=float, default=30.0,
                        help="Max sim time per trial in seconds (default: 30)")
    parser.add_argument("--out", default="skydio_x2/best_weights.json",
                        help="Output path for best config (default: skydio_x2/best_weights.json)")
    parser.add_argument("--resume", default=None,
                        help="Resume from existing best config JSON")
    parser.add_argument("--exploit-ratio", type=float, default=0.5,
                        help="Fraction of trials that perturb best vs random (default: 0.5)")
    args = parser.parse_args()

    rng = np.random.RandomState(0)

    # Fixed evaluation seeds — same across all trials for fair comparison
    eval_seeds = list(range(args.seeds_per_trial))

    # Load or init best config
    if args.resume and os.path.exists(args.resume):
        best_cfg = MPCControlConfig.from_json(args.resume)
        print(f"Resumed from {args.resume}")
    else:
        best_cfg = copy.deepcopy(DEFAULT_MPC_CONFIG)

    # Score the starting config
    print(f"\nScoring baseline config on {args.seeds_per_trial} seeds...")
    best_result = score_config(best_cfg, eval_seeds, args.max_time)
    best_score = best_result["score"]
    print(f"  Baseline: score={best_score:.2f}  hit_rate={best_result['hit_rate']:.0%}  "
          f"avg_min_dist={best_result['avg_min_dist']:.3f}m  "
          f"avg_time={best_result['avg_time']:.1f}s")

    print(f"\nStarting {args.trials} trials ({args.seeds_per_trial} seeds each)...\n")
    t_start = time.time()

    for trial in range(args.trials):
        # Exploit best or explore randomly
        if rng.random() < args.exploit_ratio:
            noise = max(0.3, 1.0 - trial / args.trials)  # anneal noise
            cfg = sample_config(rng, base=best_cfg, noise_scale=noise)
        else:
            cfg = sample_config(rng)

        result = score_config(cfg, eval_seeds, args.max_time)

        elapsed = time.time() - t_start
        eta = elapsed / (trial + 1) * (args.trials - trial - 1)

        improved = result["score"] > best_score
        marker = " ***" if improved else ""

        # Progress bar
        bar_width = 30
        filled = int(bar_width * (trial + 1) / args.trials)
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = (trial + 1) / args.trials * 100
        print(f"\r  [{bar}] {pct:5.1f}%  "
              f"score={result['score']:7.2f}  "
              f"hit={result['hit_rate']:.0%}  "
              f"best={best_score:.2f}  "
              f"eta={eta/60:.1f}min{marker}",
              end="", flush=True)

        if improved:
            best_score = result["score"]
            best_cfg = cfg
            best_result = result
            best_cfg.to_json(args.out)
            print(f"\n    *** New best: score={best_score:.2f}  "
                  f"hit={best_result['hit_rate']:.0%}  "
                  f"min_d={best_result['avg_min_dist']:.3f}m  "
                  f"-> {args.out}")

    print(f"\n{'='*60}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"  Best score:    {best_score:.2f}")
    print(f"  Hit rate:      {best_result['hit_rate']:.0%} ({best_result['hits']}/{best_result['n']})")
    print(f"  Avg min dist:  {best_result['avg_min_dist']:.3f}m")
    print(f"  Avg time:      {best_result['avg_time']:.1f}s")
    print(f"  Saved to:      {args.out}")
    print(f"{'='*60}\n")

    # Print the winning config
    print("Best config:")
    for key in SEARCH_SPACE:
        print(f"  {key} = {getattr(best_cfg, key):.4f}")


if __name__ == "__main__":
    main()
