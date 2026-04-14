"""
optimize_weights.py -- Intercept diagnostic harness.

Runs the intercept controller across diverse target paths and prints
per-frame telemetry so you can see exactly where and why things go wrong
(overshoot, lead-point error, velocity buildup, etc.).

Usage:
    python skydio_x2/optimize_weights.py                              # all scenarios
    python skydio_x2/optimize_weights.py --scenario straight          # one scenario
    python skydio_x2/optimize_weights.py --scenario evasive           # random evasive target
    python skydio_x2/optimize_weights.py --scenario evasive --visualize  # with viewer
    python skydio_x2/optimize_weights.py --config tuned.json          # custom weights
"""

import argparse
import os
import sys
import time

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skydio_x2.mpc_controller import MPCController
from skydio_x2.intercept_controller import (
    ScriptedTargetPath,
    RandomEvasiveTarget,
    build_intercept_xml,
    compute_lead_point,
    run_intercept,
)
from skydio_x2.intercept_config import InterceptMPCConfig, DEFAULT_INTERCEPT_CONFIG
from skydio_x2.skydio_x2_movement import apply_motor_mixing


# ======================================================================
# Test scenarios — diverse path types
# ======================================================================

SCENARIOS = {
    "straight": {
        "waypoints": [[5, 5, 2.0], [5, -5, 2.0]],
        "speed": 3.0,
        "desc": "Straight line — baseline, should always hit",
    },
    "straight_fast": {
        "waypoints": [[5, 5, 2.0], [5, -5, 2.0]],
        "speed": 6.0,
        "desc": "Fast straight line — tests closing speed",
    },
    "scurve": {
        "waypoints": [
            [8, 8, 2.0], [6, 4, 2.5], [4, 6, 2.0],
            [2, 2, 2.5], [0, 4, 2.0], [-2, 0, 2.5],
        ],
        "speed": 4.0,
        "desc": "Evasive S-curve — tests turn tracking",
    },
    "altitude": {
        "waypoints": [
            [6, 0, 1.5], [6, 4, 3.5], [2, 4, 1.0],
            [2, 0, 3.0], [0, 0, 2.0],
        ],
        "speed": 2.5,
        "desc": "Altitude changes — tests 3D tracking",
    },
    "figure_s": {
        "waypoints": [
            [10, 10, 2.0], [8, 6, 2.5], [6, 8, 2.0], [4, 4, 2.5],
            [2, 6, 2.0], [0, 2, 2.5], [-2, 4, 2.0], [-4, 0, 2.5],
        ],
        "speed": 4.0,
        "desc": "Figure-S at speed — the hard case",
    },
    "evasive": {
        "type": "random",
        "start_pos": [8, 8, 2.0],
        "speed": 4.0,
        "max_accel": 3.0,
        "seed": 42,
        "desc": "Random evasive — realistic unpredictable drone",
    },
}


# ======================================================================
# Diagnostic runner
# ======================================================================

def run_diagnostic(scenario_name, config, intercept_radius=1.0, max_time=30.0):
    """Run one intercept scenario and return per-frame telemetry.

    Returns a dict with:
        hit           : bool
        time          : float or None
        min_dist      : float
        max_speed     : float
        max_tilt      : float
        frames        : list of per-frame dicts
    """
    sc = SCENARIOS[scenario_name]

    # Build the target path object
    is_random = sc.get("type") == "random"
    if is_random:
        target_path = RandomEvasiveTarget(
            start_pos=sc["start_pos"],
            max_speed=sc["speed"],
            max_accel=sc["max_accel"],
            seed=sc.get("seed"),
        )
        start_pos = sc["start_pos"]
    else:
        waypoints = sc["waypoints"]
        target_path = ScriptedTargetPath(waypoints, speed=sc["speed"])
        start_pos = waypoints[0]

    drone_xml_dir = os.path.join(os.path.dirname(__file__))
    if not os.path.isdir(os.path.join(drone_xml_dir, "assets")):
        drone_xml_dir = os.path.join("skydio_x2")
    drone_xml_dir = os.path.abspath(drone_xml_dir)

    xml = build_intercept_xml(
        drone_xml_dir, "0 0 1.5", f"{start_pos[0]} {start_pos[1]} {start_pos[2]}"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    steps_per_frame = 4
    dt_ctrl = model.opt.timestep * steps_per_frame
    hover_thrust = 3.2495625

    controller = MPCController(
        horizon=config.horizon,
        n_samples=config.n_samples,
        n_elite=config.n_elite,
        n_iterations=config.n_iterations,
        dt=dt_ctrl,
    )
    config.apply_to(controller)

    max_steps = int(max_time / dt_ctrl)

    frames = []
    min_dist = float("inf")
    max_speed = 0.0
    max_tilt = 0.0
    last_thrust = 0.0
    hit = False
    hit_time = None
    phase = "cruise"

    for step in range(max_steps):
        sim_time = step * dt_ctrl

        tpos, tvel = target_path.get_state(dt_ctrl)
        data.qpos[7:10] = tpos
        data.qpos[10:14] = [1, 0, 0, 0]
        data.qvel[6:9] = tvel
        data.qvel[9:12] = [0, 0, 0]

        dpos = data.qpos[:3].copy()
        dvel = data.qvel[:3].copy()
        quat = data.qpos[3:7].copy()

        dist = np.linalg.norm(dpos - tpos)
        speed = np.linalg.norm(dvel)
        tilt = np.degrees(
            2.0 * np.arcsin(np.clip(np.sqrt(quat[1] ** 2 + quat[2] ** 2), 0, 1))
        )

        min_dist = min(min_dist, dist)
        max_speed = max(max_speed, speed)
        max_tilt = max(max_tilt, tilt)

        # Phase transition
        xy_dist = np.linalg.norm(dpos[:2] - tpos[:2])
        if phase == "cruise" and xy_dist <= config.strike_range:
            phase = "strike"
            config.apply_strike(controller)
            controller.reset()

        # Compute lead point for diagnostics
        lead = compute_lead_point(dpos, dvel, tpos, tvel)
        if phase == "strike":
            lead = tpos.copy()
        lead_err = np.linalg.norm(lead - tpos)

        # Log every 10th frame to keep output manageable
        if step % 10 == 0:
            # Angle between drone velocity and direction to target
            to_tgt = tpos - dpos
            to_tgt_dist = np.linalg.norm(to_tgt)
            if to_tgt_dist > 0.01 and speed > 0.1:
                cos_ang = np.dot(dvel, to_tgt) / (speed * to_tgt_dist)
                heading_err = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
            else:
                heading_err = 0.0

            frames.append({
                "t": sim_time,
                "dist": dist,
                "speed": speed,
                "tilt": tilt,
                "lead_err": lead_err,
                "heading_err": heading_err,
                "closing": np.dot(dvel - tvel, (tpos - dpos) / max(dist, 0.01)),
                "phase": phase,
            })

        if dist < intercept_radius:
            hit = True
            hit_time = sim_time
            break

        ctrl = controller.solve(data, lead)
        t_off, p_cmd, r_cmd, y_cmd = ctrl
        last_thrust = t_off

        apply_motor_mixing(data, p_cmd, r_cmd, y_cmd, hover_thrust + t_off)

        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        data.qpos[7:10] = tpos
        data.qpos[10:14] = [1, 0, 0, 0]
        data.qvel[6:9] = tvel
        data.qvel[9:12] = [0, 0, 0]

    return {
        "hit": hit,
        "time": hit_time,
        "min_dist": min_dist,
        "max_speed": max_speed,
        "max_tilt": max_tilt,
        "frames": frames,
    }


# ======================================================================
# Display
# ======================================================================

def print_diagnostic(scenario_name, result):
    sc = SCENARIOS[scenario_name]
    status = "HIT" if result["hit"] else "MISS"
    t_str = f'{result["time"]:.2f}s' if result["hit"] else "N/A"

    print(f"\n{'=' * 72}")
    print(f"  {scenario_name:15s}  [{status}]  t={t_str}  "
          f"min_d={result['min_dist']:.3f}m  "
          f"max_spd={result['max_speed']:.1f}m/s  "
          f"max_tilt={result['max_tilt']:.1f}deg")
    print(f"  {sc['desc']}")
    if sc.get("type") == "random":
        print(f"  max_speed={sc['speed']}m/s  max_accel={sc['max_accel']}m/s^2")
    else:
        print(f"  target_speed={sc['speed']}m/s  waypoints={len(sc['waypoints'])}")
    print(f"{'=' * 72}")

    # Column headers
    print(f"  {'t':>5s}  {'phase':>6s}  {'dist':>6s}  {'speed':>5s}  {'close':>6s}  "
          f"{'lead_e':>6s}  {'head_e':>6s}  {'tilt':>5s}  notes")
    print(f"  {'---':>5s}  {'---':>6s}  {'---':>6s}  {'---':>5s}  {'---':>6s}  "
          f"{'---':>6s}  {'---':>6s}  {'---':>5s}  ---")

    for f in result["frames"]:
        notes = []
        if f["speed"] > 8.0:
            notes.append("HIGH SPEED")
        if f["heading_err"] > 60.0 and f["dist"] < 5.0:
            notes.append("OFF HEADING")
        if f["closing"] < 0 and f["dist"] < 5.0:
            notes.append("DIVERGING")
        if f["lead_err"] > 3.0:
            notes.append("LEAD FAR")
        if f["tilt"] > 50.0:
            notes.append("STEEP TILT")

        note_str = "  ".join(notes)
        ph = f.get("phase", "?")[:6]
        print(f"  {f['t']:5.2f}  {ph:>6s}  {f['dist']:6.2f}  {f['speed']:5.2f}  "
              f"{f['closing']:+6.2f}  {f['lead_err']:6.2f}  "
              f"{f['heading_err']:6.1f}  {f['tilt']:5.1f}  {note_str}")


def print_summary(results):
    print(f"\n{'=' * 72}")
    print(f"  SUMMARY")
    print(f"{'=' * 72}")
    print(f"  {'scenario':15s}  {'result':>6s}  {'time':>7s}  {'min_d':>6s}  "
          f"{'max_spd':>7s}  {'max_tilt':>8s}")
    for name, r in results.items():
        status = "HIT" if r["hit"] else "MISS"
        t_str = f'{r["time"]:.2f}s' if r["hit"] else "---"
        print(f"  {name:15s}  {status:>6s}  {t_str:>7s}  "
              f"{r['min_dist']:6.3f}  {r['max_speed']:7.1f}  {r['max_tilt']:8.1f}")

    hits = sum(1 for r in results.values() if r["hit"])
    print(f"\n  {hits}/{len(results)} intercepted")
    print(f"{'=' * 72}\n")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Intercept diagnostic harness")
    parser.add_argument(
        "--scenario", choices=list(SCENARIOS.keys()),
        help="Run a single scenario (default: all)",
    )
    parser.add_argument(
        "--config", default=None,
        help="Load weights from JSON file (default: use intercept_config defaults)",
    )
    parser.add_argument(
        "--max-time", type=float, default=30.0,
        help="Max sim time per scenario in seconds (default: 30)",
    )
    parser.add_argument(
        "--radius", type=float, default=0.5,
        help="Intercept radius in metres (default: 0.5)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Launch MuJoCo viewer + FPV window (single scenario only)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed for evasive scenario",
    )
    args = parser.parse_args()

    if args.config:
        config = InterceptMPCConfig.from_json(args.config)
        print(f"  Loaded config from {args.config}")
    else:
        config = DEFAULT_INTERCEPT_CONFIG

    print(f"\n  Weights:")
    print(f"    w_pos={config.w_pos}  w_vel={config.w_vel}  "
          f"w_vel_run={config.w_vel_running}  w_pos_run={config.w_pos_running}")
    print(f"    att_max={config.att_max}  thrust_max={config.thrust_max}")
    print(f"    horizon={config.horizon}  samples={config.n_samples}")

    # --visualize: launch the full viewer for a single scenario
    if args.visualize:
        if not args.scenario:
            print("\n  --visualize requires --scenario. Pick one:")
            for name in SCENARIOS:
                print(f"    {name}")
            sys.exit(1)

        sc = SCENARIOS[args.scenario]
        is_random = sc.get("type") == "random"

        if is_random:
            seed = args.seed if args.seed is not None else sc.get("seed")
            target = RandomEvasiveTarget(
                start_pos=sc["start_pos"],
                max_speed=sc["speed"],
                max_accel=sc["max_accel"],
                seed=seed,
            )
            print(f"\n  Launching viewer: {args.scenario} (seed={seed})")
            run_intercept(
                interceptor_start="0 0 1.5",
                intercept_radius=args.radius,
                max_duration=args.max_time,
                target_path_override=target,
            )
        else:
            print(f"\n  Launching viewer: {args.scenario}")
            run_intercept(
                target_waypoints=sc["waypoints"],
                target_speed=sc["speed"],
                interceptor_start="0 0 1.5",
                intercept_radius=args.radius,
                max_duration=args.max_time,
            )
        return

    # Headless diagnostics
    if args.scenario:
        scenarios = {args.scenario: SCENARIOS[args.scenario]}
    else:
        scenarios = SCENARIOS

    results = {}
    for name in scenarios:
        t0 = time.time()
        result = run_diagnostic(
            name, config,
            intercept_radius=args.radius,
            max_time=args.max_time,
        )
        elapsed = time.time() - t0
        results[name] = result
        print_diagnostic(name, result)
        print(f"  ({elapsed:.1f}s)")

    if len(results) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()
