"""
test_mpc_headless.py — Headless test harness for the MPC controller.

Runs the MPC loop without any viewer or OpenCV windows and collects
quantitative metrics.  Includes robustness tests (wind, behind-drone
waypoints, rapid switching) and intercept-controller validation.

Usage:
    python skydio_x2/test_mpc_headless.py            # run all tests
    python skydio_x2/test_mpc_headless.py --test wind # run one test
"""

import argparse
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skydio_x2.mpc_controller import MPCController
from skydio_x2.intercept_controller import (
    MPCController as InterceptMPC,
    ScriptedTargetPath,
    build_intercept_xml,
    compute_lead_point,
)
from skydio_x2.intercept_config import InterceptMPCConfig, DEFAULT_INTERCEPT_CONFIG
from skydio_x2.skydio_x2_movement import (
    create_skydio_x2_simulation,
    apply_motor_mixing,
)


# ======================================================================
# Core headless evaluation harness
# ======================================================================

def evaluate_mpc(
    waypoints,
    waypoint_threshold=0.3,
    max_steps=3000,
    mpc_horizon=20,
    mpc_samples=200,
    mpc_elite=40,
    mpc_iterations=3,
    wind_force=None,
    wind_start_step=None,
    wind_end_step=None,
    cloud_path=None,
    visualize=False,
    vis_label="",
):
    """Run the MPC loop and return trajectory metrics.

    Parameters
    ----------
    waypoints           : list of [x, y, z]
    waypoint_threshold  : metres — waypoint counts as reached
    max_steps           : hard step limit to prevent infinite loops
    mpc_*               : CEM tuning knobs
    wind_force          : (3,) array — constant external force on the drone body
    wind_start_step     : step at which wind begins (None = from the start)
    wind_end_step       : step at which wind ends (None = never)
    cloud_path          : path to .ply file (auto-detected if None)
    visualize           : open MuJoCo viewer + FPV window
    vis_label           : label shown on the FPV HUD

    Returns
    -------
    dict with keys:
        success           : bool — all waypoints reached
        waypoints_reached : int
        total_steps       : int
        time_to_waypoint  : list[float] — sim-seconds to reach each waypoint
        overshoot         : list[float] — max distance past waypoint after reaching it
        steady_state_err  : list[float] — distance at the moment "reached" triggers
        max_tilt_deg      : float — peak tilt angle across the whole flight
        control_effort    : float — sum of |ctrl| over all steps (energy proxy)
        positions         : np.ndarray (N, 3)
        velocities        : np.ndarray (N, 3)
        tilts_deg         : np.ndarray (N,)
        controls          : np.ndarray (N, 4) — raw ctrl values per step
    """
    if cloud_path is None:
        cloud_path = os.path.join(os.path.dirname(__file__), "test_cube.ply")

    env = create_skydio_x2_simulation(cloud_path)
    model = env["model"]
    data = env["data"]
    hover_thrust = env["hover_thrust"]
    steps_per_frame = env["steps_per_frame"]
    renderer = env["renderer"]
    fpv_cam_id = env["fpv_cam_id"]
    dt_render = env["dt_render"]

    controller = MPCController(
        horizon=mpc_horizon,
        n_samples=mpc_samples,
        n_elite=mpc_elite,
        n_iterations=mpc_iterations,
        dt=model.opt.timestep * steps_per_frame,
    )

    waypoints = [np.array(wp, dtype=np.float64) for wp in waypoints]
    wp_idx = 0

    drone_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "x2")

    # --- Storage ---
    positions = []
    velocities = []
    tilts_deg = []
    controls_log = []
    time_to_waypoint = []
    overshoot = []
    steady_state_err = []

    wp_reached_step = None
    wp_start_step = 0
    max_tilt = 0.0
    control_effort = 0.0
    dt_ctrl = model.opt.timestep * steps_per_frame

    # --- Visualisation context ---
    viewer_ctx = (
        mujoco.viewer.launch_passive(model, data) if visualize else None
    )

    try:
        if viewer_ctx is not None:
            viewer_ctx.__enter__()
            viewer_ctx.cam.azimuth = -60
            viewer_ctx.cam.elevation = -30
            viewer_ctx.cam.distance = 5.0
            viewer_ctx.cam.lookat[:] = [0, 0, 1.5]

        for step in range(max_steps):
            step_start = time.time()

            pos = data.qpos[:3].copy()
            vel = data.qvel[:3].copy()
            quat = data.qpos[3:7].copy()

            # Tilt = arccos(body_z · world_z), using qx²+qy² shortcut
            tilt = np.degrees(2.0 * np.arcsin(np.clip(
                np.sqrt(quat[1] ** 2 + quat[2] ** 2), 0, 1)))
            max_tilt = max(max_tilt, tilt)

            positions.append(pos)
            velocities.append(vel)
            tilts_deg.append(tilt)

            # --- Wind disturbance ---
            wind_active = False
            if wind_force is not None:
                wind_active = True
                if wind_start_step is not None and step < wind_start_step:
                    wind_active = False
                if wind_end_step is not None and step >= wind_end_step:
                    wind_active = False
                if wind_active:
                    data.xfrc_applied[drone_body_id, :3] = wind_force
                else:
                    data.xfrc_applied[drone_body_id, :3] = 0.0
            else:
                data.xfrc_applied[drone_body_id, :3] = 0.0

            # --- Current target ---
            if wp_idx < len(waypoints):
                target = waypoints[wp_idx]
            else:
                target = waypoints[-1]

            dist = np.linalg.norm(pos - target)
            vel_mag = np.linalg.norm(vel)

            # --- Waypoint reached? ---
            if dist < waypoint_threshold and vel_mag < 1.0 and wp_idx < len(waypoints):
                steady_state_err.append(dist)
                time_to_waypoint.append((step - wp_start_step) * dt_ctrl)
                wp_reached_step = step
                wp_idx += 1
                controller.reset()
                wp_start_step = step

            # Track overshoot
            if wp_reached_step is not None and wp_idx > 0:
                prev_target = waypoints[wp_idx - 1]
                d = np.linalg.norm(pos - prev_target)
                if len(overshoot) < wp_idx:
                    overshoot.append(d)
                else:
                    overshoot[wp_idx - 1] = max(overshoot[wp_idx - 1], d)

            # All done + overshoot measured for 50 more steps
            if wp_idx >= len(waypoints) and wp_reached_step is not None:
                if step - wp_reached_step > 50:
                    break

            # --- MPC solve ---
            ctrl = controller.solve(data, target)
            thrust_offset, pitch_cmd, roll_cmd, yaw_cmd = ctrl

            base = hover_thrust + thrust_offset
            apply_motor_mixing(data, pitch_cmd, roll_cmd, yaw_cmd, base)

            ctrl_vec = data.ctrl[:4].copy()
            controls_log.append(ctrl_vec)
            control_effort += np.sum(np.abs(ctrl_vec))

            # --- Physics ---
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            # --- Visualisation ---
            if viewer_ctx is not None:
                if not viewer_ctx.is_running():
                    break
                viewer_ctx.sync()

                # FPV render
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=fpv_cam_id)
                fpv_img = renderer.render()
                fpv_bgr = cv2.cvtColor(fpv_img, cv2.COLOR_RGB2BGR)
                fpv_bgr = cv2.flip(fpv_bgr, 0)

                # HUD line 1: label + waypoint + distance
                sim_t = step * dt_ctrl
                wp_label = (
                    f"WP {wp_idx}/{len(waypoints)}"
                    if wp_idx < len(waypoints)
                    else "DONE"
                )
                cv2.putText(
                    fpv_bgr,
                    f"{vis_label}  {wp_label}  dist={dist:.2f}m  t={sim_t:.1f}s",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                )

                # HUD line 2: MPC commands
                cv2.putText(
                    fpv_bgr,
                    f"MPC: thr={thrust_offset:+.2f}  p={pitch_cmd:+.2f}  "
                    f"r={roll_cmd:+.2f}  y={yaw_cmd:+.2f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1,
                )

                # HUD line 3: wind + disturbance estimate
                df = controller._disturbance_force
                if wind_force is not None:
                    wf = wind_force if wind_active else np.zeros(3)
                    cv2.putText(
                        fpv_bgr,
                        f"WIND: [{wf[0]:+.1f},{wf[1]:+.1f},{wf[2]:+.1f}]N  "
                        f"EST: [{df[0]:+.1f},{df[1]:+.1f},{df[2]:+.1f}]N",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 255), 1,
                    )

                # HUD line 4: tilt
                cv2.putText(
                    fpv_bgr,
                    f"TILT: {tilt:.1f} deg  ALT: {pos[2]:.2f}m",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1,
                )

                cv2.imshow("MPC Test — FPV", fpv_bgr)
                cv2.waitKey(1)

                # Frame timing
                elapsed = time.time() - step_start
                if elapsed < dt_render:
                    time.sleep(dt_render - elapsed)

    finally:
        if viewer_ctx is not None:
            cv2.destroyAllWindows()
            viewer_ctx.__exit__(None, None, None)

    renderer.close()

    return {
        "success": wp_idx >= len(waypoints),
        "waypoints_reached": min(wp_idx, len(waypoints)),
        "total_steps": step + 1,
        "time_to_waypoint": time_to_waypoint,
        "overshoot": overshoot,
        "steady_state_err": steady_state_err,
        "max_tilt_deg": max_tilt,
        "control_effort": control_effort,
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "tilts_deg": np.array(tilts_deg),
        "controls": np.array(controls_log),
    }


# ======================================================================
# Intercept headless harness
# ======================================================================

def evaluate_intercept(
    target_waypoints,
    target_speed=1.5,
    intercept_radius=1.0,
    max_steps=3000,
    config=None,
):
    """Run the intercept controller headlessly and return metrics.

    Parameters
    ----------
    config : InterceptMPCConfig or None
        Weight configuration.  Uses DEFAULT_INTERCEPT_CONFIG when None.

    Returns
    -------
    dict with keys:
        intercepted       : bool
        time_to_intercept : float or None (sim-seconds)
        min_distance      : float — closest approach
        total_steps       : int
        positions_drone   : np.ndarray (N, 3)
        positions_target  : np.ndarray (N, 3)
        max_tilt_deg      : float
        final_closing_speed : float or None
    """
    if config is None:
        config = DEFAULT_INTERCEPT_CONFIG

    drone_xml_dir = os.path.join(os.path.dirname(__file__))
    if not os.path.isdir(os.path.join(drone_xml_dir, "assets")):
        drone_xml_dir = os.path.join("skydio_x2")
    drone_xml_dir = os.path.abspath(drone_xml_dir)

    wp0 = target_waypoints[0]
    target_start = f"{wp0[0]} {wp0[1]} {wp0[2]}"

    xml = build_intercept_xml(drone_xml_dir, "0 0 1.5", target_start)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    steps_per_frame = 4
    dt_ctrl = model.opt.timestep * steps_per_frame
    target_qpos_start = 7
    hover_thrust = 3.2495625

    controller = MPCController(
        horizon=config.horizon,
        n_samples=config.n_samples,
        n_elite=config.n_elite,
        n_iterations=config.n_iterations,
        dt=dt_ctrl,
    )
    config.apply_to(controller)

    target_path = ScriptedTargetPath(target_waypoints, speed=target_speed)

    positions_drone = []
    positions_target = []
    min_dist = float("inf")
    max_tilt = 0.0
    intercepted = False
    intercept_step = None
    final_closing = None

    last_thrust = 0.0
    phase = "cruise"

    for step in range(max_steps):
        # Advance target
        target_pos, target_vel = target_path.get_state(dt_ctrl)
        data.qpos[target_qpos_start:target_qpos_start + 3] = target_pos
        data.qpos[target_qpos_start + 3:target_qpos_start + 7] = [1, 0, 0, 0]
        data.qvel[6:9] = target_vel
        data.qvel[9:12] = [0, 0, 0]

        drone_pos = data.qpos[:3].copy()
        drone_vel = data.qvel[:3].copy()
        quat = data.qpos[3:7].copy()
        dist = np.linalg.norm(drone_pos - target_pos)
        min_dist = min(min_dist, dist)

        tilt = np.degrees(2.0 * np.arcsin(np.clip(
            np.sqrt(quat[1] ** 2 + quat[2] ** 2), 0, 1)))
        max_tilt = max(max_tilt, tilt)

        positions_drone.append(drone_pos.copy())
        positions_target.append(target_pos.copy())

        if dist < intercept_radius:
            intercepted = True
            intercept_step = step
            final_closing = np.linalg.norm(drone_vel - target_vel)
            break

        # Phase transition
        xy_dist = np.linalg.norm(drone_pos[:2] - target_pos[:2])
        if phase == "cruise" and xy_dist <= config.strike_range:
            phase = "strike"
            config.apply_strike(controller)
            controller.reset()

        # MPC toward lead point (strike: aim directly at target)
        lead = compute_lead_point(drone_pos, drone_vel, target_pos, target_vel)
        if phase == "strike":
            lead = target_pos.copy()

        ctrl = controller.solve(data, lead)
        thrust_offset, pitch_cmd, roll_cmd, yaw_cmd = ctrl
        last_thrust = thrust_offset

        base = hover_thrust + thrust_offset
        apply_motor_mixing(data, pitch_cmd, roll_cmd, yaw_cmd, base)

        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        # Re-pin target
        data.qpos[target_qpos_start:target_qpos_start + 3] = target_pos
        data.qpos[target_qpos_start + 3:target_qpos_start + 7] = [1, 0, 0, 0]
        data.qvel[6:9] = target_vel
        data.qvel[9:12] = [0, 0, 0]

    return {
        "intercepted": intercepted,
        "time_to_intercept": intercept_step * dt_ctrl if intercept_step else None,
        "min_distance": min_dist,
        "total_steps": step + 1,
        "positions_drone": np.array(positions_drone),
        "positions_target": np.array(positions_target),
        "max_tilt_deg": max_tilt,
        "final_closing_speed": final_closing,
    }


# ======================================================================
# Pretty-print helpers
# ======================================================================

def _print_header(title):
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


def _print_mpc_result(result, label=""):
    if label:
        print(f"\n  --- {label} ---")
    status = "PASS" if result["success"] else "FAIL"
    print(f"  Status:             {status}  "
          f"({result['waypoints_reached']}/{len(result['time_to_waypoint']) + (0 if result['success'] else 1)} waypoints)")
    print(f"  Total steps:        {result['total_steps']}")
    if result["time_to_waypoint"]:
        for i, t in enumerate(result["time_to_waypoint"]):
            print(f"    WP {i}: {t:.2f}s  "
                  f"err={result['steady_state_err'][i]:.4f}m  "
                  f"overshoot={result['overshoot'][i]:.4f}m")
    print(f"  Max tilt:           {result['max_tilt_deg']:.1f} deg")
    print(f"  Control effort:     {result['control_effort']:.1f}")
    return status == "PASS"


def _print_intercept_result(result, label=""):
    if label:
        print(f"\n  --- {label} ---")
    status = "HIT" if result["intercepted"] else "MISS"
    print(f"  Status:             {status}")
    if result["intercepted"]:
        print(f"  Time to intercept:  {result['time_to_intercept']:.2f}s")
        print(f"  Closing speed:      {result['final_closing_speed']:.2f} m/s")
    print(f"  Min distance:       {result['min_distance']:.3f}m")
    print(f"  Max tilt:           {result['max_tilt_deg']:.1f} deg")
    print(f"  Total steps:        {result['total_steps']}")
    return result["intercepted"]


# ======================================================================
# Test: Baseline — standard square waypoint pattern
# ======================================================================

def test_baseline():
    _print_header("TEST: Baseline — Square Waypoint Pattern")
    waypoints = [
        [2.0, 0.0, 1.5],
        [2.0, 2.0, 1.5],
        [0.0, 2.0, 2.0],
        [0.0, 0.0, 1.5],
    ]
    result = evaluate_mpc(waypoints, max_steps=4000)
    return _print_mpc_result(result)


# ======================================================================
# Test 5a: Wind disturbance
# ======================================================================

def test_wind_disturbance(visualize=False):
    _print_header("TEST 5a: Wind Disturbance Robustness")
    waypoints = [
        [2.0, 0.0, 1.5],
        [2.0, 2.0, 1.5],
        [0.0, 0.0, 1.5],
    ]

    all_passed = True

    # Light crosswind (2 N sideways)
    r = evaluate_mpc(
        waypoints, max_steps=4000,
        wind_force=np.array([0.0, 2.0, 0.0]),
        wind_start_step=50,
        visualize=visualize,
        vis_label="Crosswind 2N +Y",
    )
    if not _print_mpc_result(r, "Light crosswind (2N +Y, continuous)"):
        all_passed = False

    # Strong headwind (5 N opposing forward motion)
    r = evaluate_mpc(
        waypoints, max_steps=4000,
        wind_force=np.array([-5.0, 0.0, 0.0]),
        wind_start_step=50,
        visualize=visualize,
        vis_label="Headwind 5N -X",
    )
    if not _print_mpc_result(r, "Strong headwind (5N -X, continuous)"):
        all_passed = False

    # Gust: brief strong downward force
    r = evaluate_mpc(
        waypoints, max_steps=4000,
        wind_force=np.array([0.0, 0.0, -8.0]),
        wind_start_step=100,
        wind_end_step=200,
        visualize=visualize,
        vis_label="Downdraft gust 8N",
    )
    if not _print_mpc_result(r, "Downdraft gust (8N -Z, steps 100-200)"):
        all_passed = False

    # Diagonal turbulence
    r = evaluate_mpc(
        waypoints, max_steps=4000,
        wind_force=np.array([3.0, 3.0, -2.0]),
        wind_start_step=50,
        visualize=visualize,
        vis_label="Diagonal 3,3,-2 N",
    )
    if not _print_mpc_result(r, "Diagonal turbulence (3,3,-2 N, continuous)"):
        all_passed = False

    return all_passed


# ======================================================================
# Test 5c: Waypoint behind the drone
# ======================================================================

def test_behind_waypoint():
    _print_header("TEST 5c: Waypoints Behind the Drone")

    all_passed = True

    # Drone starts at (0, 0, 1.5) facing +X.  Target directly behind at -X.
    r = evaluate_mpc(
        [[-3.0, 0.0, 1.5]],
        max_steps=4000,
    )
    if not _print_mpc_result(r, "Single waypoint 3m directly behind (-X)"):
        all_passed = False

    # Behind and above
    r = evaluate_mpc(
        [[-2.0, 0.0, 3.0]],
        max_steps=4000,
    )
    if not _print_mpc_result(r, "Behind and above (-2, 0, 3)"):
        all_passed = False

    # Full 180: fly forward then back past origin
    r = evaluate_mpc(
        [[3.0, 0.0, 1.5], [-3.0, 0.0, 1.5]],
        max_steps=6000,
    )
    if not _print_mpc_result(r, "Forward 3m then reverse 6m"):
        all_passed = False

    # Behind + lateral offset (requires diagonal 180)
    r = evaluate_mpc(
        [[-2.0, -2.0, 1.5]],
        max_steps=4000,
    )
    if not _print_mpc_result(r, "Behind + lateral (-2, -2, 1.5)"):
        all_passed = False

    return all_passed


# ======================================================================
# Test 5e: Rapid waypoint switching
# ======================================================================

def test_rapid_waypoints():
    _print_header("TEST 5e: Rapid Waypoint Switching")

    all_passed = True

    # Dense cluster: 8 waypoints each 0.5m apart in a zigzag
    zigzag = []
    for i in range(8):
        x = 0.5 * (i + 1)
        y = 0.3 * (1 if i % 2 == 0 else -1)
        zigzag.append([x, y, 1.5])
    r = evaluate_mpc(zigzag, waypoint_threshold=0.3, max_steps=5000)
    if not _print_mpc_result(r, "Zigzag: 8 WPs, 0.5m spacing"):
        all_passed = False

    # Tight vertical stack: 0.3m altitude steps
    stack = [[0.5, 0.0, 1.5 + 0.3 * i] for i in range(6)]
    r = evaluate_mpc(stack, waypoint_threshold=0.3, max_steps=5000)
    if not _print_mpc_result(r, "Vertical stack: 6 WPs, 0.3m alt steps"):
        all_passed = False

    # Micro-square: 0.5m x 0.5m square
    micro = [
        [0.5, 0.0, 1.5],
        [0.5, 0.5, 1.5],
        [0.0, 0.5, 1.5],
        [0.0, 0.0, 1.5],
    ]
    r = evaluate_mpc(micro, waypoint_threshold=0.25, max_steps=4000)
    if not _print_mpc_result(r, "Micro-square: 0.5m sides"):
        all_passed = False

    return all_passed


# ======================================================================
# Test 6: Intercept controller validation
# ======================================================================

def test_intercept():
    _print_header("TEST 6: Intercept Controller Validation")

    # Straight-line path the target flies along
    STRAIGHT_PATH = [
        [5, 5, 2.0],
        [5, -5, 2.0],
    ]
    # Evasive S-curve
    EVASIVE_PATH = [
        [8, 8, 2.0],
        [6, 4, 2.5],
        [4, 6, 2.0],
        [2, 2, 2.5],
        [0, 4, 2.0],
        [-2, 0, 2.5],
    ]
    # Altitude-change path
    ALTITUDE_PATH = [
        [6, 0, 1.5],
        [6, 4, 3.5],
        [2, 4, 1.0],
        [2, 0, 3.0],
        [0, 0, 2.0],
    ]

    all_passed = True
    speed_results = []

    # 6a: Sweep target speeds on a straight line
    print("\n  [6a] Speed sweep — straight-line target")
    for speed in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
        r = evaluate_intercept(
            STRAIGHT_PATH,
            target_speed=speed,
            intercept_radius=1.0,
            max_steps=3000,
        )
        tag = f"speed={speed:.0f} m/s"
        hit = _print_intercept_result(r, tag)
        speed_results.append((speed, hit, r))

    hits = sum(1 for _, h, _ in speed_results if h)
    print(f"\n  Speed sweep: {hits}/{len(speed_results)} intercepted")
    if hits < len(speed_results) // 2:
        all_passed = False

    # 6b: Evasive S-curve at moderate speed
    print("\n  [6b] Evasive S-curve target")
    r = evaluate_intercept(
        EVASIVE_PATH,
        target_speed=3.0,
        intercept_radius=1.0,
        max_steps=4000,
    )
    if not _print_intercept_result(r, "S-curve at 3 m/s"):
        all_passed = False

    # 6c: Altitude-changing target
    print("\n  [6c] Altitude-varying target")
    r = evaluate_intercept(
        ALTITUDE_PATH,
        target_speed=2.5,
        intercept_radius=1.0,
        max_steps=4000,
    )
    if not _print_intercept_result(r, "Altitude path at 2.5 m/s"):
        all_passed = False

    return all_passed


# ======================================================================
# Runner
# ======================================================================

ALL_TESTS = {
    "baseline": test_baseline,
    "wind": test_wind_disturbance,
    "behind": test_behind_waypoint,
    "rapid": test_rapid_waypoints,
    "intercept": test_intercept,
}


def main():
    parser = argparse.ArgumentParser(description="Headless MPC test suite")
    parser.add_argument(
        "--test", choices=list(ALL_TESTS.keys()),
        help="Run a single test (default: all)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Open MuJoCo viewer + FPV window for wind tests",
    )
    args = parser.parse_args()

    if args.test:
        tests = {args.test: ALL_TESTS[args.test]}
    else:
        tests = ALL_TESTS

    results = {}
    t0 = time.time()

    for name, fn in tests.items():
        try:
            if name == "wind" and args.visualize:
                results[name] = fn(visualize=True)
            else:
                results[name] = fn()
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    elapsed = time.time() - t0

    # --- Summary ---
    print(f"\n{'=' * 64}")
    print(f"  SUMMARY  ({elapsed:.1f}s)")
    print(f"{'=' * 64}")
    for name, passed in results.items():
        tag = "PASS" if passed else "FAIL"
        print(f"  {name:20s}  {tag}")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} test groups passed.")
    print(f"{'=' * 64}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
