"""
point_to_path.py - PID-controlled waypoint navigation for the Skydio X2 drone.

Two main entry points:
  fly_path_relative() - waypoints as (dx, dy, dz) offsets from the drone's
                        position at call time.
  fly_path_absolute() - waypoints as (x, y, z) in world coordinates
                        (origin = where the drone started at sim begin).

Both accept an optional ending_orientation dict {'pitch', 'yaw', 'roll'} in
degrees so the drone converges to a specific attitude after the last waypoint.

All flight control is done through a PID system:
  - Horizontal position -> pitch / roll commands (via body-frame error)
  - Altitude            -> thrust command
  - Heading             -> yaw command
  - Final orientation   -> pitch / roll / yaw orientation PIDs
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import mujoco
import cv2
import time

from skydio_x2.skydio_x2_movement import create_skydio_x2_simulation


# ---------------------------------------------------------------------------
# PID Controller
# ---------------------------------------------------------------------------
class PIDController:
    """Single-axis PID controller with integral anti-windup and filtered derivative."""

    def __init__(self, kp, ki, kd, output_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.initialized = False

    def update(self, error, dt):
        if not self.initialized:
            self.prev_error = error
            self.initialized = True

        self.integral += error * dt
        # Anti-windup: clamp integral contribution
        max_integral = self.output_limit / max(abs(self.ki), 1e-6)
        self.integral = np.clip(self.integral, -max_integral, max_integral)

        raw_derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        # Low-pass filter on derivative to reduce noise-driven oscillation
        alpha = 0.3
        derivative = alpha * raw_derivative + (1 - alpha) * self.prev_derivative
        self.prev_derivative = derivative
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(output, -self.output_limit, self.output_limit)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.initialized = False


# ---------------------------------------------------------------------------
# Inner-loop attitude stabilisation (reads IMU, writes motor thrusts)
# ---------------------------------------------------------------------------
class AttitudeController:
    """Cascaded inner-loop PID that stabilises the drone at a commanded tilt.

    Outer-loop PIDs produce *commands* in [-1, 1]:
      pitch_cmd / roll_cmd  → desired tilt as fraction of max_tilt
      yaw_cmd               → desired yaw-rate as fraction of max_yaw_rate
    This controller reads the gyro (sensordata[0:3]) and quaternion
    (sensordata[6:10]), runs per-axis PIDs, and writes data.ctrl[0:4].
    """

    def __init__(self, hover_thrust=3.2495625, max_tilt_deg=25.0,
                 max_yaw_rate=2.0,
                 kp_att=6.0, ki_att=0.2, kd_att=1.5,
                 kp_yaw=3.0, ki_yaw=0.05, kd_yaw=0.5,
                 corr_limit=2.5):
        self.hover_thrust = hover_thrust
        self.max_tilt = np.radians(max_tilt_deg)
        self.max_yaw_rate = max_yaw_rate
        self.kp_att = kp_att
        self.ki_att = ki_att
        self.kd_att = kd_att
        self.kp_yaw = kp_yaw
        self.ki_yaw = ki_yaw
        self.kd_yaw = kd_yaw
        self.corr_limit = corr_limit
        self.pitch_integral = 0.0
        self.roll_integral = 0.0
        self.yaw_integral = 0.0

    def reset(self):
        self.pitch_integral = 0.0
        self.roll_integral = 0.0
        self.yaw_integral = 0.0

    def compute(self, data, pitch_cmd, roll_cmd, yaw_cmd, thrust_offset, dt):
        """Read IMU sensors, run inner PID with gyro damping, set data.ctrl[0..3]."""
        gyro = data.sensordata[0:3]   # body-frame angular velocity
        quat = data.sensordata[6:10]  # w, x, y, z
        actual_roll, actual_pitch, _ = quat_to_euler(quat)

        desired_pitch = np.clip(pitch_cmd, -1, 1) * self.max_tilt
        desired_roll = np.clip(roll_cmd, -1, 1) * self.max_tilt
        desired_yaw_rate = np.clip(yaw_cmd, -1, 1) * self.max_yaw_rate

        pitch_err = desired_pitch - actual_pitch
        roll_err = desired_roll - actual_roll
        yaw_rate_err = desired_yaw_rate - gyro[2]

        # Integrate with anti-windup
        int_limit = 0.5
        self.pitch_integral = np.clip(
            self.pitch_integral + pitch_err * dt, -int_limit, int_limit)
        self.roll_integral = np.clip(
            self.roll_integral + roll_err * dt, -int_limit, int_limit)
        self.yaw_integral = np.clip(
            self.yaw_integral + yaw_rate_err * dt, -int_limit, int_limit)

        # P on error + I + D on gyro rate (derivative-on-measurement, stable)
        pitch_corr = (self.kp_att * pitch_err
                      + self.ki_att * self.pitch_integral
                      - self.kd_att * gyro[1])
        roll_corr = (self.kp_att * roll_err
                     + self.ki_att * self.roll_integral
                     - self.kd_att * gyro[0])
        yaw_corr = (self.kp_yaw * yaw_rate_err
                    + self.ki_yaw * self.yaw_integral
                    - self.kd_yaw * gyro[2])

        pitch_corr = np.clip(pitch_corr, -self.corr_limit, self.corr_limit)
        roll_corr = np.clip(roll_corr, -self.corr_limit, self.corr_limit)
        yaw_corr = np.clip(yaw_corr, -self.corr_limit, self.corr_limit)

        base = self.hover_thrust + thrust_offset
        # Mixing derived from τ = r × F cross-product at each motor site:
        # Motor 1: rear-right  CW   (-X, -Y)  pitch+  roll-  yaw-
        # Motor 2: rear-left   CCW  (-X, +Y)  pitch+  roll+  yaw+
        # Motor 3: front-left  CW   (+X, +Y)  pitch-  roll+  yaw-
        # Motor 4: front-right CCW  (+X, -Y)  pitch-  roll-  yaw+
        data.ctrl[0] = np.clip(base + pitch_corr - roll_corr - yaw_corr, 0, 13)
        data.ctrl[1] = np.clip(base + pitch_corr + roll_corr + yaw_corr, 0, 13)
        data.ctrl[2] = np.clip(base - pitch_corr + roll_corr - yaw_corr, 0, 13)
        data.ctrl[3] = np.clip(base - pitch_corr - roll_corr + yaw_corr, 0, 13)


# ---------------------------------------------------------------------------
# Quaternion / angle helpers
# ---------------------------------------------------------------------------
def quat_to_euler(quat):
    """MuJoCo quaternion [w,x,y,z] -> (roll, pitch, yaw) in radians."""
    w, x, y, z = quat
    # Roll (X)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    # Pitch (Y)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # Yaw (Z)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw


def _angle_wrap(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _world_to_body_xy(dx, dy, yaw):
    """Project a world-frame XY error into body-frame (forward, right).

    Returns (forward, right) so that positive forward maps directly to
    positive pitch_cmd, and positive right maps to positive roll_cmd
    (matching the Skydio X2 motor-mixing convention).
    """
    c, s = np.cos(yaw), np.sin(yaw)
    forward = dx * c + dy * s
    right = dx * s - dy * c
    return forward, right


# ---------------------------------------------------------------------------
# Core path-following loop (PID)
# ---------------------------------------------------------------------------
def _fly_path(
    env,
    waypoints,
    ending_orientation=None,
    waypoint_threshold=0.5,
    position_gains=(0.8, 0.02, 0.6),
    altitude_gains=(2.5, 0.3, 1.2),
    yaw_gains=(0.5, 0.02, 0.2),
    orientation_gains=(0.8, 0.05, 0.3),
    max_time_per_waypoint=15.0,
    settle_time=3.0,
    show_fpv=True,
):
    """Internal: PID-controlled flight through absolute waypoints.

    Returns a list of waypoint indices that were successfully reached.
    """
    model = env["model"]
    data = env["data"]
    renderer = env["renderer"]
    fpv_cam_id = env["fpv_cam_id"]
    hover_thrust = env["hover_thrust"]
    steps_per_frame = env["steps_per_frame"]
    dt_render = env["dt_render"]
    dt_control = steps_per_frame * model.opt.timestep

    # -- Inner-loop attitude stabilisation --
    attitude = AttitudeController(hover_thrust=hover_thrust)

    # -- Outer-loop position PIDs (output in [-1, 1] = fraction of max tilt) --
    pid_fwd = PIDController(*position_gains, output_limit=0.8)
    pid_right = PIDController(*position_gains, output_limit=0.8)
    pid_alt = PIDController(*altitude_gains, output_limit=3.0)
    pid_yaw = PIDController(*yaw_gains, output_limit=0.6)

    # -- Orientation PIDs (used only during settling) --
    pid_orient_pitch = PIDController(*orientation_gains, output_limit=0.5)
    pid_orient_roll = PIDController(*orientation_gains, output_limit=0.5)

    wp_idx = 0
    wp_start = time.time()
    sim_start = time.time()
    navigating = True
    settle_start = None
    settle_announced = False
    reached = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = -60
        viewer.cam.elevation = -30
        viewer.cam.distance = 4.0
        viewer.cam.lookat = [0, 0, 1.0]

        while viewer.is_running():
            frame_start = time.time()
            t_sim = frame_start - sim_start

            # -- Current state --
            pos = data.qpos[0:3].copy()
            quat = data.qpos[3:7].copy()
            roll_cur, pitch_cur, yaw_cur = quat_to_euler(quat)

            # Default outputs (hover)
            pitch_out = 0.0
            roll_out = 0.0
            yaw_out = 0.0
            thrust_offset = 0.0

            # ==============================================================
            # Phase 1: Navigate through waypoints
            # ==============================================================
            if navigating and wp_idx < len(waypoints):
                target = np.array(waypoints[wp_idx], dtype=float)
                error = target - pos
                dist = np.linalg.norm(error)

                # -- Waypoint reached? --
                if dist < waypoint_threshold:
                    reached.append(wp_idx)
                    print(
                        f"Waypoint {wp_idx} reached at t={t_sim:.2f}s  "
                        f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                    )
                    wp_idx += 1
                    wp_start = time.time()
                    pid_fwd.reset()
                    pid_right.reset()
                    pid_alt.reset()
                    pid_yaw.reset()

                    if wp_idx >= len(waypoints):
                        navigating = False
                        settle_start = time.time()
                        pid_orient_pitch.reset()
                        pid_orient_roll.reset()
                        pid_yaw.reset()
                        pid_fwd.reset()
                        pid_right.reset()
                        pid_alt.reset()
                        if ending_orientation is not None:
                            print("All waypoints reached. Converging to target orientation...")
                        else:
                            print("All waypoints reached. Hovering.")

                # -- Waypoint timeout --
                elif time.time() - wp_start > max_time_per_waypoint:
                    target = np.array(waypoints[wp_idx], dtype=float)
                    err = target - pos
                    print(
                        f"Waypoint {wp_idx} timed out at t={t_sim:.2f}s, skipping.\n"
                        f"  pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  "
                        f"target=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})\n"
                        f"  displacement: dx={err[0]:.2f} dy={err[1]:.2f} dz={err[2]:.2f}  "
                        f"dist={np.linalg.norm(err):.2f}m\n"
                        f"  orientation: roll={np.degrees(roll_cur):.1f}° "
                        f"pitch={np.degrees(pitch_cur):.1f}° "
                        f"yaw={np.degrees(yaw_cur):.1f}°"
                    )
                    wp_idx += 1
                    wp_start = time.time()
                    # Only reset integral to prevent windup; keep derivative
                    # state so the controller can damp existing velocity.
                    pid_fwd.integral = 0.0
                    pid_right.integral = 0.0
                    pid_alt.integral = 0.0
                    pid_yaw.integral = 0.0

                    if wp_idx >= len(waypoints):
                        navigating = False
                        settle_start = time.time()

                # -- Compute PID commands for current waypoint --
                if navigating and wp_idx < len(waypoints):
                    target = np.array(waypoints[wp_idx], dtype=float)
                    error = target - pos
                    horiz_dist = np.linalg.norm(error[0:2])

                    fwd_err, right_err = _world_to_body_xy(
                        error[0], error[1], yaw_cur
                    )
                    alt_err = error[2]

                    # Only yaw toward waypoint when far enough to benefit;
                    # when close, hold current heading to avoid destabilising spin.
                    if horiz_dist > 1.0:
                        desired_yaw = np.arctan2(error[1], error[0])
                        yaw_err = _angle_wrap(desired_yaw - yaw_cur)
                    else:
                        yaw_err = _angle_wrap(0.0 - yaw_cur)

                    pitch_out = pid_fwd.update(fwd_err, dt_render)
                    roll_out = pid_right.update(right_err, dt_render)
                    yaw_out = pid_yaw.update(yaw_err, dt_render)
                    alt_adj = pid_alt.update(alt_err, dt_render)

                    # Tilt compensation: increase thrust to maintain vertical
                    # force when the drone is pitched/rolled.
                    tilt = np.sqrt(pitch_cur**2 + roll_cur**2)
                    cos_tilt = np.cos(np.clip(tilt, 0, np.radians(60)))
                    thrust_offset = alt_adj + hover_thrust * (1.0 / max(cos_tilt, 0.5) - 1.0)

            # ==============================================================
            # Phase 2: Hold final waypoint + converge to ending orientation
            # ==============================================================
            elif not navigating:
                # Position hold on last waypoint (or current pos if no WPs)
                hold_target = (
                    np.array(waypoints[-1], dtype=float)
                    if waypoints
                    else pos
                )
                pos_err = hold_target - pos
                fwd_err, right_err = _world_to_body_xy(
                    pos_err[0], pos_err[1], yaw_cur
                )
                alt_err = pos_err[2]

                pos_pitch = pid_fwd.update(fwd_err, dt_render)
                pos_roll = pid_right.update(right_err, dt_render)
                thrust_adj = pid_alt.update(alt_err, dt_render)

                if ending_orientation is not None:
                    t_yaw = np.radians(ending_orientation.get("yaw", 0.0))
                    t_pitch = np.radians(ending_orientation.get("pitch", 0.0))
                    t_roll = np.radians(ending_orientation.get("roll", 0.0))

                    yaw_err = _angle_wrap(t_yaw - yaw_cur)
                    pitch_err = _angle_wrap(t_pitch - pitch_cur)
                    roll_err = _angle_wrap(t_roll - roll_cur)

                    orient_pitch = pid_orient_pitch.update(pitch_err, dt_render)
                    orient_roll = pid_orient_roll.update(roll_err, dt_render)
                    yaw_adj = pid_yaw.update(yaw_err, dt_render)

                    pitch_out = pos_pitch + orient_pitch
                    roll_out = pos_roll + orient_roll
                    yaw_out = yaw_adj
                else:
                    # Just hold position with yaw toward 0
                    yaw_err = _angle_wrap(0.0 - yaw_cur)
                    pitch_out = pos_pitch
                    roll_out = pos_roll
                    yaw_out = pid_yaw.update(yaw_err, dt_render)

                # Tilt compensation
                tilt = np.sqrt(pitch_cur**2 + roll_cur**2)
                cos_tilt = np.cos(np.clip(tilt, 0, np.radians(60)))
                thrust_offset = thrust_adj + hover_thrust * (1.0 / max(cos_tilt, 0.5) - 1.0)

                # Announce when settle period is over
                if (
                    settle_start
                    and not settle_announced
                    and (time.time() - settle_start) > settle_time
                ):
                    settle_announced = True
                    print(
                        "Settle complete. Holding position. "
                        "Close the viewer window to exit."
                    )

            # ==============================================================
            # Apply controls and step physics
            # ==============================================================
            attitude.compute(data, pitch_out, roll_out, yaw_out, thrust_offset, dt_control)

            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            viewer.sync()

            # -- FPV render --
            if show_fpv:
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=fpv_cam_id)
                fpv_img = renderer.render()
                fpv_bgr = cv2.cvtColor(fpv_img, cv2.COLOR_RGB2BGR)
                fpv_bgr = cv2.flip(fpv_bgr, 0)

                if navigating:
                    label = f"WP {wp_idx}/{len(waypoints)}"
                elif ending_orientation is not None:
                    label = "ORIENT"
                else:
                    label = "HOVER"

                cv2.putText(
                    fpv_bgr,
                    f"T={t_sim:.1f}s  {label}  "
                    f"({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Drone FPV Camera", fpv_bgr)
                cv2.waitKey(1)

            # -- Frame timing --
            elapsed = time.time() - frame_start
            if elapsed < dt_render:
                time.sleep(dt_render - elapsed)

        if show_fpv:
            cv2.destroyAllWindows()
        renderer.close()

    return reached


# ---------------------------------------------------------------------------
# Public API: relative path
# ---------------------------------------------------------------------------
def fly_path_relative(
    waypoints,
    ending_orientation=None,
    cloud_path="skydio_x2/test_cube.ply",
    waypoint_threshold=0.5,
    position_gains=(0.8, 0.02, 0.6),
    altitude_gains=(2.5, 0.3, 1.2),
    yaw_gains=(0.5, 0.02, 0.2),
    orientation_gains=(0.8, 0.05, 0.3),
    max_time_per_waypoint=15.0,
    settle_time=3.0,
    show_fpv=True,
    **sim_kwargs,
):
    """Fly through waypoints given as offsets relative to the drone's start.

    Args:
        waypoints: list of (dx, dy, dz) tuples.  Each offset is relative to
                   the drone's position when the simulation begins (the hover
                   keyframe, normally (0, 0, 1.5)).
        ending_orientation: optional dict {'pitch': deg, 'yaw': deg, 'roll': deg}.
                            After the last waypoint the PID will converge to this
                            attitude while holding position.
        cloud_path: scene point-cloud / mesh file.
        waypoint_threshold: distance (m) to count a waypoint as reached.
        position_gains: (kp, ki, kd) for horizontal position PID.
        altitude_gains: (kp, ki, kd) for altitude PID.
        yaw_gains: (kp, ki, kd) for heading PID.
        orientation_gains: (kp, ki, kd) for final-orientation PID.
        max_time_per_waypoint: seconds before a waypoint is skipped.
        settle_time: seconds to hold final orientation (sim keeps running).
        show_fpv: show the FPV camera OpenCV window.
        **sim_kwargs: forwarded to create_skydio_x2_simulation (distance,
                      height, scale, fpv_width, etc.).

    Returns:
        List of waypoint indices that were successfully reached.
    """
    env = create_skydio_x2_simulation(cloud_path, **sim_kwargs)
    start_pos = env["data"].qpos[0:3].copy()

    absolute_waypoints = [(start_pos + np.array(wp)).tolist() for wp in waypoints]

    print(
        f"Relative path: {len(waypoints)} waypoints from start "
        f"({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f})"
    )
    for i, (rel, abw) in enumerate(zip(waypoints, absolute_waypoints)):
        print(
            f"  WP {i}: rel=({rel[0]:.2f}, {rel[1]:.2f}, {rel[2]:.2f}) "
            f"-> abs=({abw[0]:.2f}, {abw[1]:.2f}, {abw[2]:.2f})"
        )
    if ending_orientation:
        print(f"  Ending orientation: {ending_orientation}")

    return _fly_path(
        env,
        absolute_waypoints,
        ending_orientation=ending_orientation,
        waypoint_threshold=waypoint_threshold,
        position_gains=position_gains,
        altitude_gains=altitude_gains,
        yaw_gains=yaw_gains,
        orientation_gains=orientation_gains,
        max_time_per_waypoint=max_time_per_waypoint,
        settle_time=settle_time,
        show_fpv=show_fpv,
    )


# ---------------------------------------------------------------------------
# Public API: absolute path
# ---------------------------------------------------------------------------
def fly_path_absolute(
    waypoints,
    ending_orientation=None,
    cloud_path="skydio_x2/test_cube.ply",
    waypoint_threshold=0.5,
    position_gains=(0.8, 0.02, 0.6),
    altitude_gains=(2.5, 0.3, 1.2),
    yaw_gains=(0.5, 0.02, 0.2),
    orientation_gains=(0.8, 0.05, 0.3),
    max_time_per_waypoint=15.0,
    settle_time=3.0,
    show_fpv=True,
    **sim_kwargs,
):
    """Fly through waypoints given in absolute world coordinates.

    The world origin is where the drone was placed at the start of the
    simulation (default hover keyframe: x=0, y=0, z=1.0 lifted to 1.5
    by the start position).  Coordinates are in metres.

    Args:
        waypoints: list of (x, y, z) tuples in world frame.
        ending_orientation: optional dict {'pitch': deg, 'yaw': deg, 'roll': deg}.
        (remaining args identical to fly_path_relative)

    Returns:
        List of waypoint indices that were successfully reached.
    """
    env = create_skydio_x2_simulation(cloud_path, **sim_kwargs)
    start_pos = env["data"].qpos[0:3].copy()

    absolute_waypoints = [list(wp) for wp in waypoints]

    print(
        f"Absolute path: {len(waypoints)} waypoints, "
        f"drone at ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f})"
    )
    for i, wp in enumerate(absolute_waypoints):
        print(f"  WP {i}: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})")
    if ending_orientation:
        print(f"  Ending orientation: {ending_orientation}")

    return _fly_path(
        env,
        absolute_waypoints,
        ending_orientation=ending_orientation,
        waypoint_threshold=waypoint_threshold,
        position_gains=position_gains,
        altitude_gains=altitude_gains,
        yaw_gains=yaw_gains,
        orientation_gains=orientation_gains,
        max_time_per_waypoint=max_time_per_waypoint,
        settle_time=settle_time,
        show_fpv=show_fpv,
    )


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: fly a square path 2m on a side at 2m altitude, then settle
    # facing 90 degrees yaw (east-ish).
    fly_path_relative(
        waypoints=[
            (2.0, 0.0, 0.5),   # forward 2m, up 0.5m
            (2.0, 2.0, 0.5),   # then right 2m
            (0.0, 2.0, 0.5),   # then back 2m
            (0.0, 0.0, 0.0),   # return to start (relative)
        ],
        ending_orientation={"pitch": 0.0, "yaw": 90.0, "roll": 0.0},
        cloud_path="skydio_x2/test_cube.ply",
        distance=5.0,
        height=0.0,
        scale=1.0,
    )
