"""
intercept_controller.py — Intercept a moving target using the MPC waypoint controller.

Strategy: compute a lead intercept point each frame and feed it to the existing
MPPIController as a continuously-updating waypoint. The MPPI controller handles all the
attitude/thrust control (which it already does well for static waypoints).

Usage:
    python skydio_x2/intercept_controller.py
"""

import argparse
import numpy as np
import mujoco
import mujoco.viewer
import cv2
import time
import os
import sys
import warnings                                                         
warnings.filterwarnings("ignore", message="Could not import matplotlib")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skydio_x2.mppi_controller import MPPIController
from skydio_x2.skydio_x2_movement import apply_motor_mixing
from skydio_x2.mpc_control_config import DEFAULT_MPPI_CONFIG
from skydio_x2.optimize_weights_mppi import MPPIControlConfig


# ======================================================================
# Lead-point computation
# ======================================================================

def compute_lead_point(drone_pos, drone_vel, target_pos, target_vel,
                       max_speed=7.0, drone_speed=None):
    """Simple lead point: aim ahead of target by at most 1 second of its travel."""
    dist = np.linalg.norm(target_pos - drone_pos)
    if dist < 0.01:
        return target_pos.copy()

    speed = max(drone_speed if drone_speed is not None else 2.0, 2.0)
    target_spd = np.linalg.norm(target_vel)

    # Time to reach target at current speed, capped to 1s
    T = min(dist / speed, 1.0)

    # Further cap: lead offset can't exceed half the current distance
    if target_spd > 0:
        T = min(T, (dist * 0.5) / target_spd)

    return target_pos + target_vel * T


def predict_target_pos(target_pos, target_vel, horizon, dt):
    """Extrapolate target position over the MPC horizon (constant-velocity model).

    Gives the MPC a waypoint at where the target will be when the horizon ends —
    the MPC's own forward model solves how to intercept it.
    """
    return target_pos + target_vel * (horizon * dt)


# ======================================================================
# Scene builder — interceptor + scripted target drone
# ======================================================================

def build_intercept_xml(drone_xml_dir, interceptor_pos="0 0 1.5", target_pos="5 5 2"):
    """MJCF with two drones: one actuated (interceptor), one kinematic (target)."""
    return f"""<mujoco model="drone_intercept">
  <compiler autolimits="true" assetdir="{drone_xml_dir}/assets"/>

  <option timestep="0.005" density="1.225" viscosity="1.8e-5" gravity="0 0 -9.81"/>

  <size nconmax="2000" njmax="4000"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-20" elevation="-20" offwidth="1280" offheight="720"/>
  </visual>

  <default>
    <default class="x2">
      <geom mass="0"/>
      <motor ctrlrange="0 13"/>
      <mesh scale="0.01 0.01 0.01"/>
      <default class="visual">
        <geom group="2" type="mesh" contype="0" conaffinity="0"/>
      </default>
      <default class="collision">
        <geom group="3" type="box"/>
        <default class="rotor">
          <geom type="ellipsoid" size=".13 .13 .01"/>
        </default>
      </default>
      <site group="5"/>
    </default>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>

    <texture type="2d" file="X2_lowpoly_texture_SpinningProps_1024.png"/>
    <material name="phong3SG" texture="X2_lowpoly_texture_SpinningProps_1024"/>
    <material name="invisible" rgba="0 0 0 0"/>
    <mesh class="x2" file="X2_lowpoly.obj"/>
  </asset>

  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <!-- ===== INTERCEPTOR DRONE (actuated) ===== -->
    <light name="spotlight" mode="targetbodycom" target="interceptor" pos="0 -1 2"/>
    <body name="interceptor" pos="{interceptor_pos}" childclass="x2">
      <freejoint name="interceptor_joint"/>
      <camera name="track" pos="-1 0 .5" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="fpv" pos="0 0 0.15" mode="targetbodycom" target="target" fovy="90"/>
      <site name="imu" pos="0 0 .02"/>
      <geom material="phong3SG" mesh="X2_lowpoly" class="visual" quat="0 0 1 1"/>
      <geom class="collision" size=".06 .027 .02" pos=".04 0 .02"/>
      <geom class="collision" size=".06 .027 .02" pos=".04 0 .06"/>
      <geom class="collision" size=".05 .027 .02" pos="-.07 0 .065"/>
      <geom class="collision" size=".023 .017 .01" pos="-.137 .008 .065" quat="1 0 0 1"/>
      <geom name="rotor1" class="rotor" pos="-.14 -.18 .05" mass=".25"/>
      <geom name="rotor2" class="rotor" pos="-.14 .18 .05" mass=".25"/>
      <geom name="rotor3" class="rotor" pos=".14 .18 .08" mass=".25"/>
      <geom name="rotor4" class="rotor" pos=".14 -.18 .08" mass=".25"/>
      <geom size=".16 .04 .02" pos="0 0 0.02" type="ellipsoid" mass=".325"
            class="visual" material="invisible"/>
      <site name="thrust1" pos="-.14 -.18 .05"/>
      <site name="thrust2" pos="-.14 .18 .05"/>
      <site name="thrust3" pos=".14 .18 .08"/>
      <site name="thrust4" pos=".14 -.18 .08"/>
    </body>

    <!-- ===== TARGET DRONE (scripted, no actuators) ===== -->
    <body name="target" pos="{target_pos}">
      <freejoint name="target_joint"/>
      <geom material="phong3SG" mesh="X2_lowpoly" class="visual" quat="0 0 1 1"
            rgba="1 0.2 0.2 1"/>
      <geom class="collision" size=".06 .027 .02" pos=".04 0 .02"/>
      <geom class="collision" size=".06 .027 .02" pos=".04 0 .06"/>
      <geom class="collision" size=".05 .027 .02" pos="-.07 0 .065"/>
      <geom name="rotor1t" class="rotor" pos="-.14 -.18 .05" mass=".25"/>
      <geom name="rotor2t" class="rotor" pos="-.14 .18 .05" mass=".25"/>
      <geom name="rotor3t" class="rotor" pos=".14 .18 .08" mass=".25"/>
      <geom name="rotor4t" class="rotor" pos=".14 -.18 .08" mass=".25"/>
      <geom size=".16 .04 .02" pos="0 0 0.02" type="ellipsoid" mass=".325"
            class="visual" material="invisible"/>
    </body>
  </worldbody>

  <actuator>
    <motor class="x2" name="thrust1" site="thrust1" gear="0 0 1 0 0 -.0201"/>
    <motor class="x2" name="thrust2" site="thrust2" gear="0 0 1 0 0  .0201"/>
    <motor class="x2" name="thrust3" site="thrust3" gear="0 0 1 0 0 -.0201"/>
    <motor class="x2" name="thrust4" site="thrust4" gear="0 0 1 0 0  .0201"/>
  </actuator>

  <sensor>
    <gyro name="body_gyro" site="imu"/>
    <accelerometer name="body_linacc" site="imu"/>
    <framequat name="body_quat" objtype="site" objname="imu"/>
  </sensor>

  <keyframe>
    <key name="hover" qpos="0 0 5.0 1 0 0 0  8 8 6.0 1 0 0 0"
         ctrl="3.2495625 3.2495625 3.2495625 3.2495625"/>
  </keyframe>
</mujoco>"""


# ======================================================================
# Scripted target paths
# ======================================================================

class ScriptedTargetPath:
    """Moves the target drone along a scripted trajectory."""

    def __init__(self, waypoints, speed=1.5):
        self.waypoints = [np.array(wp, dtype=np.float64) for wp in waypoints]
        self.speed = speed
        self._seg = 0
        self._seg_progress = 0.0

    def get_state(self, dt):
        """Advance by dt seconds, return (pos, vel)."""
        if self._seg >= len(self.waypoints) - 1:
            return self.waypoints[-1].copy(), np.zeros(3)

        start = self.waypoints[self._seg]
        end = self.waypoints[self._seg + 1]
        seg_vec = end - start
        seg_len = np.linalg.norm(seg_vec)

        if seg_len < 1e-6:
            self._seg += 1
            return self.get_state(dt)

        direction = seg_vec / seg_len
        vel = direction * self.speed

        self._seg_progress += self.speed * dt
        while self._seg_progress >= seg_len:
            self._seg_progress -= seg_len
            self._seg += 1
            if self._seg >= len(self.waypoints) - 1:
                return self.waypoints[-1].copy(), np.zeros(3)
            start = self.waypoints[self._seg]
            end = self.waypoints[self._seg + 1]
            seg_vec = end - start
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < 1e-6:
                continue
            direction = seg_vec / seg_len
            vel = direction * self.speed

        pos = self.waypoints[self._seg] + direction * self._seg_progress
        return pos, vel


class RandomEvasiveTarget:
    """Target that flies with random acceleration changes, like a real drone.

    Maintains a velocity that gets perturbed by random acceleration impulses
    at random intervals. Velocity is clamped to max_speed and altitude is
    bounded. Produces smooth, unpredictable motion.
    """

    def __init__(
        self,
        start_pos=(8, 8, 2.0),
        max_speed=4.0,
        min_speed=3.0,
        max_accel=3.0,
        min_accel=1.0,
        jink_interval=(0.3, 1.0),
        altitude_range=(1.0, 3.5),
        bounds=((-12, 12), (-12, 12)),
        bias_pos=None,
        bias_strength=1.5,
        seed=None,
    ):
        self._rng = np.random.RandomState(seed)
        self.pos = np.array(start_pos, dtype=np.float64)
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_accel = max_accel
        self.min_accel = min_accel
        self.jink_interval = jink_interval
        self.altitude_range = altitude_range
        self.bounds = bounds
        self.bias_pos = np.array(bias_pos, dtype=np.float64) if bias_pos is not None else None
        self.bias_strength = bias_strength

        # Start with a random horizontal velocity
        angle = self._rng.uniform(0, 2 * np.pi)
        speed = self._rng.uniform(min_speed, max_speed)
        self.vel = np.array([np.cos(angle) * speed, np.sin(angle) * speed, 0.0])

        self._accel = np.zeros(3)
        self._time_to_jink = self._rng.uniform(*jink_interval)

    def _new_jink(self):
        """Pick a new random acceleration direction."""
        # Random horizontal acceleration between min and max
        angle = self._rng.uniform(0, 2 * np.pi)
        mag = self._rng.uniform(self.min_accel, self.max_accel)
        self._accel[0] = np.cos(angle) * mag
        self._accel[1] = np.sin(angle) * mag

        # Small vertical component
        self._accel[2] = self._rng.uniform(-self.max_accel * 0.3, self.max_accel * 0.3)

        self._time_to_jink = self._rng.uniform(*self.jink_interval)

    def get_state(self, dt):
        """Advance by dt seconds, return (pos, vel)."""
        # Count down to next jink
        self._time_to_jink -= dt
        if self._time_to_jink <= 0:
            self._new_jink()

        # Integrate
        self.vel += self._accel * dt

        # Clamp speed to [min_speed, max_speed]
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel *= self.max_speed / speed
        elif speed < self.min_speed and speed > 0.01:
            self.vel *= self.min_speed / speed

        self.pos += self.vel * dt

        # Soft altitude bounds — push back toward range
        alt_lo, alt_hi = self.altitude_range
        if self.pos[2] < alt_lo:
            self.vel[2] += 2.0 * dt
            self.pos[2] = max(self.pos[2], alt_lo - 0.5)
        elif self.pos[2] > alt_hi:
            self.vel[2] -= 2.0 * dt
            self.pos[2] = min(self.pos[2], alt_hi + 0.5)

        # Soft XY bounds — steer back toward center
        for axis in range(2):
            lo, hi = self.bounds[axis]
            if self.pos[axis] < lo:
                self.vel[axis] += 3.0 * dt
            elif self.pos[axis] > hi:
                self.vel[axis] -= 3.0 * dt

        # Bias toward target location
        if self.bias_pos is not None:
            to_bias = self.bias_pos - self.pos
            dist_bias = np.linalg.norm(to_bias)
            if dist_bias > 0.5:
                self.vel += (to_bias / dist_bias) * self.bias_strength * dt

        return self.pos.copy(), self.vel.copy()


# ======================================================================
# Runner
# ======================================================================

def run_intercept(
    target_waypoints=None,
    target_speed=2,
    interceptor_start="0 0 1.5",
    target_start=None,
    intercept_radius=1.0,
    max_duration=30.0,
    lead_gain=0.8,
    record=False,
    video_out="intercept_recording.mp4",
    target_path_override=None,
    debug=False,
    headless=False,
):
    """Launch intercept using MPC + lead-point targeting.

    If *target_path_override* is provided (e.g. a RandomEvasiveTarget), it is
    used instead of building a ScriptedTargetPath from *target_waypoints*.
    """
    drone_xml_dir = os.path.join(os.path.dirname(__file__))
    if not os.path.isdir(os.path.join(drone_xml_dir, "assets")):
        drone_xml_dir = os.path.join("skydio_x2")
    drone_xml_dir = os.path.abspath(drone_xml_dir)

    if target_start is None:
        if target_path_override is not None:
            p = target_path_override.pos
            target_start = f"{p[0]} {p[1]} {p[2]}"
        elif target_waypoints is not None:
            wp0 = target_waypoints[0]
            target_start = f"{wp0[0]} {wp0[1]} {wp0[2]}"
        else:
            target_start = "8 8 2"

    xml = build_intercept_xml(drone_xml_dir, interceptor_start, target_start)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    target_qpos_start = 7
    fpv_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv")
    renderer = mujoco.Renderer(model, height=480, width=640)

    steps_per_frame = 4
    dt_ctrl = model.opt.timestep * steps_per_frame

    # --- MPPI controller — weights loaded from mpc_control_config.py ---
    cfg = MPPIControlConfig.from_json("skydio_x2/best_weights_mppi.json")
    controller = MPPIController(
        horizon=cfg.horizon,
        n_samples=cfg.n_samples,
        dt=dt_ctrl,
    )
    cfg.apply_to(controller)

    if target_path_override is not None:
        target_path = target_path_override
    else:
        target_path = ScriptedTargetPath(target_waypoints, speed=target_speed)
    hover_thrust = 3.2495625
    dt_render = 1.0 / 60.0

    is_random = isinstance(target_path, RandomEvasiveTarget)
    print(f"\n{'='*60}")
    print(f"  DRONE INTERCEPT — MPC + Lead Point")
    print(f"{'='*60}")
    print(f"  Interceptor start: {interceptor_start}")
    print(f"  Target start:      {target_start}")
    if is_random:
        print(f"  Target mode:       RANDOM EVASIVE")
        print(f"  Max speed:         {target_path.max_speed} m/s")
        print(f"  Max accel:         {target_path.max_accel} m/s^2")
    else:
        print(f"  Target speed:      {target_speed} m/s")
        print(f"  Target waypoints:  {len(target_waypoints)}")
        for i, wp in enumerate(target_waypoints):
            print(f"    [{i}] {wp}")
    print(f"  Intercept radius:  {intercept_radius} m")
    print(f"  Max duration:      {max_duration} s")
    print(f"{'='*60}\n")

    # --- Video recording setup ---
    fpv_writer = None
    third_person_writer = None
    third_person_renderer = None
    if record:
        if os.path.isdir(video_out):
            video_out = os.path.join(video_out, "intercept_recording.mp4")
        # Use .avi + MJPG for reliable encoding on Windows
        fpv_out = video_out.replace(".mp4", "_fpv.avi")
        tp_out = video_out.replace(".mp4", "_3rdperson.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        actual_fps = 1.0 / dt_ctrl  # 50 fps — one frame per control step
        fpv_writer = cv2.VideoWriter(fpv_out, fourcc, actual_fps, (640, 480))
        # Third-person render at higher res
        third_person_renderer = mujoco.Renderer(model, height=720, width=1280)
        third_person_writer = cv2.VideoWriter(tp_out, fourcc, actual_fps, (1280, 720))
        if not fpv_writer.isOpened() or not third_person_writer.isOpened():
            print("  WARNING: Failed to open video writers!")
        print(f"  Recording FPV to:         {fpv_out}")
        print(f"  Recording 3rd-person to:  {tp_out}")

    intercepted = False
    min_dist_seen = float('inf')
    last_thrust = 0.0
    last_pitch = 0.0
    last_roll = 0.0
    last_yaw = 0.0
    phase = "cruise"
    prev_target_pos = None

    def _run_loop(viewer):
        nonlocal intercepted, min_dist_seen, last_thrust, last_pitch, last_roll, last_yaw, phase, prev_target_pos
        sim_time = 0.0
        try:
            while (viewer is None or viewer.is_running()):
                step_start = time.time()

                if sim_time > max_duration:
                    print(f"\n  TIMEOUT at t={sim_time:.1f}s  (closest: {min_dist_seen:.2f}m)")
                    break

                # --- Advance target ---
                target_pos, target_vel = target_path.get_state(dt_ctrl)

                # Fix 7: detect large target jink and reset warm-start
                if prev_target_pos is not None:
                    target_accel_est = np.linalg.norm(target_vel - data.qvel[6:9]) / dt_ctrl
                    if target_accel_est > 8.0:  # m/s² threshold
                        controller._mean = None  # drop stale warm-start
                prev_target_pos = target_pos.copy()

                # Kinematic override
                data.qpos[target_qpos_start:target_qpos_start + 3] = target_pos
                data.qpos[target_qpos_start + 3:target_qpos_start + 7] = [1, 0, 0, 0]
                data.qvel[6:9] = target_vel
                data.qvel[9:12] = [0, 0, 0]

                drone_pos = data.qpos[:3].copy()
                drone_vel = data.qvel[:3].copy()
                drone_quat = data.qpos[3:7].copy()
                tilt_deg = np.degrees(2.0 * np.arcsin(np.clip(
                    np.sqrt(drone_quat[1]**2 + drone_quat[2]**2), 0, 1)))
                dist_to_target = np.linalg.norm(drone_pos - target_pos)
                min_dist_seen = min(min_dist_seen, dist_to_target)

                # Debug: print every timestep with altitude
                if debug and phase == "strike":
                    lead_preview = predict_target_pos(target_pos, target_vel,
                                                      controller.horizon, dt_ctrl)
                    predicted_target_pos = predict_target_pos(target_pos, target_vel,
                                                              controller.horizon, dt_ctrl)
                    dist_force = np.linalg.norm(controller._disturbance_force)
                    print(
                        f"  t={sim_time:.2f}s  phase={phase:6s}  dist={dist_to_target:.3f}m\n"
                        f"    drone_pos          = [{drone_pos[0]:7.3f}, {drone_pos[1]:7.3f}, {drone_pos[2]:7.3f}]\n"
                        f"    drone_vel          = [{drone_vel[0]:7.3f}, {drone_vel[1]:7.3f}, {drone_vel[2]:7.3f}]  spd={np.linalg.norm(drone_vel):.2f}m/s\n"
                        f"    target_pos         = [{target_pos[0]:7.3f}, {target_pos[1]:7.3f}, {target_pos[2]:7.3f}]\n"
                        f"    target_vel         = [{target_vel[0]:7.3f}, {target_vel[1]:7.3f}, {target_vel[2]:7.3f}]  spd={np.linalg.norm(target_vel):.2f}m/s\n"
                        f"    predicted_tgt_pos  = [{predicted_target_pos[0]:7.3f}, {predicted_target_pos[1]:7.3f}, {predicted_target_pos[2]:7.3f}]\n"
                        f"    lead_pt            = [{lead[0]:7.3f}, {lead[1]:7.3f}, {lead[2]:7.3f}]\n"
                        f"    disturbance        = {controller._disturbance_force}  |F|={dist_force:.3f}N\n"
                        f"    tilt={tilt_deg:.1f}deg  motors=[{data.ctrl[0]:.2f},{data.ctrl[1]:.2f},{data.ctrl[2]:.2f},{data.ctrl[3]:.2f}]"
                    )

                # --- Check intercept ---
                if dist_to_target < intercept_radius:
                    drone_quat = data.qpos[3:7]
                    qw, qx, qy, qz = drone_quat
                    roll_a = np.degrees(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2)))
                    pitch_a = np.degrees(np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1)))
                    yaw_a = np.degrees(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2)))
                    closing = np.linalg.norm(drone_vel - target_vel)

                    print(f"\n{'*'*60}")
                    print(f"  TARGET INTERCEPTED at t={sim_time:.2f}s")
                    print(f"{'*'*60}")
                    print(f"  Drone pos:     [{drone_pos[0]:.3f}, {drone_pos[1]:.3f}, {drone_pos[2]:.3f}]")
                    print(f"  Target pos:    [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                    print(f"  Distance:      {dist_to_target:.4f} m")
                    print(f"  Drone vel:     [{drone_vel[0]:.2f}, {drone_vel[1]:.2f}, {drone_vel[2]:.2f}] m/s")
                    print(f"  Target vel:    [{target_vel[0]:.2f}, {target_vel[1]:.2f}, {target_vel[2]:.2f}] m/s")
                    print(f"  Closing speed: {closing:.2f} m/s")
                    print(f"  Orientation:   roll={roll_a:.1f}°  pitch={pitch_a:.1f}°  yaw={yaw_a:.1f}°")
                    print(f"  Motors:        [{data.ctrl[0]:.2f}, {data.ctrl[1]:.2f}, {data.ctrl[2]:.2f}, {data.ctrl[3]:.2f}]")
                    print(f"{'*'*60}\n")
                    intercepted = True
                    break

                # --- Two-phase guidance ---
                # CRUISE: fly toward lead point; STRIKE: close in and intercept
                drone_speed = np.linalg.norm(drone_vel)
                if phase == "cruise" and dist_to_target <= cfg.strike_range and 2.0 <= drone_speed <= 8.0 and tilt_deg < 45.0:
                    phase = "strike"
                    cfg.apply_strike(controller)
                    controller.reset()
                    controller._disturbance_force = np.zeros(3)
                    controller._predicted_vel = None
                    print(f"  t={sim_time:.2f}s  STRIKE PHASE  dist={dist_to_target:.2f}m  spd={drone_speed:.2f}m/s")

                if phase == "strike":
                    # Aim at where target will be in one control step — pure pursuit at close range
                    lead = target_pos + target_vel * dt_ctrl * 3
                else:
                    lead = compute_lead_point(
                        drone_pos, drone_vel, target_pos, target_vel,
                        drone_speed=np.linalg.norm(drone_vel),
                    )

                ctrl = controller.solve(data, lead,
                                        target_vel=target_vel if phase == "strike" else None)
                thrust_offset, pitch_cmd, roll_cmd, yaw_cmd = ctrl

                last_thrust = thrust_offset
                last_pitch = pitch_cmd
                last_roll = roll_cmd
                last_yaw = yaw_cmd

                base = hover_thrust + thrust_offset
                apply_motor_mixing(data, pitch_cmd, roll_cmd, yaw_cmd, base)

                # --- Physics --- pin target first so MuJoCo integrates with correct state
                data.qpos[target_qpos_start:target_qpos_start + 3] = target_pos
                data.qpos[target_qpos_start + 3:target_qpos_start + 7] = [1, 0, 0, 0]
                data.qvel[6:9] = target_vel
                data.qvel[9:12] = [0, 0, 0]
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)
                sim_time += dt_ctrl

                # Re-pin target after physics to correct any drift
                data.qpos[target_qpos_start:target_qpos_start + 3] = target_pos
                data.qpos[target_qpos_start + 3:target_qpos_start + 7] = [1, 0, 0, 0]
                data.qvel[6:9] = target_vel
                data.qvel[9:12] = [0, 0, 0]

                # --- Views ---
                if not headless:
                    viewer.sync()

                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=fpv_cam_id)
                    fpv_img = renderer.render()
                    fpv_bgr = cv2.cvtColor(fpv_img, cv2.COLOR_RGB2BGR)
                    fpv_bgr = cv2.flip(fpv_bgr, 0)

                    phase_color = (0, 255, 0) if phase == "cruise" else (0, 0, 255)
                    cv2.putText(
                        fpv_bgr,
                        f"{phase.upper()}  t={sim_time:.1f}s  dist={dist_to_target:.2f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, phase_color, 2,
                    )
                    cv2.putText(
                        fpv_bgr,
                        f"lead=[{lead[0]:.1f},{lead[1]:.1f},{lead[2]:.1f}]  "
                        f"thr={thrust_offset:+.2f}  p={pitch_cmd:+.2f}  r={roll_cmd:+.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1,
                    )
                    cv2.imshow("Interceptor FPV", fpv_bgr)
                    cv2.waitKey(1)

                # --- Write video frames ---
                if record and not headless:
                    fpv_writer.write(fpv_bgr)
                    # Third-person: use the same camera angles as the interactive viewer
                    third_person_renderer.update_scene(data)
                    tp_img = third_person_renderer.render()
                    tp_bgr = cv2.cvtColor(tp_img, cv2.COLOR_RGB2BGR)
                    tp_bgr = cv2.flip(tp_bgr, 0)
                    cv2.putText(
                        tp_bgr,
                        f"t={sim_time:.1f}s  dist={dist_to_target:.2f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )
                    third_person_writer.write(tp_bgr)

                # Periodic status
                if int(sim_time * 10) % 20 == 0 and int(sim_time * 10) > 0:
                    print(
                        f"  t={sim_time:.1f}s  dist={dist_to_target:.2f}m  "
                        f"lead=[{lead[0]:.1f},{lead[1]:.1f},{lead[2]:.1f}]  "
                        f"tgt=[{target_pos[0]:.1f},{target_pos[1]:.1f},{target_pos[2]:.1f}]  "
                        f"min={min_dist_seen:.2f}m"
                    )

                if not headless:
                    frame_elapsed = time.time() - step_start
                    if frame_elapsed < dt_render:
                        time.sleep(dt_render - frame_elapsed)

        finally:
            if not headless:
                cv2.destroyAllWindows()
                renderer.close()
                if record:
                    fpv_writer.release()
                    third_person_writer.release()
                    third_person_renderer.close()
                    print(f"\n  Videos saved.")

    if headless:
        _run_loop(None)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth = -60
            viewer.cam.elevation = -30
            viewer.cam.distance = 10.0
            viewer.cam.lookat = [2.5, 2.5, 2.0]
            _run_loop(viewer)

    if intercepted:
        print("Intercept scenario complete — HIT!")
    else:
        print("Intercept scenario complete — MISS.")
    return intercepted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone intercept simulation")
    parser.add_argument("--record", action="store_true", help="Enable video recording")
    parser.add_argument("--video-out", default="intercept_recording.mp4",
                        help="Output video path (default: intercept_recording.mp4)")
    parser.add_argument("--mode", choices=["scripted", "evasive"], default="scripted",
                        help="Target behaviour: scripted waypoints or random evasive (default: scripted)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for evasive target (default: random)")
    parser.add_argument("--target-speed", type=float, default=4.0,
                        help="Target max speed in m/s (default: 4.0)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-step debug info: drone pos/vel, target pos/vel, lead point, disturbance")
    parser.add_argument("--headless", action="store_true",
                        help="Run without visualization (faster, for batch testing)")
    parser.add_argument("--iters", type=int, default=1,
                        help="Number of iterations to run; stops and prints seed on first failure")
    args = parser.parse_args()

    failed_seeds = []

    for iteration in range(args.iters):
        if args.iters > 1:
            print(f"\n{'='*60}\n  ITERATION {iteration + 1}/{args.iters}\n{'='*60}")

        if args.mode == "evasive":
            seed = args.seed if args.seed is not None else np.random.randint(0, 100000)
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
            hit = run_intercept(
                interceptor_start="0 0 5.0",
                intercept_radius=1.0,
                max_duration=45.0,
                record=args.record,
                video_out=args.video_out,
                target_path_override=target,
                debug=args.debug,
                headless=args.headless,
            )
        else:
            seed = None
            TARGET_PATH = [
                [10, 10, 6.0],
                [8, 6, 6.5],
                [6, 8, 6.0],
                [4, 4, 6.5],
                [2, 6, 6.0],
                [0, 2, 6.5],
                [-2, 4, 6.0],
                [-4, 0, 6.5],
                [-6, 2, 6.0],
                [-8, -2, 6.5],
                [-6, -4, 6.0],
                [-4, -2, 6.5],
                [-2, -4, 6.0],
                [0, 0, 6.5],
            ]
            hit = run_intercept(
                target_waypoints=TARGET_PATH,
                target_speed=args.target_speed,
                interceptor_start="0 0 5.0",
                intercept_radius=1.0,
                max_duration=45.0,
                record=args.record,
                video_out=args.video_out,
                debug=args.debug,
                headless=args.headless,
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
