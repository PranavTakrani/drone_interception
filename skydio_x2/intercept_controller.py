"""
intercept_controller.py — Intercept a moving target using the MPC waypoint controller.

Strategy: compute a lead intercept point each frame and feed it to the existing
MPCController as a continuously-updating waypoint. The MPC handles all the
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skydio_x2.mpc_controller import MPCController
from skydio_x2.skydio_x2_movement import apply_motor_mixing


# ======================================================================
# Lead-point computation
# ======================================================================

def compute_lead_point(drone_pos, drone_vel, target_pos, target_vel, max_speed=7.0):
    """Predict where the drone should aim to intercept the target.

    Uses simple time-to-intercept estimate: T = dist / closing_speed.
    Lead point = target_pos + target_vel * T.
    Clamped to avoid runaway extrapolation.
    """
    to_target = target_pos - drone_pos
    dist = np.linalg.norm(to_target)
    if dist < 0.01:
        return target_pos.copy()

    los = to_target / dist
    closing = np.dot(drone_vel - target_vel, -los)
    effective_speed = max(closing, max_speed * 0.7, 3.0)
    T = np.clip(dist / effective_speed, 0.05, 2.0)

    lead = target_pos + target_vel * T
    return lead


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
      <geom type="ellipsoid" size="0.2 0.2 0.08" rgba="1 0.2 0.2 0.9"
            mass="1.0" contype="1" conaffinity="1"/>
      <geom type="sphere" size="0.05" pos="0 0 0.1" rgba="1 1 0 1"
            contype="0" conaffinity="0"/>
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
    <key name="hover" qpos="0 0 1.5 1 0 0 0  5 5 2 1 0 0 0"
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


# ======================================================================
# Runner
# ======================================================================

def run_intercept(
    target_waypoints,
    target_speed=2,
    interceptor_start="0 0 1.5",
    target_start=None,
    intercept_radius=0.5,
    max_duration=30.0,
    lead_gain=0.8,
    record=False,
    video_out="intercept_recording.mp4",
):
    """Launch intercept using MPC + lead-point targeting."""
    drone_xml_dir = os.path.join(os.path.dirname(__file__))
    if not os.path.isdir(os.path.join(drone_xml_dir, "assets")):
        drone_xml_dir = os.path.join("skydio_x2")
    drone_xml_dir = os.path.abspath(drone_xml_dir)

    if target_start is None:
        wp0 = target_waypoints[0]
        target_start = f"{wp0[0]} {wp0[1]} {wp0[2]}"

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

    # --- MPC controller (same one that flies waypoints, tuned for speed) ---
    controller = MPCController(
        horizon=15,
        n_samples=400,
        n_elite=60,
        n_iterations=5,
        dt=dt_ctrl,
    )
    # Override weights for intercept: aggressive but stable
    controller.w_pos = 25.0         # strong pull to lead point
    controller.w_vel = 1.0          # light braking (prevents runaway speed)
    controller.w_tilt = 10.0        # firm uprightness — prevents inversion
    controller.w_tilt_running = 8.0 # stay upright throughout, not just terminal
    controller.w_yaw = 1.0
    controller.w_pos_running = 8.0  # strong running position pull
    controller.w_ang_vel = 4.0      # resist spinning at terminal
    controller.w_ang_vel_running = 3.0  # resist spinning throughout
    controller.w_ctrl_rate = 0.5    # moderately snappy
    controller.att_max = 0.8        # ~46° max tilt — aggressive but recoverable
    controller.thrust_max = 6.0     # good headroom without saturation

    target_path = ScriptedTargetPath(target_waypoints, speed=target_speed)
    hover_thrust = 3.2495625
    dt_render = 1.0 / 60.0

    print(f"\n{'='*60}")
    print(f"  DRONE INTERCEPT — MPC + Lead Point")
    print(f"{'='*60}")
    print(f"  Interceptor start: {interceptor_start}")
    print(f"  Target start:      {target_start}")
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

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = -60
        viewer.cam.elevation = -30
        viewer.cam.distance = 10.0
        viewer.cam.lookat = [2.5, 2.5, 2.0]

        sim_time = 0.0

        try:
            while viewer.is_running():
                step_start = time.time()

                if sim_time > max_duration:
                    print(f"\n  TIMEOUT at t={sim_time:.1f}s  (closest: {min_dist_seen:.2f}m)")
                    break

                # --- Advance target ---
                target_pos, target_vel = target_path.get_state(dt_ctrl)

                # Kinematic override
                data.qpos[target_qpos_start:target_qpos_start + 3] = target_pos
                data.qpos[target_qpos_start + 3:target_qpos_start + 7] = [1, 0, 0, 0]
                data.qvel[6:9] = target_vel
                data.qvel[9:12] = [0, 0, 0]

                drone_pos = data.qpos[:3].copy()
                drone_vel = data.qvel[:3].copy()
                dist_to_target = np.linalg.norm(drone_pos - target_pos)
                min_dist_seen = min(min_dist_seen, dist_to_target)

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

                # --- Guidance mode: MPC or terminal coast ---
                terminal_dist = 0.1  # metres — stop replanning, coast to impact

                if dist_to_target > terminal_dist:
                    # Normal MPC guidance toward lead point
                    lead = compute_lead_point(drone_pos, drone_vel, target_pos, target_vel)
                    ctrl = controller.solve(data, lead)
                    thrust_offset, pitch_cmd, roll_cmd, yaw_cmd = ctrl
                    # Save last good controls for terminal phase
                    last_thrust = thrust_offset
                    last_pitch = pitch_cmd
                    last_roll = roll_cmd
                    last_yaw = yaw_cmd
                else:
                    # TERMINAL: lock attitude, hold last thrust, coast into target
                    lead = target_pos  # for HUD only
                    thrust_offset = last_thrust
                    pitch_cmd = 0.0   # zero out attitude commands — hold current orientation
                    roll_cmd = 0.0
                    yaw_cmd = 0.0

                base = hover_thrust + thrust_offset
                apply_motor_mixing(data, pitch_cmd, roll_cmd, yaw_cmd, base)

                # --- Physics ---
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)
                sim_time += dt_ctrl

                # Re-pin target
                data.qpos[target_qpos_start:target_qpos_start + 3] = target_pos
                data.qpos[target_qpos_start + 3:target_qpos_start + 7] = [1, 0, 0, 0]
                data.qvel[6:9] = target_vel
                data.qvel[9:12] = [0, 0, 0]

                # --- Views ---
                viewer.sync()

                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=fpv_cam_id)
                fpv_img = renderer.render()
                fpv_bgr = cv2.cvtColor(fpv_img, cv2.COLOR_RGB2BGR)
                fpv_bgr = cv2.flip(fpv_bgr, 0)

                cv2.putText(
                    fpv_bgr,
                    f"MPC INTERCEPT  t={sim_time:.1f}s  dist={dist_to_target:.2f}m",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
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
                if record:
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

                frame_elapsed = time.time() - step_start
                if frame_elapsed < dt_render:
                    time.sleep(dt_render - frame_elapsed)

        finally:
            cv2.destroyAllWindows()
            renderer.close()
            if record:
                fpv_writer.release()
                third_person_writer.release()
                third_person_renderer.close()
                print(f"\n  Videos saved.")


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
    args = parser.parse_args()

    # Figure-S pattern
    TARGET_PATH = [
        [10, 10, 2.0],
        [8, 6, 2.5],
        [6, 8, 2.0],
        [4, 4, 2.5],
        [2, 6, 2.0],
        [0, 2, 2.5],
        [-2, 4, 2.0],
        [-4, 0, 2.5],
        [-6, 2, 2.0],
        [-8, -2, 2.5],
        [-6, -4, 2.0],
        [-4, -2, 2.5],
        [-2, -4, 2.0],
        [0, 0, 2.5],
    ]

    run_intercept(
        target_waypoints=TARGET_PATH,
        target_speed=4,
        interceptor_start="0 0 1.5",
        intercept_radius=1.2,
        max_duration=45.0,
        record=args.record,
        video_out=args.video_out,
    )
