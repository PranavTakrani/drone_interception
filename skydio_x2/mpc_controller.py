"""
mpc_controller.py — Model Predictive Control for the Skydio X2 drone.

Uses the Cross-Entropy Method (CEM) to optimize control sequences over a
receding horizon, producing smooth trajectories between waypoints.

Instead of bang-bang commands ("pitch forward for 0.1s, coast 0.5s"), MPC
takes a target position and solves for the optimal motor commands at each
timestep. The result is smooth acceleration/deceleration with no overshoot.

Usage:
    from skydio_x2.mpc_controller import run_mpc_waypoints

    waypoints = [
        [2, 0, 1.5],
        [2, 2, 2.0],
        [0, 0, 1.5],
    ]
    run_mpc_waypoints("skydio_x2/test_cube.ply", waypoints)
"""

import numpy as np
import mujoco
import cv2
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skydio_x2.skydio_x2_movement import (
    create_skydio_x2_simulation,
    apply_motor_mixing,
)


class MPCController:
    """CEM-based Model Predictive Controller for quadrotor position tracking.

    At each control step the controller:
      1. Reads the drone's full state (pos, vel, orientation, angular vel).
      2. Samples many candidate control sequences over a short horizon.
      3. Predicts the resulting trajectory with a simplified dynamics model.
      4. Scores each trajectory against a cost function (position error,
         velocity, smoothness, uprightness).
      5. Refines the distribution toward the best candidates (CEM).
      6. Applies only the first control of the best sequence, then re-plans.
    """

    def __init__(
        self,
        horizon=15,
        n_samples=400,
        n_elite=60,
        n_iterations=5,
        dt=0.02,
    ):
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iterations = n_iterations
        self.dt = dt

        # --- Drone physical parameters ---
        self.mass = 1.325  # kg  (4 * 0.25 rotor + 0.325 body)
        self.gravity = 9.81
        self.hover_thrust_per_motor = 3.2495625
        self.max_motor_thrust = 13.0

        # Approximate angular-acceleration gains (from arm geometry & inertia)
        #   pitch arm ≈ 0.14 m, Iyy ≈ 0.020 → k ≈ 28
        #   roll  arm ≈ 0.18 m, Ixx ≈ 0.032 → k ≈ 22.5
        #   yaw   gear 0.0201,  Izz ≈ 0.052 → k ≈ 1.5
        self.k_pitch = 28.0
        self.k_roll = 22.5
        self.k_yaw = 1.5
        self.angular_damping = 0.5  # very low — MuJoCo freejoint has minimal damping

        # --- Cost weights ---
        self.w_pos = 12.0         # terminal position error
        self.w_vel = 4.0          # terminal velocity penalty
        self.w_tilt = 8.0         # stay upright (terminal)
        self.w_tilt_running = 5.0 # stay upright throughout trajectory
        self.w_yaw = 4.0          # penalise yaw drift
        self.w_ctrl_rate = 1.0    # penalise jerky control changes
        self.w_pos_running = 3.0  # running position cost
        self.w_ang_vel = 3.0      # terminal angular velocity
        self.w_ang_vel_running = 2.0  # running angular velocity

        # --- Control bounds ---
        self.thrust_min = -2.0
        self.thrust_max = 4.0
        self.att_max = 0.6  # cap tilt authority (~34° max)

        # --- Internal warm-start state ---
        self._mean = None
        self._prev_ctrl = np.zeros(4)

        # --- Disturbance estimator ---
        # Compares predicted vs actual state each step to estimate an
        # unmodelled external force (wind, payload shift, etc.).
        self._disturbance_force = np.zeros(3)  # world-frame force estimate (N)
        self._predicted_vel = None              # velocity predicted last step
        self._disturbance_alpha = 0.4           # EMA smoothing (0 = ignore, 1 = instant)

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------
    def get_state(self, data):
        """Read drone state from MuJoCo data."""
        pos = data.qpos[:3].copy()
        quat = data.qpos[3:7].copy()   # w, x, y, z
        vel = data.qvel[:3].copy()
        ang_vel = data.qvel[3:6].copy()
        return pos, quat, vel, ang_vel

    # ------------------------------------------------------------------
    # Vectorised forward dynamics (simplified quadrotor model)
    # ------------------------------------------------------------------
    def _predict_batch(self, pos, quat, vel, ang_vel, controls):
        """Predict trajectories for all samples in parallel.

        Parameters
        ----------
        pos      : (3,)  initial world position
        quat     : (4,)  initial quaternion  [w, x, y, z]
        vel      : (3,)  initial world-frame velocity
        ang_vel  : (3,)  initial angular velocity
        controls : (N, H, 4)  [thrust_offset, pitch, roll, yaw]

        Returns
        -------
        final_pos    : (N, 3)
        final_vel    : (N, 3)
        final_quat   : (N, 4)
        final_ang_vel: (N, 3)
        all_pos      : list[H] of (N, 3)
        """
        N = controls.shape[0]

        p = np.tile(pos, (N, 1))
        v = np.tile(vel, (N, 1))
        q = np.tile(quat, (N, 1))
        w = np.tile(ang_vel, (N, 1))

        all_pos = []
        all_quat = []
        all_ang_vel = []

        for t in range(self.horizon):
            ctrl = controls[:, t, :]
            thrust_off = ctrl[:, 0]
            pitch_cmd = ctrl[:, 1]
            roll_cmd = ctrl[:, 2]
            yaw_cmd = ctrl[:, 3]

            # Per-motor thrust → total thrust (pitch/roll/yaw cancel in sum)
            per_motor = np.clip(
                self.hover_thrust_per_motor + thrust_off,
                0,
                self.max_motor_thrust,
            )
            total_thrust = per_motor * 4

            # Body z-axis in world frame from quaternion
            qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            bz_x = 2 * (qx * qz + qw * qy)
            bz_y = 2 * (qy * qz - qw * qx)
            bz_z = 1 - 2 * (qx * qx + qy * qy)

            # Linear acceleration  (thrust along body-z minus gravity + disturbance)
            t_acc = total_thrust / self.mass
            dist_acc = self._disturbance_force / self.mass  # (3,)
            acc = np.stack(
                [bz_x * t_acc + dist_acc[0],
                 bz_y * t_acc + dist_acc[1],
                 bz_z * t_acc - self.gravity + dist_acc[2]],
                axis=1,
            )

            v = v + acc * self.dt
            p = p + v * self.dt

            # Angular dynamics  (torque – damping)
            torque = np.stack(
                [
                    -roll_cmd * self.k_roll,
                    -pitch_cmd * self.k_pitch,
                    -yaw_cmd * self.k_yaw,
                ],
                axis=1,
            )
            w = w + (torque - self.angular_damping * w) * self.dt

            # Quaternion integration
            dq = 0.5 * np.stack(
                [
                    -(w[:, 0] * q[:, 1] + w[:, 1] * q[:, 2] + w[:, 2] * q[:, 3]),
                    (w[:, 0] * q[:, 0] + w[:, 2] * q[:, 2] - w[:, 1] * q[:, 3]),
                    (w[:, 1] * q[:, 0] - w[:, 2] * q[:, 1] + w[:, 0] * q[:, 3]),
                    (w[:, 2] * q[:, 0] + w[:, 1] * q[:, 1] - w[:, 0] * q[:, 2]),
                ],
                axis=1,
            )
            q = q + dq * self.dt
            q = q / np.linalg.norm(q, axis=1, keepdims=True)

            all_pos.append(p.copy())
            all_quat.append(q.copy())
            all_ang_vel.append(w.copy())

        return p, v, q, w, all_pos, all_quat, all_ang_vel

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------
    def _compute_costs(self, final_pos, final_vel, final_quat, final_ang_vel,
                        all_pos, all_quat, all_ang_vel, controls, target):
        """Vectorised cost for all N samples."""
        # Terminal position error
        pos_err = final_pos - target[np.newaxis, :]
        cost = self.w_pos * np.sum(pos_err ** 2, axis=1)

        # Terminal velocity (want to arrive with low speed)
        cost += self.w_vel * np.sum(final_vel ** 2, axis=1)

        # Terminal uprightness  (qw² + qz² = 1 for pure yaw, penalise pitch/roll)
        tilt_err = final_quat[:, 1] ** 2 + final_quat[:, 2] ** 2  # qx² + qy²
        cost += self.w_tilt * tilt_err

        # Terminal yaw penalty (penalise yaw drift from identity)
        cost += self.w_yaw * final_quat[:, 3] ** 2  # qz² measures yaw

        # Terminal angular velocity
        cost += self.w_ang_vel * np.sum(final_ang_vel ** 2, axis=1)

        # Running costs (applied at every horizon step)
        inv_h = 1.0 / self.horizon
        for t in range(self.horizon):
            # Running position
            err = all_pos[t] - target[np.newaxis, :]
            cost += self.w_pos_running * inv_h * np.sum(err ** 2, axis=1)

            # Running tilt — stay upright throughout, not just at the end
            qt = all_quat[t]
            cost += self.w_tilt_running * inv_h * (qt[:, 1] ** 2 + qt[:, 2] ** 2)

            # Running angular velocity — resist spinning at every step
            cost += self.w_ang_vel_running * inv_h * np.sum(all_ang_vel[t] ** 2, axis=1)

        # Control smoothness within horizon
        if self.horizon > 1:
            diffs = controls[:, 1:, :] - controls[:, :-1, :]
            cost += self.w_ctrl_rate * np.sum(diffs ** 2, axis=(1, 2))

        # Smoothness across MPC steps (penalise jump from previous control)
        first_diff = controls[:, 0, :] - self._prev_ctrl[np.newaxis, :]
        cost += self.w_ctrl_rate * 2.0 * np.sum(first_diff ** 2, axis=1)

        return cost

    # ------------------------------------------------------------------
    # CEM optimisation
    # ------------------------------------------------------------------
    def solve(self, data, target_pos):
        """Run CEM and return the first-step control.

        Parameters
        ----------
        data       : MuJoCo MjData
        target_pos : array-like (3,)  desired [x, y, z]

        Returns
        -------
        ctrl : ndarray (4,)  [thrust_offset, pitch_cmd, roll_cmd, yaw_cmd]
        """
        pos, quat, vel, ang_vel = self.get_state(data)
        target = np.asarray(target_pos, dtype=np.float64)

        # --- Disturbance estimation ---
        # If we predicted a velocity last step, compare it to the actual
        # velocity now.  The residual (actual - predicted) * mass / dt
        # estimates the force NOT yet captured by self._disturbance_force.
        # We integrate the residual with smoothing so the estimate converges
        # to the true external force:  d_new = d_old + alpha * residual
        # (standard EMA: d = (1-a)*d + a*wind  rearranges to d += a*(wind-d))
        if self._predicted_vel is not None:
            vel_error = vel - self._predicted_vel  # (3,)
            force_residual = vel_error * self.mass / self.dt
            self._disturbance_force += self._disturbance_alpha * force_residual

        # Initialise / warm-start the sampling distribution
        if self._mean is None:
            mean = np.zeros((self.horizon, 4))
            std = np.tile([1.0, 0.3, 0.3, 0.2], (self.horizon, 1))
        else:
            mean = np.roll(self._mean, -1, axis=0)
            mean[-1] = mean[-2]
            std = np.tile([0.5, 0.2, 0.2, 0.15], (self.horizon, 1))

        for _ in range(self.n_iterations):
            noise = np.random.randn(self.n_samples, self.horizon, 4)
            samples = mean[np.newaxis, :, :] + noise * std[np.newaxis, :, :]

            # Clamp to control bounds
            samples[:, :, 0] = np.clip(samples[:, :, 0], self.thrust_min, self.thrust_max)
            samples[:, :, 1] = np.clip(samples[:, :, 1], -self.att_max, self.att_max)
            samples[:, :, 2] = np.clip(samples[:, :, 2], -self.att_max, self.att_max)
            samples[:, :, 3] = np.clip(samples[:, :, 3], -self.att_max, self.att_max)

            # Forward predict all samples
            final_pos, final_vel, final_quat, final_ang_vel, all_pos, all_quat, all_ang_vel = self._predict_batch(
                pos, quat, vel, ang_vel, samples,
            )

            # Evaluate cost
            costs = self._compute_costs(
                final_pos, final_vel, final_quat, final_ang_vel,
                all_pos, all_quat, all_ang_vel, samples, target,
            )

            # Elite selection & distribution update
            elite_idx = np.argsort(costs)[: self.n_elite]
            elite = samples[elite_idx]
            mean = np.mean(elite, axis=0)
            std = np.std(elite, axis=0) + 1e-3

        self._mean = mean
        best_ctrl = mean[0]
        self._prev_ctrl = best_ctrl.copy()

        # Predict next-step velocity for disturbance estimation on the next call.
        # Inline single-step forward prediction (avoids horizon mismatch with
        # _predict_batch which uses self.horizon).
        thrust_off = best_ctrl[0]
        per_motor = np.clip(
            self.hover_thrust_per_motor + thrust_off, 0, self.max_motor_thrust)
        total_thrust = per_motor * 4
        qw, qx, qy, qz = quat
        bz_x = 2 * (qx * qz + qw * qy)
        bz_y = 2 * (qy * qz - qw * qx)
        bz_z = 1 - 2 * (qx * qx + qy * qy)
        t_acc = total_thrust / self.mass
        dist_acc = self._disturbance_force / self.mass
        pred_acc = np.array([
            bz_x * t_acc + dist_acc[0],
            bz_y * t_acc + dist_acc[1],
            bz_z * t_acc - self.gravity + dist_acc[2],
        ])
        self._predicted_vel = vel + pred_acc * self.dt

        return best_ctrl

    def reset(self):
        """Clear warm-start state (call when switching targets abruptly)."""
        self._mean = None
        self._prev_ctrl = np.zeros(4)
        self._disturbance_force = np.zeros(3)
        self._predicted_vel = None


# ======================================================================
# High-level runner: fly through waypoints with MPC
# ======================================================================

def run_mpc_waypoints(
    cloud_path,
    waypoints,
    waypoint_threshold=0.3,
    hover_duration=3.0,
    mpc_horizon=20,
    mpc_samples=200,
    mpc_elite=40,
    mpc_iterations=3,
    **sim_kwargs,
):
    """Fly the Skydio X2 through a list of waypoints using MPC.

    Parameters
    ----------
    cloud_path          : str          path to .ply / .obj / .stl
    waypoints           : list[(3,)]   target positions [[x,y,z], ...]
    waypoint_threshold  : float        metres — waypoint counts as reached
    hover_duration      : float        seconds to hover after last waypoint
    mpc_horizon/samples/elite/iterations : CEM tuning knobs
    **sim_kwargs        : forwarded to create_skydio_x2_simulation
    """
    env = create_skydio_x2_simulation(cloud_path, **sim_kwargs)
    model = env["model"]
    data = env["data"]
    renderer = env["renderer"]
    fpv_cam_id = env["fpv_cam_id"]
    hover_thrust = env["hover_thrust"]
    steps_per_frame = env["steps_per_frame"]
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

    print(f"\nMPC waypoint flight: {len(waypoints)} waypoints")
    for i, wp in enumerate(waypoints):
        print(f"  WP {i}: [{wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f}]")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = -60
        viewer.cam.elevation = -30
        viewer.cam.distance = 5.0
        viewer.cam.lookat = [0, 0, 1.5]

        start_time = time.time()
        done_time = None

        while viewer.is_running():
            step_start = time.time()
            elapsed = step_start - start_time

            # Current target waypoint
            if wp_idx < len(waypoints):
                target = waypoints[wp_idx]
            else:
                target = waypoints[-1]

            # Check if waypoint is reached
            pos = data.qpos[:3]
            dist = np.linalg.norm(pos - target)
            vel_mag = np.linalg.norm(data.qvel[:3])

            if dist < waypoint_threshold and vel_mag < 1.0 and wp_idx < len(waypoints):
                quat = data.qpos[3:7]
                vel = data.qvel[:3]
                ang_vel = data.qvel[3:6]
                # Euler angles from quaternion (roll, pitch, yaw)
                qw, qx, qy, qz = quat
                roll_ang = np.degrees(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2)))
                pitch_ang = np.degrees(np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1)))
                yaw_ang = np.degrees(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2)))

                print(f"\n{'='*60}")
                print(f"  WAYPOINT {wp_idx} REACHED at t={elapsed:.2f}s")
                print(f"{'='*60}")
                print(f"  Target:      [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")
                print(f"  Position:    [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                print(f"  Distance:    {dist:.4f} m")
                print(f"  Velocity:    [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]  (mag={vel_mag:.3f} m/s)")
                print(f"  Orientation: roll={roll_ang:.1f}°  pitch={pitch_ang:.1f}°  yaw={yaw_ang:.1f}°")
                print(f"  Quaternion:  [{qw:.4f}, {qx:.4f}, {qy:.4f}, {qz:.4f}]")
                print(f"  Angular vel: [{ang_vel[0]:.3f}, {ang_vel[1]:.3f}, {ang_vel[2]:.3f}] rad/s")
                print(f"  Motors:      [{data.ctrl[0]:.3f}, {data.ctrl[1]:.3f}, {data.ctrl[2]:.3f}, {data.ctrl[3]:.3f}]")
                if wp_idx + 1 < len(waypoints):
                    nxt = waypoints[wp_idx + 1]
                    nxt_dist = np.linalg.norm(pos - nxt)
                    print(f"  Next WP:     [{nxt[0]:.2f}, {nxt[1]:.2f}, {nxt[2]:.2f}]  (dist={nxt_dist:.2f}m)")
                print(f"{'='*60}\n")

                wp_idx += 1
                controller.reset()  # clear stale warm-start for new target
                if wp_idx >= len(waypoints):
                    done_time = time.time()
                    print("  All waypoints reached — hovering at final position.")

            # Finish after hover period
            if done_time and (time.time() - done_time) > hover_duration:
                break

            # --- MPC solve ---
            ctrl = controller.solve(data, target)
            thrust_offset, pitch_cmd, roll_cmd, yaw_cmd = ctrl

            # Apply via existing motor mixer
            base = hover_thrust + thrust_offset
            apply_motor_mixing(data, pitch_cmd, roll_cmd, yaw_cmd, base)

            # --- Physics ---
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            # --- 3rd-person view ---
            viewer.sync()

            # --- FPV render ---
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=fpv_cam_id)
            fpv_img = renderer.render()
            fpv_bgr = cv2.cvtColor(fpv_img, cv2.COLOR_RGB2BGR)
            fpv_bgr = cv2.flip(fpv_bgr, 0)

            # HUD
            alt = data.qpos[2]
            wp_label = (
                f"WP {wp_idx}/{len(waypoints)}"
                if wp_idx < len(waypoints)
                else "DONE"
            )
            cv2.putText(
                fpv_bgr,
                f"ALT {alt:.1f}m   {wp_label}   DIST {dist:.2f}m",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                fpv_bgr,
                f"MPC: thr={thrust_offset:+.2f}  p={pitch_cmd:+.2f}  "
                f"r={roll_cmd:+.2f}  y={yaw_cmd:+.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 200, 200),
                1,
            )
            cv2.imshow("Drone FPV Camera", fpv_bgr)
            cv2.waitKey(1)

            # --- Timing ---
            frame_elapsed = time.time() - step_start
            if frame_elapsed < dt_render:
                time.sleep(dt_render - frame_elapsed)

        cv2.destroyAllWindows()
        renderer.close()
        print("MPC flight complete.")
