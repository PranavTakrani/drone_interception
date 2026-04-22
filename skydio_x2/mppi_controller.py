"""
mppi_controller.py — Model Predictive Path Integral (MPPI) controller for Skydio X2.

Replaces the CEM optimiser in mpc_controller.py with MPPI's information-theoretic
weighted update.  Everything else (dynamics model, cost function, disturbance
estimator, interface) is identical to MPCController.

Key difference from CEM:
  CEM:  keep top-K samples, refit Gaussian
  MPPI: weight ALL samples by exp(-cost/lambda), update mean with weighted sum
"""

import numpy as np
import mujoco
import cv2
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skydio_x2.skydio_x2_movement import create_skydio_x2_simulation, apply_motor_mixing


class MPPIController:
    """MPPI controller for quadrotor position tracking.

    At each control step:
      1. Sample N perturbations around the current mean control sequence.
      2. Roll out all trajectories with the simplified dynamics model.
      3. Score each trajectory with the same cost function as MPCController.
      4. Compute MPPI weights: w_i = exp(-(cost_i - min_cost) / lambda).
      5. Update mean: mean += sum(w_i * noise_i) / sum(w_i).
      6. Apply mean[0] to the drone, shift mean for next step.
    """

    def __init__(self, horizon: int = 15, n_samples: int = 400, dt: float = 0.02):
        self.horizon = horizon
        self.n_samples = n_samples
        self.dt = dt

        # --- Drone physical parameters ---
        self.mass = 1.325
        self.gravity = 9.81
        self.hover_thrust_per_motor = 3.2495625
        self.max_motor_thrust = 13.0
        self.k_pitch = 28.0
        self.k_roll = 22.5
        self.k_yaw = 1.5
        self.angular_damping = 0.5

        # --- Tunable params (set via MPPIControlConfig.apply_*) ---
        self.lam = 0.05
        self.sigma = np.array([0.5, 0.2, 0.2, 0.15])
        self.w_pos = 20.0
        self.w_vel = 2.0
        self.w_vel_running = 0.0
        self.w_tilt = 30.0
        self.w_tilt_running = 20.0
        self.w_yaw = 4.0
        self.w_ctrl_rate = 1.0
        self.w_pos_running = 8.0
        self.w_ang_vel = 3.0
        self.w_ang_vel_running = 2.0
        self.w_closing = 0.0
        self.att_max = 1.0
        self.thrust_min = -2.0
        self.thrust_max = 8.0
        self.z_min = 0.5
        self.w_floor = 20.0

        # --- Internal state ---
        self._mean = None
        self._prev_ctrl = np.zeros(4)
        self._disturbance_force = np.zeros(3)
        self._predicted_vel = None
        self._disturbance_alpha = 0.12

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------
    def get_state(self, data):
        pos = data.qpos[:3].copy()
        quat = data.qpos[3:7].copy()
        vel = data.qvel[:3].copy()
        ang_vel = data.qvel[3:6].copy()
        return pos, quat, vel, ang_vel

    # ------------------------------------------------------------------
    # Vectorised forward dynamics  (identical to MPCController)
    # ------------------------------------------------------------------
    def _predict_batch(self, pos, quat, vel, ang_vel, controls):
        N = controls.shape[0]
        p = np.tile(pos, (N, 1))
        v = np.tile(vel, (N, 1))
        q = np.tile(quat, (N, 1))
        w = np.tile(ang_vel, (N, 1))

        all_pos, all_vel, all_quat, all_ang_vel = [], [], [], []

        for t in range(self.horizon):
            ctrl = controls[:, t, :]
            thrust_off = ctrl[:, 0]

            per_motor = np.clip(
                self.hover_thrust_per_motor + thrust_off, 0, self.max_motor_thrust
            )
            total_thrust = per_motor * 4

            qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            bz_x = 2 * (qx * qz + qw * qy)
            bz_y = 2 * (qy * qz - qw * qx)
            bz_z = 1 - 2 * (qx * qx + qy * qy)

            t_acc = total_thrust / self.mass
            dist_acc = self._disturbance_force / self.mass
            acc = np.stack(
                [bz_x * t_acc + dist_acc[0],
                 bz_y * t_acc + dist_acc[1],
                 bz_z * t_acc - self.gravity + dist_acc[2]],
                axis=1,
            )

            v = v + acc * self.dt
            p = p + v * self.dt

            torque = np.stack(
                [-ctrl[:, 2] * self.k_roll,
                 -ctrl[:, 1] * self.k_pitch,
                 -ctrl[:, 3] * self.k_yaw],
                axis=1,
            )
            w = w + (torque - self.angular_damping * w) * self.dt

            dq = 0.5 * np.stack(
                [-(w[:, 0]*q[:, 1] + w[:, 1]*q[:, 2] + w[:, 2]*q[:, 3]),
                  (w[:, 0]*q[:, 0] + w[:, 2]*q[:, 2] - w[:, 1]*q[:, 3]),
                  (w[:, 1]*q[:, 0] - w[:, 2]*q[:, 1] + w[:, 0]*q[:, 3]),
                  (w[:, 2]*q[:, 0] + w[:, 1]*q[:, 1] - w[:, 0]*q[:, 2])],
                axis=1,
            )
            q = q + dq * self.dt
            q = q / np.linalg.norm(q, axis=1, keepdims=True)

            all_pos.append(p.copy())
            all_vel.append(v.copy())
            all_quat.append(q.copy())
            all_ang_vel.append(w.copy())

        return p, v, q, w, all_pos, all_vel, all_quat, all_ang_vel

    # ------------------------------------------------------------------
    # Cost function  (identical to MPCController)
    # ------------------------------------------------------------------
    def _compute_costs(self, final_pos, final_vel, final_quat, final_ang_vel,
                       all_pos, all_vel, all_quat, all_ang_vel, controls, target,
                       target_vel=None):
        pos_err = final_pos - target[np.newaxis, :]
        cost = self.w_pos * np.sum(pos_err ** 2, axis=1)

        rel_vel = (final_vel - target_vel[np.newaxis, :]) if target_vel is not None else final_vel
        cost += self.w_vel * np.sum(rel_vel ** 2, axis=1)

        # Uprightness: penalise deviation from qw=1 (upright).
        # (1 - qw) = 0 when upright, = 2 when fully inverted — unlike qx²+qy²
        # which is 0 for both upright AND inverted.
        tilt_err = (1.0 - final_quat[:, 0]) ** 2
        cost += self.w_tilt * tilt_err
        cost += self.w_yaw * final_quat[:, 3] ** 2
        cost += self.w_ang_vel * np.sum(final_ang_vel ** 2, axis=1)

        inv_h = 1.0 / self.horizon
        for t in range(self.horizon):
            err = all_pos[t] - target[np.newaxis, :]
            cost += self.w_pos_running * inv_h * np.sum(err ** 2, axis=1)

            if self.w_vel_running > 0:
                rv = (all_vel[t] - target_vel[np.newaxis, :]) if target_vel is not None else all_vel[t]
                cost += self.w_vel_running * inv_h * np.sum(rv ** 2, axis=1)

            # Reward closing velocity: penalise NOT moving toward target
            if self.w_closing > 0:
                to_target = target[np.newaxis, :] - all_pos[t]          # (N, 3)
                dist_t = np.linalg.norm(to_target, axis=1, keepdims=True) + 1e-6
                los = to_target / dist_t                                  # unit vector
                closing_vel = np.sum(all_vel[t] * los, axis=1)           # scalar per sample
                cost -= self.w_closing * inv_h * closing_vel             # reward = negative cost

            qt = all_quat[t]
            cost += self.w_tilt_running * inv_h * (1.0 - qt[:, 0]) ** 2
            cost += self.w_ang_vel_running * inv_h * np.sum(all_ang_vel[t] ** 2, axis=1)

            # Altitude floor: exponential penalty below z_min
            if self.w_floor > 0:
                violation = np.maximum(0.0, self.z_min - all_pos[t][:, 2])
                cost += self.w_floor * inv_h * (np.exp(violation * 4) - 1)

        if self.horizon > 1:
            diffs = controls[:, 1:, :] - controls[:, :-1, :]
            cost += self.w_ctrl_rate * np.sum(diffs ** 2, axis=(1, 2))

        first_diff = controls[:, 0, :] - self._prev_ctrl[np.newaxis, :]
        cost += self.w_ctrl_rate * 2.0 * np.sum(first_diff ** 2, axis=1)

        return cost

    # ------------------------------------------------------------------
    # MPPI solve
    # ------------------------------------------------------------------
    def solve(self, data, target_pos, target_vel=None):
        """Run one MPPI update and return the first-step control."""
        pos, quat, vel, ang_vel = self.get_state(data)
        target = np.asarray(target_pos, dtype=np.float64)
        tv = np.asarray(target_vel, dtype=np.float64) if target_vel is not None else None

        # Disturbance estimation (identical to MPCController)
        if self._predicted_vel is not None:
            vel_error = vel - self._predicted_vel
            force_residual = vel_error * self.mass / self.dt
            if np.linalg.norm(self._disturbance_force) < 3.0:
                self._disturbance_force += self._disturbance_alpha * force_residual

        # Warm-start mean
        if self._mean is None:
            mean = np.zeros((self.horizon, 4))
        else:
            mean = np.roll(self._mean, -1, axis=0)
            mean[-1] = mean[-2]

        # Sample perturbations: (N, H, 4)
        noise = np.random.randn(self.n_samples, self.horizon, 4) * self.sigma[np.newaxis, np.newaxis, :]
        samples = mean[np.newaxis, :, :] + noise

        # Clamp to control bounds
        samples[:, :, 0] = np.clip(samples[:, :, 0], self.thrust_min, self.thrust_max)
        samples[:, :, 1] = np.clip(samples[:, :, 1], -self.att_max, self.att_max)
        samples[:, :, 2] = np.clip(samples[:, :, 2], -self.att_max, self.att_max)
        samples[:, :, 3] = np.clip(samples[:, :, 3], -self.att_max, self.att_max)

        # Forward predict
        final_pos, final_vel, final_quat, final_ang_vel, all_pos, all_vel, all_quat, all_ang_vel = \
            self._predict_batch(pos, quat, vel, ang_vel, samples)

        # Costs
        costs = self._compute_costs(
            final_pos, final_vel, final_quat, final_ang_vel,
            all_pos, all_vel, all_quat, all_ang_vel, samples, target, target_vel=tv,
        )

        # MPPI weighted update
        beta = costs.min()
        weights = np.exp(-(costs - beta) / self.lam)
        weights /= weights.sum()                          # normalise

        # mean += weighted sum of perturbations (noise, not samples, to stay near mean)
        mean = mean + np.einsum("n,nhd->hd", weights, noise)

        # Re-clamp mean after update
        mean[:, 0] = np.clip(mean[:, 0], self.thrust_min, self.thrust_max)
        mean[:, 1:] = np.clip(mean[:, 1:], -self.att_max, self.att_max)

        self._mean = mean
        best_ctrl = mean[0]
        self._prev_ctrl = best_ctrl.copy()

        # Predict next-step velocity for disturbance estimator
        thrust_off = best_ctrl[0]
        per_motor = np.clip(self.hover_thrust_per_motor + thrust_off, 0, self.max_motor_thrust)
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
        self._mean = None
        self._prev_ctrl = np.zeros(4)
        self._disturbance_force = np.zeros(3)
        self._predicted_vel = None


# ======================================================================
# High-level runner
# ======================================================================

def run_mppi_waypoints(
    cloud_path,
    waypoints,
    waypoint_threshold=0.3,
    hover_duration=3.0,
    horizon=20,
    n_samples=400,
    config=None,
    **sim_kwargs,
):
    from skydio_x2.mpc_control_config import DEFAULT_MPPI_CONFIG
    cfg = config if config is not None else DEFAULT_MPPI_CONFIG

    env = create_skydio_x2_simulation(cloud_path, **sim_kwargs)
    model = env["model"]
    data = env["data"]
    renderer = env["renderer"]
    fpv_cam_id = env["fpv_cam_id"]
    hover_thrust = env["hover_thrust"]
    steps_per_frame = env["steps_per_frame"]
    dt_render = env["dt_render"]

    controller = MPPIController(
        horizon=horizon,
        n_samples=n_samples,
        dt=model.opt.timestep * steps_per_frame,
    )
    cfg.apply_to(controller)

    waypoints = [np.array(wp, dtype=np.float64) for wp in waypoints]
    wp_idx = 0

    print(f"\nMPPI waypoint flight: {len(waypoints)} waypoints")

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

            target = waypoints[min(wp_idx, len(waypoints) - 1)]
            pos = data.qpos[:3]
            dist = np.linalg.norm(pos - target)
            vel_mag = np.linalg.norm(data.qvel[:3])

            if dist < waypoint_threshold and vel_mag < 1.0 and wp_idx < len(waypoints):
                print(f"  WP {wp_idx} reached at t={elapsed:.2f}s  dist={dist:.3f}m")
                wp_idx += 1
                controller.reset()
                if wp_idx >= len(waypoints):
                    done_time = time.time()
                    print("  All waypoints reached — hovering.")

            if done_time and (time.time() - done_time) > hover_duration:
                break

            ctrl = controller.solve(data, target)
            thrust_offset, pitch_cmd, roll_cmd, yaw_cmd = ctrl

            base = hover_thrust + thrust_offset
            apply_motor_mixing(data, pitch_cmd, roll_cmd, yaw_cmd, base)

            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            viewer.sync()

            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=fpv_cam_id)
            fpv_img = renderer.render()
            fpv_bgr = cv2.cvtColor(fpv_img, cv2.COLOR_RGB2BGR)
            fpv_bgr = cv2.flip(fpv_bgr, 0)

            cv2.putText(fpv_bgr, f"ALT {data.qpos[2]:.1f}m  WP {wp_idx}/{len(waypoints)}  DIST {dist:.2f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(fpv_bgr, f"MPPI: thr={thrust_offset:+.2f}  p={pitch_cmd:+.2f}  r={roll_cmd:+.2f}  y={yaw_cmd:+.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)
            cv2.imshow("Drone FPV Camera", fpv_bgr)
            cv2.waitKey(1)

            frame_elapsed = time.time() - step_start
            if frame_elapsed < dt_render:
                time.sleep(dt_render - frame_elapsed)

        cv2.destroyAllWindows()
        renderer.close()
        print("MPPI flight complete.")
