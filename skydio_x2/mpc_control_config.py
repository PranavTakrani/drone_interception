"""
mpc_control_config.py -- Central configuration for intercept MPC/MPPI weights.
"""

from dataclasses import dataclass, asdict
import json
import numpy as np


@dataclass
class MPCControlConfig:
    # --- MPC structural parameters ---
    horizon: int = 12
    n_samples: int = 400
    n_elite: int = 60
    n_iterations: int = 5

    # --- Phase transition ---
    strike_range: float = 4.0

    # --- CRUISE phase ---
    cruise_w_pos: float = 20.0
    cruise_w_vel: float = 2.0
    cruise_w_vel_running: float = 0.0
    cruise_w_pos_running: float = 8.0
    cruise_w_tilt: float = 4.0
    cruise_w_tilt_running: float = 3.0
    cruise_w_yaw: float = 4.0
    cruise_w_ang_vel: float = 3.0
    cruise_w_ang_vel_running: float = 2.0
    cruise_w_ctrl_rate: float = 1.0
    cruise_att_max: float = 1.0
    cruise_thrust_max: float = 8.0

    # --- STRIKE phase ---
    strike_w_pos: float = 60.0
    strike_w_vel: float = 4.0
    strike_w_vel_running: float = 0.0
    strike_w_pos_running: float = 5.0
    strike_w_tilt: float = 0.5
    strike_w_tilt_running: float = 0.5
    strike_w_yaw: float = 0.1
    strike_w_ang_vel: float = 0.5
    strike_w_ang_vel_running: float = 0.2
    strike_w_ctrl_rate: float = 0.1
    strike_att_max: float = 1.1
    strike_thrust_max: float = 10.0

    def apply_cruise(self, controller):
        controller.w_pos = self.cruise_w_pos
        controller.w_vel = self.cruise_w_vel
        controller.w_vel_running = self.cruise_w_vel_running
        controller.w_tilt = self.cruise_w_tilt
        controller.w_tilt_running = self.cruise_w_tilt_running
        controller.w_yaw = self.cruise_w_yaw
        controller.w_pos_running = self.cruise_w_pos_running
        controller.w_ang_vel = self.cruise_w_ang_vel
        controller.w_ang_vel_running = self.cruise_w_ang_vel_running
        controller.w_ctrl_rate = self.cruise_w_ctrl_rate
        controller.att_max = self.cruise_att_max
        controller.thrust_max = self.cruise_thrust_max

    def apply_strike(self, controller):
        controller.w_pos = self.strike_w_pos
        controller.w_vel = self.strike_w_vel
        controller.w_vel_running = self.strike_w_vel_running
        controller.w_tilt = self.strike_w_tilt
        controller.w_tilt_running = self.strike_w_tilt_running
        controller.w_yaw = self.strike_w_yaw
        controller.w_pos_running = self.strike_w_pos_running
        controller.w_ang_vel = self.strike_w_ang_vel
        controller.w_ang_vel_running = self.strike_w_ang_vel_running
        controller.w_ctrl_rate = self.strike_w_ctrl_rate
        controller.att_max = self.strike_att_max
        controller.thrust_max = self.strike_thrust_max

    def apply_to(self, controller):
        self.apply_cruise(controller)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


DEFAULT_MPC_CONFIG = MPCControlConfig()


@dataclass
class MPPIControlConfig:
    # --- MPPI structural parameters ---
    horizon: int = 12
    n_samples: int = 400
    lam: float = 0.1
    sigma: tuple = (0.5, 0.2, 0.2, 0.15)  # noise std [thrust, pitch, roll, yaw]

    # --- Phase transition ---
    strike_range: float = 6.0

    # --- CRUISE phase ---
    cruise_w_pos: float = 20.0
    cruise_w_vel: float = 2.0
    cruise_w_vel_running: float = 0.0
    cruise_w_pos_running: float = 4.0
    cruise_w_tilt: float = 30.0
    cruise_w_tilt_running: float = 20.0
    cruise_w_yaw: float = 4.0
    cruise_w_ang_vel: float = 3.0
    cruise_w_ang_vel_running: float = 2.0
    cruise_w_ctrl_rate: float = 0.5
    cruise_w_closing: float = 0.0
    cruise_att_max: float = 1.0
    cruise_thrust_min: float = -2.0
    cruise_thrust_max: float = 8.0
    cruise_z_min: float = 0.5
    cruise_w_floor: float = 20.0

    # --- STRIKE phase ---
    strike_w_pos: float = 60.0
    strike_w_vel: float = 2.0
    strike_w_vel_running: float = 0.0
    strike_w_pos_running: float = 5.0
    strike_w_tilt: float = 0.5
    strike_w_tilt_running: float = 0.5
    strike_w_yaw: float = 0.1
    strike_w_ang_vel: float = 0.5
    strike_w_ang_vel_running: float = 0.2
    strike_w_ctrl_rate: float = 0.1
    strike_w_closing: float = 3.0
    strike_att_max: float = 1.1
    strike_thrust_min: float = -2.0
    strike_thrust_max: float = 10.0
    strike_z_min: float = 0.3
    strike_w_floor: float = 10.0

    def apply_cruise(self, controller):
        controller.w_pos = self.cruise_w_pos
        controller.w_vel = self.cruise_w_vel
        controller.w_vel_running = self.cruise_w_vel_running
        controller.w_pos_running = self.cruise_w_pos_running
        controller.w_tilt = self.cruise_w_tilt
        controller.w_tilt_running = self.cruise_w_tilt_running
        controller.w_yaw = self.cruise_w_yaw
        controller.w_ang_vel = self.cruise_w_ang_vel
        controller.w_ang_vel_running = self.cruise_w_ang_vel_running
        controller.w_ctrl_rate = self.cruise_w_ctrl_rate
        controller.w_closing = self.cruise_w_closing
        controller.att_max = self.cruise_att_max
        controller.thrust_min = self.cruise_thrust_min
        controller.thrust_max = self.cruise_thrust_max
        controller.z_min = self.cruise_z_min
        controller.w_floor = self.cruise_w_floor

    def apply_strike(self, controller):
        controller.w_pos = self.strike_w_pos
        controller.w_vel = self.strike_w_vel
        controller.w_vel_running = self.strike_w_vel_running
        controller.w_pos_running = self.strike_w_pos_running
        controller.w_tilt = self.strike_w_tilt
        controller.w_tilt_running = self.strike_w_tilt_running
        controller.w_yaw = self.strike_w_yaw
        controller.w_ang_vel = self.strike_w_ang_vel
        controller.w_ang_vel_running = self.strike_w_ang_vel_running
        controller.w_ctrl_rate = self.strike_w_ctrl_rate
        controller.w_closing = self.strike_w_closing
        controller.att_max = self.strike_att_max
        controller.thrust_min = self.strike_thrust_min
        controller.thrust_max = self.strike_thrust_max
        controller.z_min = self.strike_z_min
        controller.w_floor = self.strike_w_floor

    def apply_to(self, controller):
        controller.lam = self.lam
        controller.sigma = np.array(self.sigma)
        self.apply_cruise(controller)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
            if "sigma" in d and isinstance(d["sigma"], list):
                d["sigma"] = tuple(d["sigma"])
            return cls(**d)


DEFAULT_MPPI_CONFIG = MPPIControlConfig()
