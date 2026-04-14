"""
intercept_config.py -- Central configuration for intercept MPC weights.

Two-phase intercept strategy:
  CRUISE: fly flat and stable toward the lead point, closing distance
  STRIKE: within strike_range, tilt hard and ram the target

All intercept code imports from here so weights stay in sync.
"""

from dataclasses import dataclass, asdict
import json


@dataclass
class InterceptMPCConfig:
    # --- MPC structural parameters ---
    horizon: int = 20
    n_samples: int = 400
    n_elite: int = 60
    n_iterations: int = 5

    # --- Phase transition ---
    strike_range: float = 3.0  # metres — switch from cruise to strike

    # --- CRUISE phase: low thrust, stay flat ---
    cruise_w_pos: float = 12.0
    cruise_w_vel: float = 4.0
    cruise_w_vel_running: float = 1.0
    cruise_w_pos_running: float = 3.0
    cruise_w_tilt: float = 15.0       # heavy penalty — stay level
    cruise_w_tilt_running: float = 12.0  # stay level throughout horizon
    cruise_w_yaw: float = 4.0
    cruise_w_ang_vel: float = 3.0
    cruise_w_ang_vel_running: float = 2.0
    cruise_w_ctrl_rate: float = 1.0
    cruise_att_max: float = 0.35   # ~20° — gentle banking only
    cruise_thrust_max: float = 2.0

    # --- STRIKE phase: ram the target ---
    strike_w_pos: float = 40.0     # maximum pull to target
    strike_w_vel: float = 0.5      # barely brake — full commit
    strike_w_vel_running: float = 0.0
    strike_w_pos_running: float = 15.0
    strike_w_tilt: float = 2.0     # allow extreme tilt
    strike_w_tilt_running: float = 1.0
    strike_w_yaw: float = 0.2
    strike_w_ang_vel: float = 1.0
    strike_w_ang_vel_running: float = 0.5
    strike_w_ctrl_rate: float = 0.2  # very snappy
    strike_att_max: float = 1.2    # ~69° — hard banking allowed
    strike_thrust_max: float = 10.0  # full power

    def apply_cruise(self, controller):
        """Apply cruise-phase weights."""
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
        """Apply strike-phase weights."""
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
        """Apply cruise weights (default phase at start)."""
        self.apply_cruise(controller)

    def to_json(self, path):
        """Serialize config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        """Load config from JSON file."""
        with open(path) as f:
            return cls(**json.load(f))


# Default config instance
DEFAULT_INTERCEPT_CONFIG = InterceptMPCConfig()
