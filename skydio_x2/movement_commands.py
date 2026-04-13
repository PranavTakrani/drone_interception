"""
movement_commands.py — Demo: fly the Skydio X2 through waypoints using MPC.

Replaces the old bang-bang command sequence with smooth Model Predictive
Control.  The drone accelerates and decelerates gradually between waypoints
instead of applying fixed-duration pitch/coast pulses.

Usage:
    python skydio_x2/movement_commands.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skydio_x2.mpc_controller import run_mpc_waypoints

# Square pattern at 1.5 m altitude, then climb and return
WAYPOINTS = [
    [2.0, 0.0, 1.5],   # forward
    [2.0, 2.0, 1.5],   # right
    [0.0, 2.0, 2.0],   # back + climb
    [0.0, 0.0, 1.5],   # home
]

if __name__ == "__main__":
    run_mpc_waypoints(
        cloud_path="skydio_x2/test_cube.ply",
        waypoints=WAYPOINTS,
        waypoint_threshold=0.3,
        hover_duration=3.0,
        # MPC tuning
        mpc_horizon=50,
        mpc_samples=200,
        mpc_elite=40,
        mpc_iterations=3,
        # Simulation / mesh placement
        distance=5.0,
        height=0.0,
        scale=1.0,
        fpv_width=640,
        fpv_height=480,
    )
