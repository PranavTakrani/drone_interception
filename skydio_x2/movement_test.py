import os
import sys

# Add repo root to sys.path so imports work when running directly from file path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skydio_x2.skydio_x2_movement import run_autonomous_benchmark


if __name__ == "__main__":
    # Run a 20-second benchmark move test through the example point cloud.
    # You can adjust duration and the cloud path as needed.
    run_autonomous_benchmark(
        cloud_path="skydio_x2/test_cube.ply",
        duration=20.0,
        distance=5.0,
        height=0.0,
        rx=0.0,
        ry=0.0,
        rz=0.0,
        scale=1.0,
        fpv_width=640,
        fpv_height=480,
        cmd_strength=0.5,
        thrust_strength=0.3,
    )
