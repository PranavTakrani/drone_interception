"""
fly_through.py — Load a point cloud (PLY/OBJ/STL) into MuJoCo and fly a
Skydio X2 drone through it with keyboard control and a live FPV camera.

Usage:
    python scripts/fly_through.py path/to/cloud.ply
    python scripts/fly_through.py path/to/mesh.obj --distance 5 --height 1
    python scripts/fly_through.py mesh.stl --rx 90 --rz 45 --scale 0.5

Mesh Placement:
    --distance D    Move mesh D meters from origin along +X (default: 3)
    --height H      Raise mesh H meters above ground (default: 0)
    --rx / --ry / --rz   Rotate mesh around X/Y/Z axis (degrees)
    --scale S       Uniform scale factor (default: 1.0)

Controls (HOLD key to move — release to stop):
    W / S       — pitch forward / back
    A / D       — roll left / right
    Q / E       — yaw left / right
    SPACE       — throttle up   (hold to climb)
    Z           — throttle down (hold to descend)
    R           — reset to hover
    X           — quit

The MuJoCo viewer shows the 3rd-person view.
A separate OpenCV window shows the drone's front-facing camera.
"""

import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time
import os
import sys
from datetime import datetime
import threading
import queue
import argparse
import tempfile
import struct
import msvcrt


# ---------------------------------------------------------------------------
# Non-blocking key polling (Windows)
# ---------------------------------------------------------------------------
# We maintain a set of "currently held" keys.  A background thread polls
# msvcrt.kbhit() in a tight loop.  When a key arrives it is marked as
# pressed; a short timeout (~120 ms) after the LAST press of that key it
# is considered released.  This gives genuine hold-to-move behaviour.
# ---------------------------------------------------------------------------

class KeyState:
    """Tracks which keys are currently held down via polling timestamps."""

    def __init__(self, release_delay: float = 0.12):
        self.release_delay = release_delay  # seconds until a key counts as released
        self._last_press: dict[str, float] = {}
        self._lock = threading.Lock()
        self._quit = False

    def press(self, key: str):
        with self._lock:
            self._last_press[key] = time.perf_counter()

    def held_keys(self) -> set[str]:
        """Return the set of keys that are currently considered held."""
        now = time.perf_counter()
        with self._lock:
            held = set()
            expired = []
            for k, t in self._last_press.items():
                if now - t < self.release_delay:
                    held.add(k)
                else:
                    expired.append(k)
            for k in expired:
                del self._last_press[k]
            return held

    @property
    def quit_requested(self):
        return self._quit

    def request_quit(self):
        self._quit = True


def key_poll_loop(state: KeyState):
    """Background thread: polls msvcrt for keypresses."""
    print("--------------------------------------------------")
    print(" DRONE FLIGHT CONTROLS  (HOLD to move)")
    print("   W / S     — pitch forward / back")
    print("   A / D     — roll left / right")
    print("   Q / E     — yaw left / right")
    print("   SPACE     — throttle UP   (hold to climb)")
    print("   Z         — throttle DOWN (hold to descend)")
    print("   R         — reset to hover")
    print("   X         — quit")
    print("--------------------------------------------------")
    while not state.quit_requested:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            # Arrow keys: first char is '\xe0' or '\x00'
            if ch in ("\xe0", "\x00"):
                ch2 = msvcrt.getwch()
                arrow = {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT"}
                key = arrow.get(ch2, "")
            elif ch == "\x1b":   # ESC
                key = "x"
            elif ch == "\x03":   # Ctrl-C
                key = "x"
            else:
                key = ch.lower()

            if key == "x":
                state.request_quit()
                break
            if key:
                state.press(key)
        else:
            # Small sleep so we don't burn CPU while nothing is pressed
            time.sleep(0.005)


# ---------------------------------------------------------------------------
# Euler angles → MuJoCo quaternion
# ---------------------------------------------------------------------------
def euler_to_quat(rx_deg, ry_deg, rz_deg):
    """Convert extrinsic XYZ Euler angles (degrees) to wxyz quaternion."""
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)

    cx, sx = np.cos(rx / 2), np.sin(rx / 2)
    cy, sy = np.cos(ry / 2), np.sin(ry / 2)
    cz, sz = np.cos(rz / 2), np.sin(rz / 2)

    # Extrinsic X → Y → Z  =  intrinsic Z → Y → X
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return f"{w:.6f} {x:.6f} {y:.6f} {z:.6f}"


# ---------------------------------------------------------------------------
# PLY → OBJ conversion (supports both ASCII and binary PLY with vertex colors)
# ---------------------------------------------------------------------------
def _parse_ply_header(f):
    line = f.readline()
    if isinstance(line, bytes):
        line = line.decode("ascii", errors="replace")
    if "ply" not in line.lower():
        raise ValueError("Not a PLY file")

    fmt = "ascii"
    vertex_count = 0
    face_count = 0
    vertex_props = []
    face_props = []
    current_element = None

    while True:
        line = f.readline()
        if isinstance(line, bytes):
            line = line.decode("ascii", errors="replace")
        line = line.strip()

        if line.startswith("format"):
            fmt = line.split()[1]
        elif line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
            current_element = "vertex"
        elif line.startswith("element face"):
            face_count = int(line.split()[-1])
            current_element = "face"
        elif line.startswith("element"):
            current_element = "other"
        elif line.startswith("property"):
            parts = line.split()
            if current_element == "vertex":
                vertex_props.append((parts[-1], parts[1]))
            elif current_element == "face":
                face_props.append(parts)
        elif line == "end_header":
            break

    return fmt, vertex_count, face_count, vertex_props, face_props


def _ply_type_to_struct(type_name):
    mapping = {
        "char": "b", "int8": "b",
        "uchar": "B", "uint8": "B",
        "short": "h", "int16": "h",
        "ushort": "H", "uint16": "H",
        "int": "i", "int32": "i",
        "uint": "I", "uint32": "I",
        "float": "f", "float32": "f",
        "double": "d", "float64": "d",
    }
    return mapping.get(type_name, "f")


def load_ply_as_obj(ply_path):
    """
    Load a PLY file (ASCII or binary) and convert to a temporary OBJ file.
    Returns (obj_path, has_faces, vertex_colors) where vertex_colors is Nx3 float [0,1] or None.
    """
    vertices = []
    faces = []
    colors = []

    with open(ply_path, "rb") as f:
        fmt, vcount, fcount, vprops, fprops = _parse_ply_header(f)

        prop_names = [p[0] for p in vprops]
        prop_types = [p[1] for p in vprops]

        xi = prop_names.index("x") if "x" in prop_names else 0
        yi = prop_names.index("y") if "y" in prop_names else 1
        zi = prop_names.index("z") if "z" in prop_names else 2

        has_colors = "red" in prop_names and "green" in prop_names and "blue" in prop_names
        if has_colors:
            ri = prop_names.index("red")
            gi = prop_names.index("green")
            bi = prop_names.index("blue")

        if fmt == "ascii":
            for _ in range(vcount):
                line = f.readline().decode("ascii", errors="replace").strip().split()
                vals = [float(v) for v in line]
                vertices.append((vals[xi], vals[yi], vals[zi]))
                if has_colors:
                    r, g, b = vals[ri], vals[gi], vals[bi]
                    if r > 1 or g > 1 or b > 1:
                        r, g, b = r / 255.0, g / 255.0, b / 255.0
                    colors.append((r, g, b))

            for _ in range(fcount):
                line = f.readline().decode("ascii", errors="replace").strip().split()
                n = int(line[0])
                face_indices = [int(line[i + 1]) for i in range(n)]
                faces.append(face_indices)
        else:
            endian = "<" if "little" in fmt else ">"
            vert_fmt = endian + "".join(_ply_type_to_struct(t) for t in prop_types)
            vert_size = struct.calcsize(vert_fmt)

            for _ in range(vcount):
                raw = f.read(vert_size)
                vals = struct.unpack(vert_fmt, raw)
                vertices.append((vals[xi], vals[yi], vals[zi]))
                if has_colors:
                    r, g, b = float(vals[ri]), float(vals[gi]), float(vals[bi])
                    if r > 1 or g > 1 or b > 1:
                        r, g, b = r / 255.0, g / 255.0, b / 255.0
                    colors.append((r, g, b))

            for _ in range(fcount):
                count_byte = f.read(1)
                n = struct.unpack("B", count_byte)[0]
                idx_fmt = endian + "i" * n
                idx_data = f.read(struct.calcsize(idx_fmt))
                face_indices = list(struct.unpack(idx_fmt, idx_data))
                faces.append(face_indices)

    obj_path = os.path.join(tempfile.gettempdir(), "pointcloud_converted.obj")
    with open(obj_path, "w") as out:
        for v in vertices:
            out.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            out.write("f " + " ".join(str(i + 1) for i in face) + "\n")

    return obj_path, len(faces) > 0, np.array(colors) if colors else None


# ---------------------------------------------------------------------------
# Build MuJoCo XML
# ---------------------------------------------------------------------------
def build_xml(mesh_path, is_mesh, scale, drone_xml_dir,
              mesh_distance, mesh_height, mesh_quat,
              drone_start_pos):
    """
    Build a complete MJCF XML string.
    mesh_distance/mesh_height control where the static object sits.
    drone_start_pos is "x y z" for the drone's initial position.
    """

    mesh_section = ""
    cloud_body = ""

    if is_mesh:
        mesh_section = f"""
    <mesh name="pointcloud" file="{mesh_path}" scale="{scale} {scale} {scale}"/>"""
        cloud_body = f"""
      <body name="cloud" pos="{mesh_distance} 0 {mesh_height}" quat="{mesh_quat}">
        <geom type="mesh" mesh="pointcloud" contype="1" conaffinity="1"
              rgba="0.7 0.7 0.7 1"/>
      </body>"""
    else:
        cloud_body = "<!-- point cloud spheres added procedurally -->"

    xml = f"""<mujoco model="fly_through">
  <compiler autolimits="true" assetdir="{drone_xml_dir}/assets"/>

  <option timestep="0.005" density="1.225" viscosity="1.8e-5" gravity="0 0 -9.81"/>

  <size nconmax="2000" njmax="4000"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-20" elevation="-20" offwidth="640" offheight="480"/>
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
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>

    <texture type="2d" file="X2_lowpoly_texture_SpinningProps_1024.png"/>
    <material name="phong3SG" texture="X2_lowpoly_texture_SpinningProps_1024"/>
    <material name="invisible" rgba="0 0 0 0"/>
    <mesh class="x2" file="X2_lowpoly.obj"/>
    {mesh_section}
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    {cloud_body}

    <!-- Skydio X2 Drone -->
    <light name="spotlight" mode="targetbodycom" target="x2" pos="0 -1 2"/>
    <body name="x2" pos="{drone_start_pos}" childclass="x2">
      <freejoint/>
      <camera name="track" pos="-1 0 .5" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <!-- Front-facing FPV camera -->
      <camera name="fpv" pos="0.05 0 0.04" xyaxes="0 -1 0 0 0 1" fovy="90"/>
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
    <key name="hover" qpos="0 0 1.0 1 0 0 0" ctrl="3.2495625 3.2495625 3.2495625 3.2495625"/>
  </keyframe>
</mujoco>"""
    return xml


# ---------------------------------------------------------------------------
# Point-cloud sphere injection (for faceless PLY files)
# ---------------------------------------------------------------------------
def add_point_cloud_spheres(xml_base, vertices, colors, max_points=20000,
                            sphere_size=0.01, offset_x=0.0, offset_z=0.0,
                            mesh_quat_str="1 0 0 0"):
    """
    Insert small sphere geoms into the XML for each point cloud vertex.
    Subsamples if the cloud is too large.
    """
    if len(vertices) > max_points:
        idx = np.random.choice(len(vertices), max_points, replace=False)
        vertices = vertices[idx]
        if colors is not None:
            colors = colors[idx]

    spheres = []
    for i, v in enumerate(vertices):
        if colors is not None and len(colors) > i:
            r, g, b = colors[i]
            rgba = f"{r:.3f} {g:.3f} {b:.3f} 1"
        else:
            rgba = "0.6 0.6 0.6 1"
        spheres.append(
            f'      <geom type="sphere" size="{sphere_size}" '
            f'pos="{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" rgba="{rgba}" '
            f'contype="0" conaffinity="0"/>'
        )

    body_xml = (
        f'    <body name="cloud" pos="{offset_x} 0 {offset_z}" quat="{mesh_quat_str}">\n'
        + "\n".join(spheres)
        + "\n    </body>"
    )

    xml_base = xml_base.replace(
        "<!-- point cloud spheres added procedurally -->", body_xml
    )
    return xml_base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fly a Skydio X2 drone through a point cloud in MuJoCo."
    )
    parser.add_argument("cloud", help="Path to point cloud file (.ply, .obj, .stl)")

    # ── Mesh placement ──
    parser.add_argument("--distance", type=float, default=3.0,
                        help="Distance of mesh from origin along +X axis (default: 3)")
    parser.add_argument("--height", type=float, default=0.0,
                        help="Height offset of mesh above ground (default: 0)")
    parser.add_argument("--rx", type=float, default=0.0,
                        help="Rotate mesh around X axis (degrees, default: 0)")
    parser.add_argument("--ry", type=float, default=0.0,
                        help="Rotate mesh around Y axis (degrees, default: 0)")
    parser.add_argument("--rz", type=float, default=0.0,
                        help="Rotate mesh around Z axis (degrees, default: 0)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Uniform scale factor for the cloud (default: 1.0)")

    # ── Rendering ──
    parser.add_argument("--max-points", type=int, default=20000,
                        help="Max points to render for pure point clouds (default: 20000)")
    parser.add_argument("--sphere-size", type=float, default=0.01,
                        help="Radius of each point sphere (default: 0.01)")
    parser.add_argument("--fpv-width", type=int, default=640,
                        help="FPV camera width (default: 640)")
    parser.add_argument("--fpv-height", type=int, default=480,
                        help="FPV camera height (default: 480)")

    # ── Control tuning ──
    parser.add_argument("--cmd-strength", type=float, default=0.6,
                        help="Pitch/roll/yaw command strength (default: 0.6)")
    parser.add_argument("--thrust-strength", type=float, default=0.5,
                        help="Throttle change per held frame (default: 0.5)")
    args = parser.parse_args()

    cloud_path = os.path.abspath(args.cloud)
    if not os.path.exists(cloud_path):
        print(f"[ERROR] File not found: {cloud_path}")
        return

    ext = os.path.splitext(cloud_path)[1].lower()
    if ext not in {".ply", ".obj", ".stl"}:
        print(f"[ERROR] Unsupported format '{ext}'. Use .ply, .obj, or .stl")
        return

    # Drone assets directory (support both current tree and legacy menagerie layout).
    drone_xml_dir = os.path.join("skydio_x2")
    if not os.path.isdir(drone_xml_dir):
        drone_xml_dir = os.path.join("menagerie", "skydio_x2")
    if not os.path.isdir(drone_xml_dir):
        print("[ERROR] Skydio X2 not found! Expected skydio_x2 or menagerie/skydio_x2")
        return

    drone_xml_dir_abs = os.path.abspath(drone_xml_dir)

    # ── Mesh orientation quaternion ──
    mesh_quat = euler_to_quat(args.rx, args.ry, args.rz)

    # ── Drone start: at the origin, 1.5m up — safely away from the mesh ──
    drone_start_pos = "0 0 1.5"

    # --- Load / convert cloud ---
    is_mesh = True
    mesh_path = cloud_path
    vertex_colors = None
    raw_vertices = None

    if ext == ".ply":
        print(f"Loading PLY: {cloud_path}")
        obj_path, has_faces, vertex_colors = load_ply_as_obj(cloud_path)
        if has_faces:
            mesh_path = obj_path
            print(f"  Converted to OBJ with faces: {obj_path}")
        else:
            is_mesh = False
            raw_vertices = []
            with open(cloud_path, "rb") as f:
                fmt, vcount, _, vprops, _ = _parse_ply_header(f)
                prop_names = [p[0] for p in vprops]
                prop_types = [p[1] for p in vprops]
                xi = prop_names.index("x") if "x" in prop_names else 0
                yi = prop_names.index("y") if "y" in prop_names else 1
                zi = prop_names.index("z") if "z" in prop_names else 2
                has_colors = all(c in prop_names for c in ("red", "green", "blue"))
                colors_list = []
                if has_colors:
                    ri = prop_names.index("red")
                    gi = prop_names.index("green")
                    bi = prop_names.index("blue")

                if fmt == "ascii":
                    for _ in range(vcount):
                        line = f.readline().decode("ascii", errors="replace").strip().split()
                        vals = [float(v) for v in line]
                        raw_vertices.append((vals[xi], vals[yi], vals[zi]))
                        if has_colors:
                            r, g, b = vals[ri], vals[gi], vals[bi]
                            if r > 1 or g > 1 or b > 1:
                                r, g, b = r / 255, g / 255, b / 255
                            colors_list.append((r, g, b))
                else:
                    endian = "<" if "little" in fmt else ">"
                    vert_fmt = endian + "".join(_ply_type_to_struct(t) for t in prop_types)
                    vert_size = struct.calcsize(vert_fmt)
                    for _ in range(vcount):
                        raw = f.read(vert_size)
                        vals = struct.unpack(vert_fmt, raw)
                        raw_vertices.append((vals[xi], vals[yi], vals[zi]))
                        if has_colors:
                            r, g, b = float(vals[ri]), float(vals[gi]), float(vals[bi])
                            if r > 1 or g > 1 or b > 1:
                                r, g, b = r / 255, g / 255, b / 255
                            colors_list.append((r, g, b))

            raw_vertices = np.array(raw_vertices) * args.scale
            vertex_colors = np.array(colors_list) if colors_list else None
            print(f"  Point cloud (no faces): {len(raw_vertices)} points")
    elif ext in (".obj", ".stl"):
        print(f"Loading mesh: {cloud_path}")

    # --- Build XML ---
    xml = build_xml(
        mesh_path, is_mesh, args.scale, drone_xml_dir_abs,
        mesh_distance=args.distance,
        mesh_height=args.height,
        mesh_quat=mesh_quat,
        drone_start_pos=drone_start_pos,
    )

    if not is_mesh and raw_vertices is not None:
        xml = add_point_cloud_spheres(
            xml, raw_vertices, vertex_colors,
            max_points=args.max_points,
            sphere_size=args.sphere_size,
            offset_x=args.distance,
            offset_z=args.height,
            mesh_quat_str=mesh_quat,
        )

    # --- Load model ---
    print("Compiling MuJoCo model...")
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"[ERROR] Failed to compile model:\n{e}")
        return

    # Start from hover keyframe
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    # --- FPV offscreen renderer ---
    fpv_w, fpv_h = args.fpv_width, args.fpv_height
    fpv_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv")
    renderer = mujoco.Renderer(model, height=fpv_h, width=fpv_w)

    # --- Flight constants ---
    hover_thrust = 3.2495625
    cmd_strength = args.cmd_strength
    thrust_strength = args.thrust_strength

    # Mutable flight state
    thrust_offset = 0.0

    # --- Start keyboard polling thread ---
    key_state = KeyState(release_delay=0.12)
    poll_thread = threading.Thread(target=key_poll_loop, args=(key_state,), daemon=True)
    poll_thread.start()

    print(f"\nMesh placed at X={args.distance}, Z={args.height} "
          f"(rotation: rx={args.rx}, ry={args.ry}, rz={args.rz})")
    print(f"Drone starts at {drone_start_pos}")
    print("Starting simulation...\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = -60
        viewer.cam.elevation = -30
        viewer.cam.distance = 4.0
        viewer.cam.lookat = [0, 0, 1.0]

        steps_per_frame = 4
        dt_render = 1.0 / 60.0

        while viewer.is_running() and not key_state.quit_requested:
            step_start = time.time()

            # ── Read held keys ──
            held = key_state.held_keys()

            # Reset
            if "r" in held:
                if model.nkey > 0:
                    mujoco.mj_resetDataKeyframe(model, data, 0)
                thrust_offset = 0.0
                print("RESET to hover")

            # ── Build commands from held keys ──
            pitch_cmd = 0.0
            roll_cmd = 0.0
            yaw_cmd = 0.0

            if "w" in held or "UP" in held:
                pitch_cmd = cmd_strength
            if "s" in held or "DOWN" in held:
                pitch_cmd = -cmd_strength
            if "a" in held or "LEFT" in held:
                roll_cmd = -cmd_strength
            if "d" in held or "RIGHT" in held:
                roll_cmd = cmd_strength
            if "q" in held:
                yaw_cmd = -cmd_strength
            if "e" in held:
                yaw_cmd = cmd_strength

            # Throttle: accumulates while held, but decays back to 0 when released
            if " " in held:
                thrust_offset += thrust_strength * dt_render
            elif "z" in held:
                thrust_offset -= thrust_strength * dt_render
            else:
                # Slowly return throttle offset to zero (gravity does the rest)
                thrust_offset *= 0.995

            # ── Motor mixing ──
            base = hover_thrust + thrust_offset

            # thrust1: rear-left  CW  (-.14, -.18)
            # thrust2: rear-right CCW (-.14,  .18)
            # thrust3: front-right CW ( .14,  .18)
            # thrust4: front-left CCW ( .14, -.18)
            t1 = base - pitch_cmd + roll_cmd + yaw_cmd
            t2 = base - pitch_cmd - roll_cmd - yaw_cmd
            t3 = base + pitch_cmd - roll_cmd + yaw_cmd
            t4 = base + pitch_cmd + roll_cmd - yaw_cmd

            data.ctrl[0] = np.clip(t1, 0, 13)
            data.ctrl[1] = np.clip(t2, 0, 13)
            data.ctrl[2] = np.clip(t3, 0, 13)
            data.ctrl[3] = np.clip(t4, 0, 13)

            # ── Physics ──
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            # ── Render 3rd-person ──
            viewer.sync()

            # ── Render FPV camera ──
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=fpv_cam_id)
            fpv_img = renderer.render()
            fpv_bgr = cv2.cvtColor(fpv_img, cv2.COLOR_RGB2BGR)
            fpv_bgr = cv2.flip(fpv_bgr, 0)

            # HUD overlay
            alt = data.qpos[2]
            thr_pct = (base / 13.0) * 100
            cv2.putText(fpv_bgr, f"ALT {alt:.1f}m  THR {thr_pct:.0f}%",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Crosshair
            cx, cy = fpv_w // 2, fpv_h // 2
            cv2.line(fpv_bgr, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
            cv2.line(fpv_bgr, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)

            cv2.imshow("Drone FPV Camera", fpv_bgr)
            cv2.waitKey(1)

            # ── Timing ──
            elapsed = time.time() - step_start
            if elapsed < dt_render:
                time.sleep(dt_render - elapsed)

        cv2.destroyAllWindows()
        renderer.close()
        key_state.request_quit()
        print("Done.")


def create_skydio_x2_simulation(
    cloud_path,
    distance=3.0,
    height=0.0,
    rx=0.0,
    ry=0.0,
    rz=0.0,
    scale=1.0,
    max_points=20000,
    sphere_size=0.01,
    fpv_width=640,
    fpv_height=480,
    cmd_strength=0.6,
    thrust_strength=0.5,
):
    """Create a MuJoCo Skydio X2 simulation environment from a cloud file."""
    cloud_path = os.path.abspath(cloud_path)
    if not os.path.exists(cloud_path):
        raise FileNotFoundError(f"File not found: {cloud_path}")

    ext = os.path.splitext(cloud_path)[1].lower()
    if ext not in {".ply", ".obj", ".stl"}:
        raise ValueError(f"Unsupported format '{ext}'. Use .ply, .obj, or .stl")

    drone_xml_dir = os.path.join("skydio_x2")
    if not os.path.isdir(drone_xml_dir):
        drone_xml_dir = os.path.join("menagerie", "skydio_x2")
    if not os.path.isdir(drone_xml_dir):
        raise FileNotFoundError("Skydio X2 not found! Expected skydio_x2 or menagerie/skydio_x2")

    drone_xml_dir_abs = os.path.abspath(drone_xml_dir)
    mesh_quat = euler_to_quat(rx, ry, rz)
    drone_start_pos = "0 0 1.5"

    is_mesh = True
    mesh_path = cloud_path
    vertex_colors = None
    raw_vertices = None

    if ext == ".ply":
        obj_path, has_faces, vertex_colors = load_ply_as_obj(cloud_path)
        if has_faces:
            mesh_path = obj_path
        else:
            is_mesh = False
            raw_vertices = []
            with open(cloud_path, "rb") as f:
                fmt, vcount, _, vprops, _ = _parse_ply_header(f)
                prop_names = [p[0] for p in vprops]
                prop_types = [p[1] for p in vprops]
                xi = prop_names.index("x") if "x" in prop_names else 0
                yi = prop_names.index("y") if "y" in prop_names else 1
                zi = prop_names.index("z") if "z" in prop_names else 2
                has_colors = all(c in prop_names for c in ("red", "green", "blue"))
                colors_list = []
                if has_colors:
                    ri = prop_names.index("red")
                    gi = prop_names.index("green")
                    bi = prop_names.index("blue")

                if fmt == "ascii":
                    for _ in range(vcount):
                        line = f.readline().decode("ascii", errors="replace").strip().split()
                        vals = [float(v) for v in line]
                        raw_vertices.append((vals[xi], vals[yi], vals[zi]))
                        if has_colors:
                            r, g, b = vals[ri], vals[gi], vals[bi]
                            if r > 1 or g > 1 or b > 1:
                                r, g, b = r / 255, g / 255, b / 255
                            colors_list.append((r, g, b))
                else:
                    endian = "<" if "little" in fmt else ">"
                    vert_fmt = endian + "".join(_ply_type_to_struct(t) for t in prop_types)
                    vert_size = struct.calcsize(vert_fmt)
                    for _ in range(vcount):
                        raw = f.read(vert_size)
                        vals = struct.unpack(vert_fmt, raw)
                        raw_vertices.append((vals[xi], vals[yi], vals[zi]))
                        if has_colors:
                            r, g, b = float(vals[ri]), float(vals[gi]), float(vals[bi])
                            if r > 1 or g > 1 or b > 1:
                                r, g, b = r / 255, g / 255, b / 255
                            colors_list.append((r, g, b))

            raw_vertices = np.array(raw_vertices) * scale
            vertex_colors = np.array(colors_list) if colors_list else None

    elif ext in (".obj", ".stl"):
        pass

    xml = build_xml(
        mesh_path, is_mesh, scale, drone_xml_dir_abs,
        mesh_distance=distance,
        mesh_height=height,
        mesh_quat=mesh_quat,
        drone_start_pos=drone_start_pos,
    )

    if not is_mesh and raw_vertices is not None:
        xml = add_point_cloud_spheres(
            xml, raw_vertices, vertex_colors,
            max_points=max_points,
            sphere_size=sphere_size,
            offset_x=distance,
            offset_z=height,
            mesh_quat_str=mesh_quat,
        )

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    fpv_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv")
    renderer = mujoco.Renderer(model, height=fpv_height, width=fpv_width)

    return {
        "model": model,
        "data": data,
        "renderer": renderer,
        "fpv_cam_id": fpv_cam_id,
        "hover_thrust": 3.2495625,
        "cmd_strength": cmd_strength,
        "thrust_strength": thrust_strength,
        "steps_per_frame": 4,
        "dt_render": 1.0 / 60.0,
    }


def run_autonomous_benchmark(cloud_path, duration=20.0, **kwargs):
    """Run an autonomous benchmark flight sequence using Skydio X2 movement library."""
    env = create_skydio_x2_simulation(cloud_path, **kwargs)
    model = env["model"]
    data = env["data"]
    renderer = env["renderer"]
    fpv_cam_id = env["fpv_cam_id"]
    hover_thrust = env["hover_thrust"]
    cmd_strength = env["cmd_strength"]
    thrust_strength = env["thrust_strength"]
    steps_per_frame = env["steps_per_frame"]
    dt_render = env["dt_render"]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = -60
        viewer.cam.elevation = -30
        viewer.cam.distance = 4.0
        viewer.cam.lookat = [0, 0, 1.0]

        start_time = time.time()
        thrust_offset = 0.0

        while viewer.is_running() and (time.time() - start_time) < duration:
            elapsed = time.time() - start_time

            # Profile pattern: forward, right, back, left, up, down, yaw
            phase = int((elapsed / 2.0) % 6)
            pitch_cmd = 0.0
            roll_cmd = 0.0
            yaw_cmd = 0.0
            ascend = False
            descend = False

            if phase == 0:
                pitch_cmd = cmd_strength
            elif phase == 1:
                roll_cmd = cmd_strength
            elif phase == 2:
                pitch_cmd = -cmd_strength
            elif phase == 3:
                roll_cmd = -cmd_strength
            elif phase == 4:
                ascend = True
            elif phase == 5:
                descend = True

            if phase % 2 == 0:
                yaw_cmd = cmd_strength * 0.5

            if ascend:
                thrust_offset += thrust_strength * dt_render
            elif descend:
                thrust_offset -= thrust_strength * dt_render
            else:
                thrust_offset *= 0.995

            base = hover_thrust + thrust_offset
            t1 = base - pitch_cmd + roll_cmd + yaw_cmd
            t2 = base - pitch_cmd - roll_cmd - yaw_cmd
            t3 = base + pitch_cmd - roll_cmd + yaw_cmd
            t4 = base + pitch_cmd + roll_cmd - yaw_cmd

            data.ctrl[0] = np.clip(t1, 0, 13)
            data.ctrl[1] = np.clip(t2, 0, 13)
            data.ctrl[2] = np.clip(t3, 0, 13)
            data.ctrl[3] = np.clip(t4, 0, 13)

            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            viewer.sync()

            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=fpv_cam_id)
            fpv_img = renderer.render()
            fpv_bgr = cv2.cvtColor(fpv_img, cv2.COLOR_RGB2BGR)
            fpv_bgr = cv2.flip(fpv_bgr, 0)

            cv2.putText(fpv_bgr, f"TIME {elapsed:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Drone FPV Camera", fpv_bgr)
            cv2.waitKey(1)

            frame_time = time.time() - (start_time + elapsed)
            if frame_time < dt_render:
                time.sleep(dt_render - frame_time)

        cv2.destroyAllWindows()
        renderer.close()

def apply_motor_mixing(data, pitch_cmd, roll_cmd, yaw_cmd, base_thrust):
    """Apply motor mixing for Skydio X2 rotor commands."""
    t1 = base_thrust - pitch_cmd + roll_cmd + yaw_cmd
    t2 = base_thrust - pitch_cmd - roll_cmd - yaw_cmd
    t3 = base_thrust + pitch_cmd - roll_cmd + yaw_cmd
    t4 = base_thrust + pitch_cmd + roll_cmd - yaw_cmd
    data.ctrl[0] = np.clip(t1, 0, 13)
    data.ctrl[1] = np.clip(t2, 0, 13)
    data.ctrl[2] = np.clip(t3, 0, 13)
    data.ctrl[3] = np.clip(t4, 0, 13)


def create_reset_command(duration):
    """Create a command to reset the drone to its initial position and hover state."""
    return {
        "duration": duration,
        "reset": True,
        "pitch": 0.0,
        "roll": 0.0,
        "yaw": 0.0,
        "thrust": 0.0,
    }


def run_autonomous_command_sequence(
    cloud_path,
    command_sequence,
    duration=None,
    **kwargs,
):
    """Use an explicit command sequence for autonomous movement.

    command_sequence is a list of dicts, where each dict includes:
      duration: seconds to apply this command
      pitch: [-1,1], roll: [-1,1], yaw: [-1,1], thrust: [-1,1]
      reset: bool (optional) - if True, reset drone position to initial state
    thrust is relative to hover (positive=climb, negative=descend).
    """
    env = create_skydio_x2_simulation(cloud_path, **kwargs)
    model = env["model"]
    data = env["data"]
    renderer = env["renderer"]
    fpv_cam_id = env["fpv_cam_id"]
    hover_thrust = env["hover_thrust"]
    cmd_strength = env["cmd_strength"]
    thrust_strength = env["thrust_strength"]
    steps_per_frame = env["steps_per_frame"]
    dt_render = env["dt_render"]

    if duration is None:
        duration = sum(cmd.get("duration", 0.0) for cmd in command_sequence)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = -60
        viewer.cam.elevation = -30
        viewer.cam.distance = 4.0
        viewer.cam.lookat = [0, 0, 1.0]

        start_time = time.time()
        thrust_offset = 0.0
        command_index = 0
        command_start_time = time.time()
 
        while command_index < len(command_sequence) or (time.time() - start_time) < duration:
            current_time = time.time() - start_time
 
            # Advance command index based on wall-clock time spent on current command.
            # Use a while loop to skip zero-duration or already-expired commands cleanly.
            while (
                command_index < len(command_sequence)
                and (time.time() - command_start_time)
                >= command_sequence[command_index].get("duration", 0.0)
            ):
                print(f"Command {command_index} completed at t={current_time:.2f}s", flush=True)
                command_index += 1
                command_start_time = time.time()
 
            if command_index < len(command_sequence):
                cmd = command_sequence[command_index]
            else:
                cmd = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0, "thrust": 0.0}
 
            if cmd.get("reset", False):
                if model.nkey > 0:
                    mujoco.mj_resetDataKeyframe(model, data, 0)
                else:
                    mujoco.mj_resetData(model, data)
                thrust_offset = 0.0
        
            pitch_cmd = cmd.get("pitch", 0.0) * cmd_strength
            roll_cmd = cmd.get("roll", 0.0) * cmd_strength
            yaw_cmd = cmd.get("yaw", 0.0) * cmd_strength
            thrust_cmd = cmd.get("thrust", 0.0)
 
            thrust_offset += thrust_cmd * thrust_strength * dt_render
            if thrust_cmd == 0.0:
                thrust_offset *= 0.995
 
            base = hover_thrust + thrust_offset
            apply_motor_mixing(data, pitch_cmd, roll_cmd, yaw_cmd, base)
 
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
 
            if viewer.is_running():
                viewer.sync()
 
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=fpv_cam_id)
            fpv_img = renderer.render()
            fpv_bgr = cv2.cvtColor(fpv_img, cv2.COLOR_RGB2BGR)
            fpv_bgr = cv2.flip(fpv_bgr, 0)
 
            cv2.putText(
                fpv_bgr,
                f"TIME {current_time:.1f}s  CMD {command_index}/{len(command_sequence)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            cv2.imshow("Drone FPV Camera", fpv_bgr)
            cv2.waitKey(1)
 
            frame_time = time.time() - (start_time + current_time)
            if frame_time < dt_render:
                time.sleep(dt_render - frame_time)
 
        print(
            "Autonomous sequence completed. "
            "Keeping simulation running in hover mode. "
            "Close the viewer window to exit.",
            flush=True,
        )
        while viewer.is_running():
            # Hover command
            cmd = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0, "thrust": 0.0}

            pitch_cmd = cmd.get("pitch", 0.0) * cmd_strength
            roll_cmd = cmd.get("roll", 0.0) * cmd_strength
            yaw_cmd = cmd.get("yaw", 0.0) * cmd_strength
            thrust_cmd = cmd.get("thrust", 0.0)

            thrust_offset += thrust_cmd * thrust_strength * dt_render
            if thrust_cmd == 0.0:
                thrust_offset *= 0.995

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

            current_time = time.time() - start_time
            cv2.putText(fpv_bgr, f"TIME {current_time:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Drone FPV Camera", fpv_bgr)
            cv2.waitKey(1)

            time.sleep(dt_render)

        cv2.destroyAllWindows()
        renderer.close()

if __name__ == "__main__":
    main()