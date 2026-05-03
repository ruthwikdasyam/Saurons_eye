"""
Main capture-side entrypoint: live RGB-D mapping into a global voxel grid.

Wires together a RealSense camera, a swappable pose source, an occupancy
voxel grid (GPU when available), and Rerun visualisation.

Pose source is selected at the top of `main()`:
  - `RgbdOdometryPose(intr)` — V1 default. Estimate pose from RGB-D frames.
  - `IdentityPose()`         — Stationary debugging. Camera == world.
  - `MavlinkPose()`          — Post-v1. Drone autopilot pose over MAVLink.

When the real drone arrives, only the line `pose_source = …` changes.
Mapping, viz, wire format, headset all stay the same.

Voxels are coloured by class_id + Rerun AnnotationContext (1 byte/voxel
instead of 3, GPU-side LUT lookup; pattern from dimos).
Column carving in the occupancy grid evicts stale voxels when a column
is re-observed — solves "person walks past, leaves a ghost" cleanly.

Usage:
    python -m capture.run
    IDENTITY_POSE=1 python -m capture.run     # bypass odometry — measures
                                                pure mapping cost (camera
                                                must be held still).

Logs to logs/capture_run.log (overwrite each run) and stdout.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import open3d as o3d

from capture import viz
from capture.occupancy import OccupancyVoxelGrid
from capture.pose import IdentityPose, RgbdOdometryPose
from capture.realsense import RealSenseCapture, build_rgbd


# Map / capture parameters
DEPTH_TRUNC = 6.0          # m. Drop depth past this before integration.
VOXEL_SIZE = 0.05          # m. Occupancy voxel size — fine enough for walls/people.
DEVICE = "CUDA:0"          # GPU if available, falls back to CPU automatically.
CARVE_COLUMNS = False      # Off for global-map scanning (no moving objects to ghost out).

# Pipeline cadence
PROCESS_EVERY = 2          # Process 1 of every N camera frames (halves CPU).
EXTRACT_EVERY = 3          # Extract + log voxels every N processed frames.
                            # OccupancyVoxelGrid extract is ~ms now — bump down freely.
                            # 3 → ~1 Hz visual updates at our 3 Hz processing rate.

# Display
SPHERE_RADIUS = 0.03       # m. Sphere radius in Rerun.
COLORMAP = "turbo"         # matplotlib colormap name (turbo, viridis, plasma, …).
COLOR_AXIS = 1             # Axis index used for height colouring (Y in RDF world).
COLOR_VOXELS = True        # Set False to send uniform spheres (saves a few ms).

LOG_PATH = "logs/capture_run.log"


def _setup_logging() -> logging.Logger:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("saurons-eye")


def _depth_to_world_pcd(
    depth_raw: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    depth_scale: float,
    T_WC: np.ndarray,
    depth_trunc: float,
) -> o3d.geometry.PointCloud:
    """Project a depth image into a world-frame point cloud in one shot.

    Skips colour processing — occupancy mapping doesn't need it.
    extrinsic for create_from_depth_image is T_CW = inv(T_WC).
    """
    return o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth_raw),
        intrinsics,
        np.linalg.inv(T_WC),
        depth_scale=1.0 / depth_scale,
        depth_trunc=depth_trunc,
    )


def main() -> None:
    log = _setup_logging()

    use_identity = os.getenv("IDENTITY_POSE", "0") == "1"

    log.info(
        f"params  voxel={VOXEL_SIZE}m  depth_trunc={DEPTH_TRUNC}m  "
        f"process_every={PROCESS_EVERY}  extract_every={EXTRACT_EVERY}  "
        f"device={DEVICE}  carve={CARVE_COLUMNS}  colormap={COLORMAP}"
    )

    with RealSenseCapture() as cam:
        viz.init()
        viz.log_camera_intrinsics(cam.intrinsics)

        # === Pose source selection ===
        if use_identity:
            pose_source = IdentityPose()
            log.info("pose source: IdentityPose (camera assumed stationary)")
        else:
            pose_source = RgbdOdometryPose(cam.intrinsics)
            log.info("pose source: RgbdOdometryPose (Open3D RGB-D dense alignment)")

        grid = OccupancyVoxelGrid(
            voxel_size=VOXEL_SIZE,
            device=DEVICE,
            carve_columns=CARVE_COLUMNS,
        )
        log.info("streaming. Ctrl-C to quit.")

        raw_idx = 0
        proc_idx = 0
        last_log = time.perf_counter()
        # Per-cycle stage accumulators (reset on each extract).
        t_io = t_build = t_pose = t_project = t_integ = 0.0

        try:
            t_iter = time.perf_counter()
            for frame in cam:
                io_dt = time.perf_counter() - t_iter
                raw_idx += 1
                if raw_idx % PROCESS_EVERY != 0:
                    t_iter = time.perf_counter()
                    continue

                t0 = time.perf_counter()
                # RGBD with intensity for odometry only — occupancy grid uses depth directly.
                rgbd_odo = build_rgbd(
                    frame.color_rgb, frame.depth_raw, cam.depth_scale,
                    depth_trunc=DEPTH_TRUNC, intensity=True,
                )
                t1 = time.perf_counter()

                T_WC = pose_source.update(rgbd_odo, frame.t)
                t2 = time.perf_counter()

                pcd_world = _depth_to_world_pcd(
                    frame.depth_raw, cam.intrinsics, cam.depth_scale, T_WC, DEPTH_TRUNC,
                )
                t3 = time.perf_counter()

                grid.add_pointcloud(pcd_world)
                t4 = time.perf_counter()

                t_io += io_dt
                t_build += (t1 - t0)
                t_pose += (t2 - t1)
                t_project += (t3 - t2)
                t_integ += (t4 - t3)
                proc_idx += 1

                if proc_idx % EXTRACT_EVERY == 0:
                    t5 = time.perf_counter()
                    xyz = grid.get_voxel_centers()
                    t6 = time.perf_counter()
                    colors = (
                        viz.axis_colors(xyz, axis=COLOR_AXIS, colormap=COLORMAP)
                        if COLOR_VOXELS else None
                    )
                    t7 = time.perf_counter()

                    viz.log_pose(T_WC, proc_idx)
                    viz.log_voxels(xyz, proc_idx, radius=SPHERE_RADIUS, colors=colors)
                    t8 = time.perf_counter()

                    now = time.perf_counter()
                    rate = EXTRACT_EVERY / max(now - last_log, 1e-6)
                    last_log = now

                    n = EXTRACT_EVERY
                    pos = T_WC[:3, 3]
                    R = T_WC[:3, :3]
                    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
                    log.info(
                        f"proc {proc_idx:>4}  vox {len(xyz):>6,}  "
                        f"pos=[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}]m  "
                        f"rot={rot_deg:+5.1f}°  ~{rate:.1f}Hz"
                    )
                    log.info(
                        f"  per-frame ms  io={t_io/n*1000:5.1f}  build={t_build/n*1000:5.1f}  "
                        f"pose={t_pose/n*1000:5.1f}  project={t_project/n*1000:5.1f}  "
                        f"add={t_integ/n*1000:5.1f}   "
                        f"per-cycle ms  extract={(t6-t5)*1000:.0f}  "
                        f"colors={(t7-t6)*1000:.0f}  rerun={(t8-t7)*1000:.0f}"
                    )
                    t_io = t_build = t_pose = t_project = t_integ = 0.0

                t_iter = time.perf_counter()
        except KeyboardInterrupt:
            log.info("interrupted by user")
        finally:
            grid.dispose()

        if isinstance(pose_source, RgbdOdometryPose) and pose_source.failures:
            log.info(f"odometry failures over run: {pose_source.failures}")
        log.info(
            f"final  raw frames: {raw_idx}   processed: {proc_idx}   "
            f"log: {LOG_PATH}"
        )


if __name__ == "__main__":
    main()
