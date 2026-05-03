"""
Capture entrypoint using RTAB-Map for L (pose) and our OccupancyVoxelGrid for M.

Architecture:

  ros2 launch realsense2_camera        ─┬─► rtabmap_ros (RGB-D SLAM)
                                         │       └─► /rtabmap/odom
                                         │
                                         ▼
                       RealSenseRosCapture ──► RtabmapPose
                                         │       └─► T_WC at frame timestamp
                                         ▼
                       project depth → world ──► OccupancyVoxelGrid ──► Rerun

Same downstream pipeline as ``capture.run``; only the camera source and
pose source are different. RTAB-Map's loop closure + tighter pose tracking
should produce a materially cleaner global map than our hand-rolled
RGB-D odometry.

Usage (in three separate terminals, sourced for ROS):

    # 1. Camera
    ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

    # 2. SLAM
    ros2 launch rtabmap_launch rtabmap.launch.py \\
        rgb_topic:=/camera/camera/color/image_raw \\
        depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \\
        camera_info_topic:=/camera/camera/color/camera_info \\
        approx_sync:=false rtabmap_viz:=true frame_id:=camera_link

    # 3. Our mapper (this script)
    python -m capture.run_rtabmap

Logs to logs/capture_run_rtabmap.log (overwrite each run) and stdout.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import open3d as o3d

from capture import viz
from capture.occupancy import OccupancyVoxelGrid
from capture.pose_rtabmap import RtabmapPose
from capture.realsense_ros import RealSenseRosCapture


# Map / capture parameters
DEPTH_TRUNC = 6.0          # m. Drop depth past this before integration.
VOXEL_SIZE = 0.05          # m. Occupancy voxel size — fine enough for walls/people.
DEVICE = "CUDA:0"          # GPU if available, falls back to CPU automatically.
CARVE_COLUMNS = False      # Off for static-scene scanning; cheaper add().

# Pipeline cadence
PROCESS_EVERY = 1          # Process every Nth ROS-published frame.
EXTRACT_EVERY = 3          # Extract + log voxels every N processed frames.

# Display
SPHERE_RADIUS = 0.03       # m. Sphere radius in Rerun.
COLORMAP = "turbo"
COLOR_AXIS = 1             # Y in RDF world.
COLOR_VOXELS = True

LOG_PATH = "logs/capture_run_rtabmap.log"


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
    return logging.getLogger("saurons-eye-rtabmap")


def _depth_to_world_pcd(
    depth_raw: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    depth_scale: float,
    T_WC: np.ndarray,
    depth_trunc: float,
) -> o3d.geometry.PointCloud:
    """Project a depth image into a world-frame point cloud in one shot."""
    return o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth_raw),
        intrinsics,
        np.linalg.inv(T_WC),
        depth_scale=1.0 / depth_scale,
        depth_trunc=depth_trunc,
    )


def main() -> None:
    log = _setup_logging()
    log.info(
        f"params  voxel={VOXEL_SIZE}m  depth_trunc={DEPTH_TRUNC}m  "
        f"process_every={PROCESS_EVERY}  extract_every={EXTRACT_EVERY}  "
        f"device={DEVICE}  carve={CARVE_COLUMNS}  colormap={COLORMAP}"
    )

    with RealSenseRosCapture() as cam:
        viz.init()
        viz.log_camera_intrinsics(cam.intrinsics)

        log.info("connecting to /rtabmap/odom (5s timeout for first pose)...")
        pose_source = RtabmapPose()
        log.info(f"pose source: RtabmapPose (buffer has {len(pose_source._buffer)} pose(s))")

        grid = OccupancyVoxelGrid(
            voxel_size=VOXEL_SIZE,
            device=DEVICE,
            carve_columns=CARVE_COLUMNS,
        )
        log.info("streaming. Ctrl-C to quit.")

        raw_idx = 0
        proc_idx = 0
        last_log = time.perf_counter()
        t_io = t_pose = t_project = t_integ = 0.0

        try:
            t_iter = time.perf_counter()
            for frame in cam:
                io_dt = time.perf_counter() - t_iter
                raw_idx += 1
                if raw_idx % PROCESS_EVERY != 0:
                    t_iter = time.perf_counter()
                    continue

                t1 = time.perf_counter()
                # rgbd not needed by RtabmapPose; pass None to keep the contract.
                T_WC = pose_source.update(None, frame.t)
                t2 = time.perf_counter()

                pcd_world = _depth_to_world_pcd(
                    frame.depth_raw, cam.intrinsics, cam.depth_scale, T_WC, DEPTH_TRUNC,
                )
                t3 = time.perf_counter()

                grid.add_pointcloud(pcd_world)
                t4 = time.perf_counter()

                t_io += io_dt
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
                        f"rot={rot_deg:+5.1f}°  ~{rate:.1f}Hz  "
                        f"buffer={len(pose_source._buffer)}"
                    )
                    log.info(
                        f"  per-frame ms  io={t_io/n*1000:5.1f}  "
                        f"pose={t_pose/n*1000:5.1f}  project={t_project/n*1000:5.1f}  "
                        f"add={t_integ/n*1000:5.1f}   "
                        f"per-cycle ms  extract={(t6-t5)*1000:.0f}  "
                        f"colors={(t7-t6)*1000:.0f}  rerun={(t8-t7)*1000:.0f}"
                    )
                    t_io = t_pose = t_project = t_integ = 0.0

                t_iter = time.perf_counter()
        except KeyboardInterrupt:
            log.info("interrupted by user")
        finally:
            grid.dispose()
            pose_source.shutdown()

        log.info(
            f"final  raw frames: {raw_idx}   processed: {proc_idx}   "
            f"log: {LOG_PATH}"
        )


if __name__ == "__main__":
    main()
