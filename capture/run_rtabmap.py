"""
Capture-side data subscriber. RTAB-Map handles SLAM + visualisation; this
script is the Python consumer that has the data on hand for downstream
work (wire protocol, agentic layer, detection, etc.).

Architecture:

  ros2 launch realsense2_camera          ─┬─► imu_filter_madgwick ──► /imu/data
                                           │
                                           ├─► rtabmap_ros (RGB-D + IMU SLAM)
                                           │       ├─► /rtabmap/odom
                                           │       ├─► /rtabmap/cloud_map  (drift-corrected dense map)
                                           │       └─► rtabmap_viz GUI (or rviz)
                                           │
                                           ▼
                       RealSenseRosCapture ──► RtabmapPose
                                           │       └─► T_WC at frame timestamp
                                           ▼
                              [reserved hooks: wire protocol, agentic layer]

Visualisation now lives in rtabmap_viz / rviz. This script just collects
data and prints stats so we have a place to plug downstream consumers.

Usage (in three terminals, ROS sourced):

    # 1. Camera (with IMU)
    ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true \\
         enable_gyro:=true enable_accel:=true unite_imu_method:=2

    # 2. IMU filter (raw IMU → filtered orientation)
    ros2 run imu_filter_madgwick imu_filter_madgwick_node --ros-args \\
         -r imu/data_raw:=/camera/camera/imu \\
         -p use_mag:=false -p publish_tf:=false -p world_frame:=enu

    # 3. SLAM (with IMU + GUI)
    ros2 launch rtabmap_launch rtabmap.launch.py \\
         rgb_topic:=/camera/camera/color/image_raw \\
         depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \\
         camera_info_topic:=/camera/camera/color/camera_info \\
         imu_topic:=/imu/data approx_sync:=true wait_imu_to_init:=true \\
         rtabmap_viz:=true frame_id:=camera_link

    # 4. Our data subscriber (this script)
    python -m capture.run_rtabmap

Logs to logs/capture_run_rtabmap.log (overwrite each run) and stdout.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import open3d as o3d

from capture.occupancy import OccupancyVoxelGrid
from capture.pose_rtabmap import RtabmapPose
from capture.realsense_ros import RealSenseRosCapture
from capture.rtabmap_map import RtabmapDenseMap


# Map source: True = subscribe to RTAB-Map's drift-corrected /rtabmap/cloud_map
# (loop closures snap the map clean retroactively); False = build our own
# OccupancyVoxelGrid by integrating per-frame depth at RTAB-Map poses.
USE_RTABMAP_DENSE_MAP = True

# Used only when USE_RTABMAP_DENSE_MAP = False
DEPTH_TRUNC = 4.0
VOXEL_SIZE = 0.05
DEVICE = "CUDA:0"
CARVE_COLUMNS = True

# Per-frame cleaning before insertion into the grid.
SOR_NEIGHBORS = 20
SOR_STD_RATIO = 2.0

# Pipeline cadence
PROCESS_EVERY = 1
STATS_EVERY = 15           # print stats every N processed frames

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
        f"params  use_rtabmap_dense_map={USE_RTABMAP_DENSE_MAP}  "
        f"voxel={VOXEL_SIZE}m  depth_trunc={DEPTH_TRUNC}m  "
        f"process_every={PROCESS_EVERY}  stats_every={STATS_EVERY}  "
        f"device={DEVICE}  carve={CARVE_COLUMNS}"
    )

    with RealSenseRosCapture() as cam:
        log.info("connecting to /rtabmap/odom (5s timeout for first pose)...")
        pose_source = RtabmapPose()
        log.info(f"pose source: RtabmapPose (buffer has {len(pose_source._buffer)} pose(s))")

        if USE_RTABMAP_DENSE_MAP:
            mapper = RtabmapDenseMap()
            log.info("map source: RtabmapDenseMap (subscribed to /rtabmap/cloud_map)")
        else:
            mapper = OccupancyVoxelGrid(
                voxel_size=VOXEL_SIZE, device=DEVICE, carve_columns=CARVE_COLUMNS,
            )
            log.info(f"map source: OccupancyVoxelGrid (voxel={VOXEL_SIZE}m)")
        log.info("streaming. Visualisation in rtabmap_viz / rviz. Ctrl-C to quit.")

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
                T_WC = pose_source.update(None, frame.t)
                t2 = time.perf_counter()

                if USE_RTABMAP_DENSE_MAP:
                    t3 = t4 = t2
                else:
                    pcd_world = _depth_to_world_pcd(
                        frame.depth_raw, cam.intrinsics, cam.depth_scale, T_WC, DEPTH_TRUNC,
                    )
                    pcd_world = pcd_world.voxel_down_sample(VOXEL_SIZE)
                    if len(pcd_world.points) >= SOR_NEIGHBORS:
                        pcd_world, _ = pcd_world.remove_statistical_outlier(
                            nb_neighbors=SOR_NEIGHBORS,
                            std_ratio=SOR_STD_RATIO,
                        )
                    t3 = time.perf_counter()
                    mapper.add_pointcloud(pcd_world)
                    t4 = time.perf_counter()

                t_io += io_dt
                t_pose += (t2 - t1)
                t_project += (t3 - t2)
                t_integ += (t4 - t3)
                proc_idx += 1

                if proc_idx % STATS_EVERY == 0:
                    xyz = mapper.get_voxel_centers()

                    now = time.perf_counter()
                    rate = STATS_EVERY / max(now - last_log, 1e-6)
                    last_log = now

                    n = STATS_EVERY
                    pos = T_WC[:3, 3]
                    R = T_WC[:3, :3]
                    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
                    log.info(
                        f"proc {proc_idx:>4}  vox {xyz.shape[0]:>6,}  "
                        f"pos=[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}]m  "
                        f"rot={rot_deg:+5.1f}°  ~{rate:.1f}Hz  "
                        f"buffer={len(pose_source._buffer)}"
                    )
                    log.info(
                        f"  per-frame ms  io={t_io/n*1000:5.1f}  "
                        f"pose={t_pose/n*1000:5.1f}  project={t_project/n*1000:5.1f}  "
                        f"add={t_integ/n*1000:5.1f}"
                    )
                    t_io = t_pose = t_project = t_integ = 0.0

                t_iter = time.perf_counter()
        except KeyboardInterrupt:
            log.info("interrupted by user")
        finally:
            if hasattr(mapper, "dispose"):
                mapper.dispose()
            elif hasattr(mapper, "shutdown"):
                mapper.shutdown()
            pose_source.shutdown()

        log.info(
            f"final  raw frames: {raw_idx}   processed: {proc_idx}   "
            f"log: {LOG_PATH}"
        )


if __name__ == "__main__":
    main()
