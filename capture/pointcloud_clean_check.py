"""
Per-frame point cloud cleaning + coarse voxel display in Rerun.

Stationary camera, no fusion. Each frame:
  1. Capture aligned RGB + depth.
  2. Apply RealSense post-processing filters (spatial + temporal + holefill).
  3. Build the per-frame Open3D point cloud.
  4. Voxel-downsample to a chunky resolution (collapses small detail).
  5. Statistical outlier removal (drops flying-pixel speckles at edges).
  6. Log to Rerun as plain spheres, no colour.

Each frame replaces the last. The point of this script is to dial in the
*cleaning recipe* — voxel size, outlier thresholds — so the cloud reads
as "walls + chairs + people-shaped blobs" without small-object noise.
Fusion (TSDF) and pose tracking go on top of this once the recipe is good.

Usage:
    python capture/pointcloud_clean_check.py

Knobs at the top of the file. Try VOXEL_SIZE = 0.05–0.15 for different
levels of chunkiness.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import rerun as rr


# Cleaning knobs — tune these in place to find the right look.
VOXEL_SIZE = 0.05            # m. 0.05 = 5 cm (person-shape readable). Try 0.10 for chunkier.
SPHERE_RADIUS = 0.04         # m. Slightly < voxel size so spheres look distinct, not merged.
DEPTH_TRUNC = 4.0            # m. Drop points beyond this; noise grows quadratically.
DEPTH_NEAR = 0.3             # m. RealSense D435i is unreliable below ~30 cm.
SOR_NEIGHBORS = 20           # Statistical outlier removal: required neighbors.
SOR_STD_RATIO = 2.0          # Higher = keep more (lenient). Lower = drop more (aggressive).
LOG_EVERY = 3                # Log every Nth frame (10 Hz at 30 fps capture).


def main() -> None:
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        raise SystemExit("No RealSense device found. Run capture/realsense_check.py first.")

    dev = devices[0]
    usb = (
        dev.get_info(rs.camera_info.usb_type_descriptor)
        if dev.supports(rs.camera_info.usb_type_descriptor)
        else "?"
    )
    usb2 = usb.startswith("2.")
    fps = 6 if usb2 else 30

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
    profile = pipeline.start(config)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy,
    )

    print(
        f"Voxel: {VOXEL_SIZE*100:.0f} cm   range: {DEPTH_NEAR}-{DEPTH_TRUNC} m   "
        f"SOR: nbrs={SOR_NEIGHBORS} std={SOR_STD_RATIO}   log: every {LOG_EVERY} frames"
    )

    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    holefill = rs.hole_filling_filter()
    align = rs.align(rs.stream.color)

    rr.init("saurons-eye-clean", spawn=True)
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        "world/camera",
        rr.Pinhole(
            resolution=[intr.width, intr.height],
            focal_length=[intr.fx, intr.fy],
            principal_point=[intr.ppx, intr.ppy],
        ),
        static=True,
    )
    # Wipe stale components (any colour from previous runs/code).
    rr.log("world/voxels", rr.Clear(recursive=False))

    for _ in range(30 if not usb2 else 6):
        pipeline.try_wait_for_frames(timeout_ms=10000)
    print("Streaming. Hold camera still. Ctrl-C to quit.")

    frame_idx = 0
    last_log_t = time.perf_counter()
    try:
        while True:
            ok, frames = pipeline.try_wait_for_frames(timeout_ms=2000)
            if not ok:
                continue
            frames = spatial.process(frames).as_frameset()
            frames = temporal.process(frames).as_frameset()
            frames = holefill.process(frames).as_frameset()
            frames = align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            frame_idx += 1
            if frame_idx % LOG_EVERY != 0:
                continue

            color_rgb = cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_BGR2RGB)
            depth_raw = np.asanyarray(depth.get_data())

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_rgb),
                o3d.geometry.Image(depth_raw),
                depth_scale=1.0 / depth_scale,
                depth_trunc=DEPTH_TRUNC,
                convert_rgb_to_intensity=False,
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intr)

            # Drop near-camera garbage (D435i noisy below ~30 cm).
            pts = np.asarray(pcd.points)
            if pts.size:
                z = pts[:, 2]
                pcd = pcd.select_by_index(np.where(z >= DEPTH_NEAR)[0])

            n_raw = len(pcd.points)
            pcd = pcd.voxel_down_sample(VOXEL_SIZE)
            n_voxels = len(pcd.points)
            if n_voxels >= SOR_NEIGHBORS:
                pcd, _ = pcd.remove_statistical_outlier(
                    nb_neighbors=SOR_NEIGHBORS,
                    std_ratio=SOR_STD_RATIO,
                )
            n_clean = len(pcd.points)

            xyz = np.asarray(pcd.points, dtype=np.float32)

            now = time.perf_counter()
            rate_hz = 1.0 / max(now - last_log_t, 1e-6)
            last_log_t = now

            rr.set_time("frame", sequence=frame_idx)
            rr.log("world/voxels", rr.Points3D(positions=xyz, radii=SPHERE_RADIUS))
            print(
                f"  frame {frame_idx:>4}: raw {n_raw:>6,}  →  voxels {n_voxels:>5,}  "
                f"→  clean {n_clean:>5,}   ~{rate_hz:.1f} Hz"
            )
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
