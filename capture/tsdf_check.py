"""
Step 3 (stationary): Stream RealSense → TSDF fusion → Rerun.

Each RGB-D frame is integrated into a persistent voxel grid (Open3D
ScalableTSDFVolume). The fused cloud is extracted every N frames and pushed
to Rerun. Unlike pointcloud_check.py, the cloud accumulates and does not
flicker — but only because we assume the camera is stationary (extrinsic =
identity). Move the camera and points will land in the wrong place.

The next step is to swap the identity extrinsic for an estimated camera
pose from RGB-D odometry; the rest of this file stays the same.

Usage:
    python capture/tsdf_check.py

Hold the camera still (tripod, edge of a table, both hands locked). Sweep
slowly to expose the volume to different parts of the scene if you must;
expect drift if you do.
"""

from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import rerun as rr


VOXEL_LENGTH = 0.02       # 2 cm voxels — standard for room-scale TSDF
SDF_TRUNC = 0.06          # 3× voxel; truncation band for the signed distance
DEPTH_TRUNC = 3.0         # metres; depth past this is dropped before integration
EXTRACT_EVERY = 10        # frames between extract+log calls (3 Hz at 30 fps)


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

    print(f"Voxel: {VOXEL_LENGTH*100:.0f} cm   range: ≤{DEPTH_TRUNC} m   extract: every {EXTRACT_EVERY} frames")

    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    holefill = rs.hole_filling_filter()
    align = rs.align(rs.stream.color)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_LENGTH,
        sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    extrinsic = np.eye(4)  # camera == world (stationary assumption)

    rr.init("saurons-eye-tsdf", spawn=True)
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

    for _ in range(30 if not usb2 else 6):
        pipeline.try_wait_for_frames(timeout_ms=10000)
    print("Streaming. Hold camera still. Ctrl-C to quit.")

    frame_idx = 0
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

            color_rgb = cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_BGR2RGB)
            depth_raw = np.asanyarray(depth.get_data())

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_rgb),
                o3d.geometry.Image(depth_raw),
                depth_scale=1.0 / depth_scale,
                depth_trunc=DEPTH_TRUNC,
                convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, o3d_intr, extrinsic)
            frame_idx += 1

            if frame_idx % EXTRACT_EVERY == 0:
                pcd = volume.extract_point_cloud()
                xyz = np.asarray(pcd.points, dtype=np.float32)
                rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)
                rr.set_time("frame", sequence=frame_idx)
                rr.log("world/camera/image", rr.Image(color_rgb))
                rr.log("world/cloud", rr.Points3D(positions=xyz, colors=rgb))
                print(f"  frame {frame_idx}: {len(xyz):,} points")
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
