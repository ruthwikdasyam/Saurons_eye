"""
Step 1: Stream RealSense color + depth → Open3D point cloud → Rerun.

No SLAM, no fusion — each frame produces a fresh point cloud in the camera
frame. The cloud will "flicker" / replace itself as the camera moves; that's
expected. This script validates the intrinsics, depth scale, and Rerun
plumbing before we layer odometry + TSDF on top.

Usage:
    python capture/pointcloud_check.py

The Rerun viewer opens automatically in a separate window. Ctrl-C in the
terminal to quit.
"""

from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import rerun as rr


def main() -> None:
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        raise SystemExit(
            "No RealSense device found. Run capture/realsense_check.py first to debug."
        )

    dev = devices[0]
    usb = (
        dev.get_info(rs.camera_info.usb_type_descriptor)
        if dev.supports(rs.camera_info.usb_type_descriptor)
        else "?"
    )
    usb2 = usb.startswith("2.")
    fps = 6 if usb2 else 30
    if usb2:
        print(
            "WARNING: device is on USB 2.x. Falling back to 6 fps.\n"
            "         Use a USB 3.x port + cable for full 30 fps."
        )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
    profile = pipeline.start(config)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy,
    )

    print(
        f"Intrinsics: {intr.width}x{intr.height}  "
        f"fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}"
    )
    print(f"Depth scale: {depth_scale} m/unit (Open3D depth_scale = {1.0 / depth_scale:.0f})")

    rr.init("saurons-eye-step1", spawn=True)
    # RealSense colour frame is OpenCV-style: +X right, +Y down, +Z forward.
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

    # Depth post-processing chain. Order matters: filter on raw depth first,
    # then align to colour. Aligning before filtering injects small holes that
    # the spatial/temporal filters then smear into artifacts.
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    holefill = rs.hole_filling_filter()
    align = rs.align(rs.stream.color)

    warmup = 6 if usb2 else 30
    print(f"Warming up ({warmup} frames @ {fps} fps)...", end=" ", flush=True)
    for _ in range(warmup):
        pipeline.try_wait_for_frames(timeout_ms=10000)
    print("ready. Streaming. Ctrl-C to quit.")

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

            color_bgr = np.asanyarray(color.get_data())
            depth_raw = np.asanyarray(depth.get_data())
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_rgb),
                o3d.geometry.Image(depth_raw),
                depth_scale=1.0 / depth_scale,
                depth_trunc=5.0,
                convert_rgb_to_intensity=False,
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intr)

            xyz = np.asarray(pcd.points, dtype=np.float32)
            rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)

            rr.set_time("frame", sequence=color.get_frame_number())
            rr.log("world/camera/image", rr.Image(color_rgb))
            rr.log(
                "world/camera/depth",
                rr.DepthImage(depth_raw, meter=1.0 / depth_scale),
            )
            rr.log("world/cloud", rr.Points3D(positions=xyz, colors=rgb))
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
