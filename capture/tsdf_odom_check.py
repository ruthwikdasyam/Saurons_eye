"""
Step 2: TSDF fusion + RGB-D odometry. Camera moves; pose is tracked.

Like tsdf_check.py but lifts the stationary-camera assumption. Open3D's
compute_rgbd_odometry estimates the camera pose between consecutive frames
via dense RGB+depth alignment; the running pose is fed to the TSDF
volume's integrate() as the camera-to-world extrinsic.

World frame = wherever the camera sat on the first frame. ArUco-based
world anchoring (see shared/frames.md) is a separate add on top.

Usage:
    python capture/tsdf_odom_check.py

Move slowly. Open3D RGB-D odometry struggles with fast rotations and
texture-poor scenes. Aim for cinematic hand-held motion: slow translation
(<10 cm/sec), gentle yaw (<20°/sec).
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import rerun as rr


VOXEL_LENGTH = 0.05      # 5 cm voxels — room-scale, coarse enough to ignore small detail
SDF_TRUNC = 0.15         # 3× voxel; truncation band for the signed distance
DEPTH_TRUNC = 6.0        # m. Indoor reach; D435i is degraded but usable out to 6-8 m
PROCESS_EVERY = 2        # process 1 of every N camera frames (halves CPU load)
EXTRACT_EVERY = 15       # extract+log cloud every N PROCESSED frames
SPHERE_RADIUS = 0.04     # m. Sphere size in Rerun display.
SOR_NEIGHBORS = 15       # Statistical outlier removal: required neighbours.
SOR_STD_RATIO = 1.8      # Tighter at far range to drop sparse fly-away voxels.


def _build_rgbd(
    color_rgb: np.ndarray,
    depth_raw: np.ndarray,
    depth_scale: float,
    *,
    intensity: bool,
) -> o3d.geometry.RGBDImage:
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_rgb),
        o3d.geometry.Image(depth_raw),
        depth_scale=1.0 / depth_scale,
        depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=intensity,
    )


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
    odo_jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    # Default is [20, 10, 5] iterations across the 3 pyramid levels — too heavy
    # for live use. Most of the alignment happens at the coarse level anyway.
    odo_option = o3d.pipelines.odometry.OdometryOption(
        iteration_number_per_pyramid_level=o3d.utility.IntVector([10, 5, 3]),
    )

    # Running pose. T_WC = pose of camera in world. Identity at startup means
    # the world frame is anchored to wherever the camera was on frame 0.
    T_WC = np.eye(4)
    prev_rgbd_odo: o3d.geometry.RGBDImage | None = None

    rr.init("saurons-eye-tsdf-odom", spawn=True)
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

    # Wipe any stale components on world/cloud (Rerun's latest-at semantics
    # would otherwise re-use colour from previous logs / earlier code).
    rr.log("world/cloud", rr.Clear(recursive=False))
    print("Streaming. Move slowly. Ctrl-C to quit.")

    frame_idx = 0
    raw_idx = 0
    odo_failures = 0
    last_log_t = time.perf_counter()
    try:
        while True:
            ok, frames = pipeline.try_wait_for_frames(timeout_ms=2000)
            if not ok:
                continue
            raw_idx += 1
            if raw_idx % PROCESS_EVERY != 0:
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

            # Two RGBD images: intensity for odometry (it wants greyscale),
            # full RGB for TSDF colour integration.
            rgbd_odo = _build_rgbd(color_rgb, depth_raw, depth_scale, intensity=True)
            rgbd_tsdf = _build_rgbd(color_rgb, depth_raw, depth_scale, intensity=False)

            if prev_rgbd_odo is not None:
                # Open3D returns T such that p_prev = T · p_curr (i.e. T_prev_curr).
                # World pose update:  T_WC(t) = T_WC(t-1) · T_prev_curr.
                ok_odo, T_prev_curr, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                    rgbd_odo, prev_rgbd_odo, o3d_intr, np.eye(4),
                    odo_jacobian, odo_option,
                )
                if ok_odo:
                    T_WC = T_WC @ T_prev_curr
                else:
                    odo_failures += 1
                    print(f"  [odo fail {odo_failures}] frame {frame_idx} — skipping integrate")
                    prev_rgbd_odo = rgbd_odo
                    frame_idx += 1
                    continue

            extrinsic = np.linalg.inv(T_WC)
            volume.integrate(rgbd_tsdf, o3d_intr, extrinsic)
            prev_rgbd_odo = rgbd_odo
            frame_idx += 1

            if frame_idx % EXTRACT_EVERY == 0:
                pcd = volume.extract_point_cloud()
                n_raw = len(pcd.points)
                # Drop isolated fly-away voxels (mostly far-range noise survivors).
                if n_raw >= SOR_NEIGHBORS:
                    pcd, _ = pcd.remove_statistical_outlier(
                        nb_neighbors=SOR_NEIGHBORS,
                        std_ratio=SOR_STD_RATIO,
                    )
                xyz = np.asarray(pcd.points, dtype=np.float32)

                now = time.perf_counter()
                rate_hz = EXTRACT_EVERY / (now - last_log_t)
                last_log_t = now

                t = T_WC[:3, 3]
                rr.set_time("frame", sequence=frame_idx)
                rr.log("world/cloud", rr.Points3D(positions=xyz, radii=SPHERE_RADIUS))
                rr.log(
                    "world/camera",
                    rr.Transform3D(translation=t, mat3x3=T_WC[:3, :3]),
                )
                print(
                    f"  frame {frame_idx:>4}: {n_raw:>5,} → {len(xyz):>5,} voxels   "
                    f"pos=[{t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}] m   "
                    f"~{rate_hz:.1f} Hz"
                )
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
