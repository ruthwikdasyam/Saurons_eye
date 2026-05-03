"""
RealSense D435i wrapper. Context-managed pipeline + post-processing filter
chain + frame iterator. Yields (color_rgb, depth_raw, t) for downstream use.

The caller (mapper / pose source) decides what to do with the frames.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs


@dataclass
class Frame:
    color_rgb: np.ndarray   # H x W x 3, uint8
    depth_raw: np.ndarray   # H x W, uint16 (multiply by depth_scale for metres)
    t: float                # capture timestamp (server clock, seconds since epoch)


class RealSenseCapture:
    """Open the device on __enter__, yield filtered frames on iteration.

    Usage:
        with RealSenseCapture() as cam:
            for frame in cam:
                ...  # frame.color_rgb, frame.depth_raw, frame.t
    """

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._pipeline = rs.pipeline()
        self._profile: rs.pipeline_profile | None = None
        self._depth_scale: float | None = None
        self._intrinsics: o3d.camera.PinholeCameraIntrinsic | None = None
        self._spatial = rs.spatial_filter()
        self._temporal = rs.temporal_filter()
        self._holefill = rs.hole_filling_filter()
        self._align = rs.align(rs.stream.color)
        self._fps = 30
        self._usb2 = False

    def __enter__(self) -> "RealSenseCapture":
        ctx = rs.context()
        devices = list(ctx.query_devices())
        if not devices:
            raise SystemExit(
                "No RealSense device found. Run capture/realsense_check.py first."
            )

        dev = devices[0]
        usb = (
            dev.get_info(rs.camera_info.usb_type_descriptor)
            if dev.supports(rs.camera_info.usb_type_descriptor)
            else "?"
        )
        self._usb2 = usb.startswith("2.")
        self._fps = 6 if self._usb2 else 30

        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self._fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self._fps)
        self._profile = self._pipeline.start(config)

        self._depth_scale = (
            self._profile.get_device().first_depth_sensor().get_depth_scale()
        )
        intr = (
            self._profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self._intrinsics = o3d.camera.PinholeCameraIntrinsic(
            intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy,
        )

        # Warm up auto-exposure.
        for _ in range(self._fps):
            self._pipeline.try_wait_for_frames(timeout_ms=10000)

        print(
            f"RealSense ready: {self.width}x{self.height} @ {self._fps} fps  "
            f"(depth_scale={self._depth_scale:.6f} m/unit)"
        )
        return self

    def __exit__(self, *_) -> None:
        self._pipeline.stop()

    @property
    def intrinsics(self) -> o3d.camera.PinholeCameraIntrinsic:
        assert self._intrinsics is not None, "Use inside a `with RealSenseCapture()` block."
        return self._intrinsics

    @property
    def depth_scale(self) -> float:
        assert self._depth_scale is not None, "Use inside a `with RealSenseCapture()` block."
        return self._depth_scale

    @property
    def fps(self) -> int:
        return self._fps

    def __iter__(self) -> Iterator[Frame]:
        while True:
            ok, frames = self._pipeline.try_wait_for_frames(timeout_ms=2000)
            if not ok:
                continue
            frames = self._spatial.process(frames).as_frameset()
            frames = self._temporal.process(frames).as_frameset()
            frames = self._holefill.process(frames).as_frameset()
            frames = self._align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue
            color_bgr = np.asanyarray(color.get_data())
            depth_raw = np.asanyarray(depth.get_data())
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            yield Frame(color_rgb=color_rgb, depth_raw=depth_raw, t=time.time())


def build_rgbd(
    color_rgb: np.ndarray,
    depth_raw: np.ndarray,
    depth_scale: float,
    *,
    depth_trunc: float,
    intensity: bool,
) -> o3d.geometry.RGBDImage:
    """Build an Open3D RGBDImage from raw arrays.

    intensity=True  → greyscale (use for RGB-D odometry).
    intensity=False → 3-channel colour (use for TSDF integration so voxels
                      keep colour info even if we don't currently render it).
    """
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_rgb),
        o3d.geometry.Image(depth_raw),
        depth_scale=1.0 / depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=intensity,
    )
