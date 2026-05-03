"""
RealSense capture via ROS — context-managed wrapper that subscribes to
the realsense2_camera node's published topics and yields synchronised
(color, depth, t) frames in the same shape as ``RealSenseCapture``.

Drop-in replacement for the librealsense-direct ``RealSenseCapture`` so
``capture.run`` can be parameterised on which side fetches frames. The
mapping pipeline downstream doesn't care.

Pure helpers (``image_msg_to_np``, ``camera_info_to_intrinsics``) are
unit-tested. The ROS spinning + synchroniser path is covered by a smoke
script.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Iterator

import numpy as np
import open3d as o3d


@dataclass
class Frame:
    color_rgb: np.ndarray   # H x W x 3, uint8
    depth_raw: np.ndarray   # H x W, uint16 (multiply by depth_scale for metres)
    t: float                # capture timestamp (seconds since epoch)


def image_msg_to_np(msg: Any) -> np.ndarray:
    """Convert a sensor_msgs/Image into a numpy array.

    Supports the encodings we actually use:
      - rgb8  → (H, W, 3) uint8
      - bgr8  → (H, W, 3) uint8 (channels swapped to RGB)
      - 16UC1 / mono16 → (H, W) uint16   (depth)
      - 32FC1 → (H, W) float32           (depth in metres)
    """
    enc = msg.encoding
    if enc == "rgb8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if enc == "bgr8":
        bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return np.ascontiguousarray(bgr[..., ::-1])
    if enc in ("16UC1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    if enc == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    raise ValueError(f"unsupported encoding: {enc}")


def camera_info_to_intrinsics(info: Any) -> o3d.camera.PinholeCameraIntrinsic:
    """Build an Open3D PinholeCameraIntrinsic from a sensor_msgs/CameraInfo."""
    K = info.k   # ROS 2: lowercase 'k', flat list of 9 floats
    fx, fy = float(K[0]), float(K[4])
    cx, cy = float(K[2]), float(K[5])
    return o3d.camera.PinholeCameraIntrinsic(info.width, info.height, fx, fy, cx, cy)


class RealSenseRosCapture:
    """ROS-based RealSense capture with the same shape as RealSenseCapture.

    Usage:
        with RealSenseRosCapture() as cam:
            for frame in cam:
                ...   # frame.color_rgb, frame.depth_raw, frame.t

    Args:
        color_topic: sensor_msgs/Image, RGB.
        depth_topic: sensor_msgs/Image, aligned depth (16UC1, mm).
        info_topic:  sensor_msgs/CameraInfo for the colour stream.
        depth_scale: metres per raw depth unit (D435i = 0.001 = 1 mm).
        slop:        ApproximateTimeSynchronizer slop in seconds.
        node_name:   rclpy node name.
        frame_queue_size: bound on the in-process frame queue.
        first_frame_timeout: seconds to block in __enter__ for first frame.
    """

    def __init__(
        self,
        color_topic: str = "/camera/camera/color/image_raw",
        depth_topic: str = "/camera/camera/aligned_depth_to_color/image_raw",
        info_topic: str = "/camera/camera/color/camera_info",
        depth_scale: float = 0.001,
        slop: float = 0.05,
        node_name: str = "saurons_eye_realsense_ros",
        frame_queue_size: int = 5,
        first_frame_timeout: float = 10.0,
    ) -> None:
        self._color_topic = color_topic
        self._depth_topic = depth_topic
        self._info_topic = info_topic
        self._depth_scale = depth_scale
        self._slop = slop
        self._node_name = node_name
        self._first_frame_timeout = first_frame_timeout

        self._intrinsics: o3d.camera.PinholeCameraIntrinsic | None = None
        self._frames: Queue[Frame] = Queue(maxsize=frame_queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._node: Any = None
        self._executor: Any = None

    def __enter__(self) -> "RealSenseRosCapture":
        import rclpy
        from message_filters import ApproximateTimeSynchronizer, Subscriber
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, Image

        if not rclpy.ok():
            rclpy.init()
        self._node = Node(self._node_name)

        # CameraInfo: latch the first one we receive, store intrinsics.
        self._info_sub = self._node.create_subscription(
            CameraInfo, self._info_topic, self._on_info, 10,
        )

        color_sub = Subscriber(self._node, Image, self._color_topic)
        depth_sub = Subscriber(self._node, Image, self._depth_topic)
        self._sync = ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=self._slop,
        )
        self._sync.registerCallback(self._on_frames)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

        # Block until both intrinsics + first frame arrive.
        deadline = time.time() + self._first_frame_timeout
        while time.time() < deadline:
            if self._intrinsics is not None and not self._frames.empty():
                break
            time.sleep(0.05)

        if self._intrinsics is None:
            raise RuntimeError(
                f"No CameraInfo received from {self._info_topic} after "
                f"{self._first_frame_timeout}s. Is realsense2_camera running?"
            )
        if self._frames.empty():
            raise RuntimeError(
                f"No synchronised colour+depth frame in {self._first_frame_timeout}s. "
                f"Check: ros2 topic hz {self._color_topic}"
            )

        print(
            f"RealSense (ROS) ready: {self._intrinsics.width}x{self._intrinsics.height}  "
            f"(depth_scale={self._depth_scale} m/unit)"
        )
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._node is not None:
            self._node.destroy_node()

    def _on_info(self, msg: Any) -> None:
        if self._intrinsics is None:
            self._intrinsics = camera_info_to_intrinsics(msg)

    def _on_frames(self, color_msg: Any, depth_msg: Any) -> None:
        try:
            color_rgb = image_msg_to_np(color_msg)
            depth_raw = image_msg_to_np(depth_msg)
        except ValueError as e:
            print(f"frame decode error: {e}")
            return
        # Use the colour-frame timestamp as the canonical t.
        t = (
            float(color_msg.header.stamp.sec)
            + float(color_msg.header.stamp.nanosec) * 1e-9
        )
        frame = Frame(color_rgb=color_rgb, depth_raw=depth_raw, t=t)
        # Drop oldest if the queue is full — caller is the rate-limiter.
        if self._frames.full():
            try:
                self._frames.get_nowait()
            except Empty:
                pass
        self._frames.put(frame)

    def _spin(self) -> None:
        from rclpy.executors import ExternalShutdownException

        while not self._stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                break

    @property
    def intrinsics(self) -> o3d.camera.PinholeCameraIntrinsic:
        assert self._intrinsics is not None, "Use inside `with RealSenseRosCapture()`."
        return self._intrinsics

    @property
    def depth_scale(self) -> float:
        return self._depth_scale

    @property
    def fps(self) -> int:
        # ROS2 RealSense node typically publishes at 30 Hz; not authoritative here.
        return 30

    def __iter__(self) -> Iterator[Frame]:
        while not self._stop.is_set():
            try:
                yield self._frames.get(timeout=2.0)
            except Empty:
                continue
