"""
RTAB-Map pose source — subscribes to /rtabmap/odom (nav_msgs/Odometry)
in a background thread, buffers recent poses, returns interpolated T_WC
at any timestamp.

Implements the same `PoseSource` Protocol as `RgbdOdometryPose` /
`IdentityPose` / `MavlinkPose`. The mapping pipeline (TSDF / occupancy)
doesn't know the difference — it just calls `update(rgbd, t)` and gets
back a 4x4 pose.

Time semantics: caller passes `t` in seconds (UNIX epoch). We interpolate
between the two buffered poses bracketing `t`. If `t` is past the latest
buffered pose, we return the latest; if before the earliest, we return
the earliest. Both edge cases mean the camera frame's clock has drifted
relative to the RTAB-Map publisher — usually a sign of timing pathology.

Pure functions and `PoseBuffer` are unit-tested in tests/test_pose_rtabmap.py.
The `RtabmapPose` class itself (rclpy threading) is integration territory.
"""

from __future__ import annotations

import bisect
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import open3d as o3d  # noqa: F401  -- only for type hints


def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion (x, y, z, w) → 3x3 rotation matrix."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    s = 2.0 / n if n > 0 else 0.0
    return np.array([
        [1 - s * (qy*qy + qz*qz), s * (qx*qy - qz*qw),     s * (qx*qz + qy*qw)],
        [s * (qx*qy + qz*qw),     1 - s * (qx*qx + qz*qz), s * (qy*qz - qx*qw)],
        [s * (qx*qz - qy*qw),     s * (qy*qz + qx*qw),     1 - s * (qx*qx + qy*qy)],
    ])


def T_WC_from_pose_msg_components(
    position: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],   # (x, y, z, w)
) -> np.ndarray:
    """Build a 4x4 T from ROS pose.position + pose.orientation components."""
    T = np.eye(4)
    T[:3, :3] = quat_to_R(*quaternion)
    T[:3, 3] = position
    return T


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 R → unit quaternion (x, y, z, w). Numerically stable."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * t
    s0 = float(np.cos(theta)) - dot * float(np.sin(theta)) / sin_theta_0
    s1 = float(np.sin(theta)) / sin_theta_0
    return s0 * q0 + s1 * q1


def _interpolate_T(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate two 4x4 poses: lerp translation, slerp rotation."""
    p = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
    q0 = _R_to_quat(T0[:3, :3])
    q1 = _R_to_quat(T1[:3, :3])
    q = _slerp(q0, q1, alpha)
    T = np.eye(4)
    T[:3, :3] = quat_to_R(q[0], q[1], q[2], q[3])
    T[:3, 3] = p
    return T


class PoseBuffer:
    """Thread-safe time-indexed pose buffer with linear/SLERP interpolation."""

    def __init__(self, maxlen: int = 200) -> None:
        self._poses: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, t: float, T_WC: np.ndarray) -> None:
        with self._lock:
            self._poses.append((t, T_WC.copy()))

    def at(self, t: float) -> np.ndarray:
        with self._lock:
            if not self._poses:
                return np.eye(4)
            poses = list(self._poses)
        if t <= poses[0][0]:
            return poses[0][1].copy()
        if t >= poses[-1][0]:
            return poses[-1][1].copy()
        times = [p[0] for p in poses]
        i = bisect.bisect_right(times, t)
        t0, T0 = poses[i - 1]
        t1, T1 = poses[i]
        alpha = (t - t0) / (t1 - t0)
        return _interpolate_T(T0, T1, alpha)

    def __len__(self) -> int:
        with self._lock:
            return len(self._poses)


class RtabmapPose:
    """Subscribe to /rtabmap/odom on a background ROS thread.

    Implements the `PoseSource` Protocol — `update(rgbd, t)` returns
    interpolated T_WC at the requested timestamp. The rgbd argument is
    ignored (RTAB-Map computes pose externally from the camera stream).

    Args:
        topic: Odometry topic to subscribe to. Default `/rtabmap/odom`.
        node_name: Node name for the rclpy node we spin up.
        wait_for_first_pose: Seconds to block in __init__ waiting for at
            least one pose to arrive. 0 to skip. Helps avoid returning
            identity for the first few frames.
    """

    def __init__(
        self,
        topic: str = "/rtabmap/odom",
        node_name: str = "saurons_eye_rtabmap_pose",
        wait_for_first_pose: float = 5.0,
    ) -> None:
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node

        if not rclpy.ok():
            rclpy.init()

        self._buffer = PoseBuffer()
        self._node = Node(node_name)
        self._sub = self._node.create_subscription(Odometry, topic, self._on_odom, 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

        if wait_for_first_pose > 0:
            t0 = time.time()
            while len(self._buffer) == 0 and time.time() - t0 < wait_for_first_pose:
                time.sleep(0.05)
            if len(self._buffer) == 0:
                print(
                    f"WARNING: no poses received from {topic} after {wait_for_first_pose}s. "
                    "Is rtabmap_ros publishing? Check `ros2 topic hz {topic}`."
                )

    def _on_odom(self, msg) -> None:
        t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        pos = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        )
        quat = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        self._buffer.add(t, T_WC_from_pose_msg_components(pos, quat))

    def _spin(self) -> None:
        # rclpy.executors.ExternalShutdownException is raised on Ctrl+C; treat
        # it as a normal stop signal so we don't print a stack trace.
        from rclpy.executors import ExternalShutdownException

        while not self._stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                break

    def update(
        self,
        rgbd: "o3d.geometry.RGBDImage | None",  # noqa: F821 — string annotation, lazy import
        t: float,
    ) -> np.ndarray:
        return self._buffer.at(t)

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._node.destroy_node()


__all__ = [
    "PoseBuffer",
    "RtabmapPose",
    "T_WC_from_pose_msg_components",
    "quat_to_R",
]
