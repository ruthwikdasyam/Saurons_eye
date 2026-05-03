"""
RTAB-Map dense map subscriber.

Subscribes to /rtabmap/cloud_map (sensor_msgs/PointCloud2) — the
drift-corrected, voxelised dense map RTAB-Map publishes after each new
keyframe / loop closure.

Implements the same `get_voxel_centers()` interface as
``OccupancyVoxelGrid``, so ``run_rtabmap.py`` can swap which mapper it
uses behind a single flag. The benefit: when RTAB-Map detects a loop
closure, the dense map snaps clean retroactively — our
OccupancyVoxelGrid never gets that correction.

The PointCloud2 decoder is pure numpy and unit-tested in
tests/test_rtabmap_map.py.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np


# PointField datatype enum → numpy dtype.
# See sensor_msgs/msg/PointField for the canonical numbers.
_DATATYPE_TO_NP = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def pointcloud2_to_xyz(msg: Any) -> np.ndarray:
    """Decode sensor_msgs/PointCloud2 into an (N, 3) float32 array.

    Drops nothing — caller can filter NaN/inf as they wish.

    Raises:
        ValueError: x, y, or z field is missing.
    """
    fields = {f.name: f for f in msg.fields}
    for name in ("x", "y", "z"):
        if name not in fields:
            raise ValueError(
                f"PointCloud2 missing required field '{name}'; have {list(fields.keys())}"
            )

    n = msg.width * msg.height
    if n == 0:
        return np.empty((0, 3), dtype=np.float32)

    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)

    def extract(name: str) -> np.ndarray:
        f = fields[name]
        dtype = _DATATYPE_TO_NP[f.datatype]
        nbytes = np.dtype(dtype).itemsize
        # .copy() because raw is a view into immutable bytes
        return raw[:, f.offset:f.offset + nbytes].copy().view(dtype).flatten().astype(np.float32)

    return np.stack([extract("x"), extract("y"), extract("z")], axis=1)


class RtabmapDenseMap:
    """Subscribe to RTAB-Map's dense map topic in a background ROS thread.

    Mirrors the ``get_voxel_centers()`` interface of OccupancyVoxelGrid so
    callers can swap implementations transparently. Has no per-frame
    integration step — RTAB-Map fills the map for us, we just cache.

    Args:
        topic: PointCloud2 topic. Default `/rtabmap/cloud_map`.
        node_name: rclpy node name.
    """

    def __init__(
        self,
        topic: str = "/rtabmap/cloud_map",
        node_name: str = "saurons_eye_rtabmap_map",
    ) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import PointCloud2

        if not rclpy.ok():
            rclpy.init()

        self._lock = threading.Lock()
        self._latest_xyz: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self._cloud_count = 0

        self._node = Node(node_name)
        self._sub = self._node.create_subscription(
            PointCloud2, topic, self._on_cloud, 10,
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _on_cloud(self, msg: Any) -> None:
        try:
            xyz = pointcloud2_to_xyz(msg)
        except ValueError as e:
            print(f"PointCloud2 decode error: {e}")
            return
        # Drop NaN/inf — RTAB-Map sometimes emits invalid points.
        if xyz.shape[0]:
            valid = ~np.any(np.isnan(xyz) | np.isinf(xyz), axis=1)
            xyz = xyz[valid]
        with self._lock:
            self._latest_xyz = xyz.astype(np.float32)
            self._cloud_count += 1

    def _spin(self) -> None:
        from rclpy.executors import ExternalShutdownException

        while not self._stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                break

    def get_voxel_centers(self) -> np.ndarray:
        """Latest received dense cloud as (N, 3) float32. Empty until first message."""
        with self._lock:
            return self._latest_xyz.copy()

    def cloud_count(self) -> int:
        """Total number of PointCloud2 messages received so far."""
        with self._lock:
            return self._cloud_count

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._node.destroy_node()
