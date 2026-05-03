"""
Pure-occupancy voxel grid backed by Open3D's tensor HashMap.

No signed-distance, no per-voxel weights — just "is this voxel occupied".
Cheaper than the legacy ScalableTSDFVolume for our "show me walls and
people" use case, and supports column carving for dynamic objects:
when new points land in an XY column, existing voxels in that column
are evicted before the new ones go in.

Pattern adapted from dimos/mapping/voxels.py (see docs/comments inside).
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import open3d.core as o3c


def _pack_xy(keys: np.ndarray) -> np.ndarray:
    """Pack int32 (X, Y) into a unique int64 per row. Handles negative coords."""
    x = keys[:, 0].astype(np.int64)
    y = keys[:, 1].astype(np.int64)
    return (x << 32) | (y & 0xFFFFFFFF)


def carve_columns(existing_keys: np.ndarray, new_keys: np.ndarray) -> np.ndarray:
    """Drop existing voxel keys whose (X, Y) is shared with any new key.

    Pure numpy, vectorised via int64 packing of (X, Y) and ``np.isin``.
    Replaces the per-row Python loop that was burning ~6 µs per existing
    voxel (~130 ms at 20 K voxels). New cost is essentially flat.

    Args:
        existing_keys: int array of shape (N, 3) — current voxel keys.
        new_keys:      int array of shape (M, 3) — incoming voxel keys.

    Returns:
        int array of shape (K, 3): the subset of existing_keys whose XY
        is NOT shared with any new key.
    """
    if existing_keys.shape[0] == 0 or new_keys.shape[0] == 0:
        return existing_keys.copy()
    e_xy = _pack_xy(existing_keys)
    n_xy = _pack_xy(new_keys)
    mask = ~np.isin(e_xy, n_xy)
    return existing_keys[mask]


class OccupancyVoxelGrid:
    """Sparse occupancy voxel grid (Open3D HashMap of int voxel keys).

    Voxel coordinate system: integer index = floor(world_xyz / voxel_size).
    Voxel centre in world: (key + 0.5) * voxel_size.
    """

    def __init__(
        self,
        voxel_size: float = 0.05,
        block_count: int = 2_000_000,
        device: str = "CUDA:0",
        carve_columns: bool = True,
    ) -> None:
        self._voxel_size = float(voxel_size)
        self._carve = bool(carve_columns)
        if device.startswith("CUDA") and o3c.cuda.is_available():
            self._dev = o3c.Device(device)
        else:
            self._dev = o3c.Device("CPU:0")
        self._hashmap: o3c.HashMap | None = o3c.HashMap(
            init_capacity=block_count,
            key_dtype=o3c.int32,
            key_element_shape=o3c.SizeVector([3]),
            value_dtypes=[o3c.uint8],
            value_element_shapes=[o3c.SizeVector([1])],
            device=self._dev,
        )
        self._disposed = False

    def _check(self) -> None:
        if self._disposed:
            raise RuntimeError("OccupancyVoxelGrid has been disposed.")

    def add_pointcloud(self, pcd: o3d.geometry.PointCloud) -> None:
        """Voxelise a world-frame point cloud and merge into the grid."""
        self._check()
        assert self._hashmap is not None
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.size == 0:
            return
        # Note: hashmap.activate() dedupes natively; we don't need np.unique.
        keys = np.floor(pts / self._voxel_size).astype(np.int32)
        if keys.shape[0] == 0:
            return

        if self._carve:
            existing = self._existing_keys_np()
            if existing.shape[0]:
                e_xy = _pack_xy(existing)
                n_xy = _pack_xy(keys)
                victims = existing[np.isin(e_xy, n_xy)]
                if victims.shape[0]:
                    victims_t = o3c.Tensor(victims, o3c.int32, self._dev)
                    self._hashmap.erase(victims_t)

        keys_t = o3c.Tensor(keys, o3c.int32, self._dev)
        self._hashmap.activate(keys_t)

    def _existing_keys_np(self) -> np.ndarray:
        assert self._hashmap is not None
        active = self._hashmap.active_buf_indices()
        if active.shape[0] == 0:
            return np.empty((0, 3), dtype=np.int32)
        keys_t = self._hashmap.key_tensor()[active]
        return keys_t.cpu().numpy().astype(np.int32)

    def get_voxel_centers(self) -> np.ndarray:
        """Return active voxel centres in world frame as (N, 3) float32."""
        self._check()
        keys = self._existing_keys_np()
        if keys.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)
        return ((keys.astype(np.float32) + 0.5) * self._voxel_size).astype(np.float32)

    def size(self) -> int:
        self._check()
        assert self._hashmap is not None
        return int(self._hashmap.size())

    def __len__(self) -> int:
        return self.size()

    def dispose(self) -> None:
        """Release the GPU/CPU hashmap. Object is unusable afterwards."""
        if not self._disposed:
            self._disposed = True
            self._hashmap = None
