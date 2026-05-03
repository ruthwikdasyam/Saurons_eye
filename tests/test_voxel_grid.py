"""
Tests for the OccupancyVoxelGrid wrapper around Open3D's tensor
VoxelBlockGrid (GPU-capable, falls back to CPU).

Pattern (borrowed from dimos/mapping/voxels.py): we drop signed-distance
in favour of pure occupancy. For "show me walls and people," occupancy
is enough — and the tensor / hashmap implementation is materially
faster than the legacy ScalableTSDFVolume.

These tests run on CPU device for portability so they pass on any
machine (the production code may swap to CUDA when available).
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from capture.occupancy import OccupancyVoxelGrid


def _make_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def test_empty_grid() -> None:
    grid = OccupancyVoxelGrid(voxel_size=0.1, device="CPU:0")
    assert grid.size() == 0
    pts = grid.get_voxel_centers()
    assert pts.shape == (0, 3)
    grid.dispose()


def test_single_point_creates_single_voxel() -> None:
    grid = OccupancyVoxelGrid(voxel_size=0.1, device="CPU:0")
    grid.add_pointcloud(_make_pcd(np.array([[0.05, 0.05, 0.05]])))
    assert grid.size() == 1
    centers = grid.get_voxel_centers()
    assert centers.shape == (1, 3)
    # Centre of voxel containing (0.05, 0.05, 0.05) at 10cm voxels: ~(0.05, 0.05, 0.05).
    np.testing.assert_allclose(centers[0], [0.05, 0.05, 0.05], atol=0.06)
    grid.dispose()


def test_two_points_same_voxel_dedupe() -> None:
    """Two points landing in the same voxel cell should still produce ONE voxel."""
    grid = OccupancyVoxelGrid(voxel_size=0.1, device="CPU:0")
    grid.add_pointcloud(_make_pcd(np.array([
        [0.01, 0.01, 0.01],
        [0.09, 0.09, 0.09],
    ])))
    assert grid.size() == 1
    grid.dispose()


def test_carve_columns_in_grid() -> None:
    """Adding new points in the same XY column should evict old voxels in that column."""
    grid = OccupancyVoxelGrid(voxel_size=0.1, device="CPU:0", carve_columns=True)
    grid.add_pointcloud(_make_pcd(np.array([
        [0.05, 0.05, 0.05],
        [0.05, 0.05, 0.55],
        [0.05, 0.05, 1.05],
    ])))
    assert grid.size() == 3
    # New observation in the same XY column → existing 3 voxels should be evicted.
    grid.add_pointcloud(_make_pcd(np.array([[0.05, 0.05, 2.05]])))
    assert grid.size() == 1
    centers = grid.get_voxel_centers()
    np.testing.assert_allclose(centers[0], [0.05, 0.05, 2.05], atol=0.06)
    grid.dispose()


def test_no_carving_keeps_all() -> None:
    grid = OccupancyVoxelGrid(voxel_size=0.1, device="CPU:0", carve_columns=False)
    grid.add_pointcloud(_make_pcd(np.array([[0.05, 0.05, 0.05]])))
    grid.add_pointcloud(_make_pcd(np.array([[0.05, 0.05, 0.55]])))
    assert grid.size() == 2
    grid.dispose()


def test_dispose_makes_grid_unusable() -> None:
    grid = OccupancyVoxelGrid(voxel_size=0.1, device="CPU:0")
    grid.add_pointcloud(_make_pcd(np.array([[0.05, 0.05, 0.05]])))
    grid.dispose()
    with pytest.raises(RuntimeError):
        grid.size()


def test_disjoint_columns_survive_carving() -> None:
    """Carving an XY column shouldn't affect adjacent columns."""
    grid = OccupancyVoxelGrid(voxel_size=0.1, device="CPU:0", carve_columns=True)
    grid.add_pointcloud(_make_pcd(np.array([
        [0.05, 0.05, 0.05],   # column (0,0)
        [0.95, 0.05, 0.05],   # column (9,0)
        [0.05, 0.95, 0.05],   # column (0,9)
    ])))
    assert grid.size() == 3
    # New in column (0,0) only → others survive.
    grid.add_pointcloud(_make_pcd(np.array([[0.05, 0.05, 2.05]])))
    assert grid.size() == 3   # 1 in (0,0), 1 in (9,0), 1 in (0,9)
    grid.dispose()
