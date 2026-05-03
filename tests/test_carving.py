"""
Tests for column carving — when new voxels arrive at an (X, Y) column,
all existing voxels in that column are evicted before the new ones go in.

Pattern (borrowed from dimos/mapping/voxels.py): solves the "ghost trail"
problem for dynamic objects without needing TSDF weight decay. If a
person walks past a wall and you re-observe, the person's old voxels are
deleted automatically when fresh observations arrive in the same column.

Pure logic on integer voxel keys — no Open3D needed for these tests.
"""

from __future__ import annotations

import numpy as np

from capture.occupancy import carve_columns


def test_no_existing() -> None:
    existing = np.empty((0, 3), dtype=np.int32)
    new = np.array([[0, 0, 5]], dtype=np.int32)
    kept = carve_columns(existing, new)
    assert kept.shape == (0, 3)
    assert kept.dtype == np.int32


def test_no_new_keeps_all_existing() -> None:
    existing = np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32)
    new = np.empty((0, 3), dtype=np.int32)
    kept = carve_columns(existing, new)
    assert kept.shape == (2, 3)
    np.testing.assert_array_equal(np.sort(kept, axis=0), np.sort(existing, axis=0))


def test_carve_same_column_different_z() -> None:
    """Existing at (0,0,1), new at (0,0,5): same XY column → existing must be carved."""
    existing = np.array([[0, 0, 1]], dtype=np.int32)
    new = np.array([[0, 0, 5]], dtype=np.int32)
    kept = carve_columns(existing, new)
    assert kept.shape == (0, 3)


def test_keep_disjoint_columns() -> None:
    """Existing at (0,0) and (1,1); new only touches (0,0) → (1,1) survives."""
    existing = np.array([
        [0, 0, 1],
        [0, 0, 2],
        [1, 1, 1],
    ], dtype=np.int32)
    new = np.array([[0, 0, 5]], dtype=np.int32)
    kept = carve_columns(existing, new)
    assert kept.shape == (1, 3)
    np.testing.assert_array_equal(kept[0], [1, 1, 1])


def test_partial_overlap() -> None:
    """Existing at (0,0), (0,1), (1,0); new touches (0,0) and (1,1).
    Should keep (0,1) and (1,0); carve (0,0); (1,1) was new.
    """
    existing = np.array([
        [0, 0, 1],
        [0, 1, 2],
        [1, 0, 3],
    ], dtype=np.int32)
    new = np.array([
        [0, 0, 9],
        [1, 1, 9],
    ], dtype=np.int32)
    kept = carve_columns(existing, new)
    expected = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int32)
    np.testing.assert_array_equal(np.sort(kept, axis=0), np.sort(expected, axis=0))


def test_carves_multiple_in_column() -> None:
    """Stack of 5 voxels at (0,0,*) all carved when new (0,0,*) arrives."""
    existing = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 0, 3],
        [0, 0, 4],
    ], dtype=np.int32)
    new = np.array([[0, 0, 99]], dtype=np.int32)
    kept = carve_columns(existing, new)
    assert kept.shape == (0, 3)


def test_negative_coords() -> None:
    """Carving must work for negative voxel coords (camera frame can be negative)."""
    existing = np.array([
        [-3, -2, 0],
        [-3, -2, 5],
        [-3, +2, 0],
    ], dtype=np.int32)
    new = np.array([[-3, -2, 9]], dtype=np.int32)
    kept = carve_columns(existing, new)
    assert kept.shape == (1, 3)
    np.testing.assert_array_equal(kept[0], [-3, +2, 0])
