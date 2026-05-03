"""
Tests for the PointCloud2 → numpy decoder used by RtabmapDenseMap.

The ROS subscriber + threading is integration territory; these tests
cover only the pure decoder so we can rely on it without standing up rclpy.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from capture.rtabmap_map import pointcloud2_to_xyz


class _FakeField:
    def __init__(self, name: str, offset: int, datatype: int = 7, count: int = 1) -> None:
        self.name = name
        self.offset = offset
        self.datatype = datatype   # 7 = FLOAT32 (PointField.FLOAT32)
        self.count = count


class _FakeCloud:
    def __init__(
        self,
        fields: list[_FakeField],
        data: bytes,
        point_step: int,
        width: int,
        height: int = 1,
    ) -> None:
        self.fields = fields
        self.data = data
        self.point_step = point_step
        self.width = width
        self.height = height


def _make_xyz_only_cloud(points: list[tuple[float, float, float]]) -> _FakeCloud:
    fields = [_FakeField("x", 0), _FakeField("y", 4), _FakeField("z", 8)]
    data = b"".join(struct.pack("<fff", *p) for p in points)
    return _FakeCloud(fields, data, point_step=12, width=len(points))


def test_decode_xyz_only() -> None:
    msg = _make_xyz_only_cloud([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
    xyz = pointcloud2_to_xyz(msg)
    assert xyz.shape == (2, 3)
    assert xyz.dtype == np.float32
    np.testing.assert_allclose(xyz, [[1, 2, 3], [4, 5, 6]])


def test_decode_with_padding_after_xyz() -> None:
    """Real RTAB-Map clouds often have RGB at offset 12 or 16; xyz at 0/4/8."""
    fields = [_FakeField("x", 0), _FakeField("y", 4), _FakeField("z", 8)]
    point_step = 32   # padding to 32 bytes per point
    pts = [(1.0, 2.0, 3.0), (-4.5, 0.0, 7.25)]
    data = b""
    for p in pts:
        data += struct.pack("<fff", *p) + b"\x00" * (point_step - 12)
    msg = _FakeCloud(fields, data, point_step=point_step, width=2)
    xyz = pointcloud2_to_xyz(msg)
    np.testing.assert_allclose(xyz, [[1, 2, 3], [-4.5, 0.0, 7.25]])


def test_decode_empty() -> None:
    msg = _FakeCloud(
        fields=[_FakeField("x", 0), _FakeField("y", 4), _FakeField("z", 8)],
        data=b"",
        point_step=12,
        width=0,
    )
    xyz = pointcloud2_to_xyz(msg)
    assert xyz.shape == (0, 3)
    assert xyz.dtype == np.float32


def test_decode_missing_field_raises() -> None:
    """If x/y/z aren't all present, we can't decode."""
    msg = _FakeCloud(
        fields=[_FakeField("x", 0), _FakeField("y", 4)],   # no z
        data=b"\x00" * 8,
        point_step=8,
        width=1,
    )
    with pytest.raises(ValueError, match="missing"):
        pointcloud2_to_xyz(msg)


def test_decode_handles_organized_cloud() -> None:
    """height > 1 → organized cloud; total points = width * height."""
    pts = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)]
    fields = [_FakeField("x", 0), _FakeField("y", 4), _FakeField("z", 8)]
    data = b"".join(struct.pack("<fff", *p) for p in pts)
    msg = _FakeCloud(fields, data, point_step=12, width=2, height=2)   # 2x2 organized
    xyz = pointcloud2_to_xyz(msg)
    assert xyz.shape == (4, 3)
    np.testing.assert_allclose(xyz[0], [0, 0, 0])
    np.testing.assert_allclose(xyz[3], [3, 3, 3])
