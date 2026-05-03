"""
Tests for the pure helpers in capture.realsense_ros (no ROS spinning).

The synchronizer + iterator path is integration territory; we cover it
via a smoke script (capture/_smoke_realsense_ros.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from capture.realsense_ros import camera_info_to_intrinsics, image_msg_to_np


# ---- image_msg_to_np ----

class _FakeImageMsg:
    def __init__(self, data: bytes, width: int, height: int, encoding: str) -> None:
        self.data = data
        self.width = width
        self.height = height
        self.encoding = encoding


def test_image_rgb8() -> None:
    raw = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)  # 1x2 image
    msg = _FakeImageMsg(raw.tobytes(), width=2, height=1, encoding="rgb8")
    out = image_msg_to_np(msg)
    np.testing.assert_array_equal(out, raw)


def test_image_bgr8_swaps_to_rgb() -> None:
    bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)  # 1x2 BGR
    msg = _FakeImageMsg(bgr.tobytes(), width=2, height=1, encoding="bgr8")
    out = image_msg_to_np(msg)
    expected = bgr[..., ::-1]
    np.testing.assert_array_equal(out, expected)


def test_image_16uc1_depth() -> None:
    raw = np.array([[1000, 2000, 3000]], dtype=np.uint16)  # 1x3 depth
    msg = _FakeImageMsg(raw.tobytes(), width=3, height=1, encoding="16UC1")
    out = image_msg_to_np(msg)
    np.testing.assert_array_equal(out, raw)


def test_image_unsupported_encoding_raises() -> None:
    msg = _FakeImageMsg(b"", width=0, height=0, encoding="yuv422")
    with pytest.raises(ValueError, match="unsupported encoding"):
        image_msg_to_np(msg)


# ---- camera_info_to_intrinsics ----

class _FakeCameraInfo:
    def __init__(self, width: int, height: int, K: list[float]) -> None:
        self.width = width
        self.height = height
        self.k = K   # ROS 2 uses lowercase 'k' for the 3x3 intrinsic matrix


def test_camera_info_to_intrinsics() -> None:
    # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    info = _FakeCameraInfo(
        width=640, height=480,
        K=[600.0, 0.0, 320.0, 0.0, 605.0, 240.0, 0.0, 0.0, 1.0],
    )
    intr = camera_info_to_intrinsics(info)
    assert intr.width == 640
    assert intr.height == 480
    K = intr.intrinsic_matrix
    assert K[0, 0] == 600.0
    assert K[1, 1] == 605.0
    assert K[0, 2] == 320.0
    assert K[1, 2] == 240.0
