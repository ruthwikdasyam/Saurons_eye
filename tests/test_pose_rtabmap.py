"""
Tests for RtabmapPose math + buffer (pure logic; no ROS).

Validates:
  - Quaternion → rotation matrix conversion (multiple known cases)
  - Building a 4x4 T from ROS pose components
  - PoseBuffer time-indexed storage with linear/SLERP interpolation
"""

from __future__ import annotations

import numpy as np

from capture.pose_rtabmap import (
    PoseBuffer,
    T_WC_from_pose_msg_components,
    _R_to_quat,
    _slerp,
    quat_to_R,
)


# ---- quaternion math ----

def test_quat_identity_to_R() -> None:
    R = quat_to_R(0.0, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


def test_quat_z_90deg() -> None:
    """Quaternion (0, 0, sin(45°), cos(45°)) = 90° rotation about +Z."""
    s = np.sin(np.pi / 4)
    c = np.cos(np.pi / 4)
    R = quat_to_R(0.0, 0.0, s, c)
    expected = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    np.testing.assert_allclose(R, expected, atol=1e-10)


def test_quat_x_180deg() -> None:
    """Quaternion (1, 0, 0, 0) = 180° rotation about +X (flip Y, Z)."""
    R = quat_to_R(1.0, 0.0, 0.0, 0.0)
    expected = np.diag([1.0, -1.0, -1.0])
    np.testing.assert_allclose(R, expected, atol=1e-10)


def test_quat_to_R_is_orthogonal() -> None:
    """Random unit quaternion → orthonormal matrix (R @ R.T = I, det = 1)."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = quat_to_R(*q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


def test_R_to_quat_roundtrip() -> None:
    """quat_to_R then _R_to_quat returns the same quat (or its negative)."""
    rng = np.random.default_rng(7)
    for _ in range(10):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = quat_to_R(*q)
        q2 = _R_to_quat(R)
        # Quaternions q and -q represent the same rotation.
        if np.dot(q, q2) < 0:
            q2 = -q2
        np.testing.assert_allclose(q, q2, atol=1e-9)


# ---- T_WC builder ----

def test_T_WC_identity_pose() -> None:
    T = T_WC_from_pose_msg_components((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(T, np.eye(4), atol=1e-10)


def test_T_WC_translation_only() -> None:
    T = T_WC_from_pose_msg_components((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(T, expected, atol=1e-10)


def test_T_WC_rotation_and_translation() -> None:
    s = np.sin(np.pi / 4)
    c = np.cos(np.pi / 4)
    T = T_WC_from_pose_msg_components((1.0, 0.0, 0.0), (0.0, 0.0, s, c))
    assert T[3, 3] == 1.0
    np.testing.assert_allclose(T[:3, 3], [1.0, 0.0, 0.0])
    expected_R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    np.testing.assert_allclose(T[:3, :3], expected_R, atol=1e-10)


# ---- SLERP ----

def test_slerp_endpoints() -> None:
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    s = np.sin(np.pi / 4)
    c = np.cos(np.pi / 4)
    q1 = np.array([0.0, 0.0, s, c])
    np.testing.assert_allclose(_slerp(q0, q1, 0.0), q0, atol=1e-10)
    np.testing.assert_allclose(_slerp(q0, q1, 1.0), q1, atol=1e-10)


def test_slerp_midpoint_is_unit() -> None:
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
    qm = _slerp(q0, q1, 0.5)
    assert abs(np.linalg.norm(qm) - 1.0) < 1e-9


# ---- PoseBuffer ----

def test_buffer_empty_returns_identity() -> None:
    buf = PoseBuffer()
    np.testing.assert_array_equal(buf.at(0.0), np.eye(4))


def test_buffer_single_pose_returned_for_any_t() -> None:
    buf = PoseBuffer()
    T = np.eye(4)
    T[0, 3] = 5.0
    buf.add(1.0, T)
    np.testing.assert_array_equal(buf.at(0.5), T)
    np.testing.assert_array_equal(buf.at(1.0), T)
    np.testing.assert_array_equal(buf.at(2.0), T)


def test_buffer_translation_lerp_at_midpoint() -> None:
    buf = PoseBuffer()
    T0 = np.eye(4); T0[0, 3] = 0.0
    T1 = np.eye(4); T1[0, 3] = 2.0
    buf.add(0.0, T0)
    buf.add(1.0, T1)
    T_mid = buf.at(0.5)
    assert abs(T_mid[0, 3] - 1.0) < 1e-6


def test_buffer_returns_latest_for_t_past_end() -> None:
    buf = PoseBuffer()
    T0 = np.eye(4); T0[0, 3] = 0.0
    T1 = np.eye(4); T1[0, 3] = 2.0
    buf.add(0.0, T0)
    buf.add(1.0, T1)
    np.testing.assert_array_equal(buf.at(5.0), T1)


def test_buffer_returns_oldest_for_t_before_start() -> None:
    buf = PoseBuffer()
    T0 = np.eye(4); T0[0, 3] = 0.0
    T1 = np.eye(4); T1[0, 3] = 2.0
    buf.add(1.0, T0)
    buf.add(2.0, T1)
    np.testing.assert_array_equal(buf.at(0.0), T0)


def test_buffer_rotation_slerp_at_midpoint() -> None:
    """Buffer between identity and 90°-about-Z, query at midpoint → 45°-about-Z."""
    buf = PoseBuffer()
    buf.add(0.0, np.eye(4))
    s = np.sin(np.pi / 4)
    c = np.cos(np.pi / 4)
    T1 = np.eye(4)
    T1[:3, :3] = quat_to_R(0.0, 0.0, s, c)
    buf.add(1.0, T1)
    T_mid = buf.at(0.5)
    # 45° about Z
    s45 = np.sin(np.pi / 4)
    c45 = np.cos(np.pi / 4)
    expected_R = np.array([
        [c45, -s45, 0.0],
        [s45,  c45, 0.0],
        [0.0,  0.0, 1.0],
    ])
    np.testing.assert_allclose(T_mid[:3, :3], expected_R, atol=1e-9)


def test_buffer_maxlen_evicts_oldest() -> None:
    buf = PoseBuffer(maxlen=3)
    for i in range(5):
        T = np.eye(4); T[0, 3] = float(i)
        buf.add(float(i), T)
    assert len(buf) == 3
    # Oldest two should have been evicted; first remaining is i=2 → x=2.0
    np.testing.assert_array_equal(buf.at(2.0)[:3, 3], [2.0, 0.0, 0.0])
