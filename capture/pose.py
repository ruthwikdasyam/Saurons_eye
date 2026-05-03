"""
Pose source abstraction. Today: RGB-D odometry. Tomorrow: MAVLink. Same
contract; the rest of the pipeline doesn't know or care which.

Why this abstraction matters:
  - V1 ships with `RgbdOdometryPose` because we're using a hand-carried
    RealSense as a stand-in drone. We have to estimate L (localisation)
    ourselves.
  - On a real drone, the autopilot already runs its own VIO/SLAM and
    publishes pose over MAVLink. We then drop in `MavlinkPose` and the
    rest of the pipeline (TSDF mapping, voxel display, wire format,
    headset) doesn't move.

Contract is intentionally tiny: feed `update(rgbd, t)` and get back the
current `T_WC` (4x4 pose of camera in world).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import open3d as o3d


class PoseSource(Protocol):
    """Returns the latest pose of the camera in world frame."""

    def update(
        self,
        rgbd: o3d.geometry.RGBDImage | None,
        t: float,
    ) -> np.ndarray:
        """Advance internal state if needed; return T_WC as 4x4 matrix."""
        ...


class IdentityPose:
    """Camera == world. Pose never changes. For stationary debugging."""

    def update(self, rgbd: o3d.geometry.RGBDImage | None, t: float) -> np.ndarray:
        return np.eye(4)


class RgbdOdometryPose:
    """Frame-to-frame RGB-D odometry via Open3D dense alignment.

    First frame establishes the world origin (T_WC = identity). Subsequent
    frames are aligned against the previous; the pose is chained.

    Failures (Open3D returns ok=False) are counted in `.failures` but the
    pose is held at its last good value rather than reset.
    """

    def __init__(
        self,
        intrinsics: o3d.camera.PinholeCameraIntrinsic,
        iterations: tuple[int, int, int] = (5, 3, 1),
    ) -> None:
        self._intr = intrinsics
        self._jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        self._option = o3d.pipelines.odometry.OdometryOption(
            iteration_number_per_pyramid_level=o3d.utility.IntVector(list(iterations)),
        )
        self._T_WC = np.eye(4)
        self._prev: o3d.geometry.RGBDImage | None = None
        self.failures = 0

    def update(
        self,
        rgbd: o3d.geometry.RGBDImage | None,
        t: float,
    ) -> np.ndarray:
        if rgbd is None:
            return self._T_WC
        if self._prev is None:
            self._prev = rgbd
            return self._T_WC
        # Open3D returns T such that p_prev = T · p_curr  →  T = T_prev_curr.
        # World pose update:  T_WC(t) = T_WC(t-1) · T_prev_curr.
        ok, T_prev_curr, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd, self._prev, self._intr, np.eye(4),
            self._jacobian, self._option,
        )
        if ok:
            self._T_WC = self._T_WC @ T_prev_curr
        else:
            self.failures += 1
        self._prev = rgbd
        return self._T_WC


class MavlinkPose:
    """Post-v1 stub. Slot reserved so the swap is one class, no pipeline change.

    Implementation sketch (~50 lines when written):
      - Subscribe to a MAVLink ODOMETRY topic (mavsdk-python or pymavlink).
      - Buffer the last few poses.
      - On update(t), interpolate between the two MAVLink poses bracketing t.
      - Apply the rigid camera-on-airframe offset T_drone_camera (one matmul).
      - Return T_WC.
    """

    def __init__(self, *_, **__) -> None:
        raise NotImplementedError(
            "MavlinkPose is the post-v1 plug — see README → Ideas / Post-v1."
        )

    def update(
        self,
        rgbd: o3d.geometry.RGBDImage | None,
        t: float,
    ) -> np.ndarray:
        raise NotImplementedError
