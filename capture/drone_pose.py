"""
Drone-pose source for the segmentation publisher.

Spawns ``drone.pose_reader.DronePoseReader`` in a background thread,
keeps the latest LOCAL_POSITION_NED + ATTITUDE_QUATERNION sample, and
exposes a zero-able 4x4 ``T_world_camera`` that the silhouette transform
uses to map person points into the headset's WebXR local-floor frame.

Frame chain:

    P_world = R_axisswap_4x4 @ inv(T_NED_drone0) @ T_NED_drone_now
              @ T_drone_camera @ P_camera_optical

where:
    T_NED_drone_*    — from MAVLink. Rotation from quaternion, translation
                       from (x, y, z).
    R_axisswap       — drone-body FRD (X=fwd, Y=right, Z=down) → WebXR
                       (X=right, Y=up, -Z=forward). Treats drone-body
                       *at zero* as the world axes; world "forward" is
                       wherever the drone was pointing when zero() was
                       called. If the soldier holds the drone aimed in
                       their forward direction before launch, the WebXR
                       overlay lines up.
    T_drone_camera   — camera optical (X=right, Y=down, Z=fwd) → drone
                       body. Default: forward-facing camera mounted at
                       drone centre. Reconfigure if the camera looks
                       elsewhere or sits off-centre.

Drift caveat: NED comes from the autopilot EKF (GPS + IMU + mag).
Outdoors with GPS lock: ~10–50 cm stationary drift, ~1–2 m / minute
under motion. Indoors without VIO: meters in seconds. Press 'z' in the
cv2 window to re-zero from a known reference position.
"""

from __future__ import annotations

import threading

import numpy as np

from drone.pose_reader import DronePoseReader, DronePoseSample

# Drone-body-FRD → WebXR axis swap (rotation only).
_R_WEBXR_FROM_BODY = np.array([
    [ 0.0,  1.0,  0.0],   # webxr +X (right) = body +Y  (right)
    [ 0.0,  0.0, -1.0],   # webxr +Y (up)    = body -Z  (-down)
    [-1.0,  0.0,  0.0],   # webxr +Z (back)  = body -X  (-fwd)
])

# Camera optical → drone-body FRD, for a forward-facing camera at drone centre.
_R_BODY_FROM_OPTICAL = np.array([
    [0.0, 0.0, 1.0],   # body_X (fwd)   = optical +Z
    [1.0, 0.0, 0.0],   # body_Y (right) = optical +X
    [0.0, 1.0, 0.0],   # body_Z (down)  = optical +Y
])


def _r_to_4x4(R: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    return T


def _quat_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Hamilton (w, x, y, z) → 3x3 rotation matrix; identity if degenerate."""
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-9:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s*(qy*qy + qz*qz), s*(qx*qy - qz*qw),     s*(qx*qz + qy*qw)],
        [s*(qx*qy + qz*qw),     1 - s*(qx*qx + qz*qz), s*(qy*qz - qx*qw)],
        [s*(qx*qz - qy*qw),     s*(qy*qz + qx*qw),     1 - s*(qx*qx + qy*qy)],
    ])


def _sample_to_T_NED_body(s: DronePoseSample) -> np.ndarray | None:
    """Build a 4x4 NED-frame body pose from a sample. None if data missing."""
    if (s.x is None or s.y is None or s.z is None or
            s.qw is None or s.qx is None or s.qy is None or s.qz is None):
        return None
    T = np.eye(4)
    T[:3, :3] = _quat_to_R(s.qw, s.qx, s.qy, s.qz)
    T[:3, 3] = [float(s.x), float(s.y), float(s.z)]
    return T


class DronePoseToWorld:
    """Background MAVLink reader + zero-able T_world_camera.

    Usage:
        dpw = DronePoseToWorld("/dev/ttyUSB0", baud=57600)
        dpw.start()                      # non-blocking; spawns reader thread
        ...
        T = dpw.T_world_camera()         # None until first pose + zero
    """

    def __init__(
        self,
        connection: str,
        baud: int = 57600,
        auto_zero_on_first_pose: bool = True,
    ) -> None:
        self._reader = DronePoseReader(connection, baud=baud)
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None       # T_NED_body (4x4)
        self._zero_inv: np.ndarray | None = None     # inv(T_NED_drone0)
        self._auto_zero = auto_zero_on_first_pose
        self._T_axis = _r_to_4x4(_R_WEBXR_FROM_BODY)
        self._T_drone_camera = _r_to_4x4(_R_BODY_FROM_OPTICAL)

    def start(self) -> None:
        """Connect, request streams, spawn the reader thread. Blocks for heartbeat."""
        self._reader.connect()
        self._reader.request_streams()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        def cb(s: DronePoseSample) -> None:
            T = _sample_to_T_NED_body(s)
            if T is None:
                return
            with self._lock:
                self._latest = T
                if self._auto_zero and self._zero_inv is None:
                    self._zero_inv = np.linalg.inv(T)
                    print(f"[drone-pose] auto-zeroed at NED "
                          f"({s.x:+.2f}, {s.y:+.2f}, {s.z:+.2f})", flush=True)
        try:
            self._reader.run(cb)
        except Exception as e:
            print(f"[drone-pose] reader thread crashed: {e}")

    def zero(self) -> bool:
        """Snapshot the current pose as world origin. False if no pose yet."""
        with self._lock:
            if self._latest is None:
                return False
            self._zero_inv = np.linalg.inv(self._latest)
        print("[drone-pose] re-zeroed", flush=True)
        return True

    def T_world_camera(self) -> np.ndarray | None:
        """Latest T_world_camera 4x4. None if no pose yet, or not zeroed."""
        with self._lock:
            if self._latest is None or self._zero_inv is None:
                return None
            T_drone0_drone_now = self._zero_inv @ self._latest
        return self._T_axis @ T_drone0_drone_now @ self._T_drone_camera

    def status(self) -> str:
        """One-line status string for the HUD.

        Pre-zero: raw NED from the autopilot.
        Post-zero: drone position in WebXR world frame (~0 right after zero,
        grows as the drone moves).
        """
        with self._lock:
            if self._latest is None:
                return "no pose"
            if self._zero_inv is None:
                t = self._latest[:3, 3]
                return f"NED raw ({t[0]:+.1f},{t[1]:+.1f},{t[2]:+.1f}) zero=WAIT"
            T_world_drone = self._T_axis @ self._zero_inv @ self._latest
        t = T_world_drone[:3, 3]
        return f"world ({t[0]:+.2f},{t[1]:+.2f},{t[2]:+.2f}) zero=OK"
