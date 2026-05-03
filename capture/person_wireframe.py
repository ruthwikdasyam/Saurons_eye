"""
Extract a 3D wireframe (one OBB per person cluster) from segmented person
points and convert to headset-ready Cube dicts.

For now the camera frame is treated as the VR frame: the loaded transform
defaults to identity. When drone telemetry lands, point ``--transform`` at
the file the drone writes (4x4 T_world_camera) and nothing else changes.

Also has ``obb_edges_3d()`` so the capture pipeline can project the box
back to 2D and draw it on the live preview.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from capture.wireframe import Box, _R_to_quat_xyzw

PERSON_COLOR = 0xFF3355         # red-ish OBB outline
SILHOUETTE_COLOR = 0x00FF88     # bright green outline
SILHOUETTE_FILL = 0x00FF88      # same hue, alpha applied client-side

# Silhouette extraction
SIL_MAX_POINTS = 80              # decimate contour to this many points before sending
SIL_MIN_CONTOUR_AREA = 500       # px²; smaller blobs (limbs flickering in/out) are dropped
SIL_DEPTH_MIN = 0.2              # m
SIL_DEPTH_MAX = 8.0              # m

# DBSCAN: tighter than the room-scale wireframe — a person is a single dense blob.
# After 2 cm voxel downsample of a segmented person (front-facing surface only),
# typical density is ~20 voxels per 10-cm ball — min_points must sit below that.
DBSCAN_EPS = 0.10
DBSCAN_MIN_POINTS = 15
MIN_OBB_LARGEST_EXTENT = 0.30  # drop OBBs whose largest dim is <30 cm (sensor noise)


def points_to_person_obbs(points: np.ndarray) -> list[Box]:
    """Cluster segmented person points; return one OBB per cluster."""
    if len(points) < DBSCAN_MIN_POINTS:
        return []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    labels = np.array(pcd.cluster_dbscan(
        eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=False,
    ))
    if labels.size == 0 or labels.max() < 0:
        return []
    boxes: list[Box] = []
    for lbl in range(int(labels.max()) + 1):
        cluster = pcd.select_by_index(np.where(labels == lbl)[0])
        try:
            obb = cluster.get_oriented_bounding_box()
        except Exception:
            continue
        if max(obb.extent) >= MIN_OBB_LARGEST_EXTENT:
            boxes.append(Box(
                center=tuple(obb.center.tolist()),
                extent=tuple(obb.extent.tolist()),
                quat_xyzw=_R_to_quat_xyzw(np.asarray(obb.R)),
                label="person",
            ))
    return boxes


def mask_to_silhouettes_3d(
    mask: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    max_points: int = SIL_MAX_POINTS,
) -> list[np.ndarray]:
    """Per connected mask region, extract its silhouette as a 3D polyline.

    Steps:
      1. cv2.findContours on the binary mask → 2D outline pixels.
      2. Decimate to ``max_points`` (uniform along the perimeter).
      3. Look up depth at each contour pixel; back-project (u,v,z) → (X,Y,Z).
      4. Drop contour points whose depth is invalid (out of range or NaN).

    Returns one [N,3] float32 array per mask region (camera frame).
    """
    if not mask.any():
        return []
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = depth_m.shape[:2]

    out: list[np.ndarray] = []
    for c in contours:
        if cv2.contourArea(c) < SIL_MIN_CONTOUR_AREA:
            continue
        c2 = c[:, 0, :]                                           # [N, 2] (x, y) pixels
        if len(c2) > max_points:
            idx = np.linspace(0, len(c2) - 1, max_points).round().astype(int)
            c2 = c2[idx]
        us = np.clip(c2[:, 0].astype(np.int32), 0, w - 1)
        vs = np.clip(c2[:, 1].astype(np.int32), 0, h - 1)
        z = depth_m[vs, us]
        valid = (z >= SIL_DEPTH_MIN) & (z <= SIL_DEPTH_MAX) & np.isfinite(z)
        if valid.sum() < 8:                                       # too gappy to be useful
            continue
        us, vs, z = us[valid], vs[valid], z[valid]
        x = (us - cx) * z / fx
        y = (vs - cy) * z / fy
        out.append(np.stack([x, y, z], axis=-1).astype(np.float32))
    return out


def silhouettes_to_polylines(
    silhouettes_3d: list[np.ndarray],
    T_world_camera: np.ndarray,
    color: int = SILHOUETTE_COLOR,
    fill_color: int | None = SILHOUETTE_FILL,
    frame: str = "world",
) -> list[dict]:
    """Apply transform to each silhouette and emit polyline dicts for the headset.

    ``frame`` is forwarded to the renderer; "camera" tells it to multiply by
    the live headset pose at scene-receive time, "world" means the points are
    already in WebXR local-floor coords.
    """
    if not silhouettes_3d:
        return []
    R = T_world_camera[:3, :3]
    t = T_world_camera[:3, 3]
    out: list[dict] = []
    for pts in silhouettes_3d:
        pts_w = (R @ pts.T).T + t
        out.append({
            "points": pts_w.astype(float).tolist(),
            "color": int(color),
            "fill_color": int(fill_color) if fill_color is not None else None,
            "closed": True,
            "frame": frame,
        })
    return out


def camera_optical_to_local() -> np.ndarray:
    """Static axis swap: RealSense optical (X right, Y down, Z forward) →
    three.js camera-local (X right, Y up, -Z forward).

    Use when the publisher cannot know the headset's live pose. Emit points
    in this frame, tag them ``frame="camera"``, and let the renderer apply
    the headset pose at scene-receive time. This world-locks the silhouettes
    to wherever the user was looking when each scene arrived.
    """
    return np.array([
        [1.0,  0.0,  0.0, 0.0],
        [0.0, -1.0,  0.0, 0.0],
        [0.0,  0.0, -1.0, 0.0],
        [0.0,  0.0,  0.0, 1.0],
    ])


def load_transform(path: str | Path | None) -> np.ndarray:
    """Load a 4x4 T_world_camera. None / missing file → identity.

    Accepted JSON shapes:
        {"T_world_camera": [[...4x4...]]}
        [[...4x4...]]
        [r0c0, r0c1, ..., r3c3]   (flat 16)
    """
    if path is None:
        return np.eye(4)
    p = Path(path)
    if not p.exists():
        print(f"[transform] {p} not found — using identity")
        return np.eye(4)
    with p.open() as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("T_world_camera", data)
    arr = np.array(data, dtype=np.float64).reshape(4, 4)
    return arr


def obbs_to_cubes(
    boxes: list[Box],
    T_world_camera: np.ndarray,
    color: int = PERSON_COLOR,
) -> list[dict]:
    """Apply transform to each OBB; emit Cube dicts for the headset.

    With T = identity (current default) the cube sits in the camera frame.
    The headset renderer reads (center, size, quat) directly.
    """
    if not boxes:
        return []
    R_wc = T_world_camera[:3, :3]
    t_wc = T_world_camera[:3, 3]
    out: list[dict] = []
    for b in boxes:
        center = np.array(b.center)
        center_w = R_wc @ center + t_wc
        R_in = _quat_to_R(*b.quat_xyzw)
        R_out = R_wc @ R_in
        q = _R_to_quat_xyzw(R_out)
        out.append({
            "center": tuple(center_w.tolist()),
            "size": tuple(b.extent),
            "quat": q,
            "color": color,
        })
    return out


def obb_edges_3d(boxes: list[Box]) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each OBB, return its 8 corner points + 12 edge-index pairs.

    Lets the caller project corners through K and draw the box on a 2D image.
    """
    out = []
    for b in boxes:
        R = _quat_to_R(*b.quat_xyzw)
        center = np.array(b.center)
        ext = np.array(b.extent) * 0.5
        # 8 unit-cube corners scaled to half-extents, rotated, translated.
        signs = np.array([
            [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
        ], dtype=np.float64)
        corners = (signs * ext) @ R.T + center
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],   # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],   # top face
            [0, 4], [1, 5], [2, 6], [3, 7],   # verticals
        ], dtype=np.int32)
        out.append((corners, edges))
    return out


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = qx*qx + qy*qy + qz*qz + qw*qw
    s = 2.0 / n if n > 0 else 0.0
    return np.array([
        [1 - s*(qy*qy + qz*qz), s*(qx*qy - qz*qw),     s*(qx*qz + qy*qw)],
        [s*(qx*qy + qz*qw),     1 - s*(qx*qx + qz*qz), s*(qy*qz - qx*qw)],
        [s*(qx*qz - qy*qw),     s*(qy*qz + qx*qw),     1 - s*(qx*qx + qy*qy)],
    ])
