"""
Extract a wireframe scene from a point cloud — floor, wall OBBs, object OBBs —
and convert it into the headset's WebXR Cube format.

Two coordinate frames here:
  - input frame: gravity-aligned RTAB-Map world (+Z up)
  - WebXR frame: y-up, +X right, -Z forward (Quest 'local-floor' reference)

Pipeline in extract_wireframe():
  1. RANSAC the largest horizontal plane → floor (a thin Box).
  2. Normal-classify remaining points: |n.z|<0.30 → wall candidates.
  3. DBSCAN cluster wall candidates → one OBB per cluster.
  4. Cluster everything that's neither floor nor wall → object OBBs.

The data quality determines what survives. With a deliberately-scanned
room you'll get a clean floor + a few walls + N object boxes. With a
sloppy scan, walls fall out and you get mostly objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


# ──── Tuning ────────────────────────────────────────────────────────────────

# Floor RANSAC
FLOOR_DISTANCE_THRESHOLD = 0.05
FLOOR_MIN_INLIERS = 2000
FLOOR_HORIZONTAL_NZ = 0.85           # |n.z| > this counts as horizontal

# Wall normal-filter
NORMAL_RADIUS = 0.10
NORMAL_K = 30
WALL_NZ_THRESHOLD = 0.30             # |n.z| < this counts as wall-normal

# Wall clustering
WALL_DBSCAN_EPS = 0.30
WALL_DBSCAN_MIN_POINTS = 60
WALL_OBB_MIN_EXTENT = 0.30           # drop OBBs whose largest dim is < this

# Object clustering (run on what's left)
OBJ_DBSCAN_EPS = 0.20
OBJ_DBSCAN_MIN_POINTS = 50
OBJ_OBB_MIN_EXTENT = 0.20

# Per-label colour (0xRRGGBB) — matches the headset renderer
COLOR_FLOOR = 0x506070
COLOR_WALL = 0x00CCAA
COLOR_OBJECT = 0xFF8800


# ──── Output dataclass ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Box:
    """A 3D box in some coordinate frame. Frame-agnostic by design."""
    center: tuple[float, float, float]
    extent: tuple[float, float, float]   # full extent along each local axis
    quat_xyzw: tuple[float, float, float, float]
    label: str   # "floor" | "wall" | "object"


# ──── Helpers ───────────────────────────────────────────────────────────────

def _R_to_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    """3×3 rotation → unit quaternion (x, y, z, w). Numerically stable form."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (float(x), float(y), float(z), float(w))


def _obb_to_box(obb: o3d.geometry.OrientedBoundingBox, label: str) -> Box:
    return Box(
        center=tuple(obb.center.tolist()),
        extent=tuple(obb.extent.tolist()),
        quat_xyzw=_R_to_quat_xyzw(np.asarray(obb.R)),
        label=label,
    )


# ──── Floor as a Box ────────────────────────────────────────────────────────

def _floor_box_from_inliers(inlier_pts: np.ndarray) -> Box:
    """The largest horizontal plane → a thin Box at floor level."""
    xs = inlier_pts[:, 0]
    ys = inlier_pts[:, 1]
    z_mean = float(np.mean(inlier_pts[:, 2]))
    cx, cy = float(xs.mean()), float(ys.mean())
    width_x = float(xs.max() - xs.min())
    width_y = float(ys.max() - ys.min())
    return Box(
        center=(cx, cy, z_mean),
        extent=(width_x, width_y, 0.02),     # 2 cm thick floor slab
        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        label="floor",
    )


# ──── Main pipeline ─────────────────────────────────────────────────────────

def extract_wireframe(pcd: o3d.geometry.PointCloud) -> list[Box]:
    """Floor + walls + objects → list[Box] in the input cloud's frame."""
    pcd = o3d.geometry.PointCloud(pcd)   # don't mutate caller's cloud
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=NORMAL_K,
        ))

    boxes: list[Box] = []

    # 1. Floor
    if len(pcd.points) >= FLOOR_MIN_INLIERS:
        model, inliers = pcd.segment_plane(
            distance_threshold=FLOOR_DISTANCE_THRESHOLD,
            ransac_n=3,
            num_iterations=1000,
        )
        a, b, c, d = model
        n = np.array([a, b, c]); n /= np.linalg.norm(n)
        if abs(n[2]) > FLOOR_HORIZONTAL_NZ and len(inliers) >= FLOOR_MIN_INLIERS:
            inlier_pts = np.asarray(pcd.points)[inliers]
            boxes.append(_floor_box_from_inliers(inlier_pts))
            pcd = pcd.select_by_index(inliers, invert=True)

    # 2. Wall candidates by normal direction
    normals = np.asarray(pcd.normals)
    nz = np.abs(normals[:, 2])
    wall_mask = nz < WALL_NZ_THRESHOLD
    wall_idx = np.where(wall_mask)[0]
    other_idx = np.where(~wall_mask)[0]

    if len(wall_idx) > 0:
        wall_pcd = pcd.select_by_index(wall_idx)
        labels = np.array(wall_pcd.cluster_dbscan(
            eps=WALL_DBSCAN_EPS, min_points=WALL_DBSCAN_MIN_POINTS, print_progress=False,
        ))
        for lbl in range(int(labels.max()) + 1 if labels.size else 0):
            cluster = wall_pcd.select_by_index(np.where(labels == lbl)[0])
            try:
                obb = cluster.get_oriented_bounding_box()
            except Exception:
                continue
            if max(obb.extent) >= WALL_OBB_MIN_EXTENT:
                boxes.append(_obb_to_box(obb, "wall"))

    # 3. Objects from non-wall, non-floor points
    if len(other_idx) > 0:
        obj_pcd = pcd.select_by_index(other_idx)
        labels = np.array(obj_pcd.cluster_dbscan(
            eps=OBJ_DBSCAN_EPS, min_points=OBJ_DBSCAN_MIN_POINTS, print_progress=False,
        ))
        for lbl in range(int(labels.max()) + 1 if labels.size else 0):
            cluster = obj_pcd.select_by_index(np.where(labels == lbl)[0])
            try:
                obb = cluster.get_oriented_bounding_box()
            except Exception:
                continue
            if max(obb.extent) >= OBJ_OBB_MIN_EXTENT:
                boxes.append(_obb_to_box(obb, "object"))

    return boxes


# ──── WebXR coordinate transform ────────────────────────────────────────────

# Axis swap from gravity-aligned RTAB world (+Z up) to WebXR (+Y up, -Z forward).
# RTAB-Map with frame_id=camera_link uses REP-103 (X forward, Y left, Z up).
# We want scan-start "forward" (RTAB +X) to align with the user's "forward" in
# WebXR (-Z), so the wireframe overlays the real world when the user stands at
# the scan-start spot facing the scan-start direction.
#
#   RTAB +X (scan-start forward) → WebXR -Z (user's forward)
#   RTAB +Y (scan-start left)    → WebXR -X (user's left)
#   RTAB +Z (up)                 → WebXR +Y (up)
_T_WEBXR_FROM_RTAB = np.array([
    [ 0.0, -1.0, 0.0],
    [ 0.0,  0.0, 1.0],
    [-1.0,  0.0, 0.0],
])


def boxes_to_webxr_cubes(
    boxes: list[Box],
    floor_at_y: float = 0.0,
) -> list[dict]:
    """Convert input-frame boxes to WebXR Cube dicts.

    Auto-calibration assumption: the user stands at the spot where the camera
    was when the scan started, facing the scan-start direction, when they tap
    "Enter AR". WebXR's origin then coincides with the cloud's origin and the
    wireframe overlays the real world without further calibration.

    The only correction we apply automatically is the floor height: snap the
    cloud's floor plane to WebXR ``floor_at_y`` (default 0). This handles the
    fact that the headset is at eye-height while the camera was at hand-height.

    If the user can't stand at the exact scan-start pose, manual nudge controls
    on the headset side can compensate.
    """
    if not boxes:
        return []

    centres_webxr = np.array([_T_WEBXR_FROM_RTAB @ np.array(b.center) for b in boxes])

    # Floor alignment only — snap cloud floor to WebXR floor.
    floor_y = next(
        (c[1] for c, b in zip(centres_webxr, boxes) if b.label == "floor"),
        centres_webxr[:, 1].min(),
    )
    y_offset = floor_at_y - floor_y

    label_color = {
        "floor": COLOR_FLOOR,
        "wall": COLOR_WALL,
        "object": COLOR_OBJECT,
    }

    cubes: list[dict] = []
    for b, c_webxr in zip(boxes, centres_webxr):
        R_in = _quat_to_R(*b.quat_xyzw)
        R_out = _T_WEBXR_FROM_RTAB @ R_in @ _T_WEBXR_FROM_RTAB.T
        q = _R_to_quat_xyzw(R_out)
        center_out = (c_webxr + np.array([0.0, y_offset, 0.0])).tolist()
        cubes.append({
            "center": tuple(center_out),
            "size": b.extent,
            "quat": q,
            "color": label_color.get(b.label, 0xCCCCCC),
        })
    return cubes


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = qx*qx + qy*qy + qz*qz + qw*qw
    s = 2.0 / n if n > 0 else 0.0
    return np.array([
        [1 - s*(qy*qy + qz*qz), s*(qx*qy - qz*qw),     s*(qx*qz + qy*qw)],
        [s*(qx*qy + qz*qw),     1 - s*(qx*qx + qz*qz), s*(qy*qz - qx*qw)],
        [s*(qx*qz - qy*qw),     s*(qy*qz + qx*qw),     1 - s*(qx*qx + qy*qy)],
    ])
