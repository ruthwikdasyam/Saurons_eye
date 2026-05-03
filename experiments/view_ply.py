"""
View a saved .ply with axis-based or floor-relative colouring,
optionally first rotating it so gravity = +Z.

Usage:
    python experiments/view_ply.py [path] [--axis 0|1|2] [--floor] [--align] [--save-aligned]

Defaults to the newest recordings/final_*.ply, axis=2 (Z).

Why this is fiddly: RTAB-Map's world frame inherits the camera_link orientation
at the first frame. Without IMU / gravity reference, "up" in the saved cloud
is whatever direction was up when you pressed record — usually tilted.

  --axis 0|1|2     colour by X / Y / Z
  --floor          colour by perpendicular distance from RANSAC-detected floor
  --align          first rotate cloud so the floor normal becomes +Z (gravity-up)
  --save-aligned   if --align, also write <path>_aligned.ply for reuse

Combine: `--align --axis 2` is the "fixed" version of the default coloring.
"""

from __future__ import annotations

import argparse
import copy
import glob
import os

import matplotlib  # noqa: F401  -- pulled in for the colormap
import numpy as np
import open3d as o3d


COLORMAP = "turbo"


def latest_recording() -> str:
    files = sorted(glob.glob("recordings/final_*.ply"))
    if not files:
        raise SystemExit("No recordings/final_*.ply found.")
    return files[-1]


def detect_floor_plane(pcd: o3d.geometry.PointCloud):
    """RANSAC the largest plane (assumed floor). Returns (n, d, inliers)."""
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.05, ransac_n=3, num_iterations=1000,
    )
    a, b, c, d = plane_model
    n = np.array([a, b, c])
    n /= np.linalg.norm(n)
    # Point n upward: make sure the bulk of the cloud sits on the +n side of the plane.
    xyz = np.asarray(pcd.points)
    signed = xyz @ n + d
    if np.median(signed) < 0:
        n = -n
        d = -d
    return n, float(d), inliers


def gravity_align(pcd: o3d.geometry.PointCloud) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Rotate cloud so the floor normal becomes +Z and the floor sits at z≈0.

    Returns: (aligned_pcd, T) where T is the 4x4 applied transform.
    """
    n, d, inliers = detect_floor_plane(pcd)
    print(f"  floor normal (orig frame): {n.round(3)}")
    print(f"  floor inliers:             {len(inliers):,} / {len(pcd.points):,}")

    target = np.array([0.0, 0.0, 1.0])
    if np.allclose(n, target):
        R = np.eye(3)
    elif np.allclose(n, -target):
        R = np.diag([1.0, -1.0, -1.0])
    else:
        # Rodrigues' formula: rotation that takes n → +Z
        axis = np.cross(n, target)
        axis /= np.linalg.norm(axis)
        angle = float(np.arccos(np.clip(np.dot(n, target), -1.0, 1.0)))
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    # Translate so the floor (post-rotation) sits at Z=0.
    inlier_pts = np.asarray(pcd.points)[inliers]
    rotated_z = (inlier_pts @ R.T)[:, 2]
    floor_z = float(np.median(rotated_z))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [0.0, 0.0, -floor_z]

    aligned = copy.deepcopy(pcd).transform(T)
    print(f"  applied rotation, then translated by Δz = {-floor_z:+.3f} m")
    return aligned, T


def colour_by_axis(xyz: np.ndarray, axis: int) -> np.ndarray:
    v = xyz[:, axis]
    lo, hi = np.percentile(v, [2, 98])
    t = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    return matplotlib.colormaps[COLORMAP](t)[:, :3]


def colour_by_floor(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    n, d, _ = detect_floor_plane(pcd)
    xyz = np.asarray(pcd.points)
    signed = xyz @ n + d
    lo, hi = np.percentile(signed, [2, 98])
    t = np.clip((signed - lo) / max(hi - lo, 1e-6), 0, 1)
    return matplotlib.colormaps[COLORMAP](t)[:, :3]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default=None)
    p.add_argument("--axis", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--floor", action="store_true",
                   help="Colour by distance from RANSAC-detected floor plane.")
    p.add_argument("--align", action="store_true",
                   help="Rotate cloud so the floor normal becomes +Z before viewing.")
    p.add_argument("--save-aligned", action="store_true",
                   help="If --align, also write <path>_aligned.ply for reuse.")
    args = p.parse_args()

    path = args.path or latest_recording()
    print(f"loading {path}")
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points)
    print(f"  points: {xyz.shape[0]:,}")
    print(f"  bbox X: [{xyz[:,0].min():+.2f}, {xyz[:,0].max():+.2f}]  range={xyz[:,0].ptp():.2f}m")
    print(f"  bbox Y: [{xyz[:,1].min():+.2f}, {xyz[:,1].max():+.2f}]  range={xyz[:,1].ptp():.2f}m")
    print(f"  bbox Z: [{xyz[:,2].min():+.2f}, {xyz[:,2].max():+.2f}]  range={xyz[:,2].ptp():.2f}m")

    if args.align:
        print("gravity-aligning cloud (floor → XY plane, +Z = up)")
        pcd, T = gravity_align(pcd)
        xyz = np.asarray(pcd.points)
        print(f"  bbox X: [{xyz[:,0].min():+.2f}, {xyz[:,0].max():+.2f}]  range={xyz[:,0].ptp():.2f}m")
        print(f"  bbox Y: [{xyz[:,1].min():+.2f}, {xyz[:,1].max():+.2f}]  range={xyz[:,1].ptp():.2f}m")
        print(f"  bbox Z: [{xyz[:,2].min():+.2f}, {xyz[:,2].max():+.2f}]  range={xyz[:,2].ptp():.2f}m")
        if args.save_aligned:
            out = path.replace(".ply", "_aligned.ply")
            o3d.io.write_point_cloud(out, pcd)
            print(f"  wrote aligned cloud → {out}")

    if args.floor:
        print("colouring by distance from RANSAC floor plane")
        colors = colour_by_floor(pcd)
    else:
        print(f"colouring by axis {args.axis}")
        colors = colour_by_axis(xyz, args.axis)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    title = f"{os.path.basename(path)} — "
    if args.align: title += "aligned, "
    title += "floor-distance" if args.floor else f"axis {args.axis}"
    title += f" ({COLORMAP})"
    o3d.visualization.draw_geometries([pcd, axes], window_name=title)


if __name__ == "__main__":
    main()
