"""
Diagnostic: are there ANY wall-oriented points in the cloud, regardless
of whether they form a fittable plane?

Computes per-point normals from local neighbours, classifies each point
by its normal direction:
  - normal ≈ ±Z (vertical normal)         → horizontal surface (floor / ceiling / table)
  - normal ⊥ Z   (horizontal normal)      → vertical surface (= WALL)
  - in between                            → slanted surface

Visualises the cloud coloured by class, so you can see at a glance
whether wall-oriented points exist and where.

If you see large coherent CYAN regions in the viewer, walls are in the
data and we just need a different algorithm to extract them. If wall
points are sparse / scattered, the scan needs to be redone with more
deliberate wall coverage.

Usage:
    python experiments/wall_diagnostic.py [path]
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import open3d as o3d


NORMAL_RADIUS = 0.10        # m. Neighbourhood for normal estimation.
NORMAL_K = 30               # max neighbours to use.

VERTICAL_NORMAL_THRESHOLD = 0.85   # |n.z| > this  → horizontal surface (floor-like)
HORIZONTAL_NORMAL_THRESHOLD = 0.30 # |n.z| < this  → vertical surface  (wall-like)


def newest_input() -> str:
    candidates = sorted(glob.glob("recordings/*_aligned_cleaned.ply"))
    if candidates:
        return candidates[-1]
    candidates = sorted(glob.glob("recordings/*_aligned.ply"))
    if candidates:
        return candidates[-1]
    raise SystemExit("No suitable recordings found.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default=None)
    args = p.parse_args()

    path = args.path or newest_input()
    print(f"loading {path}")
    pcd = o3d.io.read_point_cloud(path)
    n_pts = len(pcd.points)
    print(f"  points: {n_pts:,}")

    print(f"\nestimating normals (radius={NORMAL_RADIUS}m, k={NORMAL_K})...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=NORMAL_K,
        )
    )
    pcd.orient_normals_consistent_tangent_plane(NORMAL_K)

    normals = np.asarray(pcd.normals)
    nz = np.abs(normals[:, 2])

    horizontal_surface_mask = nz > VERTICAL_NORMAL_THRESHOLD     # floors / ceilings
    wall_surface_mask = nz < HORIZONTAL_NORMAL_THRESHOLD          # walls
    slanted_mask = ~(horizontal_surface_mask | wall_surface_mask)

    n_floor = int(horizontal_surface_mask.sum())
    n_wall = int(wall_surface_mask.sum())
    n_slant = int(slanted_mask.sum())
    print()
    print(f"point classification by normal:")
    print(f"  floor/ceiling/table  (|n.z| > {VERTICAL_NORMAL_THRESHOLD}): {n_floor:>7,}  ({100*n_floor/n_pts:.1f}%)")
    print(f"  WALL                 (|n.z| < {HORIZONTAL_NORMAL_THRESHOLD}): {n_wall:>7,}  ({100*n_wall/n_pts:.1f}%)")
    print(f"  slanted              (in between):                 {n_slant:>7,}  ({100*n_slant/n_pts:.1f}%)")
    print()

    if n_wall < 100:
        print("⚠ Almost no wall-normal points. The scan really doesn't contain walls.")
        print("  Re-scan: aim the camera at walls, hold ~3-5 sec each, walk slowly along them.")
    elif n_wall < n_floor / 50:
        print("⚠ Wall points are <2% of floor points. Walls captured very thinly.")
        print("  May still cluster, but rescanning will give much better results.")
    else:
        print("✓ Reasonable wall point count. They exist but RANSAC can't fit them as planes")
        print("  (likely scattered across multiple small wall sections).")
        print("  Next step: cluster wall-normal points spatially, fit OBBs.")

    # Visualise: colour each class.
    colors = np.zeros((n_pts, 3))
    colors[horizontal_surface_mask] = [0.20, 0.40, 0.95]   # blue  = floor/ceiling/table
    colors[wall_surface_mask]       = [0.20, 0.85, 0.75]   # cyan  = walls
    colors[slanted_mask]            = [0.50, 0.50, 0.50]   # gray  = slanted / unclassified
    pcd.colors = o3d.utility.Vector3dVector(colors)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    print("\nlegend: blue=floor/ceiling/table  cyan=WALL  gray=slanted")
    o3d.visualization.draw_geometries([pcd, axes], window_name=f"wall diagnostic — {path}")


if __name__ == "__main__":
    main()
