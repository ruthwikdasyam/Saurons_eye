"""
Phase 3 — extract wall wireframes by clustering wall-normal points.

When global RANSAC fails (walls too fragmented), this works instead:
  1. Compute per-point normals.
  2. Keep points with horizontal normal (|n.z| < 0.30 = vertical surface).
  3. DBSCAN cluster them spatially.
  4. Fit oriented bounding box per cluster, render as wireframe.

Each box should correspond to one (potentially fragmented) wall.

Usage:
    python experiments/walls.py [path]
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import open3d as o3d


# Normal estimation
NORMAL_RADIUS = 0.10
NORMAL_K = 30

# Wall classification
WALL_NZ_THRESHOLD = 0.30   # |n.z| < this  → wall-like

# Clustering
DBSCAN_EPS = 0.20          # m. Max distance between neighbours in a cluster.
DBSCAN_MIN_POINTS = 100    # Min cluster size to call it a wall.

OBB_LINE_COLOR = [0.20, 0.85, 0.75]


def newest_input() -> str:
    for pattern in ("recordings/*_aligned_cleaned.ply", "recordings/*_aligned.ply"):
        files = sorted(glob.glob(pattern))
        if files:
            return files[-1]
    raise SystemExit("No suitable recordings found.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default=None)
    args = p.parse_args()

    path = args.path or newest_input()
    print(f"loading {path}")
    pcd = o3d.io.read_point_cloud(path)
    print(f"  points: {len(pcd.points):,}")

    print(f"\nestimating normals (radius={NORMAL_RADIUS}m, k={NORMAL_K})...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=NORMAL_K,
        )
    )

    normals = np.asarray(pcd.normals)
    nz = np.abs(normals[:, 2])
    wall_mask = nz < WALL_NZ_THRESHOLD
    n_wall = int(wall_mask.sum())
    print(f"wall-normal points (|n.z| < {WALL_NZ_THRESHOLD}): {n_wall:,}")

    if n_wall < DBSCAN_MIN_POINTS:
        raise SystemExit(f"Too few wall-normal points to cluster.")

    wall_pcd = pcd.select_by_index(np.where(wall_mask)[0])

    print(f"\nDBSCAN clustering (eps={DBSCAN_EPS}m, min_points={DBSCAN_MIN_POINTS})...")
    labels = np.array(wall_pcd.cluster_dbscan(
        eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=False
    ))
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    n_noise = int((labels == -1).sum())
    print(f"  found {n_clusters} clusters, {n_noise:,} noise points")

    if n_clusters == 0:
        print("⚠ No clusters survived. Try smaller eps or smaller min_points.")
        return

    # Colour each cluster a unique colour for visualisation.
    rng = np.random.default_rng(42)
    cluster_colors = rng.random((n_clusters, 3))
    point_colors = np.zeros((len(wall_pcd.points), 3))
    for label in range(n_clusters):
        point_colors[labels == label] = cluster_colors[label]
    point_colors[labels == -1] = [0.30, 0.30, 0.30]
    wall_pcd.colors = o3d.utility.Vector3dVector(point_colors)

    geometries = [wall_pcd]

    print()
    print("oriented bounding boxes per cluster:")
    obbs_kept = 0
    for label in range(n_clusters):
        idx = np.where(labels == label)[0]
        cluster = wall_pcd.select_by_index(idx)
        try:
            obb = cluster.get_oriented_bounding_box()
        except Exception as e:
            print(f"  cluster {label}: OBB failed ({e})")
            continue
        ext = obb.extent
        # Skip suspiciously tiny clusters (probably noise).
        if max(ext) < 0.3:
            continue
        ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
        ls.paint_uniform_color(OBB_LINE_COLOR)
        geometries.append(ls)
        obbs_kept += 1
        print(f"  cluster {label:>2}: {len(idx):>5,} pts  "
              f"OBB extent=[{ext[0]:.2f}, {ext[1]:.2f}, {ext[2]:.2f}] m")
    print(f"\n  → {obbs_kept} wall OBBs rendered")

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(axes)

    print("\nlegend: each cluster unique colour, dark gray = DBSCAN noise, cyan boxes = wall OBBs")
    o3d.visualization.draw_geometries(geometries, window_name=f"walls — {path}")


if __name__ == "__main__":
    main()
