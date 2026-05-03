"""
Phase 2 — iterative RANSAC plane segmentation + floor / ceiling / wall classification.

Usage:
    python experiments/planes.py [path]

Defaults to the newest recordings/*_aligned_cleaned.ply. Cleaned + aligned
input is REQUIRED (otherwise classification by normal direction is meaningless).
Run experiments/clean.py first if needed.

Prints a summary, opens an Open3D viewer with each plane coloured by category.
Saves nothing (yet) — wireframe edges in the next phase will consume this output.
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass

import numpy as np
import open3d as o3d


# RANSAC params
DISTANCE_THRESHOLD = 0.05      # m. Inlier band thickness.
MIN_INLIERS_HORIZ = 2000       # Min size for floor / ceiling / tabletop.
MIN_INLIERS_WALL = 800         # Walls are usually smaller than the floor.
MAX_HORIZONTAL_PLANES = 10
MAX_WALL_PLANES = 10
MAX_ITERATIONS_PER_PASS = 60   # Keep searching past rejections until accepted plane found.
RANSAC_N = 3
NUM_ITERATIONS = 1000

# Normal-direction thresholds (cloud must already be gravity-aligned, +Z = up)
HORIZONTAL_NZ_THRESHOLD = 0.85   # |n.z| > this  → floor / ceiling (was 0.95 — too strict for tilted floors)
VERTICAL_NZ_THRESHOLD = 0.40     # |n.z| < this  → wall (was 0.20 — accepts up to ~24° tilt now)

# Display colours
CATEGORY_COLOR = {
    "floor":    [0.20, 0.40, 0.95],
    "ceiling":  [0.95, 0.30, 0.30],
    "wall":     [0.20, 0.85, 0.75],
    "other":    [0.95, 0.65, 0.20],
    "leftover": [0.40, 0.40, 0.40],
}


@dataclass
class Plane:
    category: str
    normal: np.ndarray                # unit 3-vector
    offset: float                     # plane equation: n . p + d = 0
    inlier_pts: np.ndarray            # (M, 3)


def newest_input() -> str:
    cleaned_aligned = sorted(glob.glob("recordings/*_aligned_cleaned.ply"))
    if cleaned_aligned:
        return cleaned_aligned[-1]
    aligned = sorted(glob.glob("recordings/*_aligned.ply"))
    if aligned:
        print("WARNING: no _aligned_cleaned.ply found — using _aligned.ply directly.")
        print("         Run experiments/clean.py first for better results.")
        return aligned[-1]
    raise SystemExit("No suitable recordings found. Run experiments/view_ply.py --align --save-aligned, then experiments/clean.py.")


def classify(normal: np.ndarray, inlier_z_mean: float, cloud_z_mid: float) -> str:
    nz = abs(normal[2])
    if nz > HORIZONTAL_NZ_THRESHOLD:
        return "floor" if inlier_z_mean < cloud_z_mid else "ceiling"
    if nz < VERTICAL_NZ_THRESHOLD:
        return "wall"
    return "other"


def _ransac_pass(remaining, accept, max_accepted, min_inliers, name):
    """Iteratively RANSAC, keeping only planes whose normal passes `accept(n)`.

    Loop ends when:
      - `max_accepted` accepted planes have been kept, OR
      - `MAX_ITERATIONS_PER_PASS` total iterations done (covers many rejections), OR
      - the remaining cloud has fewer than `min_inliers` points.

    Rejections do NOT count against the accept cap — we keep trying past them.
    Inliers for both accepted and rejected planes are removed from `remaining`,
    so RANSAC doesn't keep re-finding the same surface.
    """
    planes: list[Plane] = []
    rejects = []
    for it in range(MAX_ITERATIONS_PER_PASS):
        if len(planes) >= max_accepted:
            break
        if len(remaining.points) < min_inliers:
            print(f"  stop: only {len(remaining.points):,} pts remain (< min_inliers={min_inliers})")
            break
        model, inliers = remaining.segment_plane(
            distance_threshold=DISTANCE_THRESHOLD,
            ransac_n=RANSAC_N,
            num_iterations=NUM_ITERATIONS,
        )
        if len(inliers) < min_inliers:
            print(f"  stop: largest plane only {len(inliers):,} inliers (< {min_inliers})")
            break

        a, b, c, d = model
        n = np.array([a, b, c])
        n /= np.linalg.norm(n)
        if n[2] < 0 and abs(n[2]) > HORIZONTAL_NZ_THRESHOLD:
            n, d = -n, -d

        inlier_pts = np.asarray(remaining.points)[inliers]
        if accept(n):
            planes.append(Plane(category=name, normal=n, offset=float(d), inlier_pts=inlier_pts))
            print(f"  {name:8s} ✓  n=[{n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f}]  "
                  f"d={d:+.2f}  inliers={len(inliers):,}  |n.z|={abs(n[2]):.2f}")
        else:
            rejects.append((n, len(inliers)))
            print(f"  {name:8s} ✗  n=[{n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f}]  "
                  f"|n.z|={abs(n[2]):.2f}  inliers={len(inliers):,}  (filter rejected)")
        remaining = remaining.select_by_index(inliers, invert=True)
    print(f"  → {len(planes)} accepted, {len(rejects)} rejected, {len(remaining.points):,} pts remain")
    return planes, remaining


def segment_planes(pcd: o3d.geometry.PointCloud) -> tuple[list[Plane], o3d.geometry.PointCloud]:
    """Stratified RANSAC: find horizontal planes (floor/ceiling) first, then walls.

    Without this stratification, RANSAC greedily eats every horizontal stripe
    (tabletops, counters, chair seats) before getting to walls.
    """
    xyz = np.asarray(pcd.points)
    cloud_z_mid = (xyz[:, 2].min() + xyz[:, 2].max()) / 2
    all_planes: list[Plane] = []

    print("pass 1: horizontal planes (floor / ceiling / tabletops)")
    horizontal_planes, remaining = _ransac_pass(
        pcd,
        accept=lambda n: abs(n[2]) > HORIZONTAL_NZ_THRESHOLD,
        max_accepted=MAX_HORIZONTAL_PLANES,
        min_inliers=MIN_INLIERS_HORIZ,
        name="horiz",
    )
    # Reclassify: lowest horizontal = floor, highest = ceiling, rest = "other".
    if horizontal_planes:
        zs = [pl.inlier_pts[:, 2].mean() for pl in horizontal_planes]
        i_floor = int(np.argmin(zs))
        i_ceiling = int(np.argmax(zs))
        for i, pl in enumerate(horizontal_planes):
            if i == i_floor:
                pl.category = "floor"
            elif i == i_ceiling and zs[i] - zs[i_floor] > 1.5:   # at least 1.5 m above floor
                pl.category = "ceiling"
            else:
                pl.category = "other"   # tabletop, counter, chair seat, drift slice
    all_planes.extend(horizontal_planes)
    print(f"  → {len(remaining.points):,} pts remain for wall search")
    print()

    print("pass 2: vertical planes (walls)")
    wall_planes, remaining = _ransac_pass(
        remaining,
        accept=lambda n: abs(n[2]) < VERTICAL_NZ_THRESHOLD,
        max_accepted=MAX_WALL_PLANES,
        min_inliers=MIN_INLIERS_WALL,
        name="wall",
    )
    for pl in wall_planes:
        pl.category = "wall"
    all_planes.extend(wall_planes)

    return all_planes, remaining


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default=None)
    args = p.parse_args()

    path = args.path or newest_input()
    print(f"loading {path}")
    pcd = o3d.io.read_point_cloud(path)
    print(f"  points: {len(pcd.points):,}")
    print()
    print(f"segmenting planes (threshold={DISTANCE_THRESHOLD}m, "
          f"min_inliers horiz={MIN_INLIERS_HORIZ} wall={MIN_INLIERS_WALL}, "
          f"max horiz={MAX_HORIZONTAL_PLANES} wall={MAX_WALL_PLANES})...")
    planes, leftover = segment_planes(pcd)
    print()

    by_cat: dict[str, list[Plane]] = {}
    for pl in planes:
        by_cat.setdefault(pl.category, []).append(pl)
    print(f"summary: {len(planes)} planes, {len(leftover.points):,} leftover (objects + noise)")
    for cat in ("floor", "ceiling", "wall", "other"):
        n = len(by_cat.get(cat, []))
        pts = sum(len(p.inlier_pts) for p in by_cat.get(cat, []))
        print(f"  {cat:8s}: {n} plane(s), {pts:,} pts")

    # Visualise
    geometries = []
    for pl in planes:
        plane_pcd = o3d.geometry.PointCloud()
        plane_pcd.points = o3d.utility.Vector3dVector(pl.inlier_pts)
        plane_pcd.paint_uniform_color(CATEGORY_COLOR[pl.category])
        geometries.append(plane_pcd)
    if len(leftover.points):
        leftover.paint_uniform_color(CATEGORY_COLOR["leftover"])
        geometries.append(leftover)
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]))

    print("\nlegend: floor=blue  ceiling=red  wall=cyan  other=orange  leftover=gray")
    o3d.visualization.draw_geometries(geometries, window_name=f"planes — {path}")


if __name__ == "__main__":
    main()
