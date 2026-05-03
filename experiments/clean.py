"""
Phase 1 — cleaning. Voxel downsample + statistical outlier removal.

Usage:
    python experiments/clean.py [path]

Defaults to the newest recordings/*_aligned.ply (else final_*.ply).
Output: writes <path>_cleaned.ply next to the input.

Tunable knobs at the top of the file.
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import open3d as o3d


VOXEL_SIZE = 0.05          # m. Downsample to uniform density.
SOR_NEIGHBORS = 20         # Statistical outlier removal: required neighbours.
SOR_STD_RATIO = 2.0        # Higher = lenient. Lower = aggressive.


def newest_input() -> str:
    aligned = sorted(glob.glob("recordings/*_aligned.ply"))
    if aligned:
        return aligned[-1]
    final = sorted(glob.glob("recordings/final_*.ply"))
    if final:
        return final[-1]
    raise SystemExit("No recordings found.")


def clean(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    if len(pcd.points) >= SOR_NEIGHBORS:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=SOR_NEIGHBORS, std_ratio=SOR_STD_RATIO,
        )
    return pcd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default=None)
    args = p.parse_args()

    path = args.path or newest_input()
    print(f"loading {path}")
    pcd = o3d.io.read_point_cloud(path)
    n_in = len(pcd.points)
    print(f"  input: {n_in:,} points")

    pcd = clean(pcd)
    n_out = len(pcd.points)
    print(f"  after voxel ({VOXEL_SIZE}m) + SOR ({SOR_NEIGHBORS} nbrs / {SOR_STD_RATIO}σ): {n_out:,} points")
    print(f"  reduction: {(1 - n_out / max(n_in, 1)) * 100:.1f}%")

    xyz = np.asarray(pcd.points)
    print()
    print(f"  bbox X: [{xyz[:,0].min():+.2f}, {xyz[:,0].max():+.2f}]  range={xyz[:,0].ptp():.2f}m")
    print(f"  bbox Y: [{xyz[:,1].min():+.2f}, {xyz[:,1].max():+.2f}]  range={xyz[:,1].ptp():.2f}m")
    print(f"  bbox Z: [{xyz[:,2].min():+.2f}, {xyz[:,2].max():+.2f}]  range={xyz[:,2].ptp():.2f}m")

    out = path.replace(".ply", "_cleaned.ply")
    o3d.io.write_point_cloud(out, pcd)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
