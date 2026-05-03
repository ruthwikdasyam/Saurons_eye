"""
Rerun visualisation helpers. Thin wrappers around rr.log so the main loop
isn't littered with archetype calls.

Colouring: per-point RGB via a matplotlib LUT (turbo, viridis, …).

Aside: the dimos pattern of ``class_ids`` + ``AnnotationContext`` is
better in principle (1 byte per voxel, GPU-side colour lookup) but is
blocked on this stack — Rerun 0.23 calls ``np.asarray(x, copy=...)``
inside its AnnotationContext path, which only exists in numpy>=2.0,
and Open3D 0.19 hasn't been validated on numpy 2.0. Bandwidth at our
voxel counts (~20 K) makes the 3-byte path cost ~40 KB/s extra over
the wire, which is negligible. Worth revisiting if we adopt GPU TSDF
or scale to 100 K+ voxels.
"""

from __future__ import annotations

import functools

import matplotlib  # type: ignore[import-untyped]
import numpy as np
import open3d as o3d
import rerun as rr


def init(name: str = "saurons-eye") -> None:
    rr.init(name, spawn=True)
    # OpenCV / RealSense convention: +X right, +Y down, +Z forward.
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    # Wipe stale colour components (latest-at semantics would otherwise
    # carry colour over from earlier code/runs at the same entity path).
    rr.log("world/cloud", rr.Clear(recursive=False))


def log_camera_intrinsics(intr: o3d.camera.PinholeCameraIntrinsic) -> None:
    K = intr.intrinsic_matrix
    rr.log(
        "world/camera",
        rr.Pinhole(
            resolution=[intr.width, intr.height],
            focal_length=[K[0, 0], K[1, 1]],
            principal_point=[K[0, 2], K[1, 2]],
        ),
        static=True,
    )


def log_pose(T_WC: np.ndarray, frame_idx: int) -> None:
    rr.set_time("frame", sequence=frame_idx)
    rr.log(
        "world/camera",
        rr.Transform3D(translation=T_WC[:3, 3], mat3x3=T_WC[:3, :3]),
    )


def log_voxels(
    xyz: np.ndarray,
    frame_idx: int,
    radius: float = 0.04,
    colors: np.ndarray | None = None,
) -> None:
    """Log a voxel cloud as uniform spheres with optional per-point RGB."""
    rr.set_time("frame", sequence=frame_idx)
    if colors is not None:
        rr.log(
            "world/cloud",
            rr.Points3D(positions=xyz, radii=radius, colors=colors),
        )
    else:
        rr.log("world/cloud", rr.Points3D(positions=xyz, radii=radius))


@functools.lru_cache(maxsize=16)
def get_colormap_lut(name: str = "turbo") -> np.ndarray:
    """Build a 256-entry uint8 LUT from a matplotlib colormap (one-time cost)."""
    cmap = matplotlib.colormaps[name]
    t = np.linspace(0, 1, 256)
    return (cmap(t)[:, :3] * 255).astype(np.uint8)


def axis_colors(
    xyz: np.ndarray,
    axis: int = 1,
    colormap: str = "turbo",
) -> np.ndarray:
    """Per-point uint8 RGB from a matplotlib colormap, gradient along ``axis``.

    Default axis=1 (Y) corresponds to "height" in our RDF world frame.
    Empty input → empty array of shape (0, 3). Constant input → uniform
    middle-of-LUT colour for all points.
    """
    if xyz.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    v = xyz[:, axis]
    lo = float(v.min())
    hi = float(v.max())
    lut = get_colormap_lut(colormap)
    if hi - lo < 1e-6:
        return np.broadcast_to(lut[128], (xyz.shape[0], 3)).copy()
    t = (v - lo) / (hi - lo)
    indices = (t * 255).astype(np.uint8)
    return lut[indices]
