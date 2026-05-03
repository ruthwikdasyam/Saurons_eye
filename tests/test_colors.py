"""
Tests for axis-based per-point RGB colouring via matplotlib colormaps.

These tests don't touch Rerun — they validate the pure-numpy / pure-LUT
helpers.
"""

from __future__ import annotations

import numpy as np

from capture.viz import axis_colors, get_colormap_lut


def test_lut_shape_and_dtype() -> None:
    lut = get_colormap_lut("turbo")
    assert lut.shape == (256, 3)
    assert lut.dtype == np.uint8


def test_lut_endpoints_distinct() -> None:
    """First and last colormap entries should be very different (not a constant LUT)."""
    lut = get_colormap_lut("turbo")
    diff = int(np.abs(lut[0].astype(int) - lut[-1].astype(int)).sum())
    assert diff > 100


def test_lut_caching_returns_same_object() -> None:
    """Same name → same array (lru_cache); different name → different array."""
    a = get_colormap_lut("turbo")
    b = get_colormap_lut("turbo")
    c = get_colormap_lut("viridis")
    assert a is b
    assert a is not c


def test_colors_empty() -> None:
    rgb = axis_colors(np.empty((0, 3), dtype=np.float32))
    assert rgb.shape == (0, 3)
    assert rgb.dtype == np.uint8


def test_colors_endpoints_span_lut() -> None:
    """Min value picks the first LUT entry; max picks the last."""
    xyz = np.array([
        [0.0, -1.0, 0.0],
        [0.0,  1.0, 0.0],
    ], dtype=np.float32)
    rgb = axis_colors(xyz, axis=1, colormap="turbo")
    lut = get_colormap_lut("turbo")
    np.testing.assert_array_equal(rgb[0], lut[0])
    np.testing.assert_array_equal(rgb[1], lut[255])


def test_colors_constant_axis() -> None:
    """All Y equal → all rows the same colour (middle of LUT)."""
    xyz = np.full((10, 3), 1.5, dtype=np.float32)
    rgb = axis_colors(xyz, axis=1, colormap="turbo")
    lut = get_colormap_lut("turbo")
    assert rgb.shape == (10, 3)
    for row in rgb:
        np.testing.assert_array_equal(row, lut[128])


def test_colors_axis_selection() -> None:
    """axis=0 should produce a gradient when X varies; axis=1 constant when Y is constant."""
    xyz = np.array([
        [-1.0, 0.0, 5.0],
        [ 1.0, 0.0, 5.0],
        [ 0.0, 0.0, 5.0],
    ], dtype=np.float32)
    rgb_x = axis_colors(xyz, axis=0)
    rgb_y = axis_colors(xyz, axis=1)
    # X varies → endpoint colours differ
    assert not np.array_equal(rgb_x[0], rgb_x[1])
    # Y is constant → all rows the same colour
    np.testing.assert_array_equal(rgb_y[0], rgb_y[1])
    np.testing.assert_array_equal(rgb_y[0], rgb_y[2])


def test_colors_dtype_always_uint8() -> None:
    for dtype in (np.float32, np.float64):
        xyz = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=dtype)
        rgb = axis_colors(xyz, axis=1)
        assert rgb.dtype == np.uint8
