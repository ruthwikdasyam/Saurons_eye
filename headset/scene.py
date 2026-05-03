"""
Random wireframe scene generator for the headset viewer.

Produces a list of cubes positioned in front of the WebXR ``local-floor``
reference space (y=0 at floor, +x right, -z forward, units = metres).
The scene is serialised to JSON and pushed over WebSocket to the Quest
client at ``headset/web/index.html``.

The generated scene is a stand-in. Once the SLAM pipeline emits geometry,
swap ``random_cubes()`` for a function that converts the live map into
the same Cube list — the wire format and renderer don't change.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Cube:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    quat: tuple[float, float, float, float]  # (x, y, z, w)
    color: int = 0x00FF88                    # 0xRRGGBB


def _random_quat(rng: random.Random) -> tuple[float, float, float, float]:
    # Shoemake — uniform unit-quaternion sampling.
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    s1, s2 = math.sqrt(1.0 - u1), math.sqrt(u1)
    return (
        s1 * math.sin(2.0 * math.pi * u2),
        s1 * math.cos(2.0 * math.pi * u2),
        s2 * math.sin(2.0 * math.pi * u3),
        s2 * math.cos(2.0 * math.pi * u3),
    )


def random_cubes(
    n: int = 12,
    seed: int | None = None,
    x_range: tuple[float, float] = (-1.5, 1.5),
    y_range: tuple[float, float] = (0.4, 2.2),
    z_range: tuple[float, float] = (-3.0, -0.6),
    size_range: tuple[float, float] = (0.15, 0.5),
) -> list[Cube]:
    rng = random.Random(seed)
    cubes: list[Cube] = []
    for _ in range(n):
        center = (
            rng.uniform(*x_range),
            rng.uniform(*y_range),
            rng.uniform(*z_range),
        )
        s = rng.uniform(*size_range)
        size = (
            s * rng.uniform(0.7, 1.3),
            s * rng.uniform(0.7, 1.3),
            s * rng.uniform(0.7, 1.3),
        )
        cubes.append(Cube(center=center, size=size, quat=_random_quat(rng)))
    return cubes


def to_message(cubes: list[Cube]) -> dict[str, Any]:
    return {"type": "scene", "cubes": [asdict(c) for c in cubes]}
