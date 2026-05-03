"""
Smoke test for RtabmapPose. Run this while RTAB-Map is publishing
on /rtabmap/odom.

Usage (in a fresh terminal, with the camera + rtabmap launches still running):
    source /opt/ros/humble/setup.bash
    source .venv/bin/activate
    python -m capture._smoke_rtabmap_pose

Expected: prints per-second pose lines for 15 seconds. The pose should
change as you move the camera. If it stays at identity / "no poses
received," RTAB-Map isn't publishing or the topic name is wrong.
"""

from __future__ import annotations

import time

import numpy as np

from capture.pose_rtabmap import RtabmapPose


def main() -> None:
    print("Connecting to /rtabmap/odom (5s timeout for first pose)...")
    pose = RtabmapPose()
    print(f"Init done; buffer has {len(pose._buffer)} pose(s).\n")

    print("  t                pos [m]                          rot (deg about Z)")
    print("  -----------      -----------------------------    --------")
    try:
        for _ in range(30):
            t = time.time()
            T = pose.update(None, t)
            p = T[:3, 3]
            R = T[:3, :3]
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            print(
                f"  {t:13.3f}    "
                f"[{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}]    "
                f"{yaw:+6.1f}°    "
                f"buffer={len(pose._buffer)}"
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        pose.shutdown()


if __name__ == "__main__":
    main()
