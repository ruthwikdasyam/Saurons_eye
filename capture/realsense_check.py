"""
Smoke test: verify the RealSense is connected and stream RGB + depth.

Usage:
    python capture/realsense_check.py

Press 'q' (or Esc) in the window to quit.
"""

import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> None:
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        raise SystemExit(
            "No RealSense device found.\n"
            "  - Check the USB-C cable is plugged into a USB 3.x port.\n"
            "  - Try `lsusb | grep Intel` to confirm the OS sees it.\n"
            "  - On Linux, librealsense udev rules may be missing."
        )

    print(f"Found {len(devices)} RealSense device(s):")
    usb = "?"
    for d in devices:
        usb = (
            d.get_info(rs.camera_info.usb_type_descriptor)
            if d.supports(rs.camera_info.usb_type_descriptor)
            else "?"
        )
        print(
            f"  {d.get_info(rs.camera_info.name)}"
            f"  S/N {d.get_info(rs.camera_info.serial_number)}"
            f"  FW {d.get_info(rs.camera_info.firmware_version)}"
            f"  USB {usb}"
        )

    # D435i can't carry dual 640x480@30 over USB 2.x — drop to 6 fps if needed.
    usb2 = usb.startswith("2.")
    fps = 6 if usb2 else 30
    if usb2:
        print(
            "WARNING: device is on USB 2.x. Falling back to 6 fps for both streams.\n"
            "         Use a USB 3.x (SuperSpeed / blue) port for full 30 fps."
        )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)

    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"Depth scale: {depth_scale} m/unit  (multiply raw uint16 by this for metres)")

    warmup = 6 if usb2 else 30
    print(f"Warming up ({warmup} frames @ {fps} fps)...", end=" ", flush=True)
    for _ in range(warmup):
        pipeline.try_wait_for_frames(timeout_ms=10000)
    print("ready. Press 'q' or Esc in the window to quit.")

    align = rs.align(rs.stream.color)
    drops = 0

    try:
        while True:
            ok, frames = pipeline.try_wait_for_frames(timeout_ms=2000)
            if not ok:
                drops += 1
                print(f"  [drop {drops}] no frame in 2s — continuing")
                continue
            frames = align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            color_img = np.asanyarray(color.get_data())
            depth_img = np.asanyarray(depth.get_data())

            # Visualise depth: clamp to ~5 m, then JET colormap.
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=255.0 / (5.0 / depth_scale)),
                cv2.COLORMAP_JET,
            )

            view = np.hstack((color_img, depth_vis))
            cv2.imshow("RealSense  |  color  |  depth (q to quit)", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
