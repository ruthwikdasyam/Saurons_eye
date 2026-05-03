"""Standalone YOLO smoke test for the drone-side detection path.

Modes:
    --source webcam        OpenCV VideoCapture (Mac dev loop, no depth -> 2D only)
    --source realsense     RealSense D435i with aligned depth -> 2D + 3D bboxes
    --source path/to.mp4   Video file (2D only)
    --source path/to.jpg   Single image (2D only)

Examples:
    python test_detect.py                              # webcam, yolov8n
    python test_detect.py --source realsense           # full 2D+3D pipeline
    python test_detect.py --source clip.mp4 --model yolov8s.pt
    python test_detect.py --source frame.jpg --save out.jpg --headless

Press q to quit the live window.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect import Detector, draw_detections  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="webcam", help="webcam | realsense | path to image/video")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--all-classes", action="store_true", help="don't filter to room-clearing class subset")
    p.add_argument("--device", default=None, help="cuda:0 | mps | cpu (default: ultralytics autoselect)")
    p.add_argument("--save", default=None, help="path to save the last/only annotated frame")
    p.add_argument("--headless", action="store_true", help="don't open a window (CI/SSH)")
    p.add_argument("--max-frames", type=int, default=0, help="0 = run until q / EOF")
    return p.parse_args()


def run_static_image(detector: Detector, path: str, save: str | None, headless: bool) -> None:
    import cv2

    bgr = cv2.imread(path)
    if bgr is None:
        sys.exit(f"could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t0 = time.time()
    dets = detector.infer(rgb)
    dt = (time.time() - t0) * 1000
    print(f"[image] {len(dets)} detections in {dt:.1f}ms")
    for d in dets:
        print(f"  {d.cls_name:12s} conf={d.conf:.2f} xyxy={tuple(round(v, 1) for v in d.xyxy)}")
    annotated = draw_detections(rgb, dets)
    out_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    if save:
        cv2.imwrite(save, out_bgr)
        print(f"saved {save}")
    if not headless:
        cv2.imshow("detect", out_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video_stream(detector: Detector, source, max_frames: int, save: str | None, headless: bool) -> None:
    import cv2

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"could not open video source: {source}")

    n = 0
    last = None
    fps_ema = None
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t0 = time.time()
            dets = detector.infer(rgb)
            dt = time.time() - t0
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_ema = inst_fps if fps_ema is None else 0.9 * fps_ema + 0.1 * inst_fps

            annotated = draw_detections(rgb, dets)
            out_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.putText(
                out_bgr, f"{fps_ema:5.1f} FPS  n={len(dets)}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )
            last = out_bgr

            if not headless:
                cv2.imshow("detect", out_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            n += 1
            if max_frames and n >= max_frames:
                break
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

    if save and last is not None:
        cv2.imwrite(save, last)
        print(f"saved {save}")
    print(f"processed {n} frames")


def run_realsense(detector: Detector, max_frames: int, save: str | None, headless: bool) -> None:
    try:
        import pyrealsense2 as rs
    except ImportError:
        sys.exit("pyrealsense2 not installed. pip install pyrealsense2")
    import cv2

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # meters per unit (typically 0.001)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    K = np.array(
        [[intr.fx, 0.0, intr.ppx],
         [0.0, intr.fy, intr.ppy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    print(f"RealSense ready. depth_scale={depth_scale:.5f} m/unit  K=\n{K}")

    align = rs.align(rs.stream.color)

    n = 0
    last = None
    fps_ema = None
    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            bgr = np.asarray(color_frame.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth_raw = np.asarray(depth_frame.get_data())  # uint16

            t0 = time.time()
            dets2d = detector.infer(rgb)
            dets3d = detector.infer_3d(rgb, depth_raw, K, depth_scale=depth_scale) if dets2d else []
            dt = time.time() - t0
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_ema = inst_fps if fps_ema is None else 0.9 * fps_ema + 0.1 * inst_fps

            for d3 in dets3d:
                (xmin, ymin, zmin), (xmax, ymax, zmax) = d3.bbox3d
                print(
                    f"  {d3.cls_name:10s} conf={d3.conf:.2f} "
                    f"x[{xmin:+.2f},{xmax:+.2f}] y[{ymin:+.2f},{ymax:+.2f}] z[{zmin:.2f},{zmax:.2f}] (m)"
                )

            annotated = draw_detections(rgb, dets2d, dets3d)
            out_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.putText(
                out_bgr, f"{fps_ema:5.1f} FPS  2d={len(dets2d)} 3d={len(dets3d)}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )
            last = out_bgr

            if not headless:
                cv2.imshow("detect (RealSense)", out_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            n += 1
            if max_frames and n >= max_frames:
                break
    finally:
        pipeline.stop()
        if not headless:
            cv2.destroyAllWindows()

    if save and last is not None:
        cv2.imwrite(save, last)
        print(f"saved {save}")
    print(f"processed {n} frames")


def main() -> None:
    args = parse_args()

    from detect import DEFAULT_CLASSES
    detector = Detector(
        model=args.model,
        conf=args.conf,
        classes=None if args.all_classes else DEFAULT_CLASSES,
        device=args.device,
    )

    src = args.source
    if src == "webcam":
        run_video_stream(detector, 0, args.max_frames, args.save, args.headless)
    elif src == "realsense":
        run_realsense(detector, args.max_frames, args.save, args.headless)
    elif os.path.isfile(src):
        ext = Path(src).suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            run_static_image(detector, src, args.save, args.headless)
        else:
            run_video_stream(detector, src, args.max_frames, args.save, args.headless)
    else:
        sys.exit(f"unknown source: {src}")


if __name__ == "__main__":
    main()
