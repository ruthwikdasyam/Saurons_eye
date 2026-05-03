"""Open-vocabulary object tracker.

Type a single phrase ("red backpack", "rifle", "person", "doorknob") and the
camera will only surface that object. For every frame, instances are sorted
by confidence (desc) and reported with:
    - 4 corners of the 2D bbox (TL, TR, BR, BL) in pixels
    - 2D center
    - confidence
    - 3D center + 3D bbox corners (RealSense source only)

Sources:
    --source webcam        OpenCV VideoCapture (default)
    --source realsense     D435i with aligned depth -> adds 3D coords
    --source path/to.mp4   video file
    --source path/to.jpg   single image

Output:
    Always prints to stdout.
    --json out.json        also append per-frame records to a JSON list file.

Examples:
    python track_object.py --object "person"
    python track_object.py --object "red backpack" --source realsense --json log.json
    python track_object.py --object "laptop" --source webcam --conf 0.1

    # Voice mode: speak the object name at startup; press 'v' in the viewer
    # window mid-stream to re-record and swap the query live.
    python track_object.py --voice
    python track_object.py --voice --source webcam --blur-others
    python track_object.py --voice --object "person" --source realsense   # 'person' is the initial query; 'v' swaps it
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--object", help='single phrase, e.g. "red backpack". required unless --voice is set')
    p.add_argument("--source", default="webcam", help="webcam | realsense | path to image/video")
    p.add_argument("--model", default="yolov8s-worldv2.pt", help="YOLO-World weights (auto-downloaded)")
    p.add_argument("--conf", type=float, default=0.05, help="open-vocab needs lower threshold than COCO YOLO")
    p.add_argument("--device", default=None, help="cuda:0 | mps | cpu")
    p.add_argument("--json", dest="json_path", default=None, help="append per-frame records here")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument(
        "--blur-others",
        action="store_true",
        help="blur everything outside the detected bbox(es); when no match, blur the whole frame",
    )
    p.add_argument(
        "--voice",
        action="store_true",
        help="speak the object name. Initial prompt at startup, then press 'v' "
             "in the viewer window to re-record and swap the query live.",
    )
    p.add_argument("--voice-duration", type=float, default=3.0, help="seconds to record per utterance")
    p.add_argument("--whisper-model", default="tiny.en", help="faster-whisper model size: tiny.en | base.en | small.en | …")
    return p.parse_args()


def load_world_model(model_path: str, query: str, device: str | None):
    from ultralytics import YOLOWorld

    model = YOLOWorld(model_path)
    model.set_classes([query])
    if device:
        model.to(device)
    return model


def corners_and_center(x1: float, y1: float, x2: float, y2: float):
    corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]  # TL, TR, BR, BL
    center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
    return corners, center


def project_3d(
    x1: float, y1: float, x2: float, y2: float,
    depth: np.ndarray, K: np.ndarray, depth_scale: float,
    depth_min: float = 0.2, depth_max: float = 8.0,
) -> dict | None:
    h, w = depth.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x1i = max(0, int(np.floor(x1)));  y1i = max(0, int(np.floor(y1)))
    x2i = min(w, int(np.ceil(x2)));   y2i = min(h, int(np.ceil(y2)))
    if x2i <= x1i or y2i <= y1i:
        return None
    patch = depth[y1i:y2i, x1i:x2i].astype(np.float32) * depth_scale
    valid = (patch >= depth_min) & (patch <= depth_max) & np.isfinite(patch)
    if not np.any(valid):
        return None

    zs = patch[valid]
    lo, hi = np.percentile(zs, [10, 90])
    ys, xs = np.where(valid)
    zs_full = patch[ys, xs]
    band = (zs_full >= lo) & (zs_full <= hi)
    if np.any(band):
        ys, xs, zs_full = ys[band], xs[band], zs_full[band]
    us = xs + x1i
    vs = ys + y1i
    X = (us - cx) * zs_full / fx
    Y = (vs - cy) * zs_full / fy
    Z = zs_full

    cx2d = (x1 + x2) / 2.0
    cy2d = (y1 + y2) / 2.0
    z_center = float(np.median(Z))
    X_center = (cx2d - cx) * z_center / fx
    Y_center = (cy2d - cy) * z_center / fy

    return {
        "center_3d": [X_center, Y_center, z_center],
        "bbox3d": [
            [float(X.min()), float(Y.min()), float(Z.min())],
            [float(X.max()), float(Y.max()), float(Z.max())],
        ],
    }


def boxes_from_result(r) -> list[tuple[float, tuple[float, float, float, float]]]:
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    pairs = [(float(c), (float(b[0]), float(b[1]), float(b[2]), float(b[3])))
             for b, c in zip(xyxy, confs)]
    pairs.sort(key=lambda p: p[0], reverse=True)
    return pairs


def emit_frame(
    frame_idx: int,
    query: str,
    pairs,
    depth_ctx: tuple[np.ndarray, np.ndarray, float] | None,
    json_records: list | None,
) -> dict:
    t = time.time()
    rec_dets = []
    for rank, (conf, (x1, y1, x2, y2)) in enumerate(pairs, start=1):
        corners, center = corners_and_center(x1, y1, x2, y2)
        det = {
            "rank": rank,
            "conf": round(conf, 4),
            "corners_2d": [[round(v, 1) for v in c] for c in corners],
            "center_2d": [round(v, 1) for v in center],
        }
        if depth_ctx is not None:
            depth, K, scale = depth_ctx
            d3 = project_3d(x1, y1, x2, y2, depth, K, scale)
            if d3 is not None:
                det["center_3d"] = [round(v, 3) for v in d3["center_3d"]]
                det["bbox3d"] = [[round(v, 3) for v in p] for p in d3["bbox3d"]]
        rec_dets.append(det)

    record = {"t": round(t, 3), "frame": frame_idx, "query": query, "n": len(rec_dets), "detections": rec_dets}

    print(f"[{frame_idx:05d}] '{query}' n={len(rec_dets)}")
    for d in rec_dets:
        line = f"  #{d['rank']} conf={d['conf']:.3f}  center={d['center_2d']}  corners={d['corners_2d']}"
        if "center_3d" in d:
            line += f"  center3d={d['center_3d']} (m)"
        print(line)

    if json_records is not None:
        json_records.append(record)
    return record


def draw(rgb: np.ndarray, query: str, pairs, blur_others: bool = False) -> np.ndarray:
    import cv2
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if blur_others:
        # Blur the whole frame, then alpha-composite the bbox regions of the
        # sharp original back over the blur using a feathered mask. Drawing
        # rectangles happens *after* so they don't get blurred too.
        blurred = cv2.GaussianBlur(img, (51, 51), 30)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for _, (x1, y1, x2, y2) in pairs:
            cv2.rectangle(
                mask,
                (max(0, int(x1)), max(0, int(y1))),
                (int(x2), int(y2)),
                255, -1,
            )
        # Feather so the cutout doesn't look like a hard rectangle.
        mask = cv2.GaussianBlur(mask, (21, 21), 10)
        m = (mask.astype(np.float32) / 255.0)[..., None]
        img = (img.astype(np.float32) * m + blurred.astype(np.float32) * (1.0 - m)).astype(np.uint8)

    for rank, (conf, (x1, y1, x2, y2)) in enumerate(pairs, start=1):
        x1i, y1i, x2i, y2i = (int(v) for v in (x1, y1, x2, y2))
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.rectangle(img, (x1i, y1i), (x2i, y2i), (0, 0, 255), 2)
        cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1)
        label = f"#{rank} {query} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1i, y1i - th - 6), (x1i + tw + 4, y1i), (0, 0, 255), -1)
        cv2.putText(img, label, (x1i + 2, y1i - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _maybe_voice_swap(key: int, model, listener, current_query: str) -> str:
    """If 'v' was pressed and a listener exists, record + swap the YOLO-World query.

    Returns the (possibly updated) query. Empty Whisper output keeps the old query.
    """
    if listener is None or key != ord("v"):
        return current_query
    new_q = listener.capture()
    if not new_q:
        return current_query
    model.set_classes([new_q])
    print(f"query updated: {current_query!r} -> {new_q!r}")
    return new_q


def run_video(model, query: str, source, args, json_records, listener=None):
    import time
    import cv2
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"could not open: {source}")
    print(f"[cv2] camera opened (source={source}); waiting for first frame...")
    n = 0
    # AVFoundation on Mac reports isOpened()=True immediately but the first
    # ~10-30 reads return False while the sensor physically warms up. Retry
    # silently for ~3 s before giving up.
    warmup_left = 30
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                if warmup_left > 0:
                    warmup_left -= 1
                    time.sleep(0.1)
                    continue
                if n == 0:
                    print(
                        "camera opened but no frames received in 3 s.\n"
                        "  - on Mac: System Settings → Privacy & Security → Camera, "
                        "ensure Terminal/iTerm is allowed; restart the terminal.\n"
                        "  - close any other app currently using the camera "
                        "(Zoom, Photo Booth, browser tabs).",
                        file=sys.stderr,
                    )
                break
            if n == 0:
                print(f"[cv2] first frame received, shape={bgr.shape}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb, conf=args.conf, device=args.device, verbose=False)
            pairs = boxes_from_result(results[0]) if results else []
            emit_frame(n, query, pairs, None, json_records)
            if not args.headless:
                cv2.imshow(f"track: {query}", draw(rgb, query, pairs, blur_others=args.blur_others))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                query = _maybe_voice_swap(key, model, listener, query)
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


def run_image(model, query: str, path: str, args, json_records):
    import cv2
    bgr = cv2.imread(path)
    if bgr is None:
        sys.exit(f"could not read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb, conf=args.conf, device=args.device, verbose=False)
    pairs = boxes_from_result(results[0]) if results else []
    emit_frame(0, query, pairs, None, json_records)
    if not args.headless:
        cv2.imshow(f"track: {query}", draw(rgb, query, pairs, blur_others=args.blur_others))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_realsense(model, query: str, args, json_records, listener=None):
    try:
        import pyrealsense2 as rs
    except ImportError:
        sys.exit("pyrealsense2 not installed (Linux-only on our reqs).")
    import cv2

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
    align = rs.align(rs.stream.color)

    n = 0
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            bgr = np.asarray(color_frame.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth_raw = np.asarray(depth_frame.get_data())
            results = model.predict(rgb, conf=args.conf, device=args.device, verbose=False)
            pairs = boxes_from_result(results[0]) if results else []
            emit_frame(n, query, pairs, (depth_raw, K, depth_scale), json_records)
            if not args.headless:
                cv2.imshow(f"track: {query}", draw(rgb, query, pairs, blur_others=args.blur_others))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                query = _maybe_voice_swap(key, model, listener, query)
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        pipeline.stop()
        if not args.headless:
            cv2.destroyAllWindows()


def main():
    args = parse_args()

    if not args.object and not args.voice:
        sys.exit("specify --object \"...\" or --voice (or both)")

    listener = None
    if args.voice:
        from detections.voice import VoiceListener
        print(f"voice mode: whisper '{args.whisper_model}' (subprocess; first call downloads ~75 MB)")
        listener = VoiceListener(model_size=args.whisper_model, duration_s=args.voice_duration)

    if args.object:
        query = args.object.strip()
        if not query:
            sys.exit("--object cannot be empty")
    else:
        # --voice without --object: prompt for the initial query before the YOLO model loads.
        assert listener is not None
        query = ""
        while not query:
            query = listener.capture()
            if not query:
                print("didn't catch that — try again (Ctrl-C to abort).")

    print(f"loading {args.model} for query: '{query}'")
    model = load_world_model(args.model, query, args.device)

    json_records: list | None = [] if args.json_path else None

    src = args.source
    try:
        if src == "webcam":
            run_video(model, query, 0, args, json_records, listener=listener)
        elif src == "realsense":
            run_realsense(model, query, args, json_records, listener=listener)
        elif os.path.isfile(src):
            ext = Path(src).suffix.lower()
            if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                run_image(model, query, src, args, json_records)
            else:
                run_video(model, query, src, args, json_records, listener=listener)
        else:
            sys.exit(f"unknown source: {src}")
    finally:
        if json_records is not None:
            with open(args.json_path, "w") as f:
                json.dump(json_records, f, indent=2)
            print(f"wrote {len(json_records)} frame records to {args.json_path}")


if __name__ == "__main__":
    main()
