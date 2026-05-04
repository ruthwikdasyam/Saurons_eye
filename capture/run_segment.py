"""
Standalone pipeline: detect + segment 'person' via YOLOv8-seg, back-project
masked depth to a 3D point cloud (camera frame), extract a per-cluster
oriented bounding box (the "wireframe"), and optionally push it live to
the headset server.

Left pane:  full RGB with the person mask overlaid in green.
Right pane: only the person pixels, with the projected 3D OBB drawn on top.

Usage:
    python capture/run_segment.py
    python capture/run_segment.py --model yolov8s-seg.pt --conf 0.5
    python capture/run_segment.py --vr                                     # POST to https://localhost:8443/scene
    python capture/run_segment.py --vr --vr-url https://192.168.1.7:8443
    python capture/run_segment.py --vr --transform recordings/T_world_camera.json

Transform file (4x4 T_world_camera). Identity for now since VR frame ==
camera frame; later the drone writes this and nothing else changes.

Press 'q' or Esc to quit. On exit the most recent per-frame cloud is saved
to recordings/person_TIMESTAMP.ply.
"""

import argparse
import json
import ssl
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from ultralytics import YOLO

from capture.person_wireframe import (
    camera_optical_to_local,
    load_transform,
    mask_to_silhouettes_3d,
    point_cloud_to_payload,
    silhouettes_to_polylines,
)

SAVE_DIR = "recordings"
PERSON_CLASS = 0           # COCO 'person'
DEPTH_MIN = 0.2            # metres; closer than this is sensor noise / hand on lens
DEPTH_MAX = 8.0            # metres; D435i loses accuracy past this
VOXEL_SIZE = 0.02          # 2 cm voxel downsample before save
VR_PUSH_HZ = 10.0          # cap headset POSTs to this rate

# self-signed cert on the headset server → skip verification on the publisher side
_SSL_NO_VERIFY = ssl.create_default_context()
_SSL_NO_VERIFY.check_hostname = False
_SSL_NO_VERIFY.verify_mode = ssl.CERT_NONE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n-seg.pt", help="ultralytics seg weights (auto-downloads)")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--device", default=None, help="cuda:0 | mps | cpu (default: ultralytics autoselect)")
    p.add_argument("--no-save", action="store_true", help="don't write a .ply on exit")
    p.add_argument("--vr", action="store_true", help="POST OBBs to the headset server each frame")
    p.add_argument("--vr-url", default="https://localhost:8443",
                   help="headset server base URL (POST goes to <url>/scene)")
    p.add_argument("--transform", default=None,
                   help="JSON file with 4x4 T_world_camera. Default: identity (camera frame == VR frame).")
    p.add_argument("--camera-on-headset", action="store_true",
                   help="RealSense is rigidly mounted on the Quest. Emits silhouettes in "
                        "camera-local frame; the renderer pins them using the live headset "
                        "pose at scene-receive time, so they stay world-locked when you turn "
                        "your head. Overrides --transform.")
    p.add_argument("--pose-source", choices=["none", "drone"], default="none",
                   help="none: use --transform/--camera-on-headset. "
                        "drone: subscribe to MAVLink and build T_world_camera live.")
    p.add_argument("--drone-port", default="/dev/ttyUSB0",
                   help="MAVLink connection (with --pose-source drone). e.g. /dev/ttyUSB0, "
                        "udpin:127.0.0.1:14551")
    p.add_argument("--drone-baud", type=int, default=57600)
    p.add_argument("--no-auto-zero", action="store_true",
                   help="(--pose-source drone) don't zero on first pose; press 'z' manually.")
    return p.parse_args()


def _post_scene_async(url: str, *, cubes: list[dict] | None = None,
                      polylines: list[dict] | None = None,
                      point_clouds: list[dict] | None = None) -> None:
    """Fire-and-forget POST in a daemon thread so the capture loop never blocks."""
    body: dict = {}
    if cubes:
        body["cubes"] = cubes
    if polylines:
        body["polylines"] = polylines
    if point_clouds:
        body["point_clouds"] = point_clouds
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    def _do() -> None:
        try:
            urllib.request.urlopen(req, context=_SSL_NO_VERIFY, timeout=2.0)
        except Exception as e:
            print(f"[vr] POST {url} failed: {e}", file=sys.stderr)

    threading.Thread(target=_do, daemon=True).start()


def backproject_mask(
    mask: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    color_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project masked pixels into 3D points + per-point colors (camera frame)."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    valid = mask & (depth_m >= DEPTH_MIN) & (depth_m <= DEPTH_MAX) & np.isfinite(depth_m)
    vs, us = np.where(valid)
    if vs.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    z = depth_m[vs, us]
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    points = np.stack([x, y, z], axis=-1).astype(np.float32)
    colors = color_rgb[vs, us].astype(np.float32) / 255.0
    return points, colors


def main() -> None:
    args = parse_args()

    print(f"loading {args.model}...")
    model = YOLO(args.model)
    if args.device:
        model.to(args.device)

    ctx = rs.context()
    if not list(ctx.query_devices()):
        raise SystemExit("No RealSense device found. See capture/realsense_check.py for hints.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array(
        [[intr.fx, 0.0, intr.ppx],
         [0.0, intr.fy, intr.ppy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    align = rs.align(rs.stream.color)

    print(f"depth_scale={depth_scale:.5f} m/unit  fx={intr.fx:.1f}  fy={intr.fy:.1f}  "
          f"cx={intr.ppx:.1f}  cy={intr.ppy:.1f}")

    drone_pose = None
    if args.pose_source == "drone":
        from capture.drone_pose import DronePoseToWorld
        drone_pose = DronePoseToWorld(
            args.drone_port, baud=args.drone_baud,
            auto_zero_on_first_pose=not args.no_auto_zero,
        )
        drone_pose.start()
        T_world_camera = None                              # filled per-frame
        polyline_frame = "world"
        print(f"transform: drone MAVLink ({args.drone_port}). "
              f"{'auto-zero on first pose' if not args.no_auto_zero else 'press z to zero'}.")
    elif args.camera_on_headset:
        T_world_camera = camera_optical_to_local()        # axis swap only; Quest applies head pose
        polyline_frame = "camera"
        print("transform: camera-on-headset (axis swap; renderer applies live head pose)")
    else:
        T_world_camera = load_transform(args.transform)
        polyline_frame = "world"
        is_identity = np.allclose(T_world_camera, np.eye(4))
        print(f"transform: {'identity (camera frame == VR frame)' if is_identity else args.transform}")
    if args.vr:
        print(f"vr push: ON  →  {args.vr_url}/scene  (cap {VR_PUSH_HZ:.0f} Hz)")

    print("warming up...", end=" ", flush=True)
    for _ in range(15):
        pipeline.try_wait_for_frames(timeout_ms=2000)
    print("ready. Press 'q' or Esc to quit.")

    last_cloud: tuple[np.ndarray, np.ndarray] | None = None
    last_push_t = 0.0
    push_url = f"{args.vr_url.rstrip('/')}/scene"

    try:
        while True:
            ok, frames = pipeline.try_wait_for_frames(timeout_ms=2000)
            if not ok:
                continue
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            t0 = time.time()
            bgr = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale
            t_capture = (time.time() - t0) * 1000

            t0 = time.time()
            results = model.predict(
                rgb,
                conf=args.conf,
                classes=[PERSON_CLASS],
                device=args.device,
                verbose=False,
            )
            t_seg = (time.time() - t0) * 1000

            r = results[0]
            person_mask = np.zeros(rgb.shape[:2], dtype=bool)
            n_person = 0
            if r.masks is not None and len(r.masks) > 0:
                masks = r.masks.data.cpu().numpy()  # [N,H',W'] in model space
                h, w = rgb.shape[:2]
                for m in masks:
                    if m.shape != (h, w):
                        m_full = cv2.resize(
                            m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    else:
                        m_full = m.astype(bool)
                    person_mask |= m_full
                n_person = len(masks)

            t0 = time.time()
            n_pts = 0
            pts_vox = np.empty((0, 3), dtype=np.float32)
            if person_mask.any():
                points, colors = backproject_mask(person_mask, depth_m, K, rgb)
                if len(points) > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
                    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
                    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
                    pts_vox = np.asarray(pcd.points, dtype=np.float32)
                    cls_vox = np.asarray(pcd.colors, dtype=np.float32)
                    n_pts = len(pts_vox)
                    last_cloud = (pts_vox, cls_vox)
            t_proj = (time.time() - t0) * 1000

            # 3D silhouette: trace the mask contour, back-project each pixel.
            t0 = time.time()
            silhouettes = mask_to_silhouettes_3d(person_mask, depth_m, K) if person_mask.any() else []
            t_sil = (time.time() - t0) * 1000

            # Left pane: original BGR with green mask overlay (alpha-blended).
            overlay = bgr
            if person_mask.any():
                green = np.zeros_like(bgr)
                green[..., 1] = 255  # BGR green
                m3 = person_mask[..., None].astype(np.float32)
                overlay = (bgr.astype(np.float32) * (1.0 - 0.4 * m3)
                           + green.astype(np.float32) * (0.4 * m3)).astype(np.uint8)

            # Right pane: only the person pixels, rest blacked out.
            seg_only = np.zeros_like(bgr)
            if person_mask.any():
                seg_only[person_mask] = bgr[person_mask]

            # Project each silhouette polyline back to 2D and draw it on the right pane.
            for sil in silhouettes:
                Z = sil[:, 2]
                in_front = Z > 0.05
                if in_front.sum() < 2:
                    continue
                u = (sil[in_front, 0] * K[0, 0] / sil[in_front, 2] + K[0, 2]).astype(np.int32)
                v = (sil[in_front, 1] * K[1, 1] / sil[in_front, 2] + K[1, 2]).astype(np.int32)
                pts2d = np.stack([u, v], axis=-1).reshape(-1, 1, 2)
                cv2.polylines(seg_only, [pts2d], isClosed=True,
                              color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            # Per-frame pose lookup (drone case overrides the static T set at startup).
            T_this_frame: np.ndarray | None
            if drone_pose is not None:
                T_this_frame = drone_pose.T_world_camera()       # None until first pose + zero
            else:
                T_this_frame = T_world_camera

            # Push to headset (throttled). Polylines = boundary outline + clean
            # earcut-triangulated translucent fill. The 3D point-cloud overlay
            # was dropped — visually too busy on top of the fill. PointCloud
            # payload remains in the wire format for future use.
            now = time.time()
            if (args.vr and T_this_frame is not None and silhouettes
                    and (now - last_push_t) >= (1.0 / VR_PUSH_HZ)):
                polylines = silhouettes_to_polylines(
                    silhouettes, T_this_frame,
                    frame=polyline_frame,               # default fill_color → renderer earcut-fills
                )
                _post_scene_async(push_url, polylines=polylines)
                last_push_t = now

            hud = (
                f"seg={t_seg:5.1f}ms proj={t_proj:5.1f}ms sil={t_sil:4.1f}ms cap={t_capture:4.1f}ms  "
                f"persons={n_person}  pts={n_pts}  silhouettes={len(silhouettes)}"
            )
            cv2.putText(overlay, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
            if drone_pose is not None:
                cv2.putText(overlay, f"drone: {drone_pose.status()}", (8, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            view = np.hstack((overlay, seg_only))
            cv2.imshow("segmented person  |  left: RGB+mask  |  right: person only  (q to quit)", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("z") and drone_pose is not None:
                drone_pose.zero()
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

        if args.no_save:
            return
        if last_cloud is None:
            print("(no person detected during run — nothing to save)")
            return
        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(SAVE_DIR) / f"person_{ts}.ply"
        pts, cls = last_cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cls.astype(np.float64))
        o3d.io.write_point_cloud(str(out_path), pcd)
        print(f"saved {len(pts)} points -> {out_path}")


if __name__ == "__main__":
    main()
