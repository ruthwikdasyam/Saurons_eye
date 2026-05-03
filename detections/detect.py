"""YOLOv8 detection + 2D->3D projection using an aligned depth frame.

Public surface:
    Detector(model="yolov8n.pt", conf=0.4, classes=None)
        .infer(rgb)                 -> list[Detection2D]
        .infer_3d(rgb, depth, K)    -> list[Detection3D]

Detection3D matches the v1 wire spec in README:
    {"class": str, "conf": float, "bbox3d": [[xmin,ymin,zmin],[xmax,ymax,zmax]]}
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np

# Default to COCO classes that matter for room clearing.
# 0 person, 56 chair, 57 couch, 60 dining table, 62 tv, 63 laptop, 67 cell phone, 73 book.
# Weapons aren't in COCO; v1 demos with person + furniture.
DEFAULT_CLASSES = (0, 56, 57, 60, 62, 63, 67, 73)


@dataclass
class Detection2D:
    cls_id: int
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]  # pixel coords


@dataclass
class Detection3D:
    cls_name: str
    conf: float
    bbox3d: list[list[float]]  # [[xmin,ymin,zmin],[xmax,ymax,zmax]] in camera frame, meters

    def to_wire(self) -> dict:
        return {"class": self.cls_name, "conf": self.conf, "bbox3d": self.bbox3d}


class Detector:
    def __init__(
        self,
        model: str = "yolov8n.pt",
        conf: float = 0.4,
        classes: Iterable[int] | None = DEFAULT_CLASSES,
        device: str | None = None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.conf = conf
        self.classes = list(classes) if classes is not None else None
        self.device = device
        self.names = self.model.names

    def infer(self, rgb: np.ndarray) -> list[Detection2D]:
        results = self.model.predict(
            rgb,
            conf=self.conf,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        out: list[Detection2D] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, cls):
            out.append(
                Detection2D(
                    cls_id=int(k),
                    cls_name=self.names[int(k)],
                    conf=float(c),
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return out

    def infer_3d(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        depth_scale: float = 1.0,
        depth_min: float = 0.2,
        depth_max: float = 8.0,
    ) -> list[Detection3D]:
        """Project 2D detections to 3D camera-frame bboxes using the aligned depth frame.

        depth: HxW, in meters after multiplying by depth_scale.
                For RealSense raw uint16 in mm, pass depth_scale=0.001.
        K: 3x3 intrinsics (matches the color frame the RGB came from; depth must be aligned to color).
        """
        dets2d = self.infer(rgb)
        if not dets2d:
            return []

        h, w = depth.shape[:2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        out: list[Detection3D] = []
        for d in dets2d:
            x1, y1, x2, y2 = d.xyxy
            x1i = max(0, int(np.floor(x1)))
            y1i = max(0, int(np.floor(y1)))
            x2i = min(w, int(np.ceil(x2)))
            y2i = min(h, int(np.ceil(y2)))
            if x2i <= x1i or y2i <= y1i:
                continue

            patch = depth[y1i:y2i, x1i:x2i].astype(np.float32) * depth_scale
            valid = (patch >= depth_min) & (patch <= depth_max) & np.isfinite(patch)
            if not np.any(valid):
                continue

            # Robust depth: trim 10/90 to drop background bleed and foreground spikes.
            zs = patch[valid]
            lo, hi = np.percentile(zs, [10, 90])
            band = (zs >= lo) & (zs <= hi)
            zs = zs[band] if np.any(band) else zs

            ys, xs = np.where(valid)
            zs_full = patch[ys, xs]
            band_full = (zs_full >= lo) & (zs_full <= hi)
            if np.any(band_full):
                ys, xs, zs_full = ys[band_full], xs[band_full], zs_full[band_full]

            us = xs + x1i
            vs = ys + y1i
            X = (us - cx) * zs_full / fx
            Y = (vs - cy) * zs_full / fy
            Z = zs_full

            bbox3d = [
                [float(X.min()), float(Y.min()), float(Z.min())],
                [float(X.max()), float(Y.max()), float(Z.max())],
            ]
            out.append(Detection3D(cls_name=d.cls_name, conf=d.conf, bbox3d=bbox3d))
        return out


def draw_detections(rgb: np.ndarray, dets: list[Detection2D], dets3d: list[Detection3D] | None = None) -> np.ndarray:
    import cv2

    img = rgb.copy()
    z_by_name: dict[str, float] = {}
    if dets3d:
        for d3 in dets3d:
            z_by_name.setdefault(d3.cls_name, d3.bbox3d[0][2])

    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        color = (0, 0, 255) if d.cls_name == "person" else (0, 200, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{d.cls_name} {d.conf:.2f}"
        if d.cls_name in z_by_name:
            label += f"  z~{z_by_name[d.cls_name]:.2f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


__all__ = ["Detector", "Detection2D", "Detection3D", "draw_detections", "DEFAULT_CLASSES"]
