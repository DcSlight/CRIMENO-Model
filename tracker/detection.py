import base64
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import ALL_WEAPON_CLASSES, APPEARANCE_PROMPTS


# ============================================================
# Frame helpers
# ============================================================

def recv_frame_sub(socket) -> Tuple[int, int, bytes]:
    """Receive a multipart frame message: [topic, frame_idx, video_time_ms, jpg]."""
    parts = socket.recv_multipart()
    if len(parts) < 4:
        raise ValueError(f"Expected 4 parts, got {len(parts)}")
    frame_idx = int(parts[1].decode("utf-8"))
    try:
        video_time_ms = int(parts[2].decode("utf-8"))
    except Exception:
        video_time_ms = -1
    return frame_idx, video_time_ms, parts[3]


def decode_jpg(jpg_bytes: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode JPG")
    return frame


def encode_jpg(frame_bgr: np.ndarray, jpeg_quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Failed to encode overlay JPG")
    return buf.tobytes()


# ============================================================
# IOU + overlap helpers
# ============================================================

def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def box_overlaps_any(bbox: Tuple[int, int, int, int],
                     others: List[Tuple[int, int, int, int]]) -> bool:
    """True if bbox has IOU > 0 with any box, or its centre sits inside one.
    Used to gate weapon detections to actual people."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    for o in others:
        if iou_xyxy(bbox, o) > 0.0:
            return True
        ox1, oy1, ox2, oy2 = o
        if ox1 <= cx <= ox2 and oy1 <= cy <= oy2:
            return True
    return False


def shrink_bbox_tuple(bbox: Tuple[int, int, int, int], factor: float = 0.2) -> Tuple[int, int, int, int]:
    """Shrink a bounding box by `factor` while keeping it centred."""
    x1, y1, x2, y2 = bbox
    dx = int((x2 - x1) * factor / 2)
    dy = int((y2 - y1) * factor / 2)
    return (x1 + dx, y1 + dy, x2 - dx, y2 - dy)


# ============================================================
# Track dataclass
# ============================================================

@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    cls_name: str
    conf: float
    last_seen_frame: int
    source: str = ""
    hits: int = 1
    attributes: List[str] = field(default_factory=list)
    attr_counts: Dict[str, int] = field(default_factory=dict)


def is_weapon_track(t: Track) -> bool:
    """True if this track is a weapon detection subject to person-gating + temporal confirmation."""
    return t.source in ("suspicious", "weapon") and t.cls_name in ALL_WEAPON_CLASSES


# ============================================================
# Motion fallback (MOG2)
# ============================================================

class MotionDetector:
    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32, detectShadows=False)

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        mask = self.bg.apply(frame_bgr)
        mask = cv2.medianBlur(mask, 5)
        _, mask = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)
        boxes: List[Tuple[int, int, int, int]] = []
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(c) < 600:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x + w, y + h))
        return boxes


# ============================================================
# YOLO inference
# ============================================================

def run_yolo(model, frame_bgr: np.ndarray, conf_th: float,
             device: Optional[Any] = None) -> List[Dict[str, Any]]:
    kwargs = dict(conf=conf_th, verbose=False)
    if device is not None:
        kwargs["device"] = device
    results = model.predict(frame_bgr, **kwargs)
    if not results or results[0].boxes is None:
        return []

    r = results[0]
    names = model.names if hasattr(model, "names") else {}
    dets: List[Dict[str, Any]] = []
    for b in r.boxes:
        conf = float(b.conf[0].item()) if b.conf is not None else 0.0
        if conf < conf_th:
            continue
        cls_id = int(b.cls[0].item()) if b.cls is not None else -1
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        dets.append({"bbox": (x1, y1, x2, y2), "cls_name": names.get(cls_id, str(cls_id)), "conf": conf})
    return dets


# ============================================================
# Visualisation (debug only)
# ============================================================

def draw_tracks(frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"id={t.track_id} {t.cls_name} {t.conf:.2f}",
                    (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def show_debug_frame(frame: np.ndarray, tracks: List[Track],
                     window_name: str = "DEBUG", raw_yoloe=None) -> None:
    debug = frame.copy()
    for d in (raw_yoloe or []):
        rx1, ry1, rx2, ry2 = d["bbox"]
        cv2.rectangle(debug, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)
        cv2.putText(debug, f"{d['cls_name']} {d['conf']:.2f}", (rx1, max(0, ry1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        color = (0, 0, 255) if t.source in ("suspicious", "weapon") else (0, 255, 0)
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        label = f"{t.cls_name} {t.conf:.2f}"
        if t.attributes:
            label += " [" + ",".join(t.attributes) + "]"
        cv2.putText(debug, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow(window_name, debug)
    cv2.waitKey(1)
