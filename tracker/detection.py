import base64
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


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
                     window_name: str = "DEBUG") -> None:
    debug = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{t.cls_name} {t.conf:.2f}"
        cv2.putText(debug, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(window_name, debug)
    cv2.waitKey(1)
