# tracker_worker_v2.py
# Multi-class detection + tracking for "Layer 1".
# - Subscribes to the same PUB stream as Florence (topic: "frame").
# - Runs YOLOv8 (COCO) multi-class detection and assigns stable IDs.
# - Adds a motion-based fallback (MOG2) to capture moving blobs even when YOLO misses / can't classify.
# - Writes one JSONL record per processed frame.
#
# Comments are intentionally in English only.

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import zmq


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def decode_jpg_to_bgr(jpg_bytes: bytes) -> Optional[np.ndarray]:
    buf = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return frame


def recv_frame_sub(socket) -> Tuple[int, Optional[int], bytes]:
    """
    Expects multipart: [topic, frame_idx, video_time_ms, jpg]
    """
    parts = socket.recv_multipart()
    if len(parts) < 4:
        raise ValueError(f"Expected 4 parts, got {len(parts)}")

    frame_idx = int(parts[1].decode("utf-8"))
    try:
        video_time_ms = int(parts[2].decode("utf-8"))
    except Exception:
        video_time_ms = None

    jpg_bytes = parts[3]
    return frame_idx, video_time_ms, jpg_bytes


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)

    denom = float(a_area + b_area - inter_area)
    if denom <= 0:
        return 0.0
    return float(inter_area) / denom


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    cls_name: str
    conf: float
    last_seen_frame: int


class SimpleIoUTracker:
    """
    Lightweight tracker: assigns track IDs by IoU matching between consecutive frames.
    This complements YOLO when native track IDs are missing or unstable.
    """

    def __init__(self, iou_threshold: float = 0.35, max_age_frames: int = 45):
        self.iou_threshold = iou_threshold
        self.max_age_frames = max_age_frames
        self._next_id = 1
        self._tracks: List[Track] = []

    def update(self, frame_idx: int, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Purge old tracks
        alive: List[Track] = []
        for tr in self._tracks:
            if frame_idx - tr.last_seen_frame <= self.max_age_frames:
                alive.append(tr)
        self._tracks = alive

        assigned: List[Optional[int]] = [None] * len(detections)
        used_track_ids = set()

        for det_i, det in enumerate(detections):
            bbox = tuple(det["bbox_xyxy"])
            cls_name = det.get("cls_name", "unknown")

            best_score = 0.0
            best_track: Optional[Track] = None

            for tr in self._tracks:
                if tr.track_id in used_track_ids:
                    continue
                same_class_bonus = 0.05 if tr.cls_name == cls_name else 0.0
                score = iou_xyxy(tr.bbox, bbox) + same_class_bonus
                if score > best_score:
                    best_score = score
                    best_track = tr

            if best_track is not None and best_score >= self.iou_threshold:
                assigned[det_i] = best_track.track_id
                used_track_ids.add(best_track.track_id)

                best_track.bbox = bbox
                best_track.cls_name = cls_name
                best_track.conf = float(det.get("conf", 0.0) or 0.0)
                best_track.last_seen_frame = frame_idx

        for det_i, det in enumerate(detections):
            if assigned[det_i] is not None:
                continue
            bbox = tuple(det["bbox_xyxy"])
            cls_name = det.get("cls_name", "unknown")
            conf = float(det.get("conf", 0.0) or 0.0)

            new_id = self._next_id
            self._next_id += 1
            self._tracks.append(
                Track(
                    track_id=new_id,
                    bbox=bbox,
                    cls_name=cls_name,
                    conf=conf,
                    last_seen_frame=frame_idx,
                )
            )
            assigned[det_i] = new_id

        out: List[Dict[str, Any]] = []
        for det_i, det in enumerate(detections):
            det_out = dict(det)
            det_out["track_id"] = int(assigned[det_i]) if assigned[det_i] is not None else None
            out.append(det_out)
        return out


class MotionDetector:
    """
    Motion-based blob detector (MOG2).
    Produces bounding boxes for moving regions, then tracks them using IoU.
    """

    def __init__(
        self,
        min_area: int = 900,
        history: int = 300,
        var_threshold: int = 40,
        detect_shadows: bool = True,
        morph_kernel: int = 5,
        iou_threshold: float = 0.25,
        max_age_frames: int = 20,
    ):
        self.min_area = min_area
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        self.tracker = SimpleIoUTracker(iou_threshold=iou_threshold, max_age_frames=max_age_frames)

    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
        fg = self.bg.apply(frame_bgr)

        # Remove shadows (MOG2 shadows often ~127)
        _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_OPEN, self.kernel, iterations=1)
        fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_DILATE, self.kernel, iterations=2)

        contours, _ = cv2.findContours(fg_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dets: List[Dict[str, Any]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            dets.append(
                {
                    "bbox_xyxy": [int(x), int(y), int(x + w), int(y + h)],
                    "cls_id": None,
                    "cls_name": "moving_object",
                    "conf": None,
                    "source": "motion",
                    "area_px": float(area),
                }
            )

        tracked = self.tracker.update(frame_idx, dets)
        for t in tracked:
            t["motion_track_id"] = t.pop("track_id")
        return tracked


class YoloDetector:
    """
    YOLOv8 detector with optional built-in tracking (ByteTrack when available).
    If 'track' API fails, falls back to 'predict' + IoU tracking.
    """

    def __init__(self, model_name: str, device: str, conf: float, imgsz: int, use_builtin_track: bool):
        self.available = False
        self.model = None
        self.names: Dict[int, str] = {}
        self.model_name = model_name
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.use_builtin_track = use_builtin_track

        try:
            from ultralytics import YOLO  # type: ignore
            self.model = YOLO(model_name)
            self.available = True
            self.names = getattr(self.model.model, "names", {}) or {}
            print(f"✅ YOLO loaded: {model_name} device={device} conf={conf} imgsz={imgsz}")
        except Exception as e:
            self.available = False
            print(f"⚠️ YOLO not available ({e}). Will run motion-only fallback.")

    def _boxes_to_dets(self, boxes) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for b in boxes:
            xyxy = b.xyxy[0].tolist()
            cls_id = int(b.cls[0].item()) if b.cls is not None else -1
            conf = float(b.conf[0].item()) if b.conf is not None else 0.0

            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            cls_name = self.names.get(cls_id, str(cls_id))

            det: Dict[str, Any] = {
                "bbox_xyxy": [x1, y1, x2, y2],
                "cls_id": cls_id,
                "cls_name": cls_name,
                "conf": conf,
                "source": "yolo",
            }

            tid = getattr(b, "id", None)
            if tid is not None:
                try:
                    det["track_id"] = int(tid[0].item())
                except Exception:
                    det["track_id"] = None

            out.append(det)
        return out

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if not self.available or self.model is None:
            return []

        try:
            if self.use_builtin_track:
                results = self.model.track(
                    source=frame_bgr,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    persist=True,
                    verbose=False,
                )
            else:
                results = self.model.predict(
                    source=frame_bgr,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
        except Exception as e:
            print(f"⚠️ YOLO inference failed: {e}")
            return []

        if not results:
            return []

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return []

        try:
            return self._boxes_to_dets(boxes)
        except Exception as e:
            print(f"⚠️ YOLO parse failed: {e}")
            return []


def write_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5560", help="ZeroMQ PUB endpoint (same as broadcaster).")
    parser.add_argument("--out", default="tracker.jsonl", help="Output JSONL path.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="YOLO device selection.")
    parser.add_argument("--every", type=int, default=5, help="Process every N frames.")
    parser.add_argument("--yolo_model", default="yolov8n.pt", help="YOLO model (e.g., yolov8n.pt, yolov8s.pt).")
    parser.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--yolo_imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--yolo_builtin_track", action="store_true", help="Use YOLO built-in tracking when available.")
    parser.add_argument("--min_motion_area", type=int, default=900, help="Min area (px) for motion blobs.")
    parser.add_argument("--print_every", type=int, default=30, help="Console print interval (processed frames).")
    args = parser.parse_args()

    yolo = YoloDetector(
        model_name=args.yolo_model,
        device=args.device,
        conf=args.yolo_conf,
        imgsz=args.yolo_imgsz,
        use_builtin_track=args.yolo_builtin_track,
    )

    yolo_iou_tracker = SimpleIoUTracker(iou_threshold=0.35, max_age_frames=45)
    motion = MotionDetector(min_area=args.min_motion_area)

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(args.endpoint)
    socket.setsockopt(zmq.SUBSCRIBE, b"frame")

    print(f"🔗 Tracker SUB connected to broadcaster on {args.endpoint}")
    print(f"📝 Writing tracker JSONL to: {args.out}")
    print(f"⚙️ Tracker processes every {args.every} frames")

    processed = 0
    skipped = 0

    try:
        while True:
            frame_idx, video_time_ms, jpg_bytes = recv_frame_sub(socket)

            if args.every > 1 and (frame_idx % args.every != 0):
                skipped += 1
                continue

            frame_bgr = decode_jpg_to_bgr(jpg_bytes)
            if frame_bgr is None:
                continue

            record: Dict[str, Any] = {
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "meta": {
                    "generated_at_unix_ms": now_unix_ms(),
                    "worker": "tracker_worker_v2",
                    "device": args.device,
                    "yolo_model": args.yolo_model if yolo.available else None,
                    "yolo_builtin_track": bool(args.yolo_builtin_track),
                    "every": args.every,
                },
                "detections": {
                    "yolo": [],
                    "motion": [],
                },
            }

            yolo_dets = yolo.detect(frame_bgr)

            need_iou = False
            for d in yolo_dets:
                if d.get("track_id") is None:
                    need_iou = True
                    break

            if yolo_dets and need_iou:
                sanitized: List[Dict[str, Any]] = []
                for d in yolo_dets:
                    dd = dict(d)
                    if "track_id" in dd:
                        dd.pop("track_id")
                    sanitized.append(dd)
                yolo_tracked = yolo_iou_tracker.update(frame_idx, sanitized)
            else:
                yolo_tracked = yolo_dets

            record["detections"]["yolo"] = yolo_tracked
            record["detections"]["motion"] = motion.detect(frame_bgr, frame_idx)

            write_jsonl(args.out, record)

            processed += 1
            if args.print_every > 0 and (processed % args.print_every == 0):
                y_count = len(record["detections"]["yolo"])
                m_count = len(record["detections"]["motion"])
                t_ms = f"{video_time_ms}ms" if video_time_ms is not None else "-"
                print(f"🎯 Tracker | Frame={frame_idx} t={t_ms} | yolo={y_count} motion={m_count}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (tracker_worker_v2).")
        print(f"[STATS] processed={processed}, skipped={skipped}")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
