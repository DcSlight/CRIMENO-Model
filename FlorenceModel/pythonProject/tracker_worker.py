# tracker_worker_v3.py
# YOLO multi-class detection + motion fallback, with stream-size metadata propagated from broadcaster.
#
# Supports BOTH message formats:
# Old: [topic, frame_idx, video_time_ms, jpg]
# New: [topic, frame_idx, video_time_ms, src_w, src_h, send_w, send_h, jpg]
#
# The JSONL output contains:
# - stream.src_size (original video size)
# - stream.send_size (size actually processed / bbox coordinate system)

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
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def recv_frame_sub(socket) -> Tuple[int, Optional[int], Optional[Tuple[int, int]], Optional[Tuple[int, int]], bytes]:
    """
    Returns:
      frame_idx,
      video_time_ms,
      src_size (w,h) or None,
      send_size (w,h) or None,
      jpg_bytes
    """
    parts = socket.recv_multipart()

    # Old format
    if len(parts) == 4:
        frame_idx = int(parts[1].decode("utf-8"))
        try:
            video_time_ms = int(parts[2].decode("utf-8"))
        except Exception:
            video_time_ms = None
        jpg_bytes = parts[3]
        return frame_idx, video_time_ms, None, None, jpg_bytes

    # New format
    if len(parts) >= 8:
        frame_idx = int(parts[1].decode("utf-8"))
        try:
            video_time_ms = int(parts[2].decode("utf-8"))
        except Exception:
            video_time_ms = None

        src_w = int(parts[3].decode("utf-8"))
        src_h = int(parts[4].decode("utf-8"))
        send_w = int(parts[5].decode("utf-8"))
        send_h = int(parts[6].decode("utf-8"))
        jpg_bytes = parts[7]
        return frame_idx, video_time_ms, (src_w, src_h), (send_w, send_h), jpg_bytes

    raise ValueError(f"Unexpected multipart size: {len(parts)}")


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
    def __init__(self, iou_threshold: float = 0.35, max_age_frames: int = 45):
        self.iou_threshold = iou_threshold
        self.max_age_frames = max_age_frames
        self._next_id = 1
        self._tracks: List[Track] = []

    def update(self, frame_idx: int, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._tracks = [t for t in self._tracks if frame_idx - t.last_seen_frame <= self.max_age_frames]

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
            self._tracks.append(Track(new_id, bbox, cls_name, conf, frame_idx))
            assigned[det_i] = new_id

        out: List[Dict[str, Any]] = []
        for det_i, det in enumerate(detections):
            d = dict(det)
            d["track_id"] = int(assigned[det_i]) if assigned[det_i] is not None else None
            out.append(d)
        return out


class MotionDetector:
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
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--out", default="tracker.jsonl")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--yolo_model", default="yolov8n.pt")
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--yolo_imgsz", type=int, default=640)
    parser.add_argument("--yolo_builtin_track", action="store_true")
    parser.add_argument("--min_motion_area", type=int, default=900)
    parser.add_argument("--print_every", type=int, default=30)
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
            frame_idx, video_time_ms, src_size, send_size, jpg_bytes = recv_frame_sub(socket)

            if args.every > 1 and (frame_idx % args.every != 0):
                skipped += 1
                continue

            frame_bgr = decode_jpg_to_bgr(jpg_bytes)
            if frame_bgr is None:
                continue

            if send_size is None:
                h, w = frame_bgr.shape[:2]
                send_size = (w, h)

            record: Dict[str, Any] = {
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "stream": {
                    "src_size": {"w": src_size[0], "h": src_size[1]} if src_size else None,
                    "send_size": {"w": send_size[0], "h": send_size[1]} if send_size else None,
                },
                "meta": {
                    "generated_at_unix_ms": now_unix_ms(),
                    "worker": "tracker_worker_v3",
                    "device": args.device,
                    "yolo_model": args.yolo_model if yolo.available else None,
                    "yolo_builtin_track": bool(args.yolo_builtin_track),
                    "every": args.every,
                },
                "detections": {"yolo": [], "motion": []},
            }

            yolo_dets = yolo.detect(frame_bgr)

            need_iou = any(d.get("track_id") is None for d in yolo_dets)
            if yolo_dets and need_iou:
                sanitized = []
                for d in yolo_dets:
                    dd = dict(d)
                    dd.pop("track_id", None)
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
                sz = record["stream"]["send_size"]
                sz_str = f'{sz["w"]}x{sz["h"]}' if sz else "-"
                print(f"🎯 Tracker | Frame={frame_idx} t={t_ms} send={sz_str} | yolo={y_count} motion={m_count}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (tracker_worker_v3).")
        print(f"[STATS] processed={processed}, skipped={skipped}")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
