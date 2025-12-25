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


def recv_frame_sub(sock: zmq.Socket) -> Tuple[int, Optional[int], bytes]:
    # multipart: [topic, frame_idx, video_time_ms, jpg_bytes]
    parts = sock.recv_multipart()
    if len(parts) < 4:
        raise ValueError(f"Expected 4 parts, got {len(parts)}")

    frame_idx = int(parts[1].decode("utf-8"))
    try:
        video_time_ms = int(parts[2].decode("utf-8"))
    except Exception:
        video_time_ms = None

    return frame_idx, video_time_ms, parts[3]


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
    return 0.0 if denom <= 0 else float(inter_area) / denom


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
        used = set()

        for i, det in enumerate(detections):
            bbox = tuple(det["bbox_xyxy"])
            cls_name = det.get("cls_name", "unknown")

            best_score = 0.0
            best_track: Optional[Track] = None

            for tr in self._tracks:
                if tr.track_id in used:
                    continue
                same_bonus = 0.05 if tr.cls_name == cls_name else 0.0
                score = iou_xyxy(tr.bbox, bbox) + same_bonus
                if score > best_score:
                    best_score = score
                    best_track = tr

            if best_track is not None and best_score >= self.iou_threshold:
                assigned[i] = best_track.track_id
                used.add(best_track.track_id)
                best_track.bbox = bbox
                best_track.cls_name = cls_name
                best_track.conf = float(det.get("conf", 0.0) or 0.0)
                best_track.last_seen_frame = frame_idx

        for i, det in enumerate(detections):
            if assigned[i] is not None:
                continue
            bbox = tuple(det["bbox_xyxy"])
            cls_name = det.get("cls_name", "unknown")
            conf = float(det.get("conf", 0.0) or 0.0)

            tid = self._next_id
            self._next_id += 1
            self._tracks.append(Track(tid, bbox, cls_name, conf, frame_idx))
            assigned[i] = tid

        out: List[Dict[str, Any]] = []
        for i, det in enumerate(detections):
            d = dict(det)
            d["track_id"] = int(assigned[i]) if assigned[i] is not None else None
            out.append(d)
        return out


class MotionDetector:
    def __init__(self, min_area: int = 900):
        self.min_area = min_area
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.tracker = SimpleIoUTracker(iou_threshold=0.25, max_age_frames=20)

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
    def __init__(self, model_name: str, device: str, conf: float, imgsz: int):
        self.available = False
        self.model = None
        self.names: Dict[int, str] = {}

        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(model_name)
            self.available = True
            self.names = getattr(self.model.model, "names", {}) or {}
            print(f"[TRACKER] YOLO loaded: {model_name} device={device} conf={conf} imgsz={imgsz}")
        except Exception as e:
            print(f"[TRACKER] YOLO not available ({e}). Motion-only fallback.")

        self.model_name = model_name
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if not self.available or self.model is None:
            return []

        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return []

        out: List[Dict[str, Any]] = []
        for b in boxes:
            xyxy = b.xyxy[0].tolist()
            cls_id = int(b.cls[0].item()) if b.cls is not None else -1
            conf = float(b.conf[0].item()) if b.conf is not None else 0.0
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            out.append(
                {
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "cls_id": cls_id,
                    "cls_name": self.names.get(cls_id, str(cls_id)),
                    "conf": conf,
                    "source": "yolo",
                }
            )
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--pub_endpoint", default="tcp://127.0.0.1:5571")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--yolo_model", default="yolov8n.pt")
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--yolo_imgsz", type=int, default=640)
    parser.add_argument("--min_motion_area", type=int, default=900)
    parser.add_argument("--print_every", type=int, default=30)
    args = parser.parse_args()

    # SUB frames
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(args.frames_endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"frame")

    # PUB tracker results (Nest listens on 5571)
    pub = ctx.socket(zmq.PUB)
    pub.bind(args.pub_endpoint)

    yolo = YoloDetector(args.yolo_model, args.device, args.yolo_conf, args.yolo_imgsz)
    yolo_tracker = SimpleIoUTracker(iou_threshold=0.35, max_age_frames=45)
    motion = MotionDetector(min_area=args.min_motion_area)

    print(f"[TRACKER] SUB frames: {args.frames_endpoint} topic=frame")
    print(f"[TRACKER] PUB tracker: {args.pub_endpoint} topic=tracker")
    print(f"[TRACKER] every={args.every}, device={args.device}")

    processed = 0

    try:
        while True:
            frame_idx, video_time_ms, jpg_bytes = recv_frame_sub(sub)

            if args.every > 1 and (frame_idx % args.every != 0):
                continue

            frame_bgr = decode_jpg_to_bgr(jpg_bytes)
            if frame_bgr is None:
                continue

            yolo_dets = yolo.detect(frame_bgr)
            # add stable ids if YOLO doesn't provide them
            tracked = yolo_tracker.update(frame_idx, yolo_dets) if yolo_dets else []

            payload: Dict[str, Any] = {
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "meta": {
                    "generated_at_unix_ms": now_unix_ms(),
                    "worker": "tracker_worker_realtime_zmq",
                    "device": args.device,
                    "every": args.every,
                },
                "detections": tracked,
                "motion": motion.detect(frame_bgr, frame_idx),
            }

            pub.send_multipart([b"tracker", json.dumps(payload).encode("utf-8")])

            processed += 1
            if args.print_every > 0 and (processed % args.print_every == 0):
                t_ms = f"{video_time_ms}ms" if video_time_ms is not None else "-"
                print(
                    f"[TRACKER] OK frame={frame_idx} t={t_ms} "
                    f"dets={len(payload['detections'])} motion={len(payload['motion'])}"
                )

    except KeyboardInterrupt:
        print("[TRACKER] Stopped by user.")
    finally:
        sub.close()
        pub.close()
        ctx.term()


if __name__ == "__main__":
    main()
