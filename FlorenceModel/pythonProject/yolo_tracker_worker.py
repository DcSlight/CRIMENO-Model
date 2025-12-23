# tracker_worker.py
# Detects & tracks as many objects as possible using YOLO (multi-class),
# with a motion-based fallback to detect moving objects even without class labels.
#
# Output: JSONL where each line is a per-frame record including:
# - yolo detections (class + conf + bbox + track_id)
# - motion detections (bbox + pseudo_track_id)
#
# Comments are intentionally in English only.

import json
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import zmq


def now_unix_ms() -> int:
    return int(time.time() * 1000)


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
    A minimal tracker that assigns track IDs by IoU matching frame-to-frame.
    This is not as strong as ByteTrack/DeepSORT, but it's lightweight and reliable enough
    for "fill in between Florence frames" purposes.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age_frames: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age_frames = max_age_frames
        self._next_id = 1
        self._tracks: List[Track] = []

    def update(
        self,
        frame_idx: int,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        detections: list of {bbox_xyxy, cls_name, conf}
        returns: detections with track_id injected.
        """
        # Purge old tracks
        alive_tracks: List[Track] = []
        for tr in self._tracks:
            if frame_idx - tr.last_seen_frame <= self.max_age_frames:
                alive_tracks.append(tr)
        self._tracks = alive_tracks

        assigned_track_ids: List[Optional[int]] = [None] * len(detections)
        used_track_ids = set()

        # Greedy matching by IoU (same class preferred)
        for det_i, det in enumerate(detections):
            bbox = tuple(det["bbox_xyxy"])
            cls_name = det.get("cls_name", "unknown")

            best_iou = 0.0
            best_track: Optional[Track] = None

            for tr in self._tracks:
                if tr.track_id in used_track_ids:
                    continue

                # Prefer same class when possible
                same_class_bonus = 0.05 if tr.cls_name == cls_name else 0.0
                score = iou_xyxy(tr.bbox, bbox) + same_class_bonus

                if score > best_iou:
                    best_iou = score
                    best_track = tr

            if best_track is not None and best_iou >= self.iou_threshold:
                assigned_track_ids[det_i] = best_track.track_id
                used_track_ids.add(best_track.track_id)

                best_track.bbox = bbox
                best_track.cls_name = cls_name
                best_track.conf = float(det.get("conf", 0.0))
                best_track.last_seen_frame = frame_idx

        # Create new tracks for unassigned detections
        for det_i, det in enumerate(detections):
            if assigned_track_ids[det_i] is not None:
                continue

            bbox = tuple(det["bbox_xyxy"])
            cls_name = det.get("cls_name", "unknown")
            conf = float(det.get("conf", 0.0))

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
            assigned_track_ids[det_i] = new_id

        # Attach track_id to output
        out: List[Dict[str, Any]] = []
        for det_i, det in enumerate(detections):
            det_out = dict(det)
            det_out["track_id"] = int(assigned_track_ids[det_i]) if assigned_track_ids[det_i] is not None else None
            out.append(det_out)
        return out


class MotionDetector:
    """
    Motion-based object detection fallback.
    Produces bounding boxes for moving blobs even if YOLO doesn't classify them.
    """

    def __init__(
        self,
        min_area: int = 900,
        history: int = 300,
        var_threshold: int = 40,
        detect_shadows: bool = True,
        morph_kernel: int = 5,
    ):
        self.min_area = min_area
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))

        # Track motion blobs with a separate IoU tracker (no class)
        self.tracker = SimpleIoUTracker(iou_threshold=0.25, max_age_frames=20)

    def detect(self, frame_bgr, frame_idx: int) -> List[Dict[str, Any]]:
        fg = self.bg.apply(frame_bgr)

        # Remove shadows if enabled (MOG2 shadows often ~127)
        _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
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
                    "cls_name": "moving_object",
                    "conf": None,
                    "source": "motion",
                    "area_px": float(area),
                }
            )

        # Track these blobs
        tracked = self.tracker.update(frame_idx, dets)
        for t in tracked:
            # Rename track_id key to differentiate from YOLO track_ids (optional)
            t["motion_track_id"] = t.pop("track_id")
        return tracked


def recv_frame(socket) -> Tuple[int, Optional[int], bytes]:
    """
    Supports multipart:
      - [frame_idx, jpg]
      - [frame_idx, video_time_ms, jpg]
    """
    parts = socket.recv_multipart()
    if len(parts) < 2:
        raise ValueError(f"Expected at least 2 parts, got {len(parts)}")

    frame_idx = int(parts[0].decode("utf-8"))
    video_time_ms: Optional[int] = None
    jpg_bytes = parts[-1]

    if len(parts) >= 3:
        try:
            video_time_ms = int(parts[1].decode("utf-8"))
        except Exception:
            video_time_ms = None

    return frame_idx, video_time_ms, jpg_bytes


def bgr_from_jpg(jpg_bytes: bytes):
    arr = cv2.imdecode(
        cv2.UMat(bytearray(jpg_bytes)).get(),
        cv2.IMREAD_COLOR
    )
    return arr


class YoloDetector:
    def __init__(self, model_name: str, device: str, conf: float, imgsz: int):
        self.available = False
        self.model = None
        self.names: Dict[int, str] = {}
        self.model_name = model_name
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

        try:
            from ultralytics import YOLO  # type: ignore
            self.model = YOLO(model_name)
            self.available = True
            # names is usually a dict like {0:'person', ...}
            self.names = getattr(self.model.model, "names", {}) or {}
            print(f"✅ YOLO loaded: {model_name} on device={device} conf={conf} imgsz={imgsz}")
        except Exception as e:
            self.available = False
            print(f"⚠️ YOLO not available ({e}). Will run motion-only fallback.")

    def detect(self, frame_bgr) -> List[Dict[str, Any]]:
        if not self.available or self.model is None:
            return []

        # Ultralytics accepts numpy BGR; it handles internally.
        # We intentionally do not use tracking from ultralytics here to keep dependencies minimal.
        try:
            results = self.model.predict(
                source=frame_bgr,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            print(f"⚠️ YOLO predict failed: {e}")
            return []

        if not results:
            return []

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return []

        out: List[Dict[str, Any]] = []
        try:
            for b in boxes:
                # b.xyxy is (1,4) tensor; b.cls is (1,) tensor; b.conf is (1,) tensor
                xyxy = b.xyxy[0].tolist()
                cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                conf = float(b.conf[0].item()) if b.conf is not None else 0.0

                x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                cls_name = self.names.get(cls_id, str(cls_id))

                out.append(
                    {
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "cls_id": cls_id,
                        "cls_name": cls_name,
                        "conf": conf,
                        "source": "yolo",
                    }
                )
        except Exception as e:
            print(f"⚠️ YOLO parse failed: {e}")
            return []

        return out


def write_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5561", help="ZeroMQ endpoint for tracker (PULL).")
    parser.add_argument("--out", default="tracker.jsonl", help="Output JSONL path.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="YOLO device selection.")
    parser.add_argument("--yolo_model", default="yolov8n.pt", help="YOLO model (e.g., yolov8n.pt, yolov8s.pt).")
    parser.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--yolo_imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--min_motion_area", type=int, default=900, help="Min area (px) for motion blobs.")
    parser.add_argument("--print_every", type=int, default=30, help="Console print interval in frames.")
    args = parser.parse_args()

    # YOLO detection + IoU tracking for YOLO detections
    yolo = YoloDetector(
        model_name=args.yolo_model,
        device=args.device,
        conf=args.yolo_conf,
        imgsz=args.yolo_imgsz,
    )
    yolo_tracker = SimpleIoUTracker(iou_threshold=0.35, max_age_frames=45)

    # Motion fallback + its own tracking
    motion = MotionDetector(min_area=args.min_motion_area)

    # ZMQ PULL
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect(args.endpoint)
    print(f"🔗 Tracker connected to broadcaster on {args.endpoint}")
    print(f"📝 Writing tracker JSONL to: {args.out}")

    frames_seen = 0

    try:
        while True:
            frame_idx, video_time_ms, jpg_bytes = recv_frame(socket)

            # Decode frame
            npbuf = cv2.imdecode(
                cv2.UMat(bytearray(jpg_bytes)).get(),
                cv2.IMREAD_COLOR
            )
            frame_bgr = npbuf
            if frame_bgr is None:
                continue

            record: Dict[str, Any] = {
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "meta": {
                    "generated_at_unix_ms": now_unix_ms(),
                    "yolo_model": args.yolo_model if yolo.available else None,
                    "device": args.device,
                },
                "detections": {
                    "yolo": [],
                    "motion": [],
                },
            }

            # YOLO detections (multi-class)
            yolo_dets = yolo.detect(frame_bgr)
            yolo_tracked = yolo_tracker.update(frame_idx, yolo_dets) if yolo_dets else []
            record["detections"]["yolo"] = yolo_tracked

            # Motion fallback detections
            motion_tracked = motion.detect(frame_bgr, frame_idx)
            record["detections"]["motion"] = motion_tracked

            # Write JSONL
            write_jsonl(args.out, record)

            frames_seen += 1
            if args.print_every > 0 and (frames_seen % args.print_every == 0):
                y_count = len(yolo_tracked)
                m_count = len(motion_tracked)
                t_ms = f"{video_time_ms}ms" if video_time_ms is not None else "-"
                print(f"🎯 Frame={frame_idx} t={t_ms} | yolo={y_count} motion={m_count}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (tracker).")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
