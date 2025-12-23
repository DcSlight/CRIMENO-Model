import io
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

import zmq
import cv2
import numpy as np


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def recv_frame(socket) -> Tuple[int, Optional[int], bytes]:
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


def bgr_from_jpg(jpg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPG")
    return img


def xywh_to_xyxy(x: int, y: int, w: int, h: int) -> List[int]:
    return [int(x), int(y), int(x + w), int(y + h)]


def bbox_area_xyxy(b: List[int]) -> int:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou_xyxy(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = bbox_area_xyxy(a) + bbox_area_xyxy(b) - inter
    return float(inter) / float(union) if union > 0 else 0.0


def centroid_xyxy(b: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def l2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


class CentroidTracker:
    def __init__(
        self,
        max_age: int = 15,
        min_iou: float = 0.05,
        max_center_dist: float = 120.0,
    ):
        self.max_age = max_age
        self.min_iou = min_iou
        self.max_center_dist = max_center_dist

        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Increase missed count for all existing tracks
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["missed"] += 1
            self.tracks[tid]["age_frames"] += 1

        det_used = set()
        track_ids = list(self.tracks.keys())

        # Greedy matching: for each track, pick best detection by (IoU then distance)
        for tid in track_ids:
            tb = self.tracks[tid]["bbox_xyxy"]
            tc = self.tracks[tid]["centroid"]

            best_j = None
            best_score = -1.0

            for j, det in enumerate(detections):
                if j in det_used:
                    continue
                db = det["bbox_xyxy"]
                dc = det["centroid"]

                iou = iou_xyxy(tb, db)
                dist = l2(tc, dc)

                # Score: prioritize IoU, fallback on distance gating
                if iou >= self.min_iou and dist <= self.max_center_dist:
                    score = iou * 10.0 - (dist / 1000.0)
                else:
                    score = -1.0

                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j is not None and best_score >= 0.0:
                det_used.add(best_j)
                det = detections[best_j]
                self.tracks[tid]["bbox_xyxy"] = det["bbox_xyxy"]
                self.tracks[tid]["centroid"] = det["centroid"]
                self.tracks[tid]["confidence"] = det.get("confidence", None)
                self.tracks[tid]["hits"] += 1
                self.tracks[tid]["missed"] = 0

        # Create new tracks for unused detections
        for j, det in enumerate(detections):
            if j in det_used:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "track_id": tid,
                "class": det.get("class", "person"),
                "bbox_xyxy": det["bbox_xyxy"],
                "centroid": det["centroid"],
                "confidence": det.get("confidence", None),
                "age_frames": 1,
                "hits": 1,
                "missed": 0,
            }

        # Remove dead tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["missed"] > self.max_age:
                del self.tracks[tid]

        return list(self.tracks.values())


def detect_people_hog(frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # HOG works better on slightly larger frames; your broadcaster already resizes.
    rects, weights = hog.detectMultiScale(
        frame_bgr,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )

    dets: List[Dict[str, Any]] = []
    for (x, y, w, h), wgt in zip(rects, weights):
        bbox = xywh_to_xyxy(int(x), int(y), int(w), int(h))
        c = centroid_xyxy(bbox)
        dets.append(
            {
                "class": "person",
                "bbox_xyxy": bbox,
                "centroid": [float(c[0]), float(c[1])],
                "confidence": float(wgt) if wgt is not None else None,
            }
        )

    return dets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--out", default="tracker.jsonl")
    parser.add_argument("--max_age", type=int, default=15)
    parser.add_argument("--min_iou", type=float, default=0.05)
    parser.add_argument("--max_center_dist", type=float, default=120.0)
    args = parser.parse_args()

    tracker = CentroidTracker(
        max_age=args.max_age,
        min_iou=args.min_iou,
        max_center_dist=args.max_center_dist,
    )

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect(args.endpoint)

    print(f"🔗 Connected to video broadcaster on {args.endpoint}")
    print(f"📝 Writing JSONL to: {args.out}")
    print(
        f"[INFO] Tracker params: max_age={args.max_age}, min_iou={args.min_iou}, max_center_dist={args.max_center_dist}"
    )

    try:
        while True:
            frame_idx, video_time_ms, jpg_bytes = recv_frame(socket)
            frame = bgr_from_jpg(jpg_bytes)

            detections = detect_people_hog(frame)
            tracks = tracker.update(detections)

            record: Dict[str, Any] = {
                "schema_version": "tracker_frame_v1",
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "detections": detections,  # per-frame raw detections
                "tracks": tracks,          # persistent ids across frames
                "meta": {
                    "generated_at_unix_ms": now_unix_ms(),
                    "detector": "opencv_hog_people_v1",
                    "tracker": "centroid_iou_v1",
                    "params": {
                        "max_age": args.max_age,
                        "min_iou": args.min_iou,
                        "max_center_dist": args.max_center_dist,
                    },
                },
            }

            print(
                f"🧭 Frame {frame_idx}"
                + (f" | t={video_time_ms}ms" if video_time_ms is not None else "")
                + f" | det={len(detections)}"
                + f" | tracks={len(tracks)}"
            )

            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (tracker worker).")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
