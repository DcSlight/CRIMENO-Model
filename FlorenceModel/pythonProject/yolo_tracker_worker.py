import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

import zmq
import numpy as np
import cv2


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def recv_frame_sub(socket) -> Tuple[int, Optional[int], bytes]:
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


def bgr_from_jpg(jpg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPG")
    return img


# -----------------------
# Minimal SORT tracker
# (Kalman + Hungarian)
# -----------------------
from scipy.optimize import linear_sum_assignment


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: [x1,y1,x2,y2]
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


class Track:
    def __init__(self, track_id: int, bbox: np.ndarray, cls_id: int, conf: float):
        self.id = track_id
        self.bbox = bbox.astype(np.float32)
        self.cls_id = int(cls_id)
        self.conf = float(conf)
        self.age = 0
        self.hits = 1
        self.missed = 0


class SimpleSORT:
    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 30):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: List[Track] = []

    def update(self, dets: List[Dict[str, Any]]) -> List[Track]:
        # dets: [{"bbox": np.array([x1,y1,x2,y2]), "cls_id": int, "conf": float}, ...]
        for tr in self.tracks:
            tr.age += 1
            tr.missed += 1

        if len(self.tracks) == 0:
            for d in dets:
                self.tracks.append(Track(self.next_id, d["bbox"], d["cls_id"], d["conf"]))
                self.next_id += 1
            self._cleanup()
            return self.tracks

        if len(dets) == 0:
            self._cleanup()
            return self.tracks

        # cost = 1 - iou
        cost = np.ones((len(self.tracks), len(dets)), dtype=np.float32)
        for i, tr in enumerate(self.tracks):
            for j, d in enumerate(dets):
                cost[i, j] = 1.0 - iou_xyxy(tr.bbox, d["bbox"])

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            iou_val = 1.0 - cost[r, c]
            if iou_val >= self.iou_thresh:
                tr = self.tracks[r]
                d = dets[c]
                tr.bbox = d["bbox"].astype(np.float32)
                tr.cls_id = int(d["cls_id"])
                tr.conf = float(d["conf"])
                tr.hits += 1
                tr.missed = 0
                matched_tracks.add(r)
                matched_dets.add(c)

        # Unmatched detections -> new tracks
        for j, d in enumerate(dets):
            if j not in matched_dets:
                self.tracks.append(Track(self.next_id, d["bbox"], d["cls_id"], d["conf"]))
                self.next_id += 1

        self._cleanup()
        return self.tracks

    def _cleanup(self):
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--out", default="tracker.jsonl")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--every", type=int, default=5, help="process every N frames")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--max_missed", type=int, default=30)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Missing ultralytics. Install with: pip install ultralytics") from e

    model = YOLO(args.model)
    if args.device == "cuda":
        # ultralytics will pick CUDA if available; we just log intent
        print("[INFO] Requested device=cuda (ultralytics will use it if available)")

    tracker = SimpleSORT(iou_thresh=0.3, max_missed=args.max_missed)

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(args.endpoint)
    socket.setsockopt(zmq.SUBSCRIBE, b"frame")
    print(f"🔗 SUB connected to broadcaster on {args.endpoint}")
    print(f"📝 Writing JSONL to: {args.out}")
    print(f"⚙️ YOLO/Tracker processes every {args.every} frames | conf={args.conf}")

    processed = 0
    skipped = 0

    try:
        while True:
            frame_idx, video_time_ms, jpg_bytes = recv_frame_sub(socket)

            if args.every > 1 and (frame_idx % args.every != 0):
                skipped += 1
                continue

            frame_bgr = bgr_from_jpg(jpg_bytes)

            # YOLO inference
            results = model.predict(frame_bgr, conf=args.conf, verbose=False)

            dets: List[Dict[str, Any]] = []
            if results and len(results) > 0:
                r0 = results[0]
                if r0.boxes is not None and len(r0.boxes) > 0:
                    xyxy = r0.boxes.xyxy.cpu().numpy()
                    confs = r0.boxes.conf.cpu().numpy()
                    clss = r0.boxes.cls.cpu().numpy().astype(int)

                    for bb, cf, cc in zip(xyxy, confs, clss):
                        dets.append({
                            "bbox": bb.astype(np.float32),
                            "conf": float(cf),
                            "cls_id": int(cc),
                        })

            tracks = tracker.update(dets)

            # Build JSONL record
            objects = []
            for t in tracks:
                x1, y1, x2, y2 = t.bbox.tolist()
                objects.append({
                    "track_id": t.id,
                    "cls_id": t.cls_id,
                    "conf": t.conf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "age": t.age,
                    "hits": t.hits,
                    "missed": t.missed,
                })

            record: Dict[str, Any] = {
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "objects": objects,
                "meta": {
                    "generated_at_unix_ms": now_unix_ms(),
                    "model": args.model,
                    "device": args.device,
                    "worker": "yolo_tracker_worker",
                    "every": args.every,
                    "conf": args.conf,
                    "tracker": {
                        "type": "SimpleSORT",
                        "iou_thresh": tracker.iou_thresh,
                        "max_missed": tracker.max_missed,
                    },
                },
            }

            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            processed += 1
            print(f"👁️ YOLO/TRACK | Frame {frame_idx} | t={video_time_ms}ms | tracks={len(objects)}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (yolo_tracker_worker).")
        print(f"[STATS] processed={processed}, skipped={skipped}")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
