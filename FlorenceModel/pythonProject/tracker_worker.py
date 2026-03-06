# tracker_worker.py
# Real-time multi-class detection + tracking
# - Subscribes to ZeroMQ PUB stream (topic: "frame")
# - Runs YOLOv8 (COCO) + simple IOU tracker
# - Optional motion fallback (MOG2)
# - Sends tracking data to Qwen worker via ZeroMQ PUSH (port 5580)

import argparse
import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import zmq

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# -------------------------
# ZMQ receive
# -------------------------
def recv_frame_sub(socket) -> Tuple[int, int, bytes]:
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
        video_time_ms = -1

    jpg_bytes = parts[3]
    return frame_idx, video_time_ms, jpg_bytes


def decode_jpg(jpg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode JPG")
    return frame


def encode_jpg(frame_bgr: np.ndarray, jpeg_quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Failed to encode overlay JPG")
    return buf.tobytes()


# -------------------------
# Simple tracker helpers
# -------------------------
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

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    cls_name: str
    conf: float
    last_seen_frame: int


def draw_tracks(frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"id={t.track_id} {t.cls_name} {t.conf:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


# -------------------------
# Motion fallback (optional)
# -------------------------
class MotionDetector:
    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32, detectShadows=False)

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        mask = self.bg.apply(frame_bgr)
        mask = cv2.medianBlur(mask, 5)
        _, mask = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Tuple[int, int, int, int]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 600:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x + w, y + h))
        return boxes


# -------------------------
# YOLO detection
# -------------------------
def run_yolo(model, frame_bgr: np.ndarray, conf_th: float) -> List[Dict[str, Any]]:
    results = model.predict(frame_bgr, conf=conf_th, verbose=False)
    dets: List[Dict[str, Any]] = []

    if not results:
        return dets

    r = results[0]
    if r.boxes is None:
        return dets

    names = model.names if hasattr(model, "names") else {}

    for b in r.boxes:
        xyxy = b.xyxy[0].tolist()
        cls_id = int(b.cls[0].item()) if b.cls is not None else -1
        conf = float(b.conf[0].item()) if b.conf is not None else 0.0

        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cls_name = names.get(cls_id, str(cls_id))

        if conf >= conf_th:
            dets.append({
                "bbox": (x1, y1, x2, y2),
                "cls_name": cls_name,
                "conf": conf,
            })

    return dets


# -------------------------
# Main loop
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--yolo_model", default="yolov8n.pt")
    parser.add_argument("--conf_th", type=float, default=0.35)
    parser.add_argument("--send_every_n_frames", type=int, default=1)
    parser.add_argument("--send_overlay", type=int, default=0)
    parser.add_argument("--overlay_jpeg_quality", type=int, default=80)
    parser.add_argument("--use_motion_fallback", type=int, default=1)
    parser.add_argument("--max_track_age", type=int, default=30)
    parser.add_argument("--iou_match_th", type=float, default=0.30)
    args = parser.parse_args()

    if YOLO is None:
        raise RuntimeError("ultralytics not installed")

    print(f"[TRACKER] Loading YOLO model: {args.yolo_model}...")
    model = YOLO(args.yolo_model)

    print("[TRACKER] Warming up model...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model.predict(dummy, conf=0.5, verbose=False)
    print("[TRACKER] ✓ Model ready")

    motion = MotionDetector() if args.use_motion_fallback else None

    # ZMQ SUB
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(args.sub_endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"frame")
    sub.setsockopt(zmq.RCVHWM, 5)
    sub.setsockopt(zmq.RCVTIMEO, 1000)

    print(f"[TRACKER] SUB connect: {args.sub_endpoint}")

    # Import config for Message Broker endpoint
    from config import ZMQ_MESSAGE_BROKER_ENDPOINT
    
    # ZMQ PUSH to Message Broker
    output_context = zmq.Context()
    output_socket = output_context.socket(zmq.PUSH)
    output_socket.connect(ZMQ_MESSAGE_BROKER_ENDPOINT)
    print(f"[TRACKER] Connected to Message Broker via ZMQ PUSH ({ZMQ_MESSAGE_BROKER_ENDPOINT})")

    next_track_id = 1
    tracks: List[Track] = []

    first_frame = True
    last_log_ts = time.time()
    frames_processed = 0

    try:
        while True:
            try:
                frame_idx, video_time_ms, jpg_bytes = recv_frame_sub(sub)
            except Exception:
                time.sleep(0.01)
                continue

            if first_frame:
                print(f"[TRACKER] First frame received (idx={frame_idx})")
                first_frame = False

            if args.send_every_n_frames > 1 and (frame_idx % args.send_every_n_frames) != 0:
                continue

            frame = decode_jpg(jpg_bytes)
            h, w = frame.shape[:2]

            dets = run_yolo(model, frame, args.conf_th)

            if motion is not None and len(dets) == 0:
                blobs = motion.detect(frame)
                for bb in blobs:
                    dets.append({"bbox": bb, "cls_name": "moving_object", "conf": 1.0})

            used_tracks = set()
            new_tracks: List[Track] = []

            for d in dets:
                bb = d["bbox"]
                best_iou = 0.0
                best_track = None

                for t in tracks:
                    if t.track_id in used_tracks:
                        continue
                    i = iou_xyxy(t.bbox, bb)
                    if i > best_iou:
                        best_iou = i
                        best_track = t

                if best_track is not None and best_iou >= args.iou_match_th:
                    used_tracks.add(best_track.track_id)
                    best_track.bbox = bb
                    best_track.cls_name = d["cls_name"]
                    best_track.conf = float(d["conf"])
                    best_track.last_seen_frame = frame_idx
                    new_tracks.append(best_track)
                else:
                    t = Track(
                        track_id=next_track_id,
                        bbox=bb,
                        cls_name=d["cls_name"],
                        conf=float(d["conf"]),
                        last_seen_frame=frame_idx,
                    )
                    next_track_id += 1
                    used_tracks.add(t.track_id)
                    new_tracks.append(t)
                used_tracks.add(t.track_id)
                new_tracks.append(t)

            tracks = [t for t in new_tracks if (frame_idx - t.last_seen_frame) <= args.max_track_age]

            tracks_payload = []
            for t in tracks:
                x1, y1, x2, y2 = t.bbox
                tracks_payload.append({
                    "track_id": t.track_id,
                    "cls": t.cls_name,
                    "conf": t.conf,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })

            payload = {
                "type": "tracker_frame",
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "frame_size": {"w": w, "h": h},
                "tracks": tracks_payload,
            }

            if args.send_overlay == 1:
                overlay = draw_tracks(frame, tracks)
                overlay_jpg = encode_jpg(overlay, args.overlay_jpeg_quality)
                payload["overlay_jpg_b64"] = base64.b64encode(overlay_jpg).decode("ascii")

            # Send to Message Broker
            try:
                output_socket.send(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
                print("[TRACKER] Sent to Message Broker:", payload)
            except Exception as e:
                print(f"[TRACKER] Failed to send to Message Broker: {e}")

            frames_processed += 1
            now = time.time()
            if now - last_log_ts >= 2.0:
                print(f"[TRACKER] processed={frames_processed} last_frame={frame_idx} tracks={len(tracks)}")
                last_log_ts = now

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Tracker worker).")
    finally:
        sub.close()
        qwen_socket.close()
        context.term()
        qwen_context.term()


if __name__ == "__main__":
    main()