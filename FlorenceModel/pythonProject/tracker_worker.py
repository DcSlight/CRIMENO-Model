# tracker_worker.py
# Real-time multi-class detection + tracking
# - Subscribes to ZeroMQ PUB stream (topic: "frame")
# - Runs YOLOv8 (COCO) + simple IOU tracker
# - Optional motion fallback (MOG2)
# - Sends results to NestJS via WebSocket
# - ✨ Also sends tracking data to Qwen worker via ZeroMQ PUSH (port 5580)

import argparse
import asyncio
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


def show_debug_frame(frame, tracks, window_name="DEBUG"):
    debug = frame.copy()

    for t in tracks:
        # Extract fields from Track object
        cls = t.cls_name
        conf = t.conf
        x1, y1, x2, y2 = t.bbox

        # Color by source (not by class name)
        if hasattr(t, "source") and t.source == "suspicious":
            color = (0, 0, 255)   # RED for suspicious model
        else:
            color = (0, 255, 0)   # GREEN for regular YOLO objects

        # Draw bounding box
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{cls} {conf:.2f}"
        cv2.putText(debug, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow(window_name, debug)
    cv2.waitKey(1)


def shrink_bbox_tuple(bbox, factor=0.2):
    # Shrinks a bounding box by a given factor (default 20%) while keeping it centered.
    # Used to reduce oversized detections (e.g., suspicious model outputs) before visualization or sending downstream.
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    dx = int(w * factor / 2)
    dy = int(h * factor / 2)
    return (x1 + dx, y1 + dy, x2 - dx, y2 - dy)


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
def run_yolo(model, frame_bgr: np.ndarray, conf_th: float, device: Optional[Any] = None) -> List[Dict[str, Any]]:
    if device is None:
        results = model.predict(frame_bgr, conf=conf_th, verbose=False)
    else:
        results = model.predict(frame_bgr, conf=conf_th, verbose=False, device=device)
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
# WebSocket sender
# -------------------------
async def ws_connect_loop(ws_url: str):
    import websockets

    if ws_url.lower() == "none":
        print("[WS] Disabled (ws_url=none)")
        return None

    backoff = 0.25
    while True:
        try:
            ws = await websockets.connect(ws_url, max_size=16 * 1024 * 1024)
            print(f"[WS] Connected: {ws_url}")
            return ws
        except Exception as e:
            print(f"[WS] Connect failed: {e} (retry in {backoff:.2f}s)")
            await asyncio.sleep(backoff)
            backoff = min(5.0, backoff * 1.7)


async def ws_send_json(ws, payload: Dict[str, Any]):
    if ws is not None:
        await ws.send(json.dumps(payload, ensure_ascii=False))


# -------------------------
# Main loop
# -------------------------
async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--ws_url", default="none")
    parser.add_argument("--yolo_model", default="yolov8n.pt")
    parser.add_argument("--conf_th", type=float, default=0.35)
    parser.add_argument("--send_every_n_frames", type=int, default=1)
    parser.add_argument("--send_overlay", type=int, default=0)
    parser.add_argument("--overlay_jpeg_quality", type=int, default=80)
    parser.add_argument("--use_motion_fallback", type=int, default=1)
    parser.add_argument("--max_track_age", type=int, default=30)
    parser.add_argument("--iou_match_th", type=float, default=0.30)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device for YOLO inference. Use 'cpu' to free GPU memory for Qwen.")
    parser.add_argument("--test", default="none")
    args = parser.parse_args()

    if YOLO is None:
        raise RuntimeError("ultralytics not installed")

    print(f"[TRACKER] Loading YOLO model: {args.yolo_model}...")
    model_objects = YOLO("yolov8s.pt")
    model_suspicious = YOLO("Suspicious_Activities_nano.pt")

    yolo_predict_device: Optional[Any] = None
    if args.device == "cpu":
        yolo_predict_device = "cpu"
    elif args.device == "cuda":
        yolo_predict_device = 0

    # Warmup
    print("[TRACKER] Warming up model...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)

    print("[TRACKER] Warming up object model...")
    if yolo_predict_device is None:
        _ = model_objects.predict(dummy, conf=0.5, verbose=False)
    else:
        _ = model_objects.predict(dummy, conf=0.5, verbose=False, device=yolo_predict_device)

    print("[TRACKER] Warming up suspicious model...")
    if yolo_predict_device is None:
        _ = model_suspicious.predict(dummy, conf=0.5, verbose=False)
    else:
        _ = model_suspicious.predict(dummy, conf=0.5, verbose=False, device=yolo_predict_device)

    print("[TRACKER] ✓ Both models ready")


    motion = MotionDetector() if args.use_motion_fallback else None

    # ZMQ SUB
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(args.sub_endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"frame")
    sub.setsockopt(zmq.RCVHWM, 5)
    sub.setsockopt(zmq.RCVTIMEO, 1000)

    print(f"[TRACKER] SUB connect: {args.sub_endpoint}")

    # ✨ NEW: ZMQ PUSH to Qwen worker (port 5580)
    qwen_context = zmq.Context()
    qwen_socket = qwen_context.socket(zmq.PUSH)
    qwen_socket.connect("tcp://127.0.0.1:5580")
    print("[TRACKER] Connected to Qwen worker via ZMQ PUSH (tcp://127.0.0.1:5580)")

    ws = await ws_connect_loop(args.ws_url)

    next_track_id = 1
    tracks: List[Track] = []

    first_frame = True
    last_log_ts = time.time()
    frames_processed = 0

    while True:
        try:
            frame_idx, video_time_ms, jpg_bytes = await asyncio.to_thread(recv_frame_sub, sub)
        except Exception:
            await asyncio.sleep(0.01)
            continue

        if first_frame:
            print(f"[TRACKER] First frame received (idx={frame_idx})")
            first_frame = False

        if args.send_every_n_frames > 1 and (frame_idx % args.send_every_n_frames) != 0:
            continue

        frame = decode_jpg(jpg_bytes)
        h, w = frame.shape[:2]

        # Run both YOLO models
        dets_objects = run_yolo(model_objects, frame, args.conf_th, yolo_predict_device)
        for d in dets_objects:
            d["source"] = "objects"

        dets_suspicious = run_yolo(model_suspicious, frame, args.conf_th, yolo_predict_device)
        for d in dets_suspicious:
            d["source"] = "suspicious"

        # Merge
        dets_objects = [d for d in dets_objects if d["conf"] >= 0.6]
        dets_suspicious = [d for d in dets_suspicious if d["conf"] >= 0.35]
        dets = dets_objects + dets_suspicious

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
                best_track.source = d["source"]
                new_tracks.append(best_track)
            else:
                t = Track(
                    track_id=next_track_id,
                    bbox=bb,
                    cls_name=d["cls_name"],
                    conf=float(d["conf"]),
                    last_seen_frame=frame_idx,
                )
                t.source = d["source"]
                next_track_id += 1
                used_tracks.add(t.track_id)
                new_tracks.append(t)

        tracks = [t for t in new_tracks if (frame_idx - t.last_seen_frame) <= args.max_track_age]

        # shrink AFTER tracking
        for t in tracks:
            if hasattr(t, "source") and t.source == "suspicious":
                t.bbox = shrink_bbox_tuple(t.bbox, factor=0.2)

        # DEBUG VISUALIZATION
        if args.test == "show_image":
            show_debug_frame(frame, tracks)

        tracks_payload = []
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            tracks_payload.append({
            "track_id": t.track_id,
            "cls": t.cls_name,
            "conf": t.conf,
            "source": t.source,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

        # Check if any tracks are from motion detector
        has_motion = any(t.cls_name == "moving_object" for t in tracks)

        payload = {
            "type": "tracker_frame",
            "frame_index": frame_idx,
            "video_time_ms": video_time_ms,
            "frame_size": {"w": w, "h": h},
            "tracks": tracks_payload,
            "motion_detected": has_motion,
        }

        if args.send_overlay == 1:
            overlay = draw_tracks(frame, tracks)
            overlay_jpg = encode_jpg(overlay, args.overlay_jpeg_quality)
            payload["overlay_jpg_b64"] = base64.b64encode(overlay_jpg).decode("ascii")

        # ✨ Send to Qwen worker
        try:
            qwen_socket.send(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            print("[TRACKER] Sending to Qwen:", payload)
        except Exception as e:
            print(f"[TRACKER] Failed to send to Qwen: {e}")

        # Send to WS (if enabled)
        try:
            await ws_send_json(ws, payload)
        except Exception:
            pass

        frames_processed += 1
        now = time.time()
        if now - last_log_ts >= 2.0:
            print(f"[TRACKER] processed={frames_processed} last_frame={frame_idx} tracks={len(tracks)}")
            last_log_ts = now


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()