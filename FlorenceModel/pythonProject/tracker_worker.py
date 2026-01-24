# tracker_worker.py
# Real-time multi-class detection + tracking
# Optimized for video switching with buffer clearing logic.

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
except Exception as e:
    YOLO = None

# -------------------------
# ZMQ receive helpers
# -------------------------
def recv_multipart_sub(socket) -> List[bytes]:
    return socket.recv_multipart()

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
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0

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
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < 600: continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x + w, y + h))
        return boxes

# -------------------------
# YOLO detection
# -------------------------
def run_yolo(model, frame_bgr: np.ndarray, conf_th: float) -> List[Dict[str, Any]]:
    results = model.predict(frame_bgr, conf=conf_th, verbose=False)
    dets = []
    if not results or results[0].boxes is None: return dets
    names = model.names if hasattr(model, "names") else {}
    for b in results[0].boxes:
        xyxy = [int(v) for v in b.xyxy[0].tolist()]
        cls_name = names.get(int(b.cls[0].item()), str(int(b.cls[0].item())))
        dets.append({"bbox": tuple(xyxy), "cls_name": cls_name, "conf": float(b.conf[0].item())})
    return dets

# -------------------------
# WebSocket sender
# -------------------------
async def ws_connect_loop(ws_url: str):
    import websockets
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
    await ws.send(json.dumps(payload, ensure_ascii=False))

# -------------------------
# Main loop
# -------------------------
async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--ws_url", default="ws://127.0.0.1:3000/ws/tracker")
    parser.add_argument("--yolo_model", default="yolov8n.pt")
    parser.add_argument("--conf_th", type=float, default=0.35)
    parser.add_argument("--send_every_n_frames", type=int, default=1)
    parser.add_argument("--send_overlay", type=int, default=1)
    parser.add_argument("--overlay_jpeg_quality", type=int, default=80)
    parser.add_argument("--use_motion_fallback", type=int, default=1)
    parser.add_argument("--max_track_age", type=int, default=30)
    parser.add_argument("--iou_match_th", type=float, default=0.30)
    args = parser.parse_args()

    if YOLO is None: raise RuntimeError("ultralytics is not installed.")

    model = YOLO(args.yolo_model)
    motion = MotionDetector() if args.use_motion_fallback else None

    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(args.sub_endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"frame")
    sub.setsockopt(zmq.SUBSCRIBE, b"meta")
    sub.setsockopt(zmq.RCVHWM, 5)

    print(f"[TRACKER] SUB connect: {args.sub_endpoint} topics=frame,meta")
    ws = await ws_connect_loop(args.ws_url)

    next_track_id = 1
    tracks: List[Track] = []
    last_log_ts = time.time()
    frames_processed = 0

    while True:
        # 1. Blocking wait for the next message
        parts = await asyncio.to_thread(recv_multipart_sub, sub)
        topic = parts[0].decode("utf-8")

        # --- SYNC / RESET LOGIC ---
        if topic == "meta":
            print("[TRACKER] Received 'meta'. Clearing ZMQ buffer and resetting state.")
            
            # Flush all pending frames from the ZMQ internal buffer
            try:
                while True:
                    sub.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                pass # Buffer is now empty
            
            tracks = []
            next_track_id = 1
            continue

        # --- FRAME PROCESSING ---
        if topic == "frame":
            frame_idx = int(parts[1].decode("utf-8"))
            try:
                video_time_ms = int(parts[2].decode("utf-8"))
            except:
                video_time_ms = -1
            jpg_bytes = parts[3]

            if args.send_every_n_frames > 1 and (frame_idx % args.send_every_n_frames) != 0:
                continue

            frame = decode_jpg(jpg_bytes)
            h, w = frame.shape[:2]
            dets = run_yolo(model, frame, args.conf_th)

            if motion is not None and len(dets) == 0:
                for bb in motion.detect(frame):
                    dets.append({"bbox": bb, "cls_name": "moving_object", "conf": 1.0})

            used_tracks = set()
            new_tracks: List[Track] = []

            for d in dets:
                bb, best_iou, best_track = d["bbox"], 0.0, None
                for t in tracks:
                    if t.track_id in used_tracks: continue
                    i = iou_xyxy(t.bbox, bb)
                    if i > best_iou: best_iou, best_track = i, t
                
                if best_track and best_iou >= args.iou_match_th:
                    used_tracks.add(best_track.track_id)
                    best_track.bbox, best_track.cls_name = bb, d["cls_name"]
                    best_track.conf, best_track.last_seen_frame = d["conf"], frame_idx
                    new_tracks.append(best_track)
                else:
                    new_tracks.append(Track(next_track_id, bb, d["cls_name"], d["conf"], frame_idx))
                    next_track_id += 1

            tracks = [t for t in new_tracks if (frame_idx - t.last_seen_frame) <= args.max_track_age]

            # WebSocket payload
            payload = {
                "type": "tracker_frame",
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "frame_size": {"w": w, "h": h},
                "tracks": [
                    {
                        "track_id": t.track_id, "cls": t.cls_name, "conf": t.conf,
                        "bbox": {"x1": t.bbox[0], "y1": t.bbox[1], "x2": t.bbox[2], "y2": t.bbox[3]}
                    } for t in tracks
                ]
            }

            if args.send_overlay == 1:
                overlay = draw_tracks(frame, tracks)
                payload["overlay_jpg_b64"] = base64.b64encode(encode_jpg(overlay, args.overlay_jpeg_quality)).decode("ascii")

            try:
                await ws_send_json(ws, payload)
            except:
                ws = await ws_connect_loop(args.ws_url)

            frames_processed += 1
            if time.time() - last_log_ts >= 2.0:
                print(f"[TRACKER] processed={frames_processed} frame={frame_idx} tracks={len(tracks)}")
                last_log_ts = time.time()

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()