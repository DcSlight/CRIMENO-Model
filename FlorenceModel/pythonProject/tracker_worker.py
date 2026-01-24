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

@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    cls_name: str
    conf: float
    last_seen_frame: int

# --- Helper functions for IOU and Image handling ---

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0

def decode_jpg(jpg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

async def ws_connect_loop(ws_url: str):
    import websockets
    backoff = 0.5
    while True:
        try:
            ws = await websockets.connect(ws_url, max_size=16 * 1024 * 1024)
            print(f"[WS] Connected to {ws_url}")
            return ws
        except Exception as e:
            print(f"[WS] Failed to connect: {e}. Retrying...")
            await asyncio.sleep(backoff)
            backoff = min(5.0, backoff * 1.5)

# --- Core Async Loop ---

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--ws_url", default="ws://127.0.0.1:3000/ws/tracker")
    parser.add_argument("--yolo_model", default="yolov8n.pt")
    parser.add_argument("--conf_th", type=float, default=0.35)
    parser.add_argument("--max_track_age", type=int, default=30)
    args = parser.parse_args()

    model = YOLO(args.yolo_model)
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(args.sub_endpoint)
    
    # SUBSCRIBE to both frame data and meta-data
    sub.setsockopt(zmq.SUBSCRIBE, b"frame")
    sub.setsockopt(zmq.SUBSCRIBE, b"meta")
    sub.setsockopt(zmq.RCVHWM, 5)

    ws = await ws_connect_loop(args.ws_url)

    # Tracker State
    next_track_id = 1
    tracks: List[Track] = []

    print(f"[TRACKER] Listening on {args.sub_endpoint}")

    while True:
        # Receive multipart message
        parts = await asyncio.to_thread(sub.recv_multipart)
        topic = parts[0].decode("utf-8")

        # CASE 1: NEW VIDEO SIGNAL (Reset everything)
        if topic == "meta":
            print("[TRACKER] 'meta' received. Resetting tracking IDs and state.")
            tracks = []
            next_track_id = 1
            continue

        # CASE 2: PROCESS FRAME
        if topic == "frame":
            frame_idx = int(parts[1].decode("utf-8"))
            video_time_ms = int(parts[2].decode("utf-8"))
            jpg_bytes = parts[3]

            frame = decode_jpg(jpg_bytes)
            h, w = frame.shape[:2]

            # YOLO Inference
            results = model.predict(frame, conf=args.conf_th, verbose=False)
            detections = []
            if results and results[0].boxes:
                r = results[0]
                for b in r.boxes:
                    xyxy = b.xyxy[0].tolist()
                    detections.append({
                        "bbox": tuple(int(v) for v in xyxy),
                        "cls": model.names[int(b.cls[0])],
                        "conf": float(b.conf[0])
                    })

            # Simple IOU Tracker logic
            updated_tracks = []
            used_detections = set()

            for t in tracks:
                best_iou = 0
                best_det_idx = -1
                for i, d in enumerate(detections):
                    if i in used_detections: continue
                    iou = iou_xyxy(t.bbox, d["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = i
                
                if best_det_idx != -1 and best_iou > 0.3:
                    used_detections.add(best_det_idx)
                    t.bbox = detections[best_det_idx]["bbox"]
                    t.conf = detections[best_det_idx]["conf"]
                    t.last_seen_frame = frame_idx
                    updated_tracks.append(t)
            
            # Add new detections as new tracks
            for i, d in enumerate(detections):
                if i not in used_detections:
                    new_t = Track(next_track_id, d["bbox"], d["cls"], d["conf"], frame_idx)
                    updated_tracks.append(new_t)
                    next_track_id += 1

            # Clean up old tracks
            tracks = [t for t in updated_tracks if (frame_index - t.last_seen_frame) <= args.max_track_age]

            # Send results to NestJS via WebSocket
            payload = {
                "type": "tracker_frame",
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "frame_size": {"w": w, "h": h},
                "tracks": [
                    {
                        "track_id": t.track_id,
                        "cls": t.cls_name,
                        "conf": t.conf,
                        "bbox": {"x1": t.bbox[0], "y1": t.bbox[1], "x2": t.bbox[2], "y2": t.bbox[3]}
                    } for t in tracks
                ]
            }
            
            try:
                await ws.send(json.dumps(payload))
            except:
                ws = await ws_connect_loop(args.ws_url)

if __name__ == "__main__":
    asyncio.run(main_async())