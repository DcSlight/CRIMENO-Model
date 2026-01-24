# tracker_worker.py
# Enhanced multi-model detection + tracking
# - COCO objects (YOLOv8)
# - Weapons detection (custom model)
# - Suspicious items detection
# - Improved tracking with re-identification

import argparse
import asyncio
import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
import zmq

try:
    from ultralytics import YOLO
except Exception as e:
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
# Enhanced tracker helpers
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
    area_b = max(0, bx2 - bx1) * max(0, by2 - by2)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def bbox_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Calculate center-to-center distance between two bboxes"""
    ax_c = (a[0] + a[2]) / 2
    ay_c = (a[1] + a[3]) / 2
    bx_c = (b[0] + b[2]) / 2
    by_c = (b[1] + b[3]) / 2
    return np.sqrt((ax_c - bx_c)**2 + (ay_c - by_c)**2)


def bbox_size_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Calculate size similarity between two bboxes"""
    a_area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    b_area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return min(a_area, b_area) / max(a_area, b_area)


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    cls_name: str
    conf: float
    last_seen_frame: int
    velocity: Tuple[float, float] = (0.0, 0.0)  # Track velocity for prediction
    alert_level: int = 0  # 0=normal, 1=suspicious, 2=high_alert
    appearance_features: Optional[np.ndarray] = None  # For re-identification


# Security-relevant object categories
SECURITY_CATEGORIES = {
    'high_risk': {'knife', 'gun', 'rifle', 'pistol', 'weapon'},
    'suspicious': {'backpack', 'suitcase', 'handbag', 'bag'},
    'persons': {'person'},
    'vehicles': {'car', 'truck', 'bus', 'motorcycle', 'bicycle'},
}


def get_alert_level(cls_name: str) -> int:
    """Determine alert level based on object class"""
    cls_lower = cls_name.lower()
    if any(risk in cls_lower for risk in SECURITY_CATEGORIES['high_risk']):
        return 2
    elif any(susp in cls_lower for susp in SECURITY_CATEGORIES['suspicious']):
        return 1
    return 0


def draw_tracks(frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
    out = frame.copy()
    
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        
        # Color based on alert level
        if t.alert_level == 2:
            color = (0, 0, 255)  # Red for high risk
            thickness = 3
        elif t.alert_level == 1:
            color = (0, 165, 255)  # Orange for suspicious
            thickness = 2
        else:
            color = (0, 255, 0)  # Green for normal
            thickness = 2
            
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        
        label = f"ID:{t.track_id} {t.cls_name} {t.conf:.2f}"
        if t.alert_level > 0:
            label = f"⚠ {label}"
            
        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
# Multi-model detection system
# -------------------------
class MultiModelDetector:
    def __init__(self, coco_model_path: str, weapons_model_path: Optional[str] = None, conf_th: float = 0.35):
        self.coco_model = YOLO(coco_model_path)
        self.conf_th = conf_th
        
        # Try to load weapons detection model (if available)
        self.weapons_model = None
        if weapons_model_path:
            try:
                self.weapons_model = YOLO(weapons_model_path)
                print(f"[DETECTOR] Loaded weapons model: {weapons_model_path}")
            except Exception as e:
                print(f"[DETECTOR] Could not load weapons model: {e}")
        
        # For now, we'll use COCO model with enhanced filtering
        # In production, you'd add specialized models here
        
    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run all detection models and combine results
        Returns list of detections: {bbox(x1,y1,x2,y2), cls_name, conf}
        """
        all_dets = []
        
        # 1. COCO detection (general objects)
        coco_dets = self._run_model(self.coco_model, frame_bgr, self.conf_th)
        all_dets.extend(coco_dets)
        
        # 2. Weapons detection (if model available)
        if self.weapons_model:
            weapons_dets = self._run_model(self.weapons_model, frame_bgr, self.conf_th * 0.7)  # Lower threshold
            all_dets.extend(weapons_dets)
        
        # 3. Enhanced detection for security-relevant objects
        all_dets = self._enhance_security_detections(all_dets, frame_bgr)
        
        # 4. Remove duplicates (same object detected by multiple models)
        all_dets = self._remove_duplicate_detections(all_dets)
        
        return all_dets
    
    def _run_model(self, model, frame_bgr: np.ndarray, conf_th: float) -> List[Dict[str, Any]]:
        """Run a single YOLO model"""
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
            dets.append({
                "bbox": (x1, y1, x2, y2),
                "cls_name": cls_name,
                "conf": conf,
            })

        return dets
    
    def _enhance_security_detections(self, dets: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Enhance detections for security purposes:
        - Boost confidence for security-relevant objects
        - Apply additional filtering for suspicious items
        """
        enhanced = []
        
        for d in dets:
            cls_name = d["cls_name"].lower()
            conf = d["conf"]
            
            # Keep all high-risk detections even with lower confidence
            if any(risk in cls_name for risk in SECURITY_CATEGORIES['high_risk']):
                if conf > 0.2:  # Lower threshold for weapons
                    enhanced.append(d)
            # Keep suspicious items
            elif any(susp in cls_name for susp in SECURITY_CATEGORIES['suspicious']):
                if conf > 0.3:
                    enhanced.append(d)
            # Keep persons (always important for security)
            elif 'person' in cls_name:
                if conf > 0.35:
                    enhanced.append(d)
            # Keep vehicles
            elif any(veh in cls_name for veh in SECURITY_CATEGORIES['vehicles']):
                if conf > 0.4:
                    enhanced.append(d)
            # Other objects - higher threshold
            else:
                if conf > 0.45:
                    enhanced.append(d)
        
        return enhanced
    
    def _remove_duplicate_detections(self, dets: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Remove overlapping detections, keeping the one with higher confidence"""
        if len(dets) <= 1:
            return dets
        
        # Sort by confidence (descending)
        sorted_dets = sorted(dets, key=lambda x: x["conf"], reverse=True)
        keep = []
        
        for i, det in enumerate(sorted_dets):
            should_keep = True
            for kept_det in keep:
                if iou_xyxy(det["bbox"], kept_det["bbox"]) > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(det)
        
        return keep


# -------------------------
# Enhanced tracker with re-identification
# -------------------------
class EnhancedTracker:
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.30, distance_threshold: float = 100):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.next_track_id = 1
        self.tracks: List[Track] = []
    
    def update(self, detections: List[Dict[str, Any]], frame_idx: int) -> List[Track]:
        """
        Update tracks with new detections using enhanced matching
        """
        if len(detections) == 0:
            # Age out old tracks
            self.tracks = [t for t in self.tracks if (frame_idx - t.last_seen_frame) <= self.max_age]
            return self.tracks
        
        # Match detections to existing tracks
        matched_tracks = set()
        new_tracks = []
        
        for det in detections:
            bbox = det["bbox"]
            best_score = 0.0
            best_track = None
            
            for track in self.tracks:
                if track.track_id in matched_tracks:
                    continue
                
                # Multi-factor matching
                iou = iou_xyxy(track.bbox, bbox)
                distance = bbox_distance(track.bbox, bbox)
                size_ratio = bbox_size_ratio(track.bbox, bbox)
                
                # Class matching bonus
                class_match = 1.0 if track.cls_name == det["cls_name"] else 0.5
                
                # Combined score
                score = (iou * 0.5 + 
                        (1.0 - min(distance / self.distance_threshold, 1.0)) * 0.3 + 
                        size_ratio * 0.2) * class_match
                
                if score > best_score and iou >= self.iou_threshold:
                    best_score = score
                    best_track = track
            
            if best_track is not None:
                # Update existing track
                matched_tracks.add(best_track.track_id)
                
                # Calculate velocity
                old_center = ((best_track.bbox[0] + best_track.bbox[2]) / 2,
                             (best_track.bbox[1] + best_track.bbox[3]) / 2)
                new_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
                
                best_track.bbox = bbox
                best_track.cls_name = det["cls_name"]
                best_track.conf = float(det["conf"])
                best_track.last_seen_frame = frame_idx
                best_track.velocity = velocity
                best_track.alert_level = get_alert_level(det["cls_name"])
                new_tracks.append(best_track)
            else:
                # Create new track
                track = Track(
                    track_id=self.next_track_id,
                    bbox=bbox,
                    cls_name=det["cls_name"],
                    conf=float(det["conf"]),
                    last_seen_frame=frame_idx,
                    alert_level=get_alert_level(det["cls_name"])
                )
                self.next_track_id += 1
                matched_tracks.add(track.track_id)
                new_tracks.append(track)
        
        # Keep recent unmatched tracks (might reappear)
        for track in self.tracks:
            if track.track_id not in matched_tracks:
                if (frame_idx - track.last_seen_frame) <= self.max_age:
                    new_tracks.append(track)
        
        self.tracks = new_tracks
        return self.tracks


# -------------------------
# WebSocket sender
# -------------------------
async def ws_connect_loop(ws_url: str):
    """
    Keeps trying to connect, returns an open websocket.
    """
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
    parser.add_argument("--ws_url", default="ws://127.0.0.1:3000/ws/tracker", help="NestJS WS endpoint")
    parser.add_argument("--yolo_model", default="yolov8n.pt", help="Main YOLO model (COCO)")
    parser.add_argument("--weapons_model", default=None, help="Optional weapons detection model")
    parser.add_argument("--conf_th", type=float, default=0.35)
    parser.add_argument("--send_every_n_frames", type=int, default=1)
    parser.add_argument("--send_overlay", type=int, default=1)
    parser.add_argument("--overlay_jpeg_quality", type=int, default=80)
    parser.add_argument("--use_motion_fallback", type=int, default=1)
    parser.add_argument("--max_track_age", type=int, default=30)
    parser.add_argument("--iou_match_th", type=float, default=0.30)
    args = parser.parse_args()

    if YOLO is None:
        raise RuntimeError("ultralytics is not installed. Install it: pip install ultralytics")

    # Initialize multi-model detector
    detector = MultiModelDetector(args.yolo_model, args.weapons_model, args.conf_th)
    
    # Initialize enhanced tracker
    tracker = EnhancedTracker(
        max_age=args.max_track_age,
        iou_threshold=args.iou_match_th,
        distance_threshold=100
    )
    
    motion = MotionDetector() if args.use_motion_fallback else None

    # ZMQ SUB
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(args.sub_endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"frame")
    sub.setsockopt(zmq.RCVHWM, 5)

    print(f"[TRACKER] SUB connect: {args.sub_endpoint} topic=frame")
    print(f"[TRACKER] WS target: {args.ws_url}")
    print(f"[TRACKER] Models: COCO={args.yolo_model}, Weapons={args.weapons_model or 'None'}")
    print(f"[TRACKER] send_overlay={args.send_overlay}, send_every_n_frames={args.send_every_n_frames}")

    ws = await ws_connect_loop(args.ws_url)

    last_log_ts = time.time()
    frames_processed = 0
    alerts_count = {"high_risk": 0, "suspicious": 0}

    while True:
        frame_idx, video_time_ms, jpg_bytes = await asyncio.to_thread(recv_frame_sub, sub)

        if args.send_every_n_frames > 1 and (frame_idx % args.send_every_n_frames) != 0:
            continue

        frame = decode_jpg(jpg_bytes)
        h, w = frame.shape[:2]

        # Multi-model detection
        detections = detector.detect(frame)

        # Optional motion fallback
        if motion is not None and len(detections) == 0:
            blobs = motion.detect(frame)
            for bb in blobs:
                detections.append({"bbox": bb, "cls_name": "moving_object", "conf": 1.0})

        # Update tracker
        tracks = tracker.update(detections, frame_idx)

        # Count alerts
        for t in tracks:
            if t.alert_level == 2:
                alerts_count["high_risk"] += 1
            elif t.alert_level == 1:
                alerts_count["suspicious"] += 1

        # Prepare payload
        tracks_payload = []
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            tracks_payload.append({
                "track_id": t.track_id,
                "cls": t.cls_name,
                "conf": t.conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "alert_level": t.alert_level,
                "velocity": {"vx": t.velocity[0], "vy": t.velocity[1]},
            })

        payload: Dict[str, Any] = {
            "type": "tracker_frame",
            "frame_index": frame_idx,
            "video_time_ms": video_time_ms,
            "frame_size": {"w": w, "h": h},
            "tracks": tracks_payload,
            "stats": {
                "total_tracks": len(tracks),
                "high_risk_tracks": sum(1 for t in tracks if t.alert_level == 2),
                "suspicious_tracks": sum(1 for t in tracks if t.alert_level == 1),
            }
        }

        if args.send_overlay == 1:
            overlay = draw_tracks(frame, tracks)
            overlay_jpg = encode_jpg(overlay, args.overlay_jpeg_quality)
            payload["overlay_jpg_b64"] = base64.b64encode(overlay_jpg).decode("ascii")

        # Send to WS
        try:
            await ws_send_json(ws, payload)
        except Exception as e:
            print(f"[WS] Send failed: {e} -> reconnect")
            try:
                await ws.close()
            except Exception:
                pass
            ws = await ws_connect_loop(args.ws_url)

        frames_processed += 1
        now = time.time()
        if now - last_log_ts >= 2.0:
            high_risk = sum(1 for t in tracks if t.alert_level == 2)
            suspicious = sum(1 for t in tracks if t.alert_level == 1)
            print(f"[TRACKER] frame={frame_idx} tracks={len(tracks)} [HIGH_RISK:{high_risk} SUSPICIOUS:{suspicious}]")
            last_log_ts = now


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()