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
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import zmq

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# _reset_event: set by watcher when reset arrives, cleared by main loop after applying it.
# _reset_done_event: set by main loop after applying reset, cleared by watcher before each wait.
_reset_event = threading.Event()
_reset_done_event = threading.Event()
_first_frame_pending = threading.Event()  # set on reset; cleared after first real bbox emit

ACK_TIMEOUT_S = 15  # wait up to 15s; broadcaster timeout is 20s

# ---------------------------------------------------------------------------
# Robbery-focused detection config
# ---------------------------------------------------------------------------
# COCO classes (from the YOLO26 object model) that are relevant to robbery /
# surveillance. Everything else (chairs, TVs, plants...) is dropped before it
# reaches the tracker so the downstream LLM isn't diluted with scene clutter.
# "person" is always kept regardless of this set.
ROBBERY_OBJECT_CLASSES = {
    "person", "backpack", "handbag", "suitcase",
    "knife", "cell phone", "bottle",
}

# Per-class confidence floors for the custom Suspicious_Activities model.
# Class names confirmed from the checkpoint: Fighting, Man_With_Gun,
# Man_with_Knife, Theaf_Robbery. The old code used a single blunt 0.93 gate to
# suppress the man-with-gun false positives; that hurt recall. Instead we admit
# weapon detections at a lower floor and restore precision with person-overlap
# gating + temporal confirmation (see main loop).
SUSPICIOUS_CLASS_THRESHOLDS = {
    "Man_With_Gun": 0.45,
    "Man_with_Knife": 0.80,   # knife class is weak/FP-prone on this model — keep it strict
    "Theaf_Robbery": 0.55,
    "Fighting": 0.55,
}
DEFAULT_SUSPICIOUS_TH = 0.50

# Suspicious classes (from the custom nano model) that must overlap a detected
# person to be admitted (kills floating "gun in mid-air" ghosts) and that require
# temporal confirmation across several frames before being emitted.
WEAPON_SUSPICIOUS_CLASSES = {"Man_With_Gun", "Man_with_Knife"}

# Open-vocabulary appearance prompts for the YOLOE-26 model. These describe how
# a robber typically looks; matched detections are attached as attribute tags to
# the nearest person track rather than emitted as separate boxes.
APPEARANCE_PROMPTS = [
    "hood", "mask", "balaclava", "helmet", "hooded person", "dark clothing",
]

# Open-vocabulary WEAPON prompts for the same YOLOE-26 model. No training needed:
# the model detects these by text. Weapon hits are person-gated + temporally
# confirmed exactly like the nano weapon classes (they replace them).
WEAPON_PROMPTS = ["gun", "pistol", "handgun", "rifle", "knife"]

# Everything the YOLOE open-vocab model is asked to find in a single pass.
YOLOE_PROMPTS = APPEARANCE_PROMPTS + WEAPON_PROMPTS

# All class names that count as a "weapon" for emit-gating, across both sources
# (custom nano model + open-vocab). Used by is_weapon_track().
ALL_WEAPON_CLASSES = WEAPON_SUSPICIOUS_CLASSES | set(WEAPON_PROMPTS)


def is_weapon_track(t: "Track") -> bool:
    """True if this track is a weapon detection (nano or open-vocab) subject to
    person-gating + temporal confirmation before it may be emitted."""
    return t.source in ("suspicious", "weapon") and t.cls_name in ALL_WEAPON_CLASSES

# Used by watcher thread to dispatch an immediate WS-clear into the asyncio event loop.
_main_loop: Optional[asyncio.AbstractEventLoop] = None
_clear_queue: Optional[asyncio.Queue] = None


def _reset_watcher(sub_endpoint: str, anomaly_endpoint: str, ack_endpoint: str) -> None:
    """Background thread: immediately resets Groq, then waits for the main inference
    loop to finish its current cycle before acking the broadcaster."""
    ctx = zmq.Context.instance()

    sub = ctx.socket(zmq.SUB)
    sub.connect(sub_endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"reset")

    groq_sock = ctx.socket(zmq.PUSH)
    groq_sock.connect(anomaly_endpoint)

    ack_sock = ctx.socket(zmq.PUSH)
    ack_sock.connect(ack_endpoint)

    while True:
        sub.recv_multipart()  # block until reset
        print("[TRACKER/reset-watcher] Reset received — forwarding to Groq, waiting for pipeline to clear")
        try:
            groq_sock.send(json.dumps({"type": "reset"}).encode("utf-8"))
        except Exception as e:
            print(f"[TRACKER/reset-watcher] Failed to forward reset to Groq: {e}")
        # Tell React to clear bboxes immediately — before YOLO finishes its current frame.
        if _main_loop is not None and _clear_queue is not None:
            _main_loop.call_soon_threadsafe(_clear_queue.put_nowait, "clear")
        _reset_done_event.clear()
        _first_frame_pending.set()  # main loop will ack broadcaster after first real bbox
        _reset_event.set()
        # Wait for main loop to finish current frame and drain the buffer
        if not _reset_done_event.wait(timeout=ACK_TIMEOUT_S):
            print("[TRACKER/reset-watcher] Timeout waiting for pipeline — acking anyway")
        ack_sock.send_json({"worker": "tracker", "type": "reset_ack"})
        print("[TRACKER/reset-watcher] Ack sent to broadcaster")


def show_debug_frame(frame, tracks, window_name="DEBUG", raw_yoloe=None):
    debug = frame.copy()

    # Raw open-vocab YOLOE detections (pre-gating), drawn thin + YELLOW underneath
    # the confirmed tracks so you can see what the prompts actually fire on.
    for d in (raw_yoloe or []):
        rx1, ry1, rx2, ry2 = d["bbox"]
        cv2.rectangle(debug, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)
        cv2.putText(debug, f"{d['cls_name']} {d['conf']:.2f}", (rx1, max(0, ry1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    for t in tracks:
        # Extract fields from Track object
        cls = t.cls_name
        conf = t.conf
        x1, y1, x2, y2 = t.bbox

        # Color by source
        if getattr(t, "source", "") in ("suspicious", "weapon"):
            color = (0, 0, 255)   # RED for weapon detections
        else:
            color = (0, 255, 0)   # GREEN for regular YOLO objects

        # Draw bounding box
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)

        # Draw label (include appearance attributes if any)
        label = f"{cls} {conf:.2f}"
        attrs = getattr(t, "attributes", None)
        if attrs:
            label += " [" + ",".join(attrs) + "]"
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


def box_overlaps_any(bbox: Tuple[int, int, int, int],
                     others: List[Tuple[int, int, int, int]]) -> bool:
    """True if bbox touches any box in `others` (IOU > 0) or its center sits
    inside one of them. Used to gate weapon detections to actual people."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    for o in others:
        if iou_xyxy(bbox, o) > 0.0:
            return True
        ox1, oy1, ox2, oy2 = o
        if ox1 <= cx <= ox2 and oy1 <= cy <= oy2:
            return True
    return False


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    cls_name: str
    conf: float
    last_seen_frame: int
    source: str = ""
    hits: int = 1                       # consecutive matches; for temporal confirmation
    attributes: List[str] = field(default_factory=list)  # appearance tags (hood, dark clothing...)


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
    parser.add_argument("--anomaly-endpoint", "--anomaly_endpoint", dest="anomaly_endpoint",
                        default="tcp://127.0.0.1:5580",
                        help="ZMQ PUSH endpoint for Groq anomaly worker.")
    parser.add_argument("--ack-endpoint", dest="ack_endpoint", default="tcp://127.0.0.1:5562",
                        help="ZeroMQ endpoint to send reset ack back to broadcaster (PUSH).")
    parser.add_argument("--ws_url", default="none")
    parser.add_argument("--yolo_model", default="yolo26s.pt",
                        help="YOLO26 detection weights for general objects (COCO).")
    parser.add_argument("--appearance_model", default="yoloe-26s-seg.pt",
                        help="Open-vocabulary YOLOE-26 weights. Detects both appearance "
                             "(hood, mask, dark clothing) AND weapons (gun, knife) by text prompt.")
    parser.add_argument("--use_appearance", type=int, default=1,
                        help="1 = run the open-vocab YOLOE model (appearance tags + weapon detection).")
    parser.add_argument("--appearance_every_n", type=int, default=1,
                        help="Run the (heavy) YOLOE model once every N processed frames. Default 1 "
                             "(every frame) so weapon temporal confirmation works; raise to save GPU "
                             "but weapons will then confirm more slowly.")
    parser.add_argument("--openvocab_weapon_conf", type=float, default=0.30,
                        help="Confidence floor for open-vocab weapon detections (gun/knife). Open-vocab "
                             "confidences run lower than a trained head — raise if FPs, lower if misses.")
    parser.add_argument("--weapon_confirm_frames", type=int, default=3,
                        help="A weapon (gun/knife) track must persist this many consecutive frames before it is emitted.")
    parser.add_argument("--use_suspicious", type=int, default=1,
                        help="1 = also load the custom nano suspicious model (Fighting/Theaf_Robbery). "
                             "0 = skip it entirely and run only yolo26 + open-vocab YOLOE.")
    parser.add_argument("--disable_suspicious", default="Man_With_Gun,Man_with_Knife",
                        help="Comma-separated nano suspicious classes to ignore. Defaults to the weapon "
                             "classes since open-vocab YOLOE now detects weapons better. Pass '' to re-enable.")
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

    disabled_suspicious = {c.strip() for c in args.disable_suspicious.split(",") if c.strip()}
    if disabled_suspicious:
        print(f"[TRACKER] Disabled suspicious classes: {sorted(disabled_suspicious)}")

    print(f"[TRACKER] Loading YOLO26 object model: {args.yolo_model}...")
    model_objects = YOLO(args.yolo_model)

    model_suspicious = None
    if args.use_suspicious:
        print("[TRACKER] Loading custom suspicious model: Suspicious_Activities_nano.pt...")
        model_suspicious = YOLO("Suspicious_Activities_nano.pt")
    else:
        print("[TRACKER] Suspicious nano model disabled (--use_suspicious 0)")

    yolo_predict_device: Optional[Any] = None
    if args.device == "cpu":
        yolo_predict_device = "cpu"
    elif args.device == "cuda":
        yolo_predict_device = 0

    # Open-vocabulary YOLOE-26 model: detects BOTH appearance (hood/mask/dark
    # clothing) and weapons (gun/knife) by text prompt — no training needed.
    model_appearance = None
    if args.use_appearance:
        print(f"[TRACKER] Loading YOLOE-26 open-vocab model: {args.appearance_model}...")
        try:
            model_appearance = YOLO(args.appearance_model)
            # set_classes signature differs across YOLOE builds; try the
            # text-embedding form first, fall back to the plain names form.
            try:
                model_appearance.set_classes(
                    YOLOE_PROMPTS, model_appearance.get_text_pe(YOLOE_PROMPTS)
                )
            except (AttributeError, TypeError):
                model_appearance.set_classes(YOLOE_PROMPTS)
            print(f"[TRACKER] Open-vocab prompts set: {YOLOE_PROMPTS}")
        except Exception as e:
            print(f"[TRACKER] ⚠ Failed to load open-vocab model ({e}); continuing without it.")
            model_appearance = None

    # Warmup
    print("[TRACKER] Warming up models...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)

    def _warmup(m):
        if m is None:
            return
        if yolo_predict_device is None:
            _ = m.predict(dummy, conf=0.5, verbose=False)
        else:
            _ = m.predict(dummy, conf=0.5, verbose=False, device=yolo_predict_device)

    print("[TRACKER] Warming up object model...")
    _warmup(model_objects)
    if model_suspicious is not None:
        print("[TRACKER] Warming up suspicious model...")
        _warmup(model_suspicious)
    if model_appearance is not None:
        print("[TRACKER] Warming up open-vocab model...")
        _warmup(model_appearance)

    print("[TRACKER] ✓ Models ready")


    motion = MotionDetector() if args.use_motion_fallback else None

    # ZMQ SUB — frame-only socket (reset is handled by the watcher thread)
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(args.sub_endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"frame")
    # NOTE: "reset" intentionally NOT subscribed here — the watcher thread has its own
    # SUB socket for reset so it can ack immediately even while we are mid-YOLO inference.
    # Also: with RCVHWM=5, a reset arriving when the buffer is full would be silently
    # dropped; moving it to a dedicated socket avoids that entirely.
    sub.setsockopt(zmq.RCVHWM, 5)
    sub.setsockopt(zmq.RCVTIMEO, 100)

    print(f"[TRACKER] SUB connect: {args.sub_endpoint}")

    groq_context = zmq.Context()
    groq_socket = groq_context.socket(zmq.PUSH)
    groq_socket.connect(args.anomaly_endpoint)
    print(f"[TRACKER] Connected to Groq worker via ZMQ PUSH ({args.anomaly_endpoint})")

    ack_socket = groq_context.socket(zmq.PUSH)
    ack_socket.connect(args.ack_endpoint)
    print(f"[TRACKER] Connected to broadcaster ack socket on {args.ack_endpoint}")

    threading.Thread(
        target=_reset_watcher,
        args=(args.sub_endpoint, args.anomaly_endpoint, args.ack_endpoint),
        daemon=True,
        name="tracker-reset-watcher",
    ).start()
    print("[TRACKER] Reset-watcher thread started")

    ws = await ws_connect_loop(args.ws_url)

    global _main_loop, _clear_queue
    _main_loop = asyncio.get_running_loop()
    _clear_queue = asyncio.Queue()

    async def clear_sender():
        nonlocal ws
        while True:
            await _clear_queue.get()
            clear_payload = {
                "type": "tracker_frame",
                "frame_index": -1,
                "video_time_ms": -1,
                "tracks": [],
                "motion_detected": False,
                "reset": True,
            }
            try:
                await ws_send_json(ws, clear_payload)
                print("[TRACKER] Sent UI clear payload on reset")
            except Exception as e:
                print(f"[TRACKER] Clear WS send failed: {e}; reconnecting")
                ws = await ws_connect_loop(args.ws_url)

    asyncio.create_task(clear_sender())

    next_track_id = 1
    tracks: List[Track] = []

    first_frame = True
    last_log_ts = time.time()
    frames_processed = 0

    while True:
        # Check for pending reset BEFORE blocking on recv so cold-start resets are handled
        # instantly even when no frames are flowing (avoids the 15s watcher timeout on first play).
        if _reset_event.is_set():
            _reset_event.clear()
            tracks = []
            next_track_id = 1
            motion = MotionDetector() if args.use_motion_fallback else None
            first_frame = True
            drained = 0
            while True:
                try:
                    sub.recv_multipart(zmq.NOBLOCK)
                    drained += 1
                except zmq.error.Again:
                    break
            if drained:
                print(f"[TRACKER] Drained {drained} stale frames after reset")
            print("[TRACKER] Applied pending reset — cleared tracks and state")
            _reset_done_event.set()
            continue

        try:
            parts = await asyncio.to_thread(sub.recv_multipart)
        except zmq.error.Again:
            continue  # RCVTIMEO fired; loop back to check _reset_event
        except Exception:
            await asyncio.sleep(0.01)
            continue

        topic = parts[0]

        if topic != b"frame" or len(parts) < 4:
            continue

        frame_idx = int(parts[1].decode("utf-8"))
        try:
            video_time_ms = int(parts[2].decode("utf-8"))
        except Exception:
            video_time_ms = -1
        jpg_bytes = parts[3]

        if first_frame:
            print(f"[TRACKER] First frame received (idx={frame_idx})")
            first_frame = False

        if args.send_every_n_frames > 1 and (frame_idx % args.send_every_n_frames) != 0:
            continue

        frame = decode_jpg(jpg_bytes)
        h, w = frame.shape[:2]

        # Run both YOLO models (in threads so the event loop stays free for clear_sender)
        dets_objects = await asyncio.to_thread(run_yolo, model_objects, frame, args.conf_th, yolo_predict_device)
        for d in dets_objects:
            d["source"] = "objects"

        dets_suspicious = []
        if model_suspicious is not None:
            dets_suspicious = await asyncio.to_thread(run_yolo, model_suspicious, frame, args.conf_th, yolo_predict_device)
            for d in dets_suspicious:
                d["source"] = "suspicious"

        # ---- Object model: keep only robbery-relevant classes (person always) ----
        dets_objects = [
            d for d in dets_objects
            if d["conf"] >= 0.6 and (
                d["cls_name"] == "person" or d["cls_name"] in ROBBERY_OBJECT_CLASSES
            )
        ]
        person_bboxes = [d["bbox"] for d in dets_objects if d["cls_name"] == "person"]

        # ---- Suspicious model: per-class floor + person gating for weapons ----
        # Replaces the old blunt `conf >= 0.93` gate. Weapon classes are admitted
        # at a lower floor (recall) but must overlap a detected person (precision);
        # single-frame flicker is killed later by temporal confirmation (hits).
        gated_suspicious = []
        for d in dets_suspicious:
            cls = d["cls_name"]
            if cls in disabled_suspicious:
                continue
            floor = SUSPICIOUS_CLASS_THRESHOLDS.get(cls, DEFAULT_SUSPICIOUS_TH)
            if d["conf"] < floor:
                continue
            if cls in WEAPON_SUSPICIOUS_CLASSES and not box_overlaps_any(d["bbox"], person_bboxes):
                continue
            gated_suspicious.append(d)
        dets_suspicious = gated_suspicious

        dets = dets_objects + dets_suspicious

        # Reset arrived mid-YOLO-inference — discard stale results, drain, then unblock watcher.
        if _reset_event.is_set():
            _reset_event.clear()
            tracks = []
            next_track_id = 1
            motion = MotionDetector() if args.use_motion_fallback else None
            first_frame = True
            drained = 0
            while True:
                try:
                    sub.recv_multipart(zmq.NOBLOCK)
                    drained += 1
                except zmq.error.Again:
                    break
            if drained:
                print(f"[TRACKER] Drained {drained} stale frames after reset (post-inference)")
            print("[TRACKER] Reset happened mid-inference — discarding stale results")
            _reset_done_event.set()
            continue

        if motion is not None and len(dets) == 0:
            blobs = motion.detect(frame)
            for bb in blobs:
                dets.append({"bbox": bb, "cls_name": "moving_object", "conf": 1.0})

        # ---- Open-vocab YOLOE pass: appearance tags + weapon detection ----
        # One inference, two uses. Appearance prompts → person attribute tags;
        # weapon prompts → weapon detections that go through the same person-gate
        # + temporal confirmation as the (now-retired) nano weapon classes.
        appearance_dets: List[Dict[str, Any]] = []
        weapon_dets: List[Dict[str, Any]] = []
        yoloe_dets: List[Dict[str, Any]] = []
        if (model_appearance is not None and person_bboxes
                and args.appearance_every_n > 0
                and (frames_processed % args.appearance_every_n == 0)):
            try:
                # Run at a low floor to capture both categories; filter per-category below.
                yoloe_dets = await asyncio.to_thread(
                    run_yolo, model_appearance, frame, 0.20, yolo_predict_device
                )
            except Exception as e:
                print(f"[TRACKER] Open-vocab inference failed: {e}")
                yoloe_dets = []

            for d in yoloe_dets:
                cls = d["cls_name"]
                if cls in WEAPON_PROMPTS:
                    # Person-gate + conf floor; temporal confirmation happens later via hits.
                    if d["conf"] < args.openvocab_weapon_conf:
                        continue
                    if not box_overlaps_any(d["bbox"], person_bboxes):
                        continue
                    d["source"] = "weapon"
                    weapon_dets.append(d)
                elif cls in APPEARANCE_PROMPTS:
                    appearance_dets.append(d)

            # Open-vocab weapons join the detection set so they get tracked + confirmed.
            dets = dets + weapon_dets

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
                best_track.source = d.get("source", "")
                best_track.hits += 1
                new_tracks.append(best_track)
            else:
                t = Track(
                    track_id=next_track_id,
                    bbox=bb,
                    cls_name=d["cls_name"],
                    conf=float(d["conf"]),
                    last_seen_frame=frame_idx,
                    source=d.get("source", ""),
                )
                next_track_id += 1
                used_tracks.add(t.track_id)
                new_tracks.append(t)

        tracks = [t for t in new_tracks if (frame_idx - t.last_seen_frame) <= args.max_track_age]

        # shrink AFTER tracking
        for t in tracks:
            if t.source == "suspicious":
                t.bbox = shrink_bbox_tuple(t.bbox, factor=0.2)

        # ---- Attach appearance attributes to the nearest person track (sticky) ----
        if appearance_dets:
            person_tracks = [t for t in tracks if t.cls_name == "person"]
            for ad in appearance_dets:
                tag = ad["cls_name"]
                best_t = None
                best_i = 0.0
                for t in person_tracks:
                    i = iou_xyxy(t.bbox, ad["bbox"])
                    if i > best_i:
                        best_i = i
                        best_t = t
                if best_t is not None and best_i > 0.0 and tag not in best_t.attributes:
                    best_t.attributes.append(tag)

        # ---- Temporal confirmation gate: a weapon track (nano OR open-vocab) is
        # only emitted once it has persisted `weapon_confirm_frames` frames. ----
        emitted_tracks = [
            t for t in tracks
            if not (is_weapon_track(t) and t.hits < args.weapon_confirm_frames)
        ]

        # DEBUG VISUALIZATION
        if args.test == "show_image":
            # Raw open-vocab YOLOE detections (pre-gating) so you can SEE what the
            # model finds and tune prompts — drawn under the confirmed tracks.
            if yoloe_dets:
                names = [f"{d['cls_name']} {d['conf']:.2f}" for d in yoloe_dets]
                print(f"[TRACKER/YOLOE] frame {frame_idx} raw hits: {names}")
            show_debug_frame(frame, emitted_tracks, raw_yoloe=yoloe_dets)

        tracks_payload = []
        for t in emitted_tracks:
            x1, y1, x2, y2 = t.bbox
            tracks_payload.append({
            "track_id": t.track_id,
            "cls": t.cls_name,
            "conf": t.conf,
            "source": t.source,
            "attributes": t.attributes,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

        # Check if any tracks are from motion detector
        has_motion = any(t.cls_name == "moving_object" for t in emitted_tracks)

        payload = {
            "type": "tracker_frame",
            "frame_index": frame_idx,
            "video_time_ms": video_time_ms,
            "frame_size": {"w": w, "h": h},
            "tracks": tracks_payload,
            "motion_detected": has_motion,
        }

        if args.send_overlay == 1:
            overlay = draw_tracks(frame, emitted_tracks)
            overlay_jpg = encode_jpg(overlay, args.overlay_jpeg_quality)
            payload["overlay_jpg_b64"] = base64.b64encode(overlay_jpg).decode("ascii")

        # ✨ Send to Qwen worker
        try:
            groq_socket.send(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            print("[TRACKER] Sending to Qwen:", payload)
        except Exception as e:
            print(f"[TRACKER] Failed to send to Qwen: {e}")

        # Send to WS (if enabled)
        try:
            await ws_send_json(ws, payload)
        except Exception as e:
            print(f"[TRACKER] WS send failed: {e}. Reconnecting...")
            ws = await ws_connect_loop(args.ws_url)

        # After the first real bbox is on its way to React, ack the broadcaster so it
        # can reply {ok:true} to NestJS (which unblocks React's loading spinner).
        if _first_frame_pending.is_set():
            _first_frame_pending.clear()
            try:
                ack_socket.send_json({"worker": "tracker", "type": "first_frame_ack"})
                print("[TRACKER] first_frame_ack sent to broadcaster")
            except Exception as e:
                print(f"[TRACKER] first_frame_ack send failed: {e}")

        frames_processed += 1
        now = time.time()
        if now - last_log_ts >= 2.0:
            print(f"[TRACKER] processed={frames_processed} last_frame={frame_idx} tracks={len(tracks)}")
            last_log_ts = now


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()