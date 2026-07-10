import argparse
import asyncio
import base64
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import zmq

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from config import (
    ROBBERY_OBJECT_CLASSES, SUSPICIOUS_CLASS_THRESHOLDS, DEFAULT_SUSPICIOUS_TH,
    WEAPON_SUSPICIOUS_CLASSES, APPEARANCE_PROMPTS, WEAPON_PROMPTS, YOLOE_PROMPTS,
)
from detection import (
    Track, MotionDetector, is_weapon_track,
    run_yolo, iou_xyxy, box_overlaps_any, shrink_bbox_tuple,
    decode_jpg, encode_jpg, draw_tracks, show_debug_frame,
)

_HERE = Path(__file__).resolve().parent
_OUTPUT_LOG = _HERE / "logs_output.jsonl"


def _log_output(payload: Dict[str, Any]) -> None:
    """Append the exact payload about to be sent to NestJS (minus the bulky overlay JPEG)."""
    record = {k: v for k, v in payload.items() if k != "overlay_jpg_b64"}
    with open(_OUTPUT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# Reset handshake state (shared between watcher thread + main loop)
# ============================================================

_reset_event        = threading.Event()
_reset_done_event   = threading.Event()
_first_frame_pending = threading.Event()

ACK_TIMEOUT_S = 15

_main_loop:   Optional[asyncio.AbstractEventLoop] = None
_clear_queue: Optional[asyncio.Queue]             = None


# ============================================================
# Reset watcher thread
# ============================================================

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
        sub.recv_multipart()
        print("[TRACKER/reset-watcher] Reset received — forwarding to Groq, waiting for pipeline to clear")
        try:
            groq_sock.send(json.dumps({"type": "reset"}).encode("utf-8"))
        except Exception as e:
            print(f"[TRACKER/reset-watcher] Failed to forward reset to Groq: {e}")

        if _main_loop is not None and _clear_queue is not None:
            _main_loop.call_soon_threadsafe(_clear_queue.put_nowait, "clear")

        _reset_done_event.clear()
        _first_frame_pending.set()
        _reset_event.set()

        if not _reset_done_event.wait(timeout=ACK_TIMEOUT_S):
            print("[TRACKER/reset-watcher] Timeout waiting for pipeline — acking anyway")
        ack_sock.send_json({"worker": "tracker", "type": "reset_ack"})
        print("[TRACKER/reset-watcher] Ack sent to broadcaster")


# ============================================================
# WebSocket helpers
# ============================================================

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


# ============================================================
# Reset helper (called from main loop)
# ============================================================

def _apply_reset(sub, use_motion_fallback: bool):
    """Clear all tracker state, drain stale SUB buffer, signal watcher."""
    tracks: List[Track] = []
    next_track_id = 1
    motion = MotionDetector() if use_motion_fallback else None
    drained = 0
    while True:
        try:
            sub.recv_multipart(zmq.NOBLOCK)
            drained += 1
        except zmq.error.Again:
            break
    if drained:
        print(f"[TRACKER] Drained {drained} stale frames after reset")
    _reset_done_event.set()
    return tracks, next_track_id, motion


# ============================================================
# Main loop
# ============================================================

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--anomaly-endpoint", "--anomaly_endpoint", dest="anomaly_endpoint",
                        default="tcp://127.0.0.1:5581",
                        help="ZMQ PUSH endpoint for Groq anomaly worker.")
    parser.add_argument("--ack-endpoint", dest="ack_endpoint", default="tcp://127.0.0.1:5562",
                        help="ZMQ PUSH endpoint to send reset ack back to broadcaster.")
    parser.add_argument("--ws_url", default="none")
    parser.add_argument("--yolo_model", default=str(_HERE / "yolo26s.pt"),
                        help="YOLO26 detection weights for general objects (COCO).")
    parser.add_argument("--appearance_model", default=str(_HERE / "yoloe-26s-seg.pt"),
                        help="Open-vocabulary YOLOE-26 weights (appearance + weapon detection).")
    parser.add_argument("--use_appearance", type=int, default=1,
                        help="1 = run the open-vocab YOLOE model.")
    parser.add_argument("--appearance_every_n", type=int, default=1,
                        help="Run YOLOE once every N processed frames (1 = every frame).")
    parser.add_argument("--openvocab_weapon_conf", type=float, default=0.30,
                        help="Confidence floor for open-vocab weapon detections.")
    parser.add_argument("--appearance_conf", type=float, default=0.45,
                        help="Confidence floor for open-vocab appearance tags.")
    parser.add_argument("--appearance_confirm", type=int, default=2,
                        help="Frames a tag must be seen before it sticks to a track.")
    parser.add_argument("--appearance_as_boxes", type=int, default=0,
                        help="1 = emit confirmed appearance detections as their own UI boxes.")
    parser.add_argument("--weapon_confirm_frames", type=int, default=3,
                        help="Frames a weapon track must persist before being emitted.")
    parser.add_argument("--use_suspicious", type=int, default=1,
                        help="1 = load the custom nano suspicious model.")
    parser.add_argument("--disable_suspicious", default="Man_With_Gun,Man_with_Knife",
                        help="Comma-separated nano suspicious classes to ignore.")
    parser.add_argument("--conf_th", type=float, default=0.35)
    parser.add_argument("--send_every_n_frames", type=int, default=1)
    parser.add_argument("--send_overlay", type=int, default=0)
    parser.add_argument("--overlay_jpeg_quality", type=int, default=80)
    parser.add_argument("--use_motion_fallback", type=int, default=1)
    parser.add_argument("--max_track_age", type=int, default=30)
    parser.add_argument("--iou_match_th", type=float, default=0.30)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device for YOLO inference.")
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
        model_suspicious = YOLO(str(_HERE / "Suspicious_Activities_nano.pt"))
    else:
        print("[TRACKER] Suspicious nano model disabled (--use_suspicious 0)")

    yolo_predict_device: Optional[Any] = None
    if args.device == "cpu":
        yolo_predict_device = "cpu"
    elif args.device == "cuda":
        yolo_predict_device = 0

    model_appearance = None
    if args.use_appearance:
        print(f"[TRACKER] Loading YOLOE-26 open-vocab model: {args.appearance_model}...")
        try:
            model_appearance = YOLO(args.appearance_model)
            try:
                model_appearance.set_classes(YOLOE_PROMPTS, model_appearance.get_text_pe(YOLOE_PROMPTS))
            except (AttributeError, TypeError):
                model_appearance.set_classes(YOLOE_PROMPTS)
            print(f"[TRACKER] Open-vocab prompts set: {YOLOE_PROMPTS}")
        except Exception as e:
            print(f"[TRACKER] ⚠ Failed to load open-vocab model ({e}); continuing without it.")
            model_appearance = None

    print("[TRACKER] Warming up models...")
    import numpy as np
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)

    def _warmup(m):
        if m is None:
            return
        kw = dict(conf=0.5, verbose=False)
        if yolo_predict_device is not None:
            kw["device"] = yolo_predict_device
        m.predict(dummy, **kw)

    _warmup(model_objects)
    if model_suspicious:
        _warmup(model_suspicious)
    if model_appearance:
        _warmup(model_appearance)
    print("[TRACKER] ✓ Models ready")

    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(args.sub_endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"frame")
    sub.setsockopt(zmq.RCVHWM, 5)
    sub.setsockopt(zmq.RCVTIMEO, 100)
    print(f"[TRACKER] SUB connect: {args.sub_endpoint}")

    groq_context = zmq.Context()
    groq_socket = groq_context.socket(zmq.PUSH)
    groq_socket.connect(args.anomaly_endpoint)
    print(f"[TRACKER] Connected to Groq worker ({args.anomaly_endpoint})")

    ack_socket = groq_context.socket(zmq.PUSH)
    ack_socket.connect(args.ack_endpoint)
    print(f"[TRACKER] Connected to broadcaster ack socket ({args.ack_endpoint})")

    threading.Thread(
        target=_reset_watcher,
        args=(args.sub_endpoint, args.anomaly_endpoint, args.ack_endpoint),
        daemon=True, name="tracker-reset-watcher",
    ).start()
    print("[TRACKER] Reset-watcher thread started")

    ws = await ws_connect_loop(args.ws_url)

    global _main_loop, _clear_queue
    _main_loop  = asyncio.get_running_loop()
    _clear_queue = asyncio.Queue()

    async def clear_sender():
        nonlocal ws
        while True:
            await _clear_queue.get()
            clear_payload = {
                "type": "tracker_frame", "frame_index": -1,
                "video_time_ms": -1, "tracks": [], "motion_detected": False, "reset": True,
            }
            _log_output(clear_payload)
            try:
                await ws_send_json(ws, clear_payload)
                print("[TRACKER] Sent UI clear payload on reset")
            except Exception as e:
                print(f"[TRACKER] Clear WS send failed: {e}; reconnecting")
                ws = await ws_connect_loop(args.ws_url)

    asyncio.create_task(clear_sender())

    tracks: List[Track] = []
    next_track_id = 1
    motion = MotionDetector() if args.use_motion_fallback else None
    first_frame    = True
    frames_processed = 0
    last_log_ts    = time.time()

    while True:
        # Check reset BEFORE blocking on recv (handles cold-start instantly).
        if _reset_event.is_set():
            _reset_event.clear()
            tracks, next_track_id, motion = _apply_reset(sub, bool(args.use_motion_fallback))
            first_frame = True
            print("[TRACKER] Applied pending reset — cleared tracks and state")
            continue

        try:
            parts = await asyncio.to_thread(sub.recv_multipart)
        except zmq.error.Again:
            continue
        except Exception:
            await asyncio.sleep(0.01)
            continue

        if parts[0] != b"frame" or len(parts) < 4:
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
        h, w  = frame.shape[:2]

        dets_objects = await asyncio.to_thread(run_yolo, model_objects, frame, args.conf_th, yolo_predict_device)
        for d in dets_objects:
            d["source"] = "objects"

        dets_suspicious = []
        if model_suspicious is not None:
            dets_suspicious = await asyncio.to_thread(run_yolo, model_suspicious, frame, args.conf_th, yolo_predict_device)
            for d in dets_suspicious:
                d["source"] = "suspicious"

        dets_objects = [
            d for d in dets_objects
            if d["conf"] >= 0.6 and (d["cls_name"] == "person" or d["cls_name"] in ROBBERY_OBJECT_CLASSES)
        ]
        person_bboxes = [d["bbox"] for d in dets_objects if d["cls_name"] == "person"]

        gated_suspicious = []
        for d in dets_suspicious:
            cls = d["cls_name"]
            if cls in disabled_suspicious:
                continue
            if d["conf"] < SUSPICIOUS_CLASS_THRESHOLDS.get(cls, DEFAULT_SUSPICIOUS_TH):
                continue
            if cls in WEAPON_SUSPICIOUS_CLASSES and not box_overlaps_any(d["bbox"], person_bboxes):
                continue
            gated_suspicious.append(d)
        dets_suspicious = gated_suspicious

        dets = dets_objects + dets_suspicious

        # Mid-inference reset check
        if _reset_event.is_set():
            _reset_event.clear()
            tracks, next_track_id, motion = _apply_reset(sub, bool(args.use_motion_fallback))
            first_frame = True
            print("[TRACKER] Reset happened mid-inference — discarding stale results")
            continue

        if motion is not None and not dets:
            for bb in motion.detect(frame):
                dets.append({"bbox": bb, "cls_name": "moving_object", "conf": 1.0})

        # Open-vocab YOLOE pass
        appearance_dets: List[Dict] = []
        weapon_dets:     List[Dict] = []
        yoloe_dets:      List[Dict] = []
        if (model_appearance is not None and person_bboxes
                and args.appearance_every_n > 0
                and frames_processed % args.appearance_every_n == 0):
            try:
                yoloe_dets = await asyncio.to_thread(
                    run_yolo, model_appearance, frame, 0.20, yolo_predict_device
                )
            except Exception as e:
                print(f"[TRACKER] Open-vocab inference failed: {e}")
                yoloe_dets = []

            for d in yoloe_dets:
                cls = d["cls_name"]
                if cls in WEAPON_PROMPTS:
                    if d["conf"] < args.openvocab_weapon_conf:
                        continue
                    if not box_overlaps_any(d["bbox"], person_bboxes):
                        continue
                    d["source"] = "weapon"
                    weapon_dets.append(d)
                elif cls in APPEARANCE_PROMPTS:
                    if d["conf"] < args.appearance_conf:
                        continue
                    appearance_dets.append(d)

            dets = dets + weapon_dets

        # IOU tracker
        used_tracks: set = set()
        new_tracks: List[Track] = []
        for d in dets:
            bb       = d["bbox"]
            best_iou = 0.0
            best_t   = None
            for t in tracks:
                if t.track_id in used_tracks:
                    continue
                i = iou_xyxy(t.bbox, bb)
                if i > best_iou:
                    best_iou, best_t = i, t

            if best_t is not None and best_iou >= args.iou_match_th:
                used_tracks.add(best_t.track_id)
                best_t.bbox            = bb
                best_t.cls_name        = d["cls_name"]
                best_t.conf            = float(d["conf"])
                best_t.last_seen_frame = frame_idx
                best_t.source          = d.get("source", "")
                best_t.hits           += 1
                new_tracks.append(best_t)
            else:
                t = Track(
                    track_id=next_track_id, bbox=bb, cls_name=d["cls_name"],
                    conf=float(d["conf"]), last_seen_frame=frame_idx, source=d.get("source", ""),
                )
                next_track_id += 1
                used_tracks.add(t.track_id)
                new_tracks.append(t)

        tracks = [t for t in new_tracks if (frame_idx - t.last_seen_frame) <= args.max_track_age]

        for t in tracks:
            if t.source == "suspicious":
                t.bbox = shrink_bbox_tuple(t.bbox, factor=0.2)

        # Attach appearance attributes to nearest person track
        confirmed_appearance_boxes: List[Dict] = []
        if appearance_dets:
            person_tracks = [t for t in tracks if t.cls_name == "person"]
            for ad in appearance_dets:
                best_t, best_i = None, 0.0
                for t in person_tracks:
                    i = iou_xyxy(t.bbox, ad["bbox"])
                    if i > best_i:
                        best_i, best_t = i, t
                if best_t is None or best_i <= 0.0:
                    continue
                tag = ad["cls_name"]
                best_t.attr_counts[tag] = best_t.attr_counts.get(tag, 0) + 1
                if best_t.attr_counts[tag] >= args.appearance_confirm:
                    if tag not in best_t.attributes:
                        best_t.attributes.append(tag)
                    confirmed_appearance_boxes.append({"bbox": ad["bbox"], "tag": tag, "conf": ad["conf"]})

        # Temporal confirmation gate for weapons
        emitted_tracks = [
            t for t in tracks
            if not (is_weapon_track(t) and t.hits < args.weapon_confirm_frames)
        ]

        if args.test == "show_image":
            if yoloe_dets:
                hits = [f"{d['cls_name']} {d['conf']:.2f}" for d in yoloe_dets]
                print(f"[TRACKER/YOLOE] frame {frame_idx} raw hits: {hits}")
            show_debug_frame(frame, emitted_tracks, raw_yoloe=yoloe_dets)

        tracks_payload = []
        for t in emitted_tracks:
            x1, y1, x2, y2 = t.bbox
            tracks_payload.append({
                "track_id": t.track_id, "cls": t.cls_name, "conf": t.conf,
                "source": t.source, "attributes": t.attributes,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

        if args.appearance_as_boxes and confirmed_appearance_boxes:
            for i, ab in enumerate(confirmed_appearance_boxes):
                ax1, ay1, ax2, ay2 = ab["bbox"]
                tracks_payload.append({
                    "track_id": 90000 + i, "cls": ab["tag"], "conf": ab["conf"],
                    "source": "appearance", "attributes": [],
                    "bbox": {"x1": ax1, "y1": ay1, "x2": ax2, "y2": ay2},
                })

        has_motion = any(t.cls_name == "moving_object" for t in emitted_tracks)

        payload = {
            "type": "tracker_frame", "frame_index": frame_idx,
            "video_time_ms": video_time_ms, "frame_size": {"w": w, "h": h},
            "tracks": tracks_payload, "motion_detected": has_motion,
        }

        if args.send_overlay == 1:
            overlay_jpg = encode_jpg(draw_tracks(frame, emitted_tracks), args.overlay_jpeg_quality)
            payload["overlay_jpg_b64"] = base64.b64encode(overlay_jpg).decode("ascii")

        try:
            groq_socket.send(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        except Exception as e:
            print(f"[TRACKER] Failed to send to Groq: {e}")

        _log_output(payload)
        try:
            await ws_send_json(ws, payload)
        except Exception as e:
            print(f"[TRACKER] WS send failed: {e}. Reconnecting...")
            ws = await ws_connect_loop(args.ws_url)

        if _first_frame_pending.is_set():
            _first_frame_pending.clear()
            try:
                ack_socket.send_json({"worker": "tracker", "type": "first_frame_ack"})
                print("[TRACKER] first_frame_ack sent to broadcaster")
            except Exception as e:
                print(f"[TRACKER] first_frame_ack send failed: {e}")

        frames_processed += 1
        if time.time() - last_log_ts >= 2.0:
            print(f"[TRACKER] processed={frames_processed} last_frame={frame_idx} tracks={len(tracks)}")
            last_log_ts = time.time()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
