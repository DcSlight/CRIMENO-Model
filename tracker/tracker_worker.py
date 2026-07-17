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

from config import ROBBERY_OBJECT_CLASSES
from detection import (
    Track,
    run_yolo, iou_xyxy,
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

def _apply_reset(sub):
    """Clear all tracker state, drain stale SUB buffer, signal watcher."""
    tracks: List[Track] = []
    next_track_id = 1
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
    return tracks, next_track_id


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
    parser.add_argument("--conf_th", type=float, default=0.35)
    parser.add_argument("--send_every_n_frames", type=int, default=1)
    parser.add_argument("--send_overlay", type=int, default=0)
    parser.add_argument("--overlay_jpeg_quality", type=int, default=80)
    parser.add_argument("--max_track_age", type=int, default=30)
    parser.add_argument("--iou_match_th", type=float, default=0.30)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device for YOLO inference.")
    parser.add_argument("--test", default="none")
    args = parser.parse_args()

    if YOLO is None:
        raise RuntimeError("ultralytics not installed")

    print(f"[TRACKER] Loading YOLO26 object model: {args.yolo_model}...")
    model_objects = YOLO(args.yolo_model)

    yolo_predict_device: Optional[Any] = None
    if args.device == "cpu":
        yolo_predict_device = "cpu"
    elif args.device == "cuda":
        yolo_predict_device = 0

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
    first_frame    = True
    frames_processed = 0
    last_log_ts    = time.time()

    while True:
        # Check reset BEFORE blocking on recv (handles cold-start instantly).
        if _reset_event.is_set():
            _reset_event.clear()
            tracks, next_track_id = _apply_reset(sub)
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

        dets_objects = [
            d for d in dets_objects
            if d["conf"] >= 0.6 and (d["cls_name"] == "person" or d["cls_name"] in ROBBERY_OBJECT_CLASSES)
        ]

        dets = dets_objects

        # Mid-inference reset check
        if _reset_event.is_set():
            _reset_event.clear()
            tracks, next_track_id = _apply_reset(sub)
            first_frame = True
            print("[TRACKER] Reset happened mid-inference — discarding stale results")
            continue

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

        emitted_tracks = tracks

        if args.test == "show_image":
            show_debug_frame(frame, emitted_tracks)

        tracks_payload = []
        for t in emitted_tracks:
            x1, y1, x2, y2 = t.bbox
            tracks_payload.append({
                "track_id": t.track_id, "cls": t.cls_name, "conf": t.conf,
                "source": t.source, "attributes": [],
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

        payload = {
            "type": "tracker_frame", "frame_index": frame_idx,
            "video_time_ms": video_time_ms, "frame_size": {"w": w, "h": h},
            "tracks": tracks_payload, "motion_detected": False,
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
