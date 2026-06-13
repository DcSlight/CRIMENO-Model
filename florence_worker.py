import io
import json
import re
import time
import argparse
import asyncio
import threading
from typing import Any, Dict, List, Optional, Tuple

import zmq
import torch
from PIL import Image
from transformers import pipeline

import cv2
import numpy as np

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=False
)

# _reset_event: set by watcher thread when reset arrives, cleared by main loop after applying it.
# _reset_done_event: set by main loop after applying reset, cleared by watcher before each wait.
# This lets the watcher delay the broadcaster ack until the pipeline is actually clean,
# restoring the React loading state while still forwarding the Groq reset immediately.
_reset_event = threading.Event()
_reset_done_event = threading.Event()

ACK_TIMEOUT_S = 15  # wait up to 15s for main loop to finish; broadcaster timeout is 20s


def _reset_watcher(video_endpoint: str, groq_endpoint: str, ack_endpoint: str) -> None:
    """Background thread: immediately resets Groq, then waits for the main inference
    loop to finish its current cycle before acking the broadcaster. This keeps the
    React loading state visible until the pipeline is truly clean."""
    ctx = zmq.Context.instance()

    sub = ctx.socket(zmq.SUB)
    sub.connect(video_endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"reset")

    groq_sock = ctx.socket(zmq.PUSH)
    groq_sock.connect(groq_endpoint)

    ack_sock = ctx.socket(zmq.PUSH)
    ack_sock.connect(ack_endpoint)

    while True:
        sub.recv_multipart()  # block until reset
        print("[Florence/reset-watcher] Reset received — forwarding to Groq, waiting for pipeline to clear")
        groq_sock.send(json.dumps({"type": "reset"}).encode("utf-8"))
        _reset_done_event.clear()
        _reset_event.set()
        # Wait for main loop to finish current inference and drain the buffer
        if not _reset_done_event.wait(timeout=ACK_TIMEOUT_S):
            print("[Florence/reset-watcher] Timeout waiting for pipeline — acking anyway")
        ack_sock.send_json({"worker": "florence", "type": "reset_ack"})
        print("[Florence/reset-watcher] Ack sent to broadcaster")


# --- Regex extraction ---
DATE_PATTERNS = [
    re.compile(r"\b(20\d{2})[-/\.](0[1-9]|1[0-2])[-/\.]([0-2]\d|3[01])\b"),  # YYYY-MM-DD
    re.compile(r"\b([0-2]\d|3[01])[-/\.](0[1-9]|1[0-2])[-/\.](20\d{2})\b"),  # DD-MM-YYYY
]
TIME_PATTERN = re.compile(r"\b([01]\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b")    # HH:MM(:SS)


def apply_background_subtraction(pil_image):
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, 5)
    _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

    fg = cv2.bitwise_and(frame, frame, mask=fg_mask)
    fg_rgb = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
    return Image.fromarray(fg_rgb), fg_mask


def apply_soft_background_blur(pil_image, blur_strength=15):
    # Convert PIL → OpenCV (RGB → BGR)
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Use background subtractor to get foreground mask
    fg_mask = bg_subtractor.apply(frame)

    # Clean mask (reduce noise)
    fg_mask = cv2.medianBlur(fg_mask, 5)
    _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

    # Create inverse mask for background
    bg_mask = cv2.bitwise_not(fg_mask)

    # Blur the background
    blurred_frame = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)

    # Combine sharp foreground with blurred background
    fg_part = cv2.bitwise_and(frame, frame, mask=fg_mask)
    bg_part = cv2.bitwise_and(blurred_frame, blurred_frame, mask=bg_mask)
    combined = cv2.add(fg_part, bg_part)

    # Convert back to PIL (BGR → RGB)
    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    return Image.fromarray(combined_rgb), fg_mask


def apply_focus(pil_image, expand_ratio=0.3, blur_strength=25, use_vignette=True):
    """
    The function receives a PIL image and returns an image with dynamic focus:
    - Cropping around the motion area
    - Soft edge blur (optional)
    """

    # Convert PIL → OpenCV
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 1. Background subtraction to detect motion
    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, 7)
    _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

    # 2. Find contours of motion
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # No motion detected → return original image
        return pil_image

    # 3. Compute bounding box around all motion
    x_min, y_min, x_max, y_max = frame.shape[1], frame.shape[0], 0, 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # 4. Expand bounding box to keep context
    w = x_max - x_min
    h = y_max - y_min

    expand_w = int(w * expand_ratio)
    expand_h = int(h * expand_ratio)

    x1 = max(0, x_min - expand_w)
    y1 = max(0, y_min - expand_h)
    x2 = min(frame.shape[1], x_max + expand_w)
    y2 = min(frame.shape[0], y_max + expand_h)

    # 5. Crop the frame
    cropped = frame[y1:y2, x1:x2]

    # 6. Optional: Vignette blur around edges
    if use_vignette:
        mask = np.zeros((cropped.shape[0], cropped.shape[1]), dtype=np.float32)
        cv2.circle(mask, 
                   (cropped.shape[1] // 2, cropped.shape[0] // 2),
                   int(min(cropped.shape[:2]) * 0.6),
                   1, -1)
        mask = cv2.GaussianBlur(mask, (blur_strength, blur_strength), 0)

        blurred = cv2.GaussianBlur(cropped, (blur_strength, blur_strength), 0)
        vignette = (cropped * mask[..., None] + blurred * (1 - mask[..., None])).astype(np.uint8)
        cropped = vignette

    # Convert back to PIL
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_rgb)


def show_cv_image(pil_image, window_name="Preview"):
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, img)
    cv2.waitKey(1)


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def load_florence_pipeline(model_name: str, device_str: str):
    if device_str == "cuda" and torch.cuda.is_available():
        device = 0
        torch_dtype = torch.float16
        print("Device set to use cuda")
    else:
        device = -1
        torch_dtype = torch.float32
        print("Device set to use cpu")

    vision_pipe = pipeline(
        "image-text-to-text",
        model=model_name,
        device=device,
        torch_dtype=torch_dtype,
    )
    print(f"✅ Florence-2 pipeline loaded ({model_name}) on {device_str if device != -1 else 'cpu'}")
    return vision_pipe


def recv_frame(socket) -> Tuple[int, Optional[int], bytes]:
    parts = socket.recv_multipart()

    # Supports both formats:
    # 1) [topic, frame_idx, video_time_ms, jpg]
    # 2) [frame_idx, video_time_ms, jpg]
    if len(parts) == 4:
        # New broadcaster format with topic
        _, frame_idx_b, video_time_b, jpg_bytes = parts
        frame_idx = int(frame_idx_b.decode("utf-8"))
        try:
            video_time_ms = int(video_time_b.decode("utf-8"))
        except:
            video_time_ms = None
        return frame_idx, video_time_ms, jpg_bytes

    elif len(parts) == 3:
        # Old broadcaster format
        frame_idx = int(parts[0].decode("utf-8"))
        try:
            video_time_ms = int(parts[1].decode("utf-8"))
        except:
            video_time_ms = None
        jpg_bytes = parts[2]
        return frame_idx, video_time_ms, jpg_bytes

    else:
        raise ValueError(f"Unexpected multipart format: {len(parts)} parts")



def pil_from_jpg(jpg_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpg_bytes)).convert("RGB")


def run_task(vision_pipe, image: Image.Image, task_text: str) -> str:
    out = vision_pipe(image, text=task_text)

    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict):
            if "generated_text" in first:
                return str(first["generated_text"])
            return json.dumps(first, ensure_ascii=False)
        return str(first)

    return str(out)


def extract_datetime_candidates(text: str) -> List[str]:
    cands: List[str] = []
    t = text.replace("<OCR>", "").strip()

    for pat in DATE_PATTERNS:
        for m in pat.finditer(t):
            cands.append(m.group(0))

    for m in TIME_PATTERN.finditer(t):
        cands.append(m.group(0))

    uniq: List[str] = []
    for x in cands:
        if x not in uniq:
            uniq.append(x)
    return uniq


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


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-endpoint", default="tcp://127.0.0.1:5560",
                        help="ZeroMQ endpoint to receive video frames (PULL).")
    parser.add_argument("--groq-endpoint", "--anomaly-endpoint", dest="groq_endpoint",
                        default="tcp://127.0.0.1:5580",
                        help="ZeroMQ endpoint to send text records to Groq anomaly worker (PUSH).")
    parser.add_argument("--ack-endpoint", dest="ack_endpoint", default="tcp://127.0.0.1:5562",
                        help="ZeroMQ endpoint to send reset ack back to broadcaster (PUSH).")
    parser.add_argument("--ws-url", "--ws_url", dest="ws_url", default="none",
                        help="WebSocket URL for forwarding records (or 'none' to disable).")
    parser.add_argument("--model", default="florence-community/Florence-2-base")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", default="analysis.jsonl")
    parser.add_argument("--process_every_n_frames", "--every", dest="process_every_n_frames", type=int, default=30,
                        help="Process one frame every N frames.")
    parser.add_argument("--bg", default="none")
    parser.add_argument("--test", default="none")
    args = parser.parse_args()

    vision_pipe = load_florence_pipeline(args.model, args.device)

    print("[Florence] Warming up model...")
    _dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
    for _task in ("<MORE_DETAILED_CAPTION>", "<OD>", "<OCR>"):
        run_task(vision_pipe, _dummy, _task)
    del _dummy
    print("[Florence] ✓ Warmup complete")

    # ZeroMQ – input (video frames)
    context = zmq.Context()
    video_socket = context.socket(zmq.SUB)
    video_socket.connect(args.video_endpoint)
    video_socket.setsockopt(zmq.SUBSCRIBE, b"frame")
    video_socket.setsockopt(zmq.RCVTIMEO, 100)  # wake every 100ms to check _reset_event when idle
    # NOTE: "reset" is intentionally NOT subscribed here.
    # The _reset_watcher thread has its own SUB socket for reset signals so it
    # can ack the broadcaster immediately, even while this thread is mid-inference.
    print(f"🔗 Connected to video broadcaster on {args.video_endpoint}")

    groq_socket = context.socket(zmq.PUSH)
    groq_socket.connect(args.groq_endpoint)
    print(f"🔗 Connected to Groq worker via ZMQ PUSH on {args.groq_endpoint}")

    ack_socket = context.socket(zmq.PUSH)
    ack_socket.connect(args.ack_endpoint)
    print(f"🔗 Connected to broadcaster ack socket on {args.ack_endpoint}")

    threading.Thread(
        target=_reset_watcher,
        args=(args.video_endpoint, args.groq_endpoint, args.ack_endpoint),
        daemon=True,
        name="florence-reset-watcher",
    ).start()
    print("[Florence] Reset-watcher thread started")

    ws = await ws_connect_loop(args.ws_url)

    # Tasks
    TASK_CAPTION = "<MORE_DETAILED_CAPTION>"
    TASK_OD = "<OD>"
    TASK_OCR = "<OCR>"
    WEAPON_QUERY = "gun, handgun, pistol, revolver, firearm, rifle, shotgun, knife, blade, switchblade"
    TASK_WEAPONS = f"<OPEN_VOCABULARY_DETECTION>{WEAPON_QUERY}"

    # Open output file (append)
    out_path = args.out
    print(f"📝 Writing JSONL to: {out_path}")

    def _drain_and_reset_bg():
        """Drain stale frames from the frame-only socket and recreate bg_subtractor."""
        global bg_subtractor
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        drained = 0
        while True:
            try:
                video_socket.recv_multipart(zmq.NOBLOCK)
                drained += 1
            except zmq.error.Again:
                break
        if drained:
            print(f"[Florence] Drained {drained} stale frames from buffer")

    try:
        while True:
            # Check for pending reset BEFORE blocking on recv so cold-start resets are handled
            # instantly even when no frames are flowing (avoids the 15s watcher timeout on first play).
            if _reset_event.is_set():
                _reset_event.clear()
                print("[Florence] Applying pending reset — draining stale frames")
                _drain_and_reset_bg()
                _reset_done_event.set()
                continue

            try:
                parts = await asyncio.to_thread(video_socket.recv_multipart)
            except zmq.error.Again:
                continue  # RCVTIMEO fired; loop back to check _reset_event

            topic = parts[0]
            if topic != b"frame" or len(parts) < 4:
                continue

            _, frame_idx_b, video_time_b, jpg_bytes = parts
            frame_idx = int(frame_idx_b.decode("utf-8"))
            try:
                video_time_ms = int(video_time_b.decode("utf-8"))
            except Exception:
                video_time_ms = None

            if frame_idx % args.process_every_n_frames != 0:
                continue
            image = pil_from_jpg(jpg_bytes)

            if args.bg == "blur":
                image, _ = apply_soft_background_blur(image)
            elif args.bg == "black":
                image, _ = apply_background_subtraction(image)
            elif args.bg == "focus":
                image = apply_focus(image)
            else:
                pass

            if args.test == "show_image":
                show_cv_image(image)

            record: Dict[str, Any] = {
                "type": "florence_frame",
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "raw": {},
                "meta": {
                    "generated_at_unix_ms": now_unix_ms(),
                    "model": args.model,
                }
            }

            # Caption
            try:
                caption = run_task(vision_pipe, image, TASK_CAPTION)
            except Exception as e:
                caption = f"[ERROR running {TASK_CAPTION}] {e}"
            record["raw"]["more_detailed_caption"] = caption

            # OD
            try:
                od = run_task(vision_pipe, image, TASK_OD)
            except Exception as e:
                od = f"[ERROR running {TASK_OD}] {e}"
            record["raw"]["object_detection"] = od

            # OCR
            try:
                ocr = run_task(vision_pipe, image, TASK_OCR)
            except Exception as e:
                ocr = f"[ERROR running {TASK_OCR}] {e}"
            record["raw"]["ocr"] = ocr
            record["text_overlay"] = {
                "datetime_candidates": extract_datetime_candidates(ocr),
            }

            # Open vocab weapons
            try:
                weapons = run_task(vision_pipe, image, TASK_WEAPONS)
            except Exception as e:
                weapons = f"[ERROR running <OPEN_VOCABULARY_DETECTION>] {e}"
            record["raw"]["open_vocab_weapons"] = weapons

            # Reset arrived mid-inference — discard stale results, drain, then unblock watcher.
            if _reset_event.is_set():
                _reset_event.clear()
                print("[Florence] Reset happened mid-inference — discarding stale results, draining buffer")
                _drain_and_reset_bg()
                _reset_done_event.set()
                continue

            # Parse objects and weapons for NestJS
            objects_list = []
            try:
                od_str = record["raw"]["object_detection"]
                if od_str and not od_str.startswith("[ERROR"):
                    # Try to parse if it's JSON-like
                    import ast
                    try:
                        objects_list = ast.literal_eval(od_str) if isinstance(od_str, str) else []
                    except:
                        # If not parseable, just use empty list
                        pass
            except:
                pass

            weapons_list = []
            try:
                weapons_str = record["raw"]["open_vocab_weapons"]
                if weapons_str and not weapons_str.startswith("[ERROR"):
                    import ast
                    try:
                        weapons_list = ast.literal_eval(weapons_str) if isinstance(weapons_str, str) else []
                    except:
                        pass
            except:
                pass

            # Add NestJS-friendly fields
            record["caption"] = caption
            record["objects"] = objects_list
            record["weapons_detected"] = weapons_list

            # Console output — print the FULL caption (no truncation)
            dt = record["text_overlay"]["datetime_candidates"]
            dt_str = dt[0] if dt else "-"
            caption_full = caption.replace("\n", " ").strip()

            print(f"🎬 Frame {frame_idx}"
                  + (f" | t={video_time_ms}ms" if video_time_ms is not None else "")
                  + f" | dt={dt_str}"
                  + f" | caption={caption_full}")

            # ✨ NEW: send to Qwen worker via ZeroMQ (as JSON-line string)
            msg = json.dumps(record, ensure_ascii=False).encode("utf-8")
            groq_socket.send(msg)

            # Send to WebSocket (if enabled)
            try:
                await ws_send_json(ws, record)
            except Exception as e:
                print(f"[WS] Send failed: {e}. Reconnecting...")
                ws = await ws_connect_loop(args.ws_url)

            # --- Write JSONL ---
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Florence worker).")
    finally:
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        video_socket.close()
        groq_socket.close()
        ack_socket.close()
        context.term()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()