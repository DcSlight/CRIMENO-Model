import io
import json
import re
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

import zmq
import torch
import numpy as np
from PIL import Image
from transformers import pipeline


DATE_PATTERNS = [
    re.compile(r"\b(20\d{2})[-/\.](0[1-9]|1[0-2])[-/\.]([0-2]\d|3[01])\b"),
    re.compile(r"\b([0-2]\d|3[01])[-/\.](0[1-9]|1[0-2])[-/\.](20\d{2})\b"),
]
TIME_PATTERN = re.compile(r"\b([01]\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b")


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def load_florence_pipeline(model_name: str, device_str: str):
    print(f"[FLORENCE] Loading model: {model_name}...")
    load_start = time.time()
    
    if device_str == "cuda" and torch.cuda.is_available():
        device = 0
        torch_dtype = torch.float16
        actual_device = "cuda"
        print("[FLORENCE] Device set to use CUDA")
    else:
        device = -1
        torch_dtype = torch.float32
        actual_device = "cpu"
        print("[FLORENCE] Device set to use CPU")

    vision_pipe = pipeline(
        "image-text-to-text",
        model=model_name,
        device=device,
        torch_dtype=torch_dtype,
    )
    
    load_time = time.time() - load_start
    print(f"[FLORENCE] ✓ Model loaded ({load_time:.2f}s)")
    
    # CRITICAL: Warm up the model with dummy inference
    print("[FLORENCE] Warming up model with dummy inference...")
    warmup_start = time.time()
    
    # Create a realistic dummy image (640x480 RGB)
    dummy_img = Image.fromarray(
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    )
    
    # Run all three tasks to warm up everything
    tasks = [
        "<MORE_DETAILED_CAPTION>",
        "<OD>",
        "<OCR>"
    ]
    
    for task in tasks:
        try:
            _ = vision_pipe(dummy_img, text=task)
        except Exception as e:
            print(f"[FLORENCE] Warmup task {task} failed: {e}")
    
    warmup_time = time.time() - warmup_start
    print(f"[FLORENCE] ✓ Warmup complete ({warmup_time:.2f}s)")
    print(f"[FLORENCE] ✓ Total initialization: {load_time + warmup_time:.2f}s")
    print("[FLORENCE] Ready to process frames!")
    
    return vision_pipe, actual_device


def recv_frame_sub(socket) -> Tuple[int, Optional[int], bytes]:
    parts = socket.recv_multipart()
    # [topic, frame_idx, video_time_ms, jpg]
    if len(parts) < 4:
        raise ValueError(f"Expected 4 parts, got {len(parts)}")

    frame_idx = int(parts[1].decode("utf-8"))
    try:
        video_time_ms = int(parts[2].decode("utf-8"))
    except Exception:
        video_time_ms = None

    jpg_bytes = parts[3]
    return frame_idx, video_time_ms, jpg_bytes


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


# -------------------------
# WebSocket sender (NestJS PromptsGateway)
# -------------------------
async def ws_connect_loop(ws_url: str):
    """
    Keeps trying to connect, returns an open websocket.
    """
    import asyncio
    import websockets  # lazy import

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
    import json as _json
    await ws.send(_json.dumps(payload, ensure_ascii=False))


# -------------------------
# Main loop
# -------------------------
async def main_async():
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5562")  # Updated default to real-time endpoint
    parser.add_argument("--out", default="florence.jsonl")
    parser.add_argument("--model", default="florence-community/Florence-2-base")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--every", type=int, default=60, help="process every N frames")

    # New: send the SAME record to NestJS over WS (PromptsGateway)
    parser.add_argument(
        "--ws_url",
        default="ws://127.0.0.1:3000/ws/prompts",
        help="NestJS WS endpoint (PromptsGateway)",
    )
    parser.add_argument(
        "--ws_enable",
        type=int,
        default=1,
        help="1=send to NestJS via WS, 0=disable WS sending",
    )

    args = parser.parse_args()

    # Load and warm up model BEFORE connecting to ZMQ
    vision_pipe, actual_device = load_florence_pipeline(args.model, args.device)

    # Now connect to ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(args.endpoint)
    socket.setsockopt(zmq.SUBSCRIBE, b"frame")
    # Set receive timeout to avoid blocking forever
    socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
    
    print(f"🔗 SUB connected to broadcaster on {args.endpoint}")
    print(f"📝 Writing JSONL to: {args.out}")
    print(f"⚙️ Florence processes every {args.every} frames")
    if args.ws_enable == 1:
        print(f"🌐 WS send enabled -> {args.ws_url}")
    else:
        print("🌐 WS send disabled")

    TASK_CAPTION = "<MORE_DETAILED_CAPTION>"
    TASK_OD = "<OD>"
    TASK_OCR = "<OCR>"

    processed = 0
    skipped = 0
    first_frame = True

    ws = None
    if args.ws_enable == 1:
        try:
            ws = await ws_connect_loop(args.ws_url)
        except Exception as e:
            print(f"[WS] Disabled (failed to init): {e}")
            ws = None

    try:
        while True:
            try:
                # ZMQ recv is blocking; run it in a thread to not block asyncio loop
                frame_idx, video_time_ms, jpg_bytes = await asyncio.to_thread(recv_frame_sub, socket)
            except Exception as e:
                # Timeout or other error - continue waiting
                await asyncio.sleep(0.01)
                continue

            if first_frame:
                print(f"[FLORENCE] ✓ First frame received (idx={frame_idx}) - processing started!")
                first_frame = False

            # Skip fast without decoding
            if args.every > 1 and (frame_idx % args.every != 0):
                skipped += 1
                continue

            # Track processing time for first few frames
            process_start = time.time()
            
            image = pil_from_jpg(jpg_bytes)

            record: Dict[str, Any] = {
                "type": "florence_frame",  # for NestJS PromptsGateway log routing
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "raw": {},
                "text_overlay": {},
                "meta": {
                    "generated_at_unix_ms": now_unix_ms(),
                    "model": args.model,
                    "device": actual_device,
                    "worker": "florence_worker",
                    "every": args.every,
                },
            }

            try:
                caption = run_task(vision_pipe, image, TASK_CAPTION)
            except Exception as e:
                caption = f"[ERROR running {TASK_CAPTION}] {e}"
            record["raw"]["more_detailed_caption"] = caption

            try:
                od = run_task(vision_pipe, image, TASK_OD)
            except Exception as e:
                od = f"[ERROR running {TASK_OD}] {e}"
            record["raw"]["object_detection"] = od

            try:
                ocr = run_task(vision_pipe, image, TASK_OCR)
            except Exception as e:
                ocr = f"[ERROR running {TASK_OCR}] {e}"
            record["raw"]["ocr"] = ocr

            record["text_overlay"]["datetime_candidates"] = extract_datetime_candidates(record["raw"]["ocr"])

            # Append JSONL (same as before)
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Send to NestJS via WS (same record)
            if ws is not None:
                try:
                    await ws_send_json(ws, record)
                except Exception as e:
                    print(f"[WS] Send failed: {e} -> reconnect")
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    try:
                        ws = await ws_connect_loop(args.ws_url)
                    except Exception as e2:
                        print(f"[WS] Reconnect failed, disabling WS: {e2}")
                        ws = None

            processed += 1
            process_time = time.time() - process_start
            
            dt = record["text_overlay"]["datetime_candidates"]
            dt_str = dt[0] if dt else "-"
            caption_short = caption.replace("\n", " ").strip()
            if len(caption_short) > 120:
                caption_short = caption_short[:120] + "..."
            
            # Show processing time for first 5 frames to verify warmup worked
            if processed <= 5:
                print(f"🧠 Florence | Frame {frame_idx} | t={video_time_ms}ms | dt={dt_str} | Process time: {process_time:.2f}s")
                print(f"   Caption: {caption_short}")
            else:
                print(f"🧠 Florence | Frame {frame_idx} | t={video_time_ms}ms | dt={dt_str} | caption={caption_short}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (florence_worker).")
        print(f"[STATS] processed={processed}, skipped={skipped}")
    finally:
        try:
            if ws is not None:
                await ws.close()
        except Exception:
            pass
        socket.close()
        context.term()


def main():
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()