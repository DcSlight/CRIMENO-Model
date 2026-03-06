import io
import json
import re
import time
import argparse
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import zmq
import torch
from PIL import Image
from transformers import pipeline

# --- Regex extraction ---
DATE_PATTERNS = [
    re.compile(r"\b(20\d{2})[-/\.](0[1-9]|1[0-2])[-/\.]([0-2]\d|3[01])\b"),  # YYYY-MM-DD
    re.compile(r"\b([0-2]\d|3[01])[-/\.](0[1-9]|1[0-2])[-/\.](20\d{2})\b"),  # DD-MM-YYYY
]
TIME_PATTERN = re.compile(r"\b([01]\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b")    # HH:MM(:SS)


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
    parser.add_argument("--qwen-endpoint", default="tcp://127.0.0.1:5580",
                        help="ZeroMQ endpoint to send text records to Qwen worker (PUSH).")
    parser.add_argument("--ws-url", "--ws_url", dest="ws_url", default="none",
                        help="WebSocket URL for forwarding records (or 'none' to disable).")
    parser.add_argument("--model", default="florence-community/Florence-2-base")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", default="analysis.jsonl")
    parser.add_argument("--process_every_n_frames", "--every", dest="process_every_n_frames", type=int, default=30,
                        help="Process one frame every N frames.")
    args = parser.parse_args()

    vision_pipe = load_florence_pipeline(args.model, args.device)

    # ZeroMQ – input (video frames)
    context = zmq.Context()
    video_socket = context.socket(zmq.SUB)
    video_socket.connect(args.video_endpoint)
    video_socket.setsockopt(zmq.SUBSCRIBE, b"frame")
    print(f"🔗 Connected to video broadcaster on {args.video_endpoint}")

    # ZeroMQ – output (text to Qwen)
    qwen_socket = context.socket(zmq.PUSH)
    qwen_socket.connect(args.qwen_endpoint)
    print(f"🔗 Connected Qwen text output PUSH on {args.qwen_endpoint}")

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

    try:
        while True:
            frame_idx, video_time_ms, jpg_bytes = await asyncio.to_thread(recv_frame, video_socket)
            if frame_idx % args.process_every_n_frames != 0:
                continue
            image = pil_from_jpg(jpg_bytes)

            record: Dict[str, Any] = {
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

            # Console output
            dt = record["text_overlay"]["datetime_candidates"]
            dt_str = dt[0] if dt else "-"
            caption_short = caption.replace("\n", " ").strip()
            if len(caption_short) > 120:
                caption_short = caption_short[:120] + "..."

            print(f"🎬 Frame {frame_idx}"
                  + (f" | t={video_time_ms}ms" if video_time_ms is not None else "")
                  + f" | dt={dt_str}"
                  + f" | caption={caption_short}")

            # ✨ NEW: send to Qwen worker via ZeroMQ (as JSON-line string)
            msg = json.dumps(record, ensure_ascii=False).encode("utf-8")
            qwen_socket.send(msg)

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
        qwen_socket.close()
        context.term()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()