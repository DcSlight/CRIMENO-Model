# vlm_worker.py
# Full-frame Vision-Language reasoning layer for the CRIMENO pipeline.
# - Subscribes to the broadcaster PUB stream (topic: "frame")
# - Sends each (throttled) full frame to a LOCAL VLM (PaliGemma 2) and asks a
#   fixed set of robbery-focused questions (behavior + weapons + concealment)
# - PUSHes a compact "vlm_frame" record to the Groq anomaly worker (port 5581)
# - Optionally forwards the same record to NestJS over WebSocket
#
# Runs fully locally — no API key. One-time: `huggingface-cli login` and accept
# the (free) PaliGemma license on the model page.
#
# This is an ADDITIVE layer: Florence (scene caption + OCR) and the tracker
# (weapons/appearance) keep running; Groq fuses all three.

import argparse
import asyncio
import io
import json
import threading
import time
from typing import Any, Dict, List, Optional

import zmq
from PIL import Image

try:
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    import transformers
    # Quiet the per-call processor/deprecation warnings so the VLM output is readable.
    transformers.logging.set_verbosity_error()
except Exception:  # pragma: no cover - import guard
    torch = None
    AutoProcessor = None
    PaliGemmaForConditionalGeneration = None

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Fixed robbery-focused question set asked of the full frame each cycle.
# PaliGemma-friendly: ONE free-form "describe" (its strongest mode) + atomic yes/no
# cues. Never list multiple distinct actions with "or" in one question — PaliGemma
# echoes the last option instead of reasoning (the old "...or running?" always
# answered "running"). yes/no questions may join near-synonyms (one concept) safely.
QUESTIONS: List[tuple] = [
    ("actions", "Describe what each person is doing."),
    ("gun", "Is anyone holding a gun?"),
    ("counter", "Is a person reaching over the counter or into a display case?"),
    ("handsup", "Does anyone have their hands raised in the air?"),
    ("mask", "Is anyone's face covered by a mask, hood, or helmet?"),
]

# The describe question gets room; yes/no answers are capped short to stay terse + fast.
SHORT_ANSWER_TOKENS = 12


def now_unix_ms() -> int:
    return int(time.time() * 1000)


# -------------------------
# VLM loading + inference
# -------------------------
def load_vlm(model_id: str, device_str: str):
    if PaliGemmaForConditionalGeneration is None:
        raise RuntimeError("transformers/torch not installed (need PaliGemma support)")

    use_cuda = device_str == "cuda" and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    device = "cuda:0" if use_cuda else "cpu"

    print(f"[VLM] Loading {model_id} on {device} ({dtype})...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"✅ [VLM] {model_id} ready on {device}")
    return model, processor, device, dtype


def ask_vlm(model, processor, device, dtype, image: Image.Image, question: str,
            max_new_tokens: int = 48) -> str:
    """Ask PaliGemma one question about the image; return only the generated answer.
    PaliGemma VQA convention: prompt = 'answer en <question>' (the processor adds the
    image tokens). Verify against the model card if you change checkpoints."""
    prompt = f"answer en {question}"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    if "pixel_values" in inputs and dtype is not None:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = out[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def analyze_frame(model, processor, device, dtype, image: Image.Image,
                  max_new_tokens: int = 64) -> Dict[str, str]:
    answers: Dict[str, str] = {}
    for key, q in QUESTIONS:
        # Free-form describe gets the full budget; yes/no cues stay short.
        tokens = max_new_tokens if key == "actions" else SHORT_ANSWER_TOKENS
        try:
            answers[key] = ask_vlm(model, processor, device, dtype, image, q,
                                   max_new_tokens=tokens)
        except Exception as e:
            answers[key] = f"[ERROR] {e}"
    return answers


def build_summary(answers: Dict[str, str]) -> str:
    return (
        f"People's actions: {answers.get('actions', '-')}. "
        f"Gun visible: {answers.get('gun', '-')}. "
        f"Reaching over counter/display case: {answers.get('counter', '-')}. "
        f"Hands raised (possible victim): {answers.get('handsup', '-')}. "
        f"Face concealed: {answers.get('mask', '-')}."
    )


def pil_from_jpg(jpg_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpg_bytes)).convert("RGB")


# -------------------------
# WebSocket sender (mirrors florence_worker)
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
    parser.add_argument("--video-endpoint", "--video_endpoint", dest="video_endpoint",
                        default="tcp://127.0.0.1:5560",
                        help="ZeroMQ endpoint to receive video frames (SUB).")
    parser.add_argument("--anomaly-endpoint", "--anomaly_endpoint", dest="anomaly_endpoint",
                        default="tcp://127.0.0.1:5580",
                        help="ZeroMQ PUSH endpoint of the Groq/Qwen anomaly worker.")
    parser.add_argument("--ws-url", "--ws_url", dest="ws_url", default="none",
                        help="WebSocket URL for forwarding VLM records (or 'none').")
    parser.add_argument("--vlm_model", default="google/paligemma2-3b-mix-448",
                        help="Local VLM weights (PaliGemma 2 mix). Use -896 for more small-figure detail.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--process_every_n_frames", "--every", dest="process_every_n_frames",
                        type=int, default=60, help="Analyze one frame every N frames.")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Token budget for the free-form 'describe' answer; yes/no cues are capped short.")
    parser.add_argument("--test", default="none")
    args = parser.parse_args()

    model, processor, device, dtype = load_vlm(args.vlm_model, args.device)

    print("[VLM] Warming up...")
    _dummy = Image.new("RGB", (448, 448), color=(128, 128, 128))
    try:
        _ = ask_vlm(model, processor, device, dtype, _dummy, "What is happening?", max_new_tokens=8)
    except Exception as e:
        print(f"[VLM] Warmup failed (continuing): {e}")
    print("[VLM] ✓ Warmup complete")

    # ZeroMQ — input (video frames). We also subscribe to "reset" so we can drain
    # stale frames on a video switch. No ack/first_frame handshake: the broadcaster
    # only waits for florence+tracker, and Groq clears our buffer when they reset.
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(args.video_endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"frame")
    sub.setsockopt(zmq.SUBSCRIBE, b"reset")
    sub.setsockopt(zmq.RCVHWM, 5)
    sub.setsockopt(zmq.RCVTIMEO, 200)
    print(f"🔗 [VLM] Connected to broadcaster on {args.video_endpoint}")

    groq_socket = context.socket(zmq.PUSH)
    groq_socket.connect(args.anomaly_endpoint)
    print(f"🔗 [VLM] Connected to anomaly worker via ZMQ PUSH on {args.anomaly_endpoint}")

    ws = await ws_connect_loop(args.ws_url)

    def _drain():
        drained = 0
        while True:
            try:
                sub.recv_multipart(zmq.NOBLOCK)
                drained += 1
            except zmq.error.Again:
                break
        if drained:
            print(f"[VLM] Drained {drained} stale frames after reset")

    frames_processed = 0
    last_log_ts = time.time()

    try:
        while True:
            try:
                parts = await asyncio.to_thread(sub.recv_multipart)
            except zmq.error.Again:
                continue
            except Exception:
                await asyncio.sleep(0.01)
                continue

            topic = parts[0]
            if topic == b"reset":
                print("[VLM] Reset received — draining stale frames")
                _drain()
                continue
            if topic != b"frame" or len(parts) < 4:
                continue

            frame_idx = int(parts[1].decode("utf-8"))
            try:
                video_time_ms = int(parts[2].decode("utf-8"))
            except Exception:
                video_time_ms = -1
            jpg_bytes = parts[3]

            if args.process_every_n_frames > 1 and (frame_idx % args.process_every_n_frames) != 0:
                continue

            image = pil_from_jpg(jpg_bytes)

            # Heavy VLM inference off the event loop so WS stays responsive.
            answers = await asyncio.to_thread(
                analyze_frame, model, processor, device, dtype, image, args.max_new_tokens
            )
            summary = build_summary(answers)

            record = {
                "type": "vlm_frame",
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "qa": answers,
                "summary": summary,
                "meta": {"generated_at_unix_ms": now_unix_ms(), "model": args.vlm_model},
            }

            print(f"🤖 [VLM] frame {frame_idx} | {summary}")

            try:
                groq_socket.send(json.dumps(record, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                print(f"[VLM] Failed to send to anomaly worker: {e}")

            try:
                await ws_send_json(ws, record)
            except Exception as e:
                print(f"[WS] Send failed: {e}. Reconnecting...")
                ws = await ws_connect_loop(args.ws_url)

            frames_processed += 1
            now = time.time()
            if now - last_log_ts >= 5.0:
                print(f"[VLM] processed={frames_processed} last_frame={frame_idx}")
                last_log_ts = now

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (VLM worker).")
    finally:
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        sub.close()
        groq_socket.close()
        context.term()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
