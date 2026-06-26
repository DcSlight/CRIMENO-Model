# vlm_worker.py
# Full-frame Vision-Language reasoning layer for the CRIMENO pipeline.
# - Subscribes to the broadcaster PUB stream (topic: "frame")
# - Sends each (throttled) full frame to a LOCAL VLM (Qwen2.5-VL-Instruct) with a single
#   structured-JSON call that returns scene description + weapon/behavior cues
# - PUSHes a compact "vlm_frame" record to the Groq anomaly worker (port 5581)
# - Optionally forwards the same record to NestJS over WebSocket
#
# Runs fully locally — no API key needed.
# One-time install:
#   pip install "transformers>=4.49" accelerate
#   (torch and pillow are already installed from the tracker/previous VLM)
#
# Model options:
#   --vlm_model Qwen/Qwen2.5-VL-3B-Instruct   (~8 GB VRAM, recommended)
#   --vlm_model Qwen/Qwen2.5-VL-7B-Instruct   (~16 GB VRAM, higher quality)

import argparse
import asyncio
import io
import json
import re
import time
from typing import Any, Dict, Optional

import zmq
from PIL import Image

try:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import transformers
    transformers.logging.set_verbosity_error()
except Exception:  # pragma: no cover
    torch = None
    Qwen2_5_VLForConditionalGeneration = None
    AutoProcessor = None

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Structured analysis prompt
# ============================================================

ANALYSIS_INSTRUCTION = """You are a surveillance camera analyst. Analyze this security camera frame from a store.
Return ONLY a valid JSON object with these exact keys — no extra text before or after:

{
  "description": "1-2 sentence factual description of the scene",
  "people_actions": "what each visible person is doing",
  "appearance": "what each person is wearing (clothing, headwear, face covering)",
  "weapon": "describe any visible weapon (gun, knife, etc.) or 'none'",
  "gun": "yes or no",
  "knife": "yes or no",
  "reaching_counter": "yes or no — is anyone reaching over the counter or into a display case",
  "hands_up": "yes or no — does anyone have hands raised (possible victim or surrender pose)",
  "face_concealed": "yes or no — is anyone's face covered by a mask, hood, or helmet",
  "aggression": "yes or no — is anyone being physically aggressive or threatening"
}

Rules:
- Base every answer ONLY on what is clearly visible in the frame.
- Answer "no" for binary fields if you are uncertain.
- Do NOT guess, hallucinate, or invent details.
- Return ONLY the JSON object with no other text."""


def now_unix_ms() -> int:
    return int(time.time() * 1000)


# ============================================================
# VLM loading + inference
# ============================================================

def load_vlm(model_id: str, device_str: str):
    if Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError(
            "Qwen2.5-VL requires transformers>=4.49 and accelerate. "
            "Run: pip install \"transformers>=4.49\" accelerate"
        )

    use_cuda = device_str == "cuda" and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    device = "cuda:0" if use_cuda else "cpu"

    print(f"[VLM] Loading {model_id} on {device} ({dtype})...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"✅ [VLM] {model_id} ready on {device}")
    return model, processor, device, dtype


def analyze_frame(model, processor, device, dtype, image: Image.Image,
                  max_new_tokens: int = 256) -> Dict[str, str]:
    """Single Qwen2.5-VL call → returns structured dict.
    On parse failure returns a safe default dict (no crash, no junk cues)."""

    # Build the chat message with the image. For single PIL images the processor
    # can handle them directly without qwen-vl-utils.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": ANALYSIS_INSTRUCTION},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = out[0][input_len:]
    raw_text = processor.decode(generated, skip_special_tokens=True).strip()

    return _parse_vlm_output(raw_text)


def _parse_vlm_output(text: str) -> Dict[str, str]:
    """Strip markdown fences and parse the JSON. Returns a safe fallback on failure."""
    # Strip ```json ... ``` fences
    cleaned = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    # Try full parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return _sanitize(parsed)
    except Exception:
        pass

    # Try extracting the first {...} block
    match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return _sanitize(parsed)
        except Exception:
            pass

    # Parse failure — safe fallback (raw text in description, all cues default to "no")
    print(f"[VLM] ⚠️ JSON parse failed. Raw output: {text[:200]!r}")
    return {
        "description": text[:300] if text and not text.lower().startswith("sorry") else "Scene analysis unavailable.",
        "people_actions": "-",
        "appearance": "-",
        "weapon": "none",
        "gun": "no",
        "knife": "no",
        "reaching_counter": "no",
        "hands_up": "no",
        "face_concealed": "no",
        "aggression": "no",
    }


def _sanitize(d: Dict) -> Dict[str, str]:
    """Ensure all values are strings and no 'Sorry...' junk is treated as a yes cue."""
    result = {}
    binary_keys = {"gun", "knife", "reaching_counter", "hands_up", "face_concealed", "aggression"}
    for k, v in d.items():
        s = str(v).strip()
        # If a binary cue is a refusal/error phrase, default to "no"
        if k in binary_keys and any(
            s.lower().startswith(p)
            for p in ("sorry", "unanswerable", "i cannot", "i can't", "i am not")
        ):
            s = "no"
        result[k] = s
    return result


def build_summary(qa: Dict[str, str]) -> str:
    """Flatten the structured QA dict into the single-line summary consumed by build_vlm_sentence."""
    return (
        f"Scene: {qa.get('description', '-')}. "
        f"People: {qa.get('people_actions', '-')}. "
        f"Appearance: {qa.get('appearance', '-')}. "
        f"Weapon: {qa.get('weapon', 'none')}. "
        f"Gun visible: {qa.get('gun', 'no')}. "
        f"Knife visible: {qa.get('knife', 'no')}. "
        f"Reaching over counter: {qa.get('reaching_counter', 'no')}. "
        f"Hands raised (possible victim): {qa.get('hands_up', 'no')}. "
        f"Face concealed: {qa.get('face_concealed', 'no')}. "
        f"Aggressive: {qa.get('aggression', 'no')}."
    )


def pil_from_jpg(jpg_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpg_bytes)).convert("RGB")


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
# Main loop
# ============================================================

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-endpoint", "--video_endpoint", dest="video_endpoint",
                        default="tcp://127.0.0.1:5560",
                        help="ZeroMQ endpoint to receive video frames (SUB).")
    parser.add_argument("--anomaly-endpoint", "--anomaly_endpoint", dest="anomaly_endpoint",
                        default="tcp://127.0.0.1:5580",
                        help="ZeroMQ PUSH endpoint of the Groq anomaly worker.")
    parser.add_argument("--ws-url", "--ws_url", dest="ws_url", default="none",
                        help="WebSocket URL for forwarding VLM records (or 'none').")
    parser.add_argument("--vlm_model", default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Qwen2.5-VL model weights. 3B ~8GB VRAM, 7B ~16GB VRAM.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--process_every_n_frames", "--every", dest="process_every_n_frames",
                        type=int, default=60, help="Analyze one frame every N frames.")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Token budget for the structured JSON response.")
    parser.add_argument("--test", default="none")
    args = parser.parse_args()

    model, processor, device, dtype = load_vlm(args.vlm_model, args.device)

    print("[VLM] Warming up...")
    _dummy = Image.new("RGB", (448, 448), color=(128, 128, 128))
    try:
        _ = analyze_frame(model, processor, device, dtype, _dummy, max_new_tokens=16)
    except Exception as e:
        print(f"[VLM] Warmup failed (continuing): {e}")
    print("[VLM] ✓ Warmup complete")

    # ZeroMQ — subscribe to frames + reset signals.
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
            qa = await asyncio.to_thread(
                analyze_frame, model, processor, device, dtype, image, args.max_new_tokens
            )
            summary = build_summary(qa)

            record = {
                "type": "vlm_frame",
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "qa": qa,
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
