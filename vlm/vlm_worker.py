import argparse
import asyncio
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_OUTPUT_LOG = _HERE / "logs_output.jsonl"

sys.path.insert(0, str(_PROJECT_ROOT))
import session_log

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

import zmq

from vlm_model import load_vlm, analyze_frame, build_summary, now_unix_ms


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
                        default="tcp://127.0.0.1:5581",
                        help="ZeroMQ PUSH endpoint of the Groq anomaly worker.")
    parser.add_argument("--ws-url", "--ws_url", dest="ws_url", default="none",
                        help="WebSocket URL for forwarding VLM records (or 'none').")
    parser.add_argument("--vlm_model", default="qwen2.5vl:7b",
                        help="Ollama vision model tag. Runs locally via Ollama — pull it first "
                             "with `ollama pull <tag>`. Hosted vision was dropped after every "
                             "free tier failed in practice (Groq deprecated Llama 4 Scout, "
                             "Groq's qwen/qwen3.6-27b is a flaky preview reasoning model, "
                             "Gemini's free tier caps out at 5 requests/minute). Two local "
                             "models were also tried and rejected: minicpm-v4.5/4.6 crash "
                             "official Ollama (exit 0xc0000005) — that architecture needs an "
                             "unofficial fork to run at all — and llama3.2-vision:11b fails to "
                             "load (`unknown model architecture: 'mllama'`) because Ollama's "
                             "new inference engine dropped mllama support with no fix/ETA. "
                             "Ollama's new engine only natively supports Llama 4, Gemma 3, "
                             "Qwen 2.5 VL, and Mistral Small 3.1 — qwen2.5vl:7b is in that set "
                             "and strong at structured/OCR-style output, which matches this "
                             "worker's schema-constrained JSON use case. gemma3:12b is the "
                             "fallback if needed. Do NOT use qwen2-vl (no '.5') — different, "
                             "buggier model. NOTE: on Pascal-class GPUs (e.g. Tesla/GRID P40, "
                             "compute capability 6.1) qwen2.5vl also crashed with "
                             "`exit 0xc0000005` — that was NOT the model's fault, it was "
                             "Ollama's CUDA backend segfaulting on Pascal in the new engine. "
                             "Fix: set env var CUDA_VISIBLE_DEVICES=-1 (persist with "
                             "`setx CUDA_VISIBLE_DEVICES -1`, then fully restart the Ollama "
                             "service). This hides the device from the CUDA backend only — "
                             "Ollama then falls back to its Vulkan backend, which supports "
                             "Pascal correctly and still runs on GPU (confirmed: 29/29 layers "
                             "offloaded, flash attention enabled, no crash). Do not mistake "
                             "this for a CPU fallback.")
    parser.add_argument("--ollama-host", "--ollama_host", dest="ollama_host", default="",
                        help="Ollama server URL (or set OLLAMA_HOST env var). "
                             "Defaults to http://localhost:11434.")
    parser.add_argument("--process_every_n_frames", "--every", dest="process_every_n_frames",
                        type=int, default=60, help="Analyze one frame every N frames.")
    parser.add_argument("--max_new_tokens", "--max_output_tokens", dest="max_new_tokens",
                        type=int, default=512, help="Token budget for the structured JSON response.")
    args = parser.parse_args()

    client = load_vlm(args.vlm_model, args.ollama_host)

    # Loads the model into VRAM and surfaces load/architecture-compatibility errors here,
    # before the rest of the pipeline (broadcaster/tracker/anomaly worker) is up and running —
    # mirrors tracker_worker.py's model warm-up.
    print("[VLM] Warming up model...")
    _warmup_jpg = io.BytesIO()
    Image.new("RGB", (64, 64)).save(_warmup_jpg, format="JPEG")
    _warmup_qa = await asyncio.to_thread(
        analyze_frame, client, args.vlm_model, _warmup_jpg.getvalue(), args.max_new_tokens
    )
    print(f"[VLM] Warm-up result: {build_summary(_warmup_qa)}")
    print("[VLM] ✓ Model ready")

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

    # Resolve the current session's log path; falls back to the legacy flat file if
    # no session exists yet (e.g. broadcaster hasn't played a video, or session_log
    # hiccuped).
    current_log = session_log.resolve_log_path("vlm") or _OUTPUT_LOG

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
                # Broadcaster may have started a new business/session before this reset.
                current_log = session_log.resolve_log_path("vlm") or _OUTPUT_LOG
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

            qa = await asyncio.to_thread(
                analyze_frame, client, args.vlm_model, jpg_bytes, args.max_new_tokens
            )
            summary = build_summary(qa)

            record = {
                "type":          "vlm_frame",
                "frame_index":   frame_idx,
                "video_time_ms": video_time_ms,
                "qa":            qa,
                "summary":       summary,
                "meta": {"generated_at_unix_ms": now_unix_ms(), "model": args.vlm_model},
            }

            print(f"🤖 [VLM] frame {frame_idx} | {summary}")

            with open(current_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

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
