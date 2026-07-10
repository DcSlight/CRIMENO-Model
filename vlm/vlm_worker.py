import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

_HERE = Path(__file__).resolve().parent
_OUTPUT_LOG = _HERE / "vlm_output.jsonl"

import zmq

from vlm_model import load_vlm, analyze_frame, build_summary, pil_from_jpg, now_unix_ms


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
    parser.add_argument("--vlm_model", default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Qwen2.5-VL model weights. 3B ~8GB VRAM, 7B ~16GB VRAM.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--process_every_n_frames", "--every", dest="process_every_n_frames",
                        type=int, default=60, help="Analyze one frame every N frames.")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Token budget for the structured JSON response.")
    args = parser.parse_args()

    model, processor, device, dtype = load_vlm(args.vlm_model, args.device)

    print("[VLM] Warming up...")
    from PIL import Image as _Image
    try:
        analyze_frame(model, processor, device, dtype,
                      _Image.new("RGB", (448, 448), color=(128, 128, 128)),
                      max_new_tokens=16)
    except Exception as e:
        print(f"[VLM] Warmup failed (continuing): {e}")
    print("[VLM] ✓ Warmup complete")

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

            qa = await asyncio.to_thread(
                analyze_frame, model, processor, device, dtype, image, args.max_new_tokens
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

            with open(_OUTPUT_LOG, "a", encoding="utf-8") as f:
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
