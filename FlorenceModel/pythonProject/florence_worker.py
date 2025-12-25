"""
Florence Worker (Layer 1)

- Subscribes to ZeroMQ video frames
- Every N frames runs Florence image-to-text
- Writes results to JSONL (light I/O)
"""

import argparse
import json
import time
from typing import Dict, Any

import zmq
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


# -------------------------
# ZMQ helpers
# -------------------------
def recv_frame(socket):
    """
    Expected multipart:
    [topic, frame_index, video_time_ms, jpg_bytes]
    """
    parts = socket.recv_multipart()
    if len(parts) < 4:
        raise RuntimeError("Invalid ZMQ frame message")

    frame_index = int(parts[1].decode())
    video_time_ms = int(parts[2].decode())
    jpg_bytes = parts[3]
    return frame_index, video_time_ms, jpg_bytes


def decode_jpg(jpg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode JPG")
    return frame


# -------------------------
# Florence inference
# -------------------------
def run_florence(
    model,
    processor,
    image_bgr: np.ndarray,
    prompt: str,
    device: str,
) -> str:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    inputs = processor(
        text=prompt,
        images=image_rgb,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
        )

    result = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return result.strip()


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--every_n_frames", type=int, default=60)
    parser.add_argument("--out", default="florence.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--prompt",
        default="Describe the scene and any suspicious or criminal activity.",
    )
    args = parser.parse_args()

    print("[FLORENCE] Loading model...")
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base",
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    ).to(args.device)
    model.eval()

    print(f"[FLORENCE] Device: {args.device}")

    # ZMQ SUB
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(args.sub_endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"frame")
    sub.setsockopt(zmq.RCVHWM, 5)

    print(f"[FLORENCE] Subscribed to {args.sub_endpoint}")
    print(f"[FLORENCE] Writing to {args.out} every {args.every_n_frames} frames")

    last_log = time.time()

    while True:
        frame_index, video_time_ms, jpg_bytes = recv_frame(sub)

        if frame_index % args.every_n_frames != 0:
            continue

        frame = decode_jpg(jpg_bytes)

        try:
            caption = run_florence(
                model=model,
                processor=processor,
                image_bgr=frame,
                prompt=args.prompt,
                device=args.device,
            )
        except Exception as e:
            print(f"[FLORENCE] Inference failed: {e}")
            continue

        record: Dict[str, Any] = {
            "frame_index": frame_index,
            "video_time_ms": video_time_ms,
            "prompt": args.prompt,
            "caption": caption,
            "ts_unix_ms": int(time.time() * 1000),
        }

        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        now = time.time()
        if now - last_log > 2:
            print(f"[FLORENCE] wrote frame={frame_index}")
            last_log = now


if __name__ == "__main__":
    main()
