import io
import json
import time
import argparse
from typing import Dict, Any, List, Tuple

import zmq
import torch
from PIL import Image
from transformers import pipeline


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def load_florence_pipeline(model_name: str, device_str: str):
    if device_str == "cuda" and torch.cuda.is_available():
        device = 0
        torch_dtype = torch.float16
        actual_device = "cuda"
    else:
        device = -1
        torch_dtype = torch.float32
        actual_device = "cpu"

    vision_pipe = pipeline(
        "image-to-text",
        model=model_name,
        device=device,
        torch_dtype=torch_dtype,
    )
    return vision_pipe, actual_device


def parse_frame_multipart(msg: List[bytes]) -> Tuple[int, int, int, int, bytes]:
    if len(msg) == 6:
        _, f_id, t_ms, w, h, jpg = msg
        return int(f_id), int(t_ms), int(w), int(h), jpg
    if len(msg) == 4:
        _, f_id, t_ms, jpg = msg
        return int(f_id), int(t_ms), 0, 0, jpg
    raise ValueError(f"Unexpected frame multipart size: {len(msg)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--push_endpoint", default="tcp://127.0.0.1:5572")
    parser.add_argument("--model", default="florence-community/Florence-2-base")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--every", type=int, default=60, help="process every N frames")
    args = parser.parse_args()

    vision_pipe, actual_device = load_florence_pipeline(args.model, args.device)
    print(f"[FLORENCE] model={args.model} device={actual_device}")

    context = zmq.Context()

    sub = context.socket(zmq.SUB)
    sub.connect(args.frames_endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"frame")

    push = context.socket(zmq.PUSH)
    push.connect(args.push_endpoint)

    print(f"[FLORENCE] SUB frames: {args.frames_endpoint}")
    print(f"[FLORENCE] PUSH results -> Nest: {args.push_endpoint}")

    processed = 0

    try:
        while True:
            msg = sub.recv_multipart()
            frame_id, video_time_ms, width, height, jpg_bytes = parse_frame_multipart(msg)

            if args.every > 1 and (frame_id % args.every != 0):
                continue

            image = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")

            # Florence output format can vary; keep it safe.
            out = vision_pipe(image)
            # Usually: [{"generated_text": "..."}]
            generated_text = ""
            if isinstance(out, list) and out:
                if isinstance(out[0], dict):
                    generated_text = str(out[0].get("generated_text", ""))
                else:
                    generated_text = str(out[0])
            else:
                generated_text = str(out)

            payload: Dict[str, Any] = {
                "type": "florence",
                "ts_unix_ms": now_unix_ms(),
                "frame_id": frame_id,
                "video_time_ms": video_time_ms,
                "frame_width": width,
                "frame_height": height,
                "caption": generated_text,
                "raw": out,
            }

            push.send_multipart([b"florence", json.dumps(payload).encode("utf-8")])

            processed += 1
            if processed % 20 == 0:
                print(f"[FLORENCE] processed={processed}")

    except KeyboardInterrupt:
        print("\n[FLORENCE] stopped.")
    finally:
        sub.close(0)
        push.close(0)
        context.term()


if __name__ == "__main__":
    main()
