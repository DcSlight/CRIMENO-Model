import argparse
import json
import time
from typing import Any, Dict, Optional, Tuple

import zmq
from PIL import Image
import io


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def recv_frame_sub(sub_socket) -> Tuple[int, Optional[int], bytes]:
    parts = sub_socket.recv_multipart()
    frame_idx = int(parts[1].decode("utf-8"))
    video_time_ms = int(parts[2].decode("utf-8")) if parts[2] else None
    jpg_bytes = parts[3]
    return frame_idx, video_time_ms, jpg_bytes


def safe_extract_text(model_out: Any) -> str:
    """
    Florence output format can vary; keep it safe.
    """
    if model_out is None:
        return ""

    # common HF pipeline output: list[dict]
    if isinstance(model_out, list) and model_out:
        item = model_out[0]
        if isinstance(item, dict):
            for k in ["generated_text", "text", "caption", "answer"]:
                if k in item and isinstance(item[k], str):
                    return item[k]
        if isinstance(item, str):
            return item

    if isinstance(model_out, dict):
        for k in ["generated_text", "text", "caption", "answer"]:
            if k in model_out and isinstance(model_out[k], str):
                return model_out[k]

    if isinstance(model_out, str):
        return model_out

    return str(model_out)[:2000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--pub_endpoint", default="tcp://127.0.0.1:5572")
    parser.add_argument("--every", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_jsonl", default="", help="Optional JSONL output path (debug)")
    parser.add_argument("--print_every", type=int, default=10)
    args = parser.parse_args()

    # Your existing pipeline init (keep yours if already working)
    from transformers import pipeline
    vision_pipe = pipeline("image-to-text", model="microsoft/Florence-2-base", device=args.device)

    ctx = zmq.Context()

    sub = ctx.socket(zmq.SUB)
    sub.connect(args.frames_endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"frame")

    pub = ctx.socket(zmq.PUB)
    pub.bind(args.pub_endpoint)

    print(f"[FLORENCE] SUB frames: {args.frames_endpoint} (topic=frame)")
    print(f"[FLORENCE] PUB results bind: {args.pub_endpoint} (topic=florence)")
    print(f"[FLORENCE] every={args.every} device={args.device}")

    processed = 0
    skipped = 0

    def write_jsonl(path: str, rec: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    try:
        while True:
            frame_idx, video_time_ms, jpg_bytes = recv_frame_sub(sub)

            if args.every > 1 and (frame_idx % args.every != 0):
                skipped += 1
                continue

            image = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")

            # Florence output format can vary; keep it safe.
            out = vision_pipe(image)
            text = safe_extract_text(out)

            record: Dict[str, Any] = {
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "meta": {
                    "generated_at_unix_ms": now_unix_ms(),
                    "worker": "florence_worker_rt",
                    "device": args.device,
                    "every": args.every,
                },
                "caption": text,
                "raw": out if isinstance(out, (dict, list, str)) else str(out),
            }

            pub.send_multipart([b"florence", json.dumps(record).encode("utf-8")])

            if args.out_jsonl:
                write_jsonl(args.out_jsonl, record)

            processed += 1
            if args.print_every > 0 and (processed % args.print_every == 0):
                t_ms = f"{video_time_ms}ms" if video_time_ms is not None else "-"
                print(f"[FLORENCE] processed={processed} skipped={skipped} frame={frame_idx} t={t_ms} text_len={len(text)}")

    except KeyboardInterrupt:
        print("\n[FLORENCE] Stopped by user.")
    finally:
        sub.close()
        pub.close()
        ctx.term()


if __name__ == "__main__":
    main()
