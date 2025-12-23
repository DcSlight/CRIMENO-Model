import io
import json
import re
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

import zmq
import torch
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
    if len(parts) < 2:
        raise ValueError(f"Expected at least 2 parts, got {len(parts)}")

    frame_idx = int(parts[0].decode("utf-8"))
    video_time_ms: Optional[int] = None
    jpg_bytes = parts[-1]

    if len(parts) >= 3:
        try:
            video_time_ms = int(parts[1].decode("utf-8"))
        except Exception:
            video_time_ms = None

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--out", default="florence.jsonl")
    parser.add_argument("--model", default="florence-community/Florence-2-base")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    vision_pipe = load_florence_pipeline(args.model, args.device)

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect(args.endpoint)
    print(f"🔗 Connected to video broadcaster on {args.endpoint}")
    print(f"📝 Writing JSONL to: {args.out}")

    TASK_CAPTION = "<MORE_DETAILED_CAPTION>"
    TASK_OD = "<OD>"
    TASK_OCR = "<OCR>"

    try:
        while True:
            frame_idx, video_time_ms, jpg_bytes = recv_frame(socket)
            image = pil_from_jpg(jpg_bytes)

            record: Dict[str, Any] = {
                "schema_version": "florence_frame_v1",
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "raw": {
                    "more_detailed_caption": None,
                    "object_detection": None,
                    "ocr": None,
                },
                "derived": {
                    "datetime_candidates": [],
                },
                "meta": {
                    "generated_at_unix_ms": now_unix_ms(),
                    "model": args.model,
                    "device_requested": args.device,
                },
            }

            try:
                record["raw"]["more_detailed_caption"] = run_task(vision_pipe, image, TASK_CAPTION)
            except Exception as e:
                record["raw"]["more_detailed_caption"] = f"[ERROR running {TASK_CAPTION}] {e}"

            try:
                record["raw"]["object_detection"] = run_task(vision_pipe, image, TASK_OD)
            except Exception as e:
                record["raw"]["object_detection"] = f"[ERROR running {TASK_OD}] {e}"

            try:
                record["raw"]["ocr"] = run_task(vision_pipe, image, TASK_OCR)
            except Exception as e:
                record["raw"]["ocr"] = f"[ERROR running {TASK_OCR}] {e}"

            record["derived"]["datetime_candidates"] = extract_datetime_candidates(record["raw"]["ocr"] or "")

            caption_short = (record["raw"]["more_detailed_caption"] or "").replace("\n", " ").strip()
            if len(caption_short) > 120:
                caption_short = caption_short[:120] + "..."

            dt = record["derived"]["datetime_candidates"]
            dt_str = dt[0] if dt else "-"

            print(
                f"🎬 Frame {frame_idx}"
                + (f" | t={video_time_ms}ms" if video_time_ms is not None else "")
                + f" | dt={dt_str}"
                + f" | caption={caption_short}"
            )

            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (florence worker).")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
