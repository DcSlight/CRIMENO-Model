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


# --- Regex extraction (לא "הסקה" — רק חילוץ תבניות) ---
DATE_PATTERNS = [
    re.compile(r"\b(20\d{2})[-/\.](0[1-9]|1[0-2])[-/\.]([0-2]\d|3[01])\b"),  # YYYY-MM-DD
    re.compile(r"\b([0-2]\d|3[01])[-/\.](0[1-9]|1[0-2])[-/\.](20\d{2})\b"),  # DD-MM-YYYY
    re.compile(r"\b(0[0-9]|1[0-9]|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b"),      # HH:MM(:SS)
]


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def load_pipeline(model_name: str, device_str: str):
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
    """
    Supports:
      - [frame_idx, jpg]
      - [frame_idx, video_time_ms, jpg]
      - any longer: uses first as idx, last as jpg, second as time if numeric
    """
    parts = socket.recv_multipart()

    if len(parts) < 2:
        raise RuntimeError(f"Expected at least 2 parts, got {len(parts)}")

    frame_idx = int(parts[0].decode("utf-8", errors="ignore"))
    jpg_bytes = parts[-1]

    video_time_ms: Optional[int] = None
    if len(parts) >= 3:
        try:
            video_time_ms = int(parts[1].decode("utf-8", errors="ignore"))
        except Exception:
            video_time_ms = None

    return frame_idx, video_time_ms, jpg_bytes


def pil_from_jpg(jpg_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpg_bytes)).convert("RGB")


def run_task(vision_pipe, image: Image.Image, task_text: str) -> str:
    """
    Florence returns list[dict|str]. We normalize into a single string.

    IMPORTANT:
    Some transformers versions DO NOT accept keyword argument `text=...` for this pipeline.
    To avoid:
      ImageToTextPipeline._sanitize_parameters() got an unexpected keyword argument 'text'
    we pass the prompt/task as a positional argument.
    """
    try:
        out = vision_pipe(image, task_text)  # safest
    except TypeError:
        out = vision_pipe(image, prompt=task_text)  # fallback for other versions

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

    seen = set()
    uniq: List[str] = []
    for x in cands:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5555", help="ZMQ endpoint from broadcaster")
    p.add_argument("--out", default="analysis.jsonl", help="Output JSONL file path")
    p.add_argument("--model", default="microsoft/Florence-2-large", help="Florence model name")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="cpu|cuda")
    return p


def main():
    args = build_arg_parser().parse_args()

    vision_pipe = load_pipeline(args.model, args.device)

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(args.endpoint)
    print(f"🔗 Connected to video broadcaster on {args.endpoint}")

    out_path = args.out
    print(f"📝 Writing JSONL to: {out_path}")

    TASK_CAPTION = "<MORE_DETAILED_CAPTION>"
    TASK_OCR = "<OCR>"

    try:
        while True:
            frame_idx, video_time_ms, jpg_bytes = recv_frame(socket)
            image = pil_from_jpg(jpg_bytes)

            record: Dict[str, Any] = {
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "raw": {},
                "meta": {
                    "generated_at_unix_ms": now_unix_ms(),
                    "model": args.model,
                    "device": args.device,
                },
            }

            # Caption
            try:
                caption = run_task(vision_pipe, image, TASK_CAPTION)
            except Exception as e:
                caption = f"[ERROR running {TASK_CAPTION}] {e}"
            record["raw"]["caption"] = caption

            # OCR
            try:
                ocr = run_task(vision_pipe, image, TASK_OCR)
            except Exception as e:
                ocr = f"[ERROR running {TASK_OCR}] {e}"
            record["raw"]["ocr"] = ocr
            record["text_overlay"] = {
                "datetime_candidates": extract_datetime_candidates(ocr),
            }

            # Console output
            dt = record["text_overlay"]["datetime_candidates"]
            dt_str = dt[0] if dt else "-"
            caption_short = caption.replace("\n", " ").strip()
            if len(caption_short) > 120:
                caption_short = caption_short[:120] + "..."

            print(
                f"🎬 Frame {frame_idx}"
                + (f" | t={video_time_ms}ms" if video_time_ms is not None else "")
                + f" | dt={dt_str}"
                + f" | caption={caption_short}"
            )

            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (worker).")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
