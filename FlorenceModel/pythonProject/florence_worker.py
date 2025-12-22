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
    re.compile(r"\b(20\d{2})[-/\.](0[1-9]|1[0-2])\b"),  # YYYY-MM
    re.compile(r"\b(0[1-9]|1[0-2])[-/\.](20\d{2})\b"),  # MM-YYYY
]

TIME_PATTERNS = [
    re.compile(r"\b([01]\d|2[0-3]):([0-5]\d):([0-5]\d)\b"),  # HH:MM:SS
    re.compile(r"\b([01]\d|2[0-3]):([0-5]\d)\b"),  # HH:MM
]

PLATE_PATTERNS = [
    # Israel-like 7/8 digits with optional hyphens (best-effort only; might false positive)
    re.compile(r"\b\d{2,3}-?\d{2,3}-?\d{2,3}\b"),
    re.compile(r"\b\d{7,8}\b"),
]


def extract_datetime_candidates(text: str) -> List[str]:
    if not text:
        return []
    candidates: List[str] = []
    for pat in DATE_PATTERNS:
        candidates.extend([m.group(0) for m in pat.finditer(text)])
    for pat in TIME_PATTERNS:
        candidates.extend([m.group(0) for m in pat.finditer(text)])
    # de-dup while preserving order
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def extract_plate_candidates(text: str) -> List[str]:
    if not text:
        return []
    candidates: List[str] = []
    for pat in PLATE_PATTERNS:
        candidates.extend([m.group(0) for m in pat.finditer(text)])
    # de-dup while preserving order
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def pil_from_jpg(jpg_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpg_bytes)).convert("RGB")


def load_florence_pipeline(model_name: str, device: str):
    use_device = 0 if (device == "cuda" and torch.cuda.is_available()) else -1
    if device == "cuda" and use_device == -1:
        print("⚠️  CUDA requested but not available. Falling back to CPU.")
    pipe = pipeline(
        task="image-to-text",
        model=model_name,
        device=use_device,
        trust_remote_code=True
    )
    return pipe


def run_task(pipe, image: Image.Image, prompt: str) -> Any:
    # Florence style: provide prompt text as "text" input.
    # Keep it minimal and deterministic.
    out = pipe(image, text=prompt)
    return out


def recv_frame(socket) -> Tuple[int, Optional[int], bytes]:
    """
    Receives multipart:
      [frame_idx (ascii), video_time_ms (ascii or empty), jpg_bytes]
    """
    parts = socket.recv_multipart()
    if len(parts) != 3:
        raise ValueError(f"Expected 3 parts, got {len(parts)}")
    frame_idx = int(parts[0].decode("utf-8"))
    ts_raw = parts[1].decode("utf-8").strip()
    video_time_ms = int(ts_raw) if ts_raw else None
    jpg_bytes = parts[2]
    return frame_idx, video_time_ms, jpg_bytes


def safe_get_first_text(out: Any) -> str:
    """
    Florence pipeline output can vary; normalize to a string.
    """
    if out is None:
        return ""
    # common: list of dicts with 'generated_text'
    if isinstance(out, list) and len(out) > 0:
        first = out[0]
        if isinstance(first, dict):
            if "generated_text" in first and isinstance(first["generated_text"], str):
                return first["generated_text"]
            # some variants might use 'text'
            if "text" in first and isinstance(first["text"], str):
                return first["text"]
        # sometimes it's plain strings
        if isinstance(first, str):
            return first
    # fallback: dict
    if isinstance(out, dict):
        for k in ("generated_text", "text"):
            if k in out and isinstance(out[k], str):
                return out[k]
    # last resort
    return str(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--out", default="analysis.jsonl")
    parser.add_argument("--model", default="florence-community/Florence-2-base")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    vision_pipe = load_florence_pipeline(args.model, args.device)

    # ZeroMQ
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect(args.endpoint)
    print(f"🔗 Connected to video broadcaster on {args.endpoint}")

    # Open output file (append)
    out_path = args.out
    print(f"📝 Writing JSONL to: {out_path}")

    # Pure data extraction tasks (no inference)
    # IMPORTANT: <MORE_DETAILED_CAPTION> must be the ONLY content!
    TASK_CAPTION = "<MORE_DETAILED_CAPTION>"
    TASK_OD = "<OD>"
    TASK_OCR = "<OCR>"

    try:
        while True:
            frame_idx, video_time_ms, jpg_bytes = recv_frame(socket)
            image = pil_from_jpg(jpg_bytes)

            record: Dict[str, Any] = {
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "ts_worker_ms": int(time.time() * 1000),
                "raw": {},
                "text_overlay": {
                    "datetime_candidates": [],
                    "plate_candidates": [],
                    "ocr_text": ""
                },
            }

            # Caption (more detailed)
            try:
                cap_out = run_task(vision_pipe, image, TASK_CAPTION)
                caption = safe_get_first_text(cap_out)
            except Exception as e:
                caption = f"[ERROR running {TASK_CAPTION}] {e}"
                cap_out = None

            record["raw"]["more_detailed_caption"] = caption

            # Object detection (structured-ish)
            try:
                od_out = run_task(vision_pipe, image, TASK_OD)
            except Exception as e:
                od_out = f"[ERROR running {TASK_OD}] {e}"
            record["raw"]["od"] = od_out

            # OCR
            try:
                ocr_out = run_task(vision_pipe, image, TASK_OCR)
                ocr_text = safe_get_first_text(ocr_out)
            except Exception as e:
                ocr_text = f"[ERROR running {TASK_OCR}] {e}"
                ocr_out = None

            record["raw"]["ocr"] = ocr_out
            record["text_overlay"]["ocr_text"] = ocr_text

            # Extract patterns from OCR text
            record["text_overlay"]["datetime_candidates"] = extract_datetime_candidates(ocr_text)
            record["text_overlay"]["plate_candidates"] = extract_plate_candidates(ocr_text)

            # --- Console output (compact but useful) ---
            dt = record["text_overlay"]["datetime_candidates"]
            dt_str = dt[0] if dt else "-"
            caption_short = caption.replace("\n", " ").strip()
            if len(caption_short) > 120:
                caption_short = caption_short[:120] + "..."

            print(f"🎬 Frame {frame_idx}"
                  + (f" | t={video_time_ms}ms" if video_time_ms is not None else "")
                  + f" | dt={dt_str}"
                  + f" | caption={caption_short}")

            # --- Write JSONL ---
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (worker).")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
