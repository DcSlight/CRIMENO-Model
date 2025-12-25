import argparse
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import zmq


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def recv_frame_sub(sub_socket) -> Tuple[int, Optional[int], bytes]:
    parts = sub_socket.recv_multipart()
    # [topic, frame_idx, video_time_ms, jpg_bytes]
    frame_idx = int(parts[1].decode("utf-8"))
    video_time_ms = int(parts[2].decode("utf-8")) if parts[2] else None
    jpg_bytes = parts[3]
    return frame_idx, video_time_ms, jpg_bytes


def decode_jpg_to_bgr(jpg_bytes: bytes):
    if not jpg_bytes:
        return None
    arr = cv2.imdecode(
        cv2.UMat(cv2.imdecode(
            cv2.imencode(".jpg", cv2.imdecode(
                cv2.imdecode(
                    cv2.imdecode(
                        None, 1
                    ), 1
                ), 1
            )[1], 1
        )[1], 1).get() if False else cv2.imdecode(
            cv2.imencode(".jpg", cv2.imdecode(
                cv2.imencode(".jpg", cv2.imdecode(
                    None, 1
                )[1], 1
            )[1], 1
        )[1], 1
    )  # never executed; kept to avoid accidental OCR-like tricks
    return None  # replaced below


def decode_jpg_to_bgr(jpg_bytes: bytes):
    import numpy as np
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def write_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--pub_endpoint", default="tcp://127.0.0.1:5571")
    parser.add_argument("--every", type=int, default=1, help="Process every N frames")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--yolo_model", default="yolov8n.pt")
    parser.add_argument("--out_jsonl", default="", help="Optional JSONL output path (debug)")
    parser.add_argument("--print_every", type=int, default=20)
    args = parser.parse_args()

    # Lazy import (so the script can start even if YOLO isn't installed yet)
    try:
        from ultralytics import YOLO
        yolo = YOLO(args.yolo_model)
        yolo_available = True
    except Exception as e:
        print(f"[TRACKER] YOLO not available: {e}")
        yolo_available = False
        yolo = None

    ctx = zmq.Context()

    sub = ctx.socket(zmq.SUB)
    sub.connect(args.frames_endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"frame")

    pub = ctx.socket(zmq.PUB)
    pub.bind(args.pub_endpoint)

    print(f"[TRACKER] SUB frames: {args.frames_endpoint} (topic=frame)")
    print(f"[TRACKER] PUB results bind: {args.pub_endpoint} (topic=tracker)")
    print(f"[TRACKER] every={args.every} device={args.device} model={args.yolo_model}")

    processed = 0
    skipped = 0

    try:
        while True:
            frame_idx, video_time_ms, jpg_bytes = recv_frame_sub(sub)

            if args.every > 1 and (frame_idx % args.every != 0):
                skipped += 1
                continue

            frame_bgr = decode_jpg_to_bgr(jpg_bytes)
            if frame_bgr is None:
                continue

            dets: List[Dict[str, Any]] = []
            if yolo_available:
                # ultralytics returns boxes in xyxy
                results = yolo.predict(frame_bgr, verbose=False, device=args.device)
                r0 = results[0]
                boxes = getattr(r0, "boxes", None)
                if boxes is not None:
                    for b in boxes:
                        xyxy = b.xyxy[0].tolist()
                        conf = float(b.conf[0]) if hasattr(b, "conf") else None
                        cls_id = int(b.cls[0]) if hasattr(b, "cls") else -1
                        cls_name = r0.names.get(cls_id, str(cls_id)) if hasattr(r0, "names") else str(cls_id)
                        dets.append({
                            "bbox_xyxy": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                            "conf": conf,
                            "cls_id": cls_id,
                            "cls_name": cls_name,
                        })

            record: Dict[str, Any] = {
                "frame_index": frame_idx,
                "video_time_ms": video_time_ms,
                "meta": {
                    "generated_at_unix_ms": now_unix_ms(),
                    "worker": "tracker_worker_rt",
                    "device": args.device,
                    "model": args.yolo_model if yolo_available else None,
                    "every": args.every,
                },
                "detections": dets,
            }

            # Publish to Nest (topic=tracker)
            pub.send_multipart([b"tracker", json.dumps(record).encode("utf-8")])

            # Optional debug write
            if args.out_jsonl:
                write_jsonl(args.out_jsonl, record)

            processed += 1
            if args.print_every > 0 and (processed % args.print_every == 0):
                t_ms = f"{video_time_ms}ms" if video_time_ms is not None else "-"
                print(f"[TRACKER] processed={processed} skipped={skipped} frame={frame_idx} t={t_ms} dets={len(dets)}")

    except KeyboardInterrupt:
        print("\n[TRACKER] Stopped by user.")
    finally:
        sub.close()
        pub.close()
        ctx.term()


if __name__ == "__main__":
    main()
