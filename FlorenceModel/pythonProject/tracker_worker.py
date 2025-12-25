# tracker_worker.py
# - SUBscribes to frames (topic "frame")
# - Runs YOLO tracking/detection
# - PUSHes results to NestJS aggregator (topic "tracker")

import argparse
import json
import time
from typing import Any, Dict, List, Tuple, Optional

import zmq
import cv2

# If you already use ultralytics in your current tracker_worker.py, keep it.
# This version assumes you have ultralytics installed.
from ultralytics import YOLO


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def parse_frame_multipart(msg: List[bytes]) -> Tuple[int, int, int, int, bytes]:
    """
    Expected:
      [b"frame", frame_id, video_time_ms, width, height, jpg_bytes]
    Backward compatible with old format:
      [b"frame", frame_id, video_time_ms, jpg_bytes]
    """
    if len(msg) == 6:
        _, f_id, t_ms, w, h, jpg = msg
        return int(f_id), int(t_ms), int(w), int(h), jpg
    if len(msg) == 4:
        _, f_id, t_ms, jpg = msg
        return int(f_id), int(t_ms), 0, 0, jpg
    raise ValueError(f"Unexpected frame multipart size: {len(msg)}")


def decode_jpeg(jpg_bytes: bytes) -> Optional[Any]:
    arr = cv2.imdecode(
        # pylint: disable=no-member
        # OpenCV expects numpy array; imdecode accepts buffer too, but safest:
        # We'll use frombuffer
        __import__("numpy").frombuffer(jpg_bytes, dtype=__import__("numpy").uint8),
        cv2.IMREAD_COLOR,
    )
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_endpoint", default="tcp://127.0.0.1:5560", help="ZMQ PUB endpoint from broadcaster")
    parser.add_argument("--push_endpoint", default="tcp://127.0.0.1:5571", help="ZMQ PUSH endpoint to NestJS (Nest binds PULL)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--every", type=int, default=5, help="process every N frames")
    parser.add_argument("--yolo_model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    model = YOLO(args.yolo_model)

    context = zmq.Context()

    sub = context.socket(zmq.SUB)
    sub.connect(args.frames_endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"frame")

    push = context.socket(zmq.PUSH)
    push.connect(args.push_endpoint)

    print(f"[TRACKER] SUB frames: {args.frames_endpoint}")
    print(f"[TRACKER] PUSH results -> Nest: {args.push_endpoint}")

    processed = 0

    try:
        while True:
            msg = sub.recv_multipart()
            frame_id, video_time_ms, width, height, jpg_bytes = parse_frame_multipart(msg)

            if args.every > 1 and (frame_id % args.every != 0):
                continue

            frame = decode_jpeg(jpg_bytes)
            if frame is None:
                continue

            # YOLO inference
            # We keep it simple: detection only. If you use built-in track(), swap it in.
            results = model.predict(frame, conf=args.conf, verbose=False, device=0 if args.device == "cuda" else "cpu")

            dets: List[Dict[str, Any]] = []
            r0 = results[0]
            if r0.boxes is not None and len(r0.boxes) > 0:
                boxes = r0.boxes
                # boxes.xyxy, boxes.conf, boxes.cls
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()

                names = model.names if hasattr(model, "names") else {}

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    cls_id = int(clss[i])
                    dets.append({
                        "id": None,  # if you have a tracker with IDs, put it here
                        "cls": names.get(cls_id, str(cls_id)),
                        "cls_id": cls_id,
                        "conf": float(confs[i]),
                        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    })

            payload = {
                "type": "tracker",
                "ts_unix_ms": now_unix_ms(),
                "frame_id": frame_id,
                "video_time_ms": video_time_ms,
                "frame_width": width,
                "frame_height": height,
                "detections": dets,
            }

            push.send_multipart([b"tracker", json.dumps(payload).encode("utf-8")])

            processed += 1
            if processed % 50 == 0:
                print(f"[TRACKER] processed={processed}")

    except KeyboardInterrupt:
        print("\n[TRACKER] stopped.")
    finally:
        sub.close(0)
        push.close(0)
        context.term()


if __name__ == "__main__":
    main()
