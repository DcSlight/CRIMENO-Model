# visualize_tracker.py
# Overlay tracker.jsonl detections on top of a video (for debugging).
# Comments in English only.

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import cv2


def load_latest_records_by_frame(jsonl_path: str) -> Dict[int, Dict[str, Any]]:
    """
    If the video was looped during debugging, there may be multiple records with the same frame_index.
    We keep the latest record per frame_index using meta.generated_at_unix_ms.
    """
    latest: Dict[int, Dict[str, Any]] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            frame_idx = rec.get("frame_index")
            if frame_idx is None:
                continue

            meta = rec.get("meta", {}) or {}
            ts = meta.get("generated_at_unix_ms", 0) or 0

            prev = latest.get(int(frame_idx))
            if prev is None:
                latest[int(frame_idx)] = rec
            else:
                prev_ts = (prev.get("meta", {}) or {}).get("generated_at_unix_ms", 0) or 0
                if ts >= prev_ts:
                    latest[int(frame_idx)] = rec

    return latest


def draw_box(
    img,
    bbox_xyxy: List[int],
    label: str,
    color: Tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(
        img,
        label,
        (x1 + 3, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to the original video file.")
    parser.add_argument("--tracker_jsonl", default="tracker.jsonl", help="Path to tracker JSONL output.")
    parser.add_argument("--show_yolo", action="store_true", help="Overlay YOLO detections.")
    parser.add_argument("--show_motion", action="store_true", help="Overlay motion detections.")
    parser.add_argument("--out", default="", help="Optional output video path (mp4). If empty, just displays.")
    parser.add_argument("--start_frame", type=int, default=0, help="Start from a specific frame index.")
    parser.add_argument("--end_frame", type=int, default=-1, help="Stop at a specific frame index (-1 = no limit).")
    parser.add_argument("--fps", type=float, default=0.0, help="Override display/output fps (0 = use source fps).")
    args = parser.parse_args()

    if not args.show_yolo and not args.show_motion:
        # Default: show both
        args.show_yolo = True
        args.show_motion = True

    records = load_latest_records_by_frame(args.tracker_jsonl)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.fps if args.fps and args.fps > 0 else src_fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    frame_idx = 0

    # Jump to start_frame if requested
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        frame_idx = args.start_frame

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.end_frame != -1 and frame_idx > args.end_frame:
            break

        rec = records.get(frame_idx)
        if rec is not None:
            # draw YOLO
            if args.show_yolo:
                yolo_list = ((rec.get("detections") or {}).get("yolo") or [])
                for det in yolo_list:
                    bbox = det.get("bbox_xyxy")
                    if not bbox:
                        continue
                    cls_name = det.get("cls_name", "unknown")
                    conf = det.get("conf", None)
                    tid = det.get("track_id", None)
                    if conf is None:
                        label = f"YOLO {cls_name} id={tid}"
                    else:
                        label = f"YOLO {cls_name} {conf:.2f} id={tid}"
                    draw_box(frame, bbox, label, (0, 200, 0))

            # draw Motion
            if args.show_motion:
                motion_list = ((rec.get("detections") or {}).get("motion") or [])
                for det in motion_list:
                    bbox = det.get("bbox_xyxy")
                    if not bbox:
                        continue
                    mid = det.get("motion_track_id", None)
                    area = det.get("area_px", None)
                    if area is None:
                        label = f"MOTION id={mid}"
                    else:
                        label = f"MOTION id={mid} area={int(area)}"
                    draw_box(frame, bbox, label, (0, 140, 255))

        # HUD
        cv2.putText(
            frame,
            f"frame={frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if writer is not None:
            writer.write(frame)
        else:
            cv2.imshow("Tracker Overlay", frame)
            key = cv2.waitKey(int(1000 / max(1.0, fps))) & 0xFF
            if key == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved overlay video to: {args.out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
