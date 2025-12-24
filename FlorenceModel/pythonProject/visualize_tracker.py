# visualize_tracker.py
# Offline visualization for tracker.jsonl
# Resizes video frames to the SAME resolution used by the tracker
# so bounding boxes are aligned correctly.
#
# Usage:
#   python visualize_tracker.py --video videos/shop.mp4 --tracker tracker.jsonl --show
#   python visualize_tracker.py --video videos/shop.mp4 --tracker tracker.jsonl --out annotated.mp4

import argparse
import json
from typing import Dict, Any, List

import cv2


# MUST match the resize width used in video_broadcaster.py
TRACKER_RESIZE_WIDTH = 640


def load_tracker_index(path: str) -> Dict[int, Dict[str, Any]]:
    """
    Load tracker.jsonl into {frame_index -> record}
    If duplicate frame_index exists, last one wins.
    """
    index = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                fi = int(rec["frame_index"])
                index[fi] = rec
            except Exception:
                continue
    return index


def resize_like_tracker(frame):
    h, w = frame.shape[:2]
    if w == TRACKER_RESIZE_WIDTH:
        return frame

    scale = TRACKER_RESIZE_WIDTH / w
    new_h = int(h * scale)
    return cv2.resize(frame, (TRACKER_RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)


def draw_box(frame, bbox_xyxy: List[int], label: str, color, thickness: int):
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    t = 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, t)

    y = max(0, y1 - 6)
    cv2.rectangle(frame, (x1, y - th - 6), (x1 + tw + 6, y), color, -1)
    cv2.putText(frame, label, (x1 + 3, y - 4), font, scale, (0, 0, 0), t, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Original video file")
    parser.add_argument("--tracker", required=True, help="tracker.jsonl path")
    parser.add_argument("--out", default="", help="Output video (mp4). If empty, no file is saved.")
    parser.add_argument("--show", action="store_true", help="Show preview window")
    args = parser.parse_args()

    tracker = load_tracker_index(args.tracker)
    if not tracker:
        print("❌ tracker.jsonl is empty or invalid")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    frame_index = 0
    drawn = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 🔑 CRITICAL: resize video frame to tracker resolution
        frame = resize_like_tracker(frame)

        rec = tracker.get(frame_index)
        if rec:
            dets = rec.get("detections", {})

            # YOLO detections (green)
            for d in dets.get("yolo", []):
                bbox = d.get("bbox_xyxy")
                if not bbox:
                    continue
                cls = d.get("cls_name", "obj")
                tid = d.get("track_id")
                conf = d.get("conf")
                label = f"yolo:{cls} id={tid} {conf:.2f}" if isinstance(conf, (int, float)) else f"yolo:{cls} id={tid}"
                draw_box(frame, bbox, label, (0, 255, 0), 2)

            # Motion detections (orange)
            for d in dets.get("motion", []):
                bbox = d.get("bbox_xyxy")
                if not bbox:
                    continue
                mid = d.get("motion_track_id")
                area = d.get("area_px")
                label = f"motion:id={mid} area={int(area)}" if isinstance(area, (int, float)) else f"motion:id={mid}"
                draw_box(frame, bbox, label, (0, 165, 255), 1)

            drawn += 1

        if args.out:
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
            writer.write(frame)

        if args.show:
            cv2.imshow("Tracker Debug Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1

    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"✅ Done. Frames={frame_index}, frames_with_detections={drawn}")
    if args.out:
        print(f"📼 Saved to: {args.out}")


if __name__ == "__main__":
    main()
