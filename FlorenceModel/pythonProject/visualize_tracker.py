# visualize_tracker.py
# Offline visualization: draws tracker detections on top of the original video.
# Reads tracker.jsonl (our schema) and overlays YOLO + motion boxes.
#
# Usage:
#   python visualize_tracker.py --video videos/shop.mp4 --tracker tracker.jsonl --out annotated.mp4
#
# Notes:
# - This is OFFLINE: run after you generate tracker.jsonl.
# - If you want realtime, you'd need to subscribe to the ZMQ stream and read tracker output live.

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import cv2


def load_tracker_index(tracker_jsonl_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Returns: {frame_index: record}
    If there are duplicates for the same frame_index, the last one wins.
    """
    index: Dict[int, Dict[str, Any]] = {}
    with open(tracker_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                fi = int(rec.get("frame_index"))
                index[fi] = rec
            except Exception:
                continue
    return index


def draw_box(
    frame,
    bbox_xyxy: List[int],
    label: str,
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    # Simple colors (BGR). You can change if needed.
    color = (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_th = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_th)
    y_text = max(0, y1 - 5)
    cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 6, y_text), color, -1)
    cv2.putText(frame, label, (x1 + 3, y_text - 4), font, font_scale, (0, 0, 0), text_th, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--tracker", required=True, help="tracker.jsonl path")
    p.add_argument("--out", default="", help="Output annotated video path (mp4). If empty, shows a window only.")
    p.add_argument("--show", action="store_true", help="Show preview window while processing (press q to quit).")
    p.add_argument("--fps", type=float, default=0.0, help="Override output FPS (0 = use source FPS).")
    args = p.parse_args()

    index = load_tracker_index(args.tracker)
    if not index:
        print("⚠️ tracker index is empty. Did you generate tracker.jsonl before running visualization?")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Could not open video: {args.video}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.fps if args.fps and args.fps > 0 else src_fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
        if not writer.isOpened():
            print(f"❌ Could not open VideoWriter: {args.out}")
            writer = None

    frame_index = 0
    drawn_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rec = index.get(frame_index)
        if rec is not None:
            dets = rec.get("detections", {}) or {}
            yolo = dets.get("yolo", []) or []
            motion = dets.get("motion", []) or []

            # YOLO
            for d in yolo:
                bbox = d.get("bbox_xyxy")
                if not bbox:
                    continue
                cls_name = str(d.get("cls_name", "obj"))
                tid = d.get("track_id", None)
                conf = d.get("conf", None)
                if conf is not None:
                    label = f"yolo:{cls_name} id={tid} conf={conf:.2f}" if isinstance(conf, (int, float)) else f"yolo:{cls_name} id={tid}"
                else:
                    label = f"yolo:{cls_name} id={tid}"
                draw_box(frame, bbox, label, thickness=2)

            # Motion fallback (draw thinner)
            for d in motion:
                bbox = d.get("bbox_xyxy")
                if not bbox:
                    continue
                mid = d.get("motion_track_id", None)
                area = d.get("area_px", None)
                label = f"motion:id={mid} area={int(area)}" if isinstance(area, (int, float)) else f"motion:id={mid}"
                draw_box(frame, bbox, label, thickness=1)

            drawn_frames += 1

        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow("tracker overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"✅ Done. frames={frame_index}, frames_with_overlay={drawn_frames}")
    if args.out:
        print(f"📼 Output: {args.out}")


if __name__ == "__main__":
    main()
