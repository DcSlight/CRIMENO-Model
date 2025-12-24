# visualize_tracker_v2.py
# Offline visualization: draws tracker detections on top of the original video.
#
# Key point:
# - tracker bboxes are in the coordinate system of the JPEG that was sent by the broadcaster (send_size).
# - this script scales bboxes to match the actual video resolution.
#
# Usage:
#   python visualize_tracker_v2.py --video videos/shop.mp4 --tracker tracker.jsonl --show
#   python visualize_tracker_v2.py --video videos/shop.mp4 --tracker tracker.jsonl --out annotated.mp4

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import cv2


def load_tracker_index(tracker_jsonl_path: str) -> Dict[int, Dict[str, Any]]:
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


def get_send_size_from_index(index: Dict[int, Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    for _, rec in index.items():
        stream = rec.get("stream") or {}
        send = stream.get("send_size")
        if isinstance(send, dict) and "w" in send and "h" in send:
            try:
                return int(send["w"]), int(send["h"])
            except Exception:
                continue
    return None


def scale_bbox(bbox_xyxy: List[int], sx: float, sy: float) -> List[int]:
    x1, y1, x2, y2 = bbox_xyxy
    return [
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    ]


def draw_box(frame, bbox_xyxy: List[int], label: str, thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_th = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_th)
    y_text = max(0, y1 - 5)
    cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 6, y_text), color, -1)
    cv2.putText(frame, label, (x1 + 3, y_text - 4), font, font_scale, (0, 0, 0), text_th, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--tracker", required=True)
    p.add_argument("--out", default="", help="If empty -> window only. If set -> writes mp4.")
    p.add_argument("--show", action="store_true")
    p.add_argument("--fps", type=float, default=0.0, help="Override output FPS (0 = use source FPS).")
    args = p.parse_args()

    index = load_tracker_index(args.tracker)
    if not index:
        print("⚠️ tracker index is empty. Generate tracker.jsonl first.")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Could not open video: {args.video}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.fps if args.fps and args.fps > 0 else src_fps
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    send_size = get_send_size_from_index(index)
    if send_size is None:
        send_w, send_h = video_w, video_h
        print("⚠️ No stream.send_size found in JSONL. Assuming no resize.")
    else:
        send_w, send_h = send_size

    sx = video_w / float(send_w) if send_w > 0 else 1.0
    sy = video_h / float(send_h) if send_h > 0 else 1.0
    print(f"📐 video={video_w}x{video_h} | tracker_coords={send_w}x{send_h} | scale=({sx:.3f},{sy:.3f})")

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (video_w, video_h))
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

            for d in yolo:
                bbox = d.get("bbox_xyxy")
                if not bbox:
                    continue
                bbox2 = scale_bbox(bbox, sx, sy)
                cls_name = str(d.get("cls_name", "obj"))
                tid = d.get("track_id", None)
                conf = d.get("conf", None)
                if isinstance(conf, (int, float)):
                    label = f"yolo:{cls_name} id={tid} conf={conf:.2f}"
                else:
                    label = f"yolo:{cls_name} id={tid}"
                draw_box(frame, bbox2, label, thickness=2)

            for d in motion:
                bbox = d.get("bbox_xyxy")
                if not bbox:
                    continue
                bbox2 = scale_bbox(bbox, sx, sy)
                mid = d.get("motion_track_id", None)
                area = d.get("area_px", None)
                label = f"motion:id={mid} area={int(area)}" if isinstance(area, (int, float)) else f"motion:id={mid}"
                draw_box(frame, bbox2, label, thickness=1)

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
