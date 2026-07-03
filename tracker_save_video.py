from ultralytics import YOLO
import cv2
from pathlib import Path


def run_yolo_on_video(
    video_path: str,
    output_path: str = "output_with_bbox.mp4",
    model_path: str = "yolo26n.pt",  # אפשר להחליף ל-yolo26s.pt וכו'
    conf: float = 0.25,
):
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=conf, verbose=False)

        # מצייר bbox, label, confidence
        annotated_frame = results[0].plot()

        writer.write(annotated_frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()

    print(f"Done. Saved video to: {output_path}")


if __name__ == "__main__":
    run_yolo_on_video(
        video_path="videos/market_b.mp4",
        output_path="videos/market_b_with_bbox.mp4",
        model_path="tracker/yolo26n.pt",
        conf=0.65,
    )