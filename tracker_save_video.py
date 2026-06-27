from ultralytics import YOLO
import cv2
import math
from pathlib import Path


def calc_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c

    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    if mag1 == 0 or mag2 == 0:
        return None

    cos_val = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_val))


def run(video_path, output_path="crimeno_yolo_output.mp4"):
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    detect_model = YOLO("yolo26s.pt")
    pose_model = YOLO("yolo26s-pose.pt")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detect_results = detect_model(frame, conf=0.25, verbose=False)
        pose_results = pose_model(frame, conf=0.25, verbose=False)

        annotated = detect_results[0].plot()

        pose_annotated = pose_results[0].plot()
        annotated = cv2.addWeighted(annotated, 0.75, pose_annotated, 0.25, 0)

        keypoints = pose_results[0].keypoints

        if keypoints is not None:
            for person in keypoints.xy:
                pts = person.cpu().numpy()

                if len(pts) < 11:
                    continue

                left_arm = calc_angle(pts[5], pts[7], pts[9])
                right_arm = calc_angle(pts[6], pts[8], pts[10])

                if left_arm is not None:
                    x, y = pts[7]
                    cv2.putText(
                        annotated,
                        f"L arm {left_arm:.0f}",
                        (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                if right_arm is not None:
                    x, y = pts[8]
                    cv2.putText(
                        annotated,
                        f"R arm {right_arm:.0f}",
                        (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                left_wrist_y = pts[9][1]
                right_wrist_y = pts[10][1]
                left_shoulder_y = pts[5][1]
                right_shoulder_y = pts[6][1]

                if left_wrist_y < left_shoulder_y and right_wrist_y < right_shoulder_y:
                    cv2.putText(
                        annotated,
                        "RAISED HANDS?",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )

        cv2.putText(
            annotated,
            f"Frame: {frame_idx}/{total_frames}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames}")

    cap.release()
    writer.release()

    print(f"Saved video: {output_path}")


if __name__ == "__main__":
    run(
        video_path="jewerly_store_short.mp4",
        output_path="crimeno_yolo_output.mp4",
    )