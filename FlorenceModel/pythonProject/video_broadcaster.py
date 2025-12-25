import cv2
import zmq
import time
import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", nargs="?", default="videos/shop.mp4")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--resize_width", type=int, default=640)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--max_fps", type=float, default=0.0, help="0 = no throttling, otherwise cap send rate")
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(args.endpoint)

    print(f"[INFO] Broadcasting video: {args.video_path}")
    print(f"[INFO] ZeroMQ PUB bind: {args.endpoint}")
    print(f"[INFO] resize_width={args.resize_width}, jpeg_quality={args.jpeg_quality}, max_fps={args.max_fps}")

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video_path}")

    frame_idx = 0
    last_send_ts = time.time()
    min_interval = (1.0 / args.max_fps) if args.max_fps and args.max_fps > 0 else 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("🔁 End of video — restarting from beginning...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            # optional throttling
            if min_interval > 0:
                now = time.time()
                elapsed = now - last_send_ts
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                last_send_ts = time.time()

            video_time_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # resize to reduce load
            h, w = frame.shape[:2]
            if w > args.resize_width:
                new_h = int(h * (args.resize_width / w))
                frame = cv2.resize(frame, (args.resize_width, new_h), interpolation=cv2.INTER_AREA)

            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            if not ok2:
                frame_idx += 1
                continue

            jpg_bytes = buf.tobytes()

            # Topic-based multipart:
            # [topic, frame_idx, video_time_ms, jpg]
            socket.send_multipart([
                b"frame",
                str(frame_idx).encode("utf-8"),
                str(video_time_ms).encode("utf-8"),
                jpg_bytes
            ])

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (broadcaster).")
    finally:
        cap.release()
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
