import cv2
import zmq
import time
import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", nargs="?", default="videos/shop.mp4")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5560")  # ZMQ PUB bind
    parser.add_argument("--resize_width", type=int, default=640)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--max_fps", type=float, default=0.0, help="0 = no throttling")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video_path}")

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(args.endpoint)

    print(f"[INFO] Broadcasting: {args.video_path}")
    print(f"[INFO] ZMQ PUB bind: {args.endpoint}")

    frame_idx = 0
    last_send = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[INFO] End of video. Rewinding...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Resize (keep aspect ratio)
            h, w = frame.shape[:2]
            if args.resize_width > 0 and w != args.resize_width:
                scale = args.resize_width / float(w)
                new_w = args.resize_width
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Encode JPEG
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue
            jpg_bytes = buf.tobytes()

            # Video time in ms (best-effort)
            video_time_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # Optional FPS cap
            if args.max_fps and args.max_fps > 0:
                now = time.time()
                min_dt = 1.0 / args.max_fps
                dt = now - last_send
                if dt < min_dt:
                    time.sleep(min_dt - dt)
                last_send = time.time()

            # Multipart:
            # [topic, frame_id, video_time_ms, width, height, jpg_bytes]
            fh, fw = frame.shape[:2]
            socket.send_multipart([
                b"frame",
                str(frame_idx).encode("utf-8"),
                str(video_time_ms).encode("utf-8"),
                str(fw).encode("utf-8"),
                str(fh).encode("utf-8"),
                jpg_bytes,
            ])

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (broadcast).")
    finally:
        cap.release()
        socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
