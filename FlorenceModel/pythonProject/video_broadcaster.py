import cv2
import zmq
import time
import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", nargs="?", default="videos/shop.mp4")
    parser.add_argument("--endpoint_florence", default="tcp://127.0.0.1:5560")
    parser.add_argument("--endpoint_tracker", default="tcp://127.0.0.1:5561")
    parser.add_argument("--send_every_n_frames", type=int, default=60)
    parser.add_argument("--resize_width", type=int, default=640)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    args = parser.parse_args()

    video_path = args.video_path

    context = zmq.Context()

    sock_florence = context.socket(zmq.PUSH)
    sock_florence.bind(args.endpoint_florence)

    sock_tracker = context.socket(zmq.PUSH)
    sock_tracker.bind(args.endpoint_tracker)

    print(f"[INFO] Broadcasting video: {video_path}")
    print(f"[INFO] ZeroMQ PUSH bind (florence): {args.endpoint_florence}")
    print(f"[INFO] ZeroMQ PUSH bind (tracker):  {args.endpoint_tracker}")
    print(f"[INFO] Sending every {args.send_every_n_frames} frames, resize width={args.resize_width}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("🔁 End of video — restarting from beginning...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            if frame_idx % args.send_every_n_frames == 0:
                video_time_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

                h, w = frame.shape[:2]
                if w > args.resize_width:
                    new_h = int(h * (args.resize_width / w))
                    frame = cv2.resize(frame, (args.resize_width, new_h), interpolation=cv2.INTER_AREA)

                ok2, buf = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
                )
                if ok2:
                    jpg_bytes = buf.tobytes()

                    payload = [
                        str(frame_idx).encode("utf-8"),
                        str(video_time_ms).encode("utf-8"),
                        jpg_bytes,
                    ]

                    # Send the SAME frame to both workers
                    sock_florence.send_multipart(payload)
                    sock_tracker.send_multipart(payload)

            frame_idx += 1
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (broadcaster).")
    finally:
        cap.release()
        sock_florence.close()
        sock_tracker.close()
        context.term()


if __name__ == "__main__":
    main()
