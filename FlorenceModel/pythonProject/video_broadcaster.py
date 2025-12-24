# video_broadcaster_v2.py
# Publishes video frames over ZeroMQ.
# Adds dynamic stream metadata so downstream workers/visualizers can stay consistent
# without hardcoding any video size.
#
# Multipart message (topic="frame"):
#   [topic, frame_idx, video_time_ms, src_w, src_h, send_w, send_h, jpg_bytes]
#
# - src_w/src_h: original frame size read from the file
# - send_w/send_h: size of the JPEG that is actually sent (after optional resize)

import argparse
import time
import cv2
import zmq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", nargs="?", default="videos/shop.mp4")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--resize_width", type=int, default=640, help="0 = no resize, otherwise resize to this width")
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

            src_h, src_w = frame.shape[:2]

            # optional resize to reduce load
            send_frame = frame
            if args.resize_width and args.resize_width > 0:
                if src_w > args.resize_width:
                    new_h = int(src_h * (args.resize_width / src_w))
                    send_frame = cv2.resize(send_frame, (args.resize_width, new_h), interpolation=cv2.INTER_AREA)

            send_h, send_w = send_frame.shape[:2]

            ok2, buf = cv2.imencode(
                ".jpg",
                send_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
            )
            if not ok2:
                frame_idx += 1
                continue

            jpg_bytes = buf.tobytes()

            socket.send_multipart(
                [
                    b"frame",
                    str(frame_idx).encode("utf-8"),
                    str(video_time_ms).encode("utf-8"),
                    str(src_w).encode("utf-8"),
                    str(src_h).encode("utf-8"),
                    str(send_w).encode("utf-8"),
                    str(send_h).encode("utf-8"),
                    jpg_bytes,
                ]
            )

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (broadcaster).")
    finally:
        cap.release()
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
