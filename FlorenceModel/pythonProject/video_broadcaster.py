import cv2
import zmq
import time
import argparse


def encode_jpg(frame_bgr, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return b""
    return buf.tobytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", nargs="?", default="videos/shop.mp4")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--resize_width", type=int, default=640)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--max_fps", type=float, default=15.0, help="Cap send rate (0 = no throttling)")
    parser.add_argument("--print_every", type=int, default=30)
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(args.endpoint)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")

    fps_sleep = 0.0 if args.max_fps <= 0 else (1.0 / args.max_fps)

    print(f"[BROADCAST] Video: {args.video_path}")
    print(f"[BROADCAST] ZMQ PUB bind: {args.endpoint}")
    print(f"[BROADCAST] resize_width={args.resize_width} jpeg_quality={args.jpeg_quality} max_fps={args.max_fps}")

    frame_idx = 0
    sent = 0
    last_log = time.time()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[BROADCAST] End of video. Looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if args.resize_width > 0:
                h, w = frame_bgr.shape[:2]
                if w != args.resize_width:
                    new_h = int(h * (args.resize_width / w))
                    frame_bgr = cv2.resize(frame_bgr, (args.resize_width, new_h), interpolation=cv2.INTER_AREA)

            video_time_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            jpg_bytes = encode_jpg(frame_bgr, args.jpeg_quality)
            if not jpg_bytes:
                frame_idx += 1
                continue

            # Multipart: [topic, frame_index, video_time_ms, jpg_bytes]
            socket.send_multipart([
                b"frame",
                str(frame_idx).encode("utf-8"),
                str(video_time_ms).encode("utf-8"),
                jpg_bytes
            ])

            sent += 1
            frame_idx += 1

            if args.print_every > 0 and (sent % args.print_every == 0):
                now = time.time()
                dt = now - last_log
                last_log = now
                approx_fps = args.print_every / dt if dt > 0 else 0.0
                print(f"[BROADCAST] sent={sent} last_frame={frame_idx-1} t_ms={video_time_ms} ~fps={approx_fps:.1f}")

            if fps_sleep > 0:
                time.sleep(fps_sleep)

    except KeyboardInterrupt:
        print("\n[BROADCAST] Stopped by user.")
    finally:
        cap.release()
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
