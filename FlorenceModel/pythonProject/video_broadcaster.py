import cv2
import zmq
import time
import argparse


def encode_jpg(frame_bgr, jpeg_quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Failed to encode JPG")
    return buf.tobytes()


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

    # Give SUB sockets time to connect (PUB/SUB pattern)
    time.sleep(0.5)

    frame_index = 0
    last_send_ts = 0.0
    min_dt = (1.0 / args.max_fps) if args.max_fps and args.max_fps > 0 else 0.0

    # Read original video meta
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    frame_time_ms = 1000.0 / fps

    # Send one "meta" message at the start (optional, useful for consumers)
    # topic: meta, [w,h,fps]
    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("Empty video")
    h0, w0 = frame0.shape[:2]
    socket.send_multipart([b"meta", str(w0).encode(), str(h0).encode(), str(fps).encode()])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop for "realtime dashboard" demo
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_index = 0
            continue

        # Resize while keeping aspect ratio
        h, w = frame.shape[:2]
        if args.resize_width and w > args.resize_width:
            scale = args.resize_width / float(w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        now = time.time()
        if min_dt > 0 and (now - last_send_ts) < min_dt:
            continue
        last_send_ts = now

        video_time_ms = int(frame_index * frame_time_ms)
        jpg = encode_jpg(frame, args.jpeg_quality)

        # topic: frame, [frame_idx, video_time_ms, jpg_bytes]
        socket.send_multipart([
            b"frame",
            str(frame_index).encode("utf-8"),
            str(video_time_ms).encode("utf-8"),
            jpg,
        ])

        if frame_index % 60 == 0:
            print(f"[BROADCAST] frame={frame_index} t={video_time_ms}ms size={len(jpg)}B")

        frame_index += 1


if __name__ == "__main__":
    main()
