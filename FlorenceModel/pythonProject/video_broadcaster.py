# video_broadcaster.py
import cv2
import zmq
import time
import argparse
import yt_dlp

WORKER_ACK_TIMEOUT_S = 20    # seconds to wait for reset_ack from all workers
FIRST_FRAME_ACK_TIMEOUT_S = 5  # seconds to wait for tracker to emit its first real bbox


def resolve_video_source(video_path: str, video_type: str) -> str:
    """Returns a source URL/path that cv2.VideoCapture can open."""
    if video_type == "online":
        print(f"[YT-DLP] Extracting stream URL from: {video_path}")
        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_path, download=False)
            stream_url = info["url"]
        print(f"[YT-DLP] Stream URL resolved successfully")
        return stream_url
    return video_path


def encode_jpg(frame_bgr, jpeg_quality: int) -> bytes:
    """Encodes a BGR frame into a JPEG buffer."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Failed to encode JPG")
    return buf.tobytes()


def main():
    parser = argparse.ArgumentParser()
    # No default video is loaded at startup
    parser.add_argument("video_path", nargs="?", default=None)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--cmd_endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--ack_endpoint", default="tcp://127.0.0.1:5562")
    parser.add_argument("--resize_width", type=int, default=640)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--max_fps", type=float, default=0.0, help="Legacy param - now uses video FPS automatically")
    args = parser.parse_args()

    context = zmq.Context()

    # PUB socket for broadcasting frames (Tracker/Florence)
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(args.endpoint)

    # REP socket for receiving commands from NestJS
    cmd_socket = context.socket(zmq.REP)
    cmd_socket.bind(args.cmd_endpoint)

    # PULL socket for worker reset acks
    ack_socket = context.socket(zmq.PULL)
    ack_socket.bind(args.ack_endpoint)
    ack_socket.setsockopt(zmq.RCVTIMEO, 500)

    poller = zmq.Poller()
    poller.register(cmd_socket, zmq.POLLIN)

    cap = None
    current_video = None
    frame_index = 0
    video_fps = 25.0
    last_send_ts = 0.0
    stream_start_time = 0.0

    print(f"[INFO] Broadcaster initialized. PUB: {args.endpoint}, CMD: {args.cmd_endpoint}")
    print(f"[STATUS] Waiting for 'play' command...")

    # AUTO-PLAY MODE: if video_path was provided directly, start streaming immediately (always local)
    if args.video_path is not None:
        print(f"[AUTO-PLAY] Starting video immediately: {args.video_path}")
        source = resolve_video_source(args.video_path, "local")
        cap = cv2.VideoCapture(source)
        current_video = args.video_path
        frame_index = 0
        stream_start_time = time.time()

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or video_fps > 120:
            video_fps = 25.0

        print(f"[AUTO-PLAY] FPS={video_fps}")

    while True:
        # 1. Check for commands from NestJS
        socks = dict(poller.poll(1))
        if cmd_socket in socks:
            msg = cmd_socket.recv_json()
            if msg.get("cmd") == "play":
                new_path = msg.get("video")
                video_type = msg.get("videoType", "local")
                print(f"[CONTROL] Received play command: {new_path} (type={video_type})")

                if cap is not None:
                    cap.release()

                try:
                    source = resolve_video_source(new_path, video_type)
                except Exception as e:
                    print(f"[ERROR] Failed to resolve video source: {e}")
                    cmd_socket.send_json({"status": "error", "msg": str(e)})
                    continue

                cap = cv2.VideoCapture(source)
                current_video = new_path
                frame_index = 0
                # stream_start_time is set AFTER the first-frame ack so pacing starts from
                # the moment React receives {ok:true} — avoids a catch-up burst on first play.

                # Get video FPS
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                if video_fps <= 0 or video_fps > 120:
                    video_fps = 25.0  # Fallback to 25 FPS

                print(f"[INFO] Video FPS: {video_fps}")

                # Signal all workers to flush their buffers
                pub_socket.send_multipart([b"reset"])
                print("[CONTROL] Sent reset — waiting for worker acks...")

                expected_acks = {"florence", "tracker"}
                received_acks = set()
                deadline = time.time() + WORKER_ACK_TIMEOUT_S
                while received_acks < expected_acks and time.time() < deadline:
                    try:
                        ack = ack_socket.recv_json()
                        worker = ack.get("worker", "")
                        if worker in expected_acks:
                            received_acks.add(worker)
                            print(f"[CONTROL] Ack from '{worker}' ({len(received_acks)}/{len(expected_acks)})")
                    except zmq.error.Again:
                        continue

                missing = expected_acks - received_acks
                if missing:
                    print(f"[CONTROL] Timeout — missing acks from: {missing}. Starting anyway.")
                else:
                    print("[CONTROL] All workers ready — starting stream")

                # Publish warmup frame 0 so tracker can run YOLO on it before we reply to NestJS.
                ret, frame0 = cap.read()
                if not ret:
                    cmd_socket.send_json({"status": "error", "msg": "failed to read first frame"})
                    continue
                h0, w0 = frame0.shape[:2]
                if args.resize_width and w0 > args.resize_width:
                    scale = args.resize_width / float(w0)
                    frame0 = cv2.resize(frame0, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
                    h0, w0 = frame0.shape[:2]
                jpg0 = encode_jpg(frame0, args.jpeg_quality)
                pub_socket.send_multipart([b"frame", b"0", b"0", jpg0])
                print("[CONTROL] Warmup frame 0 published — waiting for tracker first_frame_ack")

                # Wait for tracker to confirm it emitted its first bbox over WS (or time out).
                first_frame_deadline = time.time() + FIRST_FRAME_ACK_TIMEOUT_S
                got_first_frame = False
                while time.time() < first_frame_deadline:
                    try:
                        ack = ack_socket.recv_json()
                        if ack.get("type") == "first_frame_ack" and ack.get("worker") == "tracker":
                            got_first_frame = True
                            print("[CONTROL] first_frame_ack received — tracker has a bbox ready")
                            break
                    except zmq.error.Again:
                        continue
                if not got_first_frame:
                    print("[CONTROL] Timeout waiting for tracker first_frame_ack — replying anyway")

                # Reply to NestJS — by now tracker has sent the first bbox over WS to React.
                cmd_socket.send_json({
                    "status": "ok",
                    "video": new_path,
                    "workers_ready": sorted(received_acks),
                    "workers_timeout": sorted(missing),
                    "first_frame_ready": got_first_frame,
                })

                # Meta broadcast for the stream
                pub_socket.send_multipart([b"meta", str(w0).encode(), str(h0).encode(), str(video_fps).encode()])

                # Anchor FPS pacing to now — frame 1 onward is paced from when React started playing.
                stream_start_time = time.time()
                frame_index = 1  # frame 0 was already published as the warmup frame
            else:
                cmd_socket.send_json({"status": "error", "msg": "unknown command"})

        # 2. Idle check
        if cap is None or not cap.isOpened():
            time.sleep(0.1)
            continue

        # 3. Read frame
        ret, frame = cap.read()
        if not ret:
            # Video reached the end - stop broadcasting and wait for next command
            print(f"[INFO] Video {current_video} finished.")
            cap.release()
            cap = None
            current_video = None
            continue

        # Resize logic
        h, w = frame.shape[:2]
        if args.resize_width and w > args.resize_width:
            scale = args.resize_width / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # CRITICAL: Synchronize with video timeline
        # Calculate when this frame should be sent based on video FPS
        expected_time = stream_start_time + (frame_index / video_fps)
        current_time = time.time()

        # If we're ahead of schedule, wait
        time_diff = expected_time - current_time
        if time_diff > 0:
            time.sleep(time_diff)

        # Calculate video timestamp
        video_time_ms = int(frame_index * (1000.0 / video_fps))

        # Encode and broadcast
        jpg = encode_jpg(frame, args.jpeg_quality)
        pub_socket.send_multipart([
            b"frame",
            str(frame_index).encode(),
            str(video_time_ms).encode(),
            jpg,
        ])

        if frame_index % 60 == 0:
            print(f"[STREAMING] {current_video} | frame={frame_index} | time={video_time_ms}ms | fps={video_fps}")

        frame_index += 1


if __name__ == "__main__":
    main()