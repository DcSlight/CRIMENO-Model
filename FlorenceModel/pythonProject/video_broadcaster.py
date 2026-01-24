import cv2
import zmq
import time
import argparse

def encode_jpg(frame_bgr, jpeg_quality: int) -> bytes:
    """Encodes a BGR frame into a JPEG buffer."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Failed to encode JPG")
    return buf.tobytes()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", nargs="?", default=None)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--cmd_endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--resize_width", type=int, default=640)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--max_fps", type=float, default=0.0)
    args = parser.parse_args()

    context = zmq.Context()
    
    # PUB socket for broadcasting frames and meta-data
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(args.endpoint)

    # REP socket for receiving commands from NestJS
    cmd_socket = context.socket(zmq.REP)
    cmd_socket.bind(args.cmd_endpoint)
    
    poller = zmq.Poller()
    poller.register(cmd_socket, zmq.POLLIN)

    cap = None
    current_video = None
    frame_index = 0
    last_send_ts = 0.0
    
    print(f"[INFO] Broadcaster initialized. PUB: {args.endpoint}, CMD: {args.cmd_endpoint}")
    print(f"[STATUS] Waiting for 'play' command...")

    while True:
        # 1. Check for commands from NestJS
        socks = dict(poller.poll(1)) 
        if cmd_socket in socks:
            msg = cmd_socket.recv_json()
            if msg.get("cmd") == "play":
                new_path = msg.get("video")
                print(f"[CONTROL] Received play command: {new_path}")
                
                if cap is not None:
                    cap.release()
                
                cap = cv2.VideoCapture(new_path)
                current_video = new_path
                frame_index = 0
                
                cmd_socket.send_json({"status": "ok", "video": new_path})
                
                # IMPORTANT: Broadcast 'meta' message to signal a new stream
                # This triggers the Tracker to reset its internal state
                ret, frame0 = cap.read()
                if ret:
                    h0, w0 = frame0.shape[:2]
                    fps0 = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    pub_socket.send_multipart([
                        b"meta", 
                        str(w0).encode(), 
                        str(h0).encode(), 
                        str(fps0).encode()
                    ])
                    # Reset to start of video after reading first frame for meta
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                cmd_socket.send_json({"status": "error", "msg": "unknown command"})

        # 2. Idle check
        if cap is None or not cap.isOpened():
            time.sleep(0.1) 
            continue

        # 3. Read and Broadcast
        ret, frame = cap.read()
        if not ret:
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

        # FPS Throttling
        min_dt = (1.0 / args.max_fps) if args.max_fps > 0 else 0.0
        now = time.time()
        if min_dt > 0 and (now - last_send_ts) < min_dt:
            continue
        last_send_ts = now

        # Broadcast frame to subscribers
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        video_time_ms = int(frame_index * (1000.0 / fps))
        jpg = encode_jpg(frame, args.jpeg_quality)

        pub_socket.send_multipart([
            b"frame",
            str(frame_index).encode(),
            str(video_time_ms).encode(),
            jpg,
        ])

        if frame_index % 60 == 0:
            print(f"[STREAMING] {current_video} | frame={frame_index}")

        frame_index += 1

if __name__ == "__main__":
    main()