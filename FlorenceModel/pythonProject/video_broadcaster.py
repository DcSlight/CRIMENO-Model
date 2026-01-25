# video_broadcaster.py
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
    
    # Two separate endpoints
    parser.add_argument("--tracker_endpoint", default="tcp://127.0.0.1:5560", 
                        help="PUB endpoint for tracker (time-synced)")
    parser.add_argument("--florence_endpoint", default="tcp://127.0.0.1:5562", 
                        help="PUB endpoint for florence (real-time)")
    parser.add_argument("--cmd_endpoint", default="tcp://127.0.0.1:5561")
    
    parser.add_argument("--resize_width", type=int, default=640)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--max_fps", type=float, default=0.0, help="Legacy param - now uses video FPS automatically")
    args = parser.parse_args()

    context = zmq.Context()
    
    # PUB socket for tracker (time-synchronized)
    tracker_socket = context.socket(zmq.PUB)
    tracker_socket.bind(args.tracker_endpoint)

    # PUB socket for florence (real-time, no delay)
    florence_socket = context.socket(zmq.PUB)
    florence_socket.bind(args.florence_endpoint)

    # REP socket for receiving commands from NestJS
    cmd_socket = context.socket(zmq.REP)
    cmd_socket.bind(args.cmd_endpoint)
    
    poller = zmq.Poller()
    poller.register(cmd_socket, zmq.POLLIN)

    cap = None
    current_video = None
    frame_index = 0
    video_fps = 25.0
    stream_start_time = 0.0
    
    print(f"[INFO] Broadcaster initialized.")
    print(f"[INFO] Tracker (time-synced): {args.tracker_endpoint}")
    print(f"[INFO] Florence (real-time): {args.florence_endpoint}")
    print(f"[INFO] Command: {args.cmd_endpoint}")
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
                stream_start_time = time.time()
                
                # Get video FPS
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                if video_fps <= 0 or video_fps > 120:
                    video_fps = 25.0  # Fallback to 25 FPS
                
                print(f"[INFO] Video FPS: {video_fps}")
                
                cmd_socket.send_json({"status": "ok", "video": new_path})
                
                # Meta-data broadcast for the new stream (both sockets)
                ret, frame0 = cap.read()
                if ret:
                    h0, w0 = frame0.shape[:2]
                    meta_msg = [b"meta", str(w0).encode(), str(h0).encode(), str(video_fps).encode()]
                    tracker_socket.send_multipart(meta_msg)
                    florence_socket.send_multipart(meta_msg)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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

        # Calculate video timestamp
        video_time_ms = int(frame_index * (1000.0 / video_fps))
        
        # Encode ONCE for both streams
        jpg = encode_jpg(frame, args.jpeg_quality)
        
        # Create message parts
        msg_parts = [
            b"frame",
            str(frame_index).encode(),
            str(video_time_ms).encode(),
            jpg,
        ]

        # Send to FLORENCE immediately (no delay)
        florence_socket.send_multipart(msg_parts)

        # For TRACKER: synchronize with video timeline
        expected_time = stream_start_time + (frame_index / video_fps)
        current_time = time.time()
        
        # If we're ahead of schedule, wait
        time_diff = expected_time - current_time
        if time_diff > 0:
            time.sleep(time_diff)

        # Send to TRACKER (after timing delay)
        tracker_socket.send_multipart(msg_parts)

        if frame_index % 60 == 0:
            print(f"[STREAMING] {current_video} | frame={frame_index} | time={video_time_ms}ms | fps={video_fps}")

        frame_index += 1

if __name__ == "__main__":
    main()