import cv2
import zmq
import time
import sys


def main():
    # Read video path from CLI or use default
    if len(sys.argv) < 2:
        video_path = "videos/shop.mp4"
        print(f"[INFO] No video path provided, using default: {video_path}")
    else:
        video_path = sys.argv[1]

    # ZeroMQ PUSH socket
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind("tcp://127.0.0.1:5560")
    print("📡 Video broadcaster bound on tcp://127.0.0.1:5560")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return

    EVERY_N_FRAMES = 120  # send one frame every N frames to reduce load
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()

            # If we reached the end of the video – restart from beginning
            if not ret:
                print("🔁 End of video — restarting from beginning...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Downscale frame to speed up processing
            h, w = frame.shape[:2]
            target_width = 640
            if w > target_width:
                target_height = int(h * target_width / w)
                frame = cv2.resize(frame, (target_width, target_height))

            # Send only every Nth frame
            if frame_idx % EVERY_N_FRAMES == 0:
                # Extract video time in milliseconds (position of current frame)
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                pos_msec_int = int(pos_msec) if pos_msec is not None else -1

                success, buffer = cv2.imencode(".jpg", frame)
                if not success:
                    print(f"[WARN] Failed to encode frame {frame_idx}")
                else:
                    jpg_bytes = buffer.tobytes()

                    # Send multipart:
                    # [frame_index, pos_msec, jpeg_bytes]
                    socket.send_multipart(
                        [
                            str(frame_idx).encode("utf-8"),
                            str(pos_msec_int).encode("utf-8"),
                            jpg_bytes,
                        ]
                    )
                    print(f"📤 Sent frame {frame_idx} (t={pos_msec_int}ms)")

            frame_idx += 1
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (broadcaster).")
    finally:
        cap.release()
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
