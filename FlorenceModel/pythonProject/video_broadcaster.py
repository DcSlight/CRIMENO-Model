import cv2
import zmq
import time
import sys


def main():
    # Usage: python video_broadcaster.py videos/shop.mp4
    if len(sys.argv) < 2:
        video_path = "videos/shop.mp4"
        print(f"[INFO] No video path provided, using default: {video_path}")
    else:
        video_path = sys.argv[1]

    endpoint = "tcp://127.0.0.1:5560"
    send_every_n_frames = 60         # אפשר לשנות (60 / 120 וכו')
    resize_width = 640               # כדי להקטין עומס

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind(endpoint)

    print(f"[INFO] Broadcasting video: {video_path}")
    print(f"[INFO] ZeroMQ PUSH bind: {endpoint}")
    print(f"[INFO] Sending every {send_every_n_frames} frames, resize width={resize_width}")

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

            if frame_idx % send_every_n_frames == 0:
                # timestamp in ms inside the video
                video_time_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

                # resize
                h, w = frame.shape[:2]
                if w > resize_width:
                    new_h = int(h * (resize_width / w))
                    frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)

                # encode jpeg
                ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok2:
                    frame_idx += 1
                    continue

                jpg_bytes = buf.tobytes()

                # multipart: [frame_idx, video_time_ms, jpg]
                socket.send_multipart([
                    str(frame_idx).encode("utf-8"),
                    str(video_time_ms).encode("utf-8"),
                    jpg_bytes
                ])

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
