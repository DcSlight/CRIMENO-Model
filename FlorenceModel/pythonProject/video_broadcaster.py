import cv2
import zmq
import time
import threading
import queue
import json

BROADCAST_ENDPOINT = "tcp://127.0.0.1:5560"
CONTROL_ENDPOINT = "tcp://127.0.0.1:5561"

RESIZE_WIDTH = 640
JPEG_QUALITY = 85
MAX_FPS = 0.0  # 0 = unlimited


def encode_jpg(frame_bgr):
    ok, buf = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("Failed to encode JPG")
    return buf.tobytes()


class VideoBroadcaster:
    def __init__(self):
        self.ctx = zmq.Context()

        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(BROADCAST_ENDPOINT)

        self.rep = self.ctx.socket(zmq.REP)
        self.rep.bind(CONTROL_ENDPOINT)

        self.command_queue = queue.Queue()
        self.current_video = None
        self.stop_flag = False

    def control_loop(self):
        print("[CONTROL] Listening for play commands...")
        while not self.stop_flag:
            msg = self.rep.recv_string()
            try:
                data = json.loads(msg)
                if data.get("cmd") == "play":
                    video = data.get("video")
                    self.command_queue.put(video)
                    self.rep.send_string("OK")
                    print(f"[CONTROL] Play video: {video}")
                else:
                    self.rep.send_string("UNKNOWN_CMD")
            except Exception as e:
                self.rep.send_string(f"ERROR: {e}")

    def broadcast_loop(self):
        print("[BROADCAST] Ready")
        last_send = 0.0

        while not self.stop_flag:
            video_path = self.command_queue.get()
            self.current_video = video_path

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open video {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25.0
            frame_time_ms = 1000.0 / fps

            ret, first = cap.read()
            if not ret:
                cap.release()
                continue

            h0, w0 = first.shape[:2]
            self.pub.send_multipart(
                [b"meta", str(w0).encode(), str(h0).encode(), str(fps).encode()]
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_index = 0

            while not self.stop_flag:
                if not self.command_queue.empty():
                    print("[BROADCAST] Switching video")
                    cap.release()
                    break

                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_index = 0
                    continue

                h, w = frame.shape[:2]
                if RESIZE_WIDTH and w > RESIZE_WIDTH:
                    scale = RESIZE_WIDTH / float(w)
                    frame = cv2.resize(
                        frame,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                now = time.time()
                if MAX_FPS > 0:
                    min_dt = 1.0 / MAX_FPS
                    if now - last_send < min_dt:
                        continue
                last_send = now

                video_time_ms = int(frame_index * frame_time_ms)
                jpg = encode_jpg(frame)

                self.pub.send_multipart(
                    [
                        b"frame",
                        str(frame_index).encode(),
                        str(video_time_ms).encode(),
                        jpg,
                    ]
                )

                frame_index += 1

    def run(self):
        t1 = threading.Thread(target=self.control_loop, daemon=True)
        t1.start()
        self.broadcast_loop()


if __name__ == "__main__":
    VideoBroadcaster().run()
