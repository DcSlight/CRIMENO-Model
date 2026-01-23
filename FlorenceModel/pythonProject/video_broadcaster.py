import cv2
import zmq
import time
import threading
import json
from queue import Queue, Empty


# =========================
# CONFIG
# =========================
BROADCAST_ENDPOINT = "tcp://127.0.0.1:5560"   # YOLO SUB
CONTROL_ENDPOINT   = "tcp://127.0.0.1:5561"   # NestJS REQ

JPEG_QUALITY = 85
RESIZE_WIDTH = 640
MAX_FPS = 0.0   # 0 = unlimited


# =========================
# UTILS
# =========================
def encode_jpg(frame_bgr):
    ok, buf = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("Failed to encode JPG")
    return buf.tobytes()


# =========================
# BROADCASTER
# =========================
class VideoBroadcaster:
    def __init__(self):
        self.ctx = zmq.Context()

        # PUB → YOLO
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(BROADCAST_ENDPOINT)

        # REP ← NestJS
        self.rep = self.ctx.socket(zmq.REP)
        self.rep.bind(CONTROL_ENDPOINT)

        self.switch_queue = Queue()

    # -------------------------
    # Control loop (NestJS)
    # -------------------------
    def control_loop(self):
        print("[CONTROL] Waiting for switch commands from NestJS...")
        while True:
            msg = self.rep.recv_string()
            try:
                data = json.loads(msg)
                if data.get("cmd") == "switch":
                    video = data.get("video")
                    print(f"[CONTROL] Switch video → {video}")
                    self.switch_queue.put(video)
                    self.rep.send_string("OK")
                else:
                    self.rep.send_string("UNKNOWN_CMD")
            except Exception as e:
                self.rep.send_string(f"ERROR: {e}")

    # -------------------------
    # Broadcast loop
    # -------------------------
    def broadcast_loop(self):
        print("[BROADCAST] Idle (no video selected yet)")
        cap = None
        frame_index = 0
        last_send = 0.0
        frame_time_ms = 40.0  # fallback

        while True:
            # Wait for a video if none is active
            if cap is None:
                video_path = self.switch_queue.get()
                cap = self.open_video(video_path)
                frame_index = 0

                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = 25.0
                frame_time_ms = 1000.0 / fps

                # Send META (YOLO expects this)
                ret, frame0 = cap.read()
                if not ret:
                    cap.release()
                    cap = None
                    continue

                h0, w0 = frame0.shape[:2]
                self.pub.send_multipart([
                    b"meta",
                    str(w0).encode(),
                    str(h0).encode(),
                    str(fps).encode(),
                ])
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                print(f"[BROADCAST] Started streaming: {video_path}")

            # Check for video switch
            try:
                new_video = self.switch_queue.get_nowait()
                cap.release()
                cap = self.open_video(new_video)
                frame_index = 0

                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = 25.0
                frame_time_ms = 1000.0 / fps

                ret, frame0 = cap.read()
                if ret:
                    h0, w0 = frame0.shape[:2]
                    self.pub.send_multipart([
                        b"meta",
                        str(w0).encode(),
                        str(h0).encode(),
                        str(fps).encode(),
                    ])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                print(f"[BROADCAST] Switched to: {new_video}")

            except Empty:
                pass

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_index = 0
                continue

            if RESIZE_WIDTH:
                h, w = frame.shape[:2]
                if w > RESIZE_WIDTH:
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

            self.pub.send_multipart([
                b"frame",
                str(frame_index).encode(),
                str(video_time_ms).encode(),
                jpg,
            ])

            frame_index += 1

    def open_video(self, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        return cap

    def run(self):
        threading.Thread(target=self.control_loop, daemon=True).start()
        self.broadcast_loop()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("[INFO] Video Broadcaster running (idle until switch)")
    VideoBroadcaster().run()
