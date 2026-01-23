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
    parser.add_argument("--cmd_endpoint", default="tcp://127.0.0.1:5561") # הפורט של NestJS
    parser.add_argument("--resize_width", type=int, default=640)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--max_fps", type=float, default=0.0)
    args = parser.parse_args()

    context = zmq.Context()
    
    # סוקט שידור (PUB)
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(args.endpoint)

    # סוקט פקודות (REP)
    cmd_socket = context.socket(zmq.REP)
    cmd_socket.bind(args.cmd_endpoint)
    
    # מאפשר לנו לבדוק אם יש הודעות בלי לחכות (non-blocking)
    poller = zmq.Poller()
    poller.register(cmd_socket, zmq.POLLIN)

    current_video = args.video_path
    cap = cv2.VideoCapture(current_video)
    
    print(f"[INFO] Initial video: {current_video}")
    print(f"[INFO] Command server: {args.cmd_endpoint}")

    frame_index = 0
    last_send_ts = 0.0
    
    while True:
        # --- בדיקה אם הגיעה פקודה מ-NestJS ---
        socks = dict(poller.poll(0)) # poll(0) לא מחכה בכלל
        if cmd_socket in socks:
            msg = cmd_socket.recv_json()
            if msg.get("cmd") == "play":
                new_path = msg.get("video")
                print(f"[CONTROL] Switching to: {new_path}")
                
                # סגירה ופתיחה של סרטון חדש
                cap.release()
                cap = cv2.VideoCapture(new_path)
                current_video = new_path
                frame_index = 0
                
                # שליחת אישור ל-NestJS כדי שלא יתקע
                cmd_socket.send_json({"status": "ok", "video": new_path})
                
                # אופציונלי: שליחת Meta חדש אחרי החלפה
                ret, frame0 = cap.read()
                if ret:
                    h0, w0 = frame0.shape[:2]
                    fps0 = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    pub_socket.send_multipart([b"meta", str(w0).encode(), str(h0).encode(), str(fps0).encode()])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                cmd_socket.send_json({"status": "error", "msg": "unknown command"})

        # --- לוגיקת שידור הוידאו המקורית שלך ---
        if not cap.isOpened():
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_index = 0
            continue

        # Resize
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

        # Encoding ושליחה
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
            print(f"[BROADCAST] {current_video} | frame={frame_index}")

        frame_index += 1

if __name__ == "__main__":
    main()