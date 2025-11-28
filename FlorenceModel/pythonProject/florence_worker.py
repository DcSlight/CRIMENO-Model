import io
import zmq
import torch
from PIL import Image
from transformers import pipeline


def load_florence_pipeline():
    """Load Florence-2 as an image-text-to-text pipeline."""
    has_cuda = torch.cuda.is_available()

    if has_cuda:
        vision_pipe = pipeline(
            "image-text-to-text",
            model="florence-community/Florence-2-base",
            device=0,  # GPU 0
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("✅ Florence-2 pipeline loaded on GPU")
    else:
        vision_pipe = pipeline(
            "image-text-to-text",
            model="florence-community/Florence-2-base",
            device=-1,  # CPU
            trust_remote_code=True,
        )
        print("✅ Florence-2 pipeline loaded on CPU")

    return vision_pipe


def main():
    vision_pipe = load_florence_pipeline()

    # ZeroMQ PULL socket
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://127.0.0.1:5560")
    print("🔗 Connected to video broadcaster on tcp://127.0.0.1:5560")

    # Florence expects ONLY the task token here
    task_prompt = "<MORE_DETAILED_CAPTION>"

    try:
        while True:
            # Receive multipart message: [frame_idx, jpg_bytes]
            frame_idx_bytes, jpg_bytes = socket.recv_multipart()
            frame_idx = int(frame_idx_bytes.decode("utf-8"))

            # Decode JPEG to PIL image
            image = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")

            # Run Florence-2
            result = vision_pipe(image, text=task_prompt)

            # Typical pipeline output: list with a string or dict
            if isinstance(result, list) and len(result) > 0:
                first = result[0]
                if isinstance(first, str):
                    caption = first
                elif isinstance(first, dict) and "generated_text" in first:
                    caption = first["generated_text"]
                else:
                    caption = str(first)
            else:
                caption = str(result)

            print(f"🎬 Frame {frame_idx}: {caption}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (worker).")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
