# CRIMENO Surveillance System

Professional video surveillance anomaly detection system using Computer Vision + LLM analysis.

## Quick Start

```bash
# Terminal 1: Message Broker (start FIRST)
python message_broker.py

# Terminal 2: Video Broadcaster
python video_broadcaster.py videos/shop.mp4

# Terminal 3: Florence Worker (vision AI)
python florence_worker.py --device cuda --every 30

# Terminal 4: Tracker Worker (object tracking)
python tracker_worker.py --send_every_n_frames 1

# Terminal 5: Qwen Worker (anomaly detection)
python qwen_anomaly_worker.py --device cuda
```

Then open NestJS frontend at `http://localhost:3000`

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

**Key Points:**

- ✅ Clean layer separation
- ✅ Workers communicate only via ZMQ
- ✅ WebSocket isolated in message broker
- ✅ Scalable and maintainable

## Configuration

All settings centralized in `config.py`:

```python
ZMQ_VIDEO_BROADCASTER_ENDPOINT = "tcp://127.0.0.1:5560"
ZMQ_MESSAGE_BROKER_ENDPOINT = "tcp://127.0.0.1:5580"
WEBSOCKET_PORT = 3000
```

## Files

- `config.py` - Centralized configuration
- `utils.py` - Shared utilities (logging, ZMQ, messaging)
- `message_broker.py` - Central communication hub
- `florence_worker.py` - Vision AI (captions, OCR, weapons detection)
- `tracker_worker.py` - YOLOv8 multi-object tracking
- `qwen_anomaly_worker.py` - LLM-based anomaly scoring
- `video_broadcaster.py` - Video source
- `ARCHITECTURE.md` - System design documentation
- `VERIFICATION.md` - Code quality checklist

## Workers

### Florence Worker

- **Input**: Video frames (5560 SUB)
- **Output**: Vision AI results (5580 PUSH)
- **Tasks**: Captions, object detection, OCR, weapon detection
- **Command**: `python florence_worker.py --device cuda --every 30`

### Tracker Worker

- **Input**: Video frames (5560 SUB)
- **Output**: Tracking results (5580 PUSH)
- **Tasks**: YOLOv8 detection + IOU tracking
- **Command**: `python tracker_worker.py --send_every_n_frames 1`

### Qwen Anomaly Worker

- **Input**: All worker results (5580 PULL)
- **Output**: Anomaly scores (5580 PUSH)
- **Tasks**: Scene analysis, anomaly scoring
- **Command**: `python qwen_anomaly_worker.py --device cuda`

### Message Broker

- **Input**: All results (5580 PULL)
- **Output**: WebSocket broadcast (3000)
- **Tasks**: Route messages to frontend
- **Command**: `python message_broker.py`

## Ports

| Port | Type          | Purpose          |
| ---- | ------------- | ---------------- |
| 5560 | ZMQ PUB/SUB   | Video frames     |
| 5561 | ZMQ REP/REQ   | Video commands   |
| 5580 | ZMQ PULL/PUSH | Results hub      |
| 3000 | WebSocket     | Frontend updates |

## Output

Florence worker writes to `analysis.jsonl`:

```json
{
  "type": "florence_frame",
  "frame_index": 123,
  "video_time_ms": 5000,
  "raw": {
    "more_detailed_caption": "...",
    "object_detection": "...",
    "ocr": "...",
    "open_vocab_weapons": "..."
  },
  "text_overlay": {
    "datetime_candidates": ["14:30"]
  }
}
```

## Adding a New Worker

1. Create `my_worker.py`
2. Connect to ZMQ 5580 for input/output
3. Don't use WebSocket (broker handles it)
4. Run it alongside other workers

Example:

```python
import zmq
from config import ZMQ_MESSAGE_BROKER_ENDPOINT
from utils import ZMQConnector, MessageHandler, Logger

def main():
    socket = ZMQConnector.create_pull_socket(ZMQ_MESSAGE_BROKER_ENDPOINT)
    while True:
        msg = socket.recv()
        # Process...
        result = {"type": "my_result", "data": "..."}
        socket.send(MessageHandler.encode_message(result))
```

## Troubleshooting

**Workers can't connect:**

- Start message broker first
- Check ports are available: `netstat -an | grep 558`

**No data flowing:**

- Check video broadcaster is running
- Verify video file exists: `ls videos/shop.mp4`

**NestJS not receiving data:**

- Check message broker is running
- Check WebSocket connection in browser console

## Performance

- Florence: ~1-2 FPS (depends on model size and GPU)
- Tracker: ~10-30 FPS
- Qwen: ~0.5 FPS (LLM inference)
- Message Broker: <1ms latency

## License

Internal project - CRIMENO research group

## Support

See `ARCHITECTURE.md` for design details and `VERIFICATION.md` for code quality.
