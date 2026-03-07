# Quick Start Guide

## Updated Workflow Summary

The system now follows this data flow:

```
Broadcast → Tracker → Qwen & NestJS ✅
Broadcast → Florence → Qwen & NestJS ✅
Qwen → NestJS ✅
```

## What Changed

### ✅ Qwen Worker Now Sends Data to NestJS

The `qwen_anomaly_worker.py` has been updated to send anomaly detection results to NestJS via WebSocket.

**New Features:**

- Added async/await support
- Added WebSocket connection with auto-reconnect
- Sends structured anomaly results to NestJS

**New Command Line Argument:**

```bash
--ws-url <websocket_url>
```

Example:

```bash
python qwen_anomaly_worker.py --ws-url ws://localhost:3000/qwen
```

## Data Sent to NestJS

### From Qwen Worker (NEW!)

```json
{
  "type": "qwen_anomaly",
  "frame_range": {
    "start": 120,
    "end": 150
  },
  "result": {
    "anomaly_score": 0.85,
    "label": "suspicious",
    "reason": "Person loitering near entrance without clear purpose",
    "key_moments": [
      "individual looking around nervously",
      "repeated pacing in same area"
    ]
  }
}
```

### From Florence Worker (Existing)

```json
{
  "type": "florence_frame",
  "frame_index": 123,
  "video_time_ms": 4100,
  "caption": "A person standing near counter...",
  "objects": [...],
  "weapons_detected": [...],
  "raw": {
    "more_detailed_caption": "...",
    "object_detection": "...",
    "ocr": "...",
    "open_vocab_weapons": "..."
  },
  "text_overlay": {
    "datetime_candidates": ["2024-03-06"]
  },
  "meta": {...}
}
```

### From Tracker Worker (Existing)

```json
{
  "type": "tracker_frame",
  "frame_index": 123,
  "video_time_ms": 4100,
  "frame_size": { "w": 640, "h": 480 },
  "motion_detected": false,
  "tracks": [
    {
      "track_id": 1,
      "cls": "person",
      "conf": 0.92,
      "bbox": { "x1": 100, "y1": 150, "x2": 200, "y2": 350 }
    }
  ]
}
```

## Complete Startup Sequence

### 1. Start NestJS Backend (if not running)

```bash
cd path/to/nestjs-backend
npm run start
```

### 2. Start Qwen Worker (MUST START FIRST)

```bash
python qwen_anomaly_worker.py --ws-url ws://localhost:3000/ws/qwen
```

### 3. Start Florence Worker

```bash
python florence_worker.py --ws-url ws://localhost:3000/ws/florence
```

### 4. Start Tracker Worker

```bash
python tracker_worker.py --ws_url ws://localhost:3000/ws/tracker
```

### 5. Start Video Broadcaster (Last)

```bash
python video_broadcaster.py videos/shop.mp4
```

## Testing Without NestJS

To test without NestJS, omit the `--ws-url` argument or set it to "none":

```bash
# Qwen will only process data, not send to WebSocket
python qwen_anomaly_worker.py

# Florence will only process and send to Qwen
python florence_worker.py

# Tracker will only process and send to Qwen
python tracker_worker.py
```

## Network Ports

- **5560**: Video broadcast (ZMQ PUB)
- **5561**: Video control commands (ZMQ REP)
- **5580**: Qwen input (ZMQ PULL) - receives from Florence & Tracker
- **3000**: NestJS WebSocket server (configurable)

## Notes

- Qwen MUST be started before Florence and Tracker (it binds to port 5580)
- All WebSocket connections auto-reconnect on failure
- The system gracefully handles missing WebSocket connections
- Press Ctrl+C to stop any worker
