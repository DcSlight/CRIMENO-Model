# Data Flow Workflow

## Overview

This document describes the complete data flow between all components in the CRIMENO Model system.

## Components

1. **Video Broadcaster** - Streams video frames via ZeroMQ PUB
2. **Florence Worker** - Vision-language model for scene understanding
3. **Tracker Worker** - YOLOv8 object detection and tracking
4. **Qwen Worker** - Anomaly detection and reasoning
5. **NestJS Backend** - Receives and processes all data via WebSocket

## Data Flow Diagram

```
┌─────────────────┐
│ Video Broadcast │
│   (ZMQ PUB)     │
└────────┬────────┘
         │
         │ (frames)
         ├────────────────┬────────────────┐
         │                │                │
         ▼                ▼                │
   ┌──────────┐     ┌──────────┐         │
   │ Florence │     │ Tracker  │         │
   │  Worker  │     │  Worker  │         │
   └────┬─────┘     └────┬─────┘         │
        │                │                │
        │ (captions,     │ (tracks,       │
        │  OCR, etc.)    │  detections)   │
        │                │                │
        ├────────────┬───┼───────────┐    │
        │            │   │           │    │
        ▼            ▼   ▼           ▼    │
   ┌─────────┐   ┌──────────────┐  ┌────────┐
   │ NestJS  │   │ Qwen Worker  │  │ NestJS │
   │ (WS)    │   │ (ZMQ PULL)   │  │ (WS)   │
   └─────────┘   └──────┬───────┘  └────────┘
                        │
                        │ (anomaly
                        │  detection)
                        │
                        ▼
                   ┌─────────┐
                   │ NestJS  │
                   │ (WS)    │
                   └─────────┘
```

## Detailed Workflow

### 1. Video Broadcaster → Florence Worker

- **Protocol**: ZeroMQ SUB (topic: "frame")
- **Endpoint**: tcp://127.0.0.1:5560
- **Data**: Frame index, video timestamp, JPG bytes
- **Frequency**: Every Nth frame (configurable via --process_every_n_frames)

### 2. Video Broadcaster → Tracker Worker

- **Protocol**: ZeroMQ SUB (topic: "frame")
- **Endpoint**: tcp://127.0.0.1:5560
- **Data**: Frame index, video timestamp, JPG bytes
- **Frequency**: Real-time (every frame or configurable via --send_every_n_frames)

### 3. Florence Worker → Qwen Worker ✅

- **Protocol**: ZeroMQ PUSH
- **Endpoint**: tcp://127.0.0.1:5580
- **Data Format**: JSON
  ```json
  {
    "frame_index": 123,
    "video_time_ms": 4100,
    "raw": {
      "more_detailed_caption": "...",
      "object_detection": "...",
      "ocr": "...",
      "open_vocab_weapons": "..."
    },
    "text_overlay": {
      "datetime_candidates": ["2024-03-06", "14:30"]
    },
    "meta": {
      "generated_at_unix_ms": 1709736000000,
      "model": "florence-community/Florence-2-base"
    }
  }
  ```

### 4. Florence Worker → NestJS ✅

- **Protocol**: WebSocket
- **Endpoint**: Configurable via --ws-url (e.g., ws://localhost:3000/ws/florence)
- **Data Format**: JSON (matches NestJS expectations)
  ```json
  {
    "type": "florence_frame",
    "frame_index": 123,
    "video_time_ms": 4100,
    "caption": "A person standing near a counter...",
    "objects": [...],
    "weapons_detected": [...],
    "raw": {
      "more_detailed_caption": "...",
      "object_detection": "...",
      "ocr": "...",
      "open_vocab_weapons": "..."
    },
    "text_overlay": {
      "datetime_candidates": ["2024-03-06", "14:30"]
    },
    "meta": {
      "generated_at_unix_ms": 1709736000000,
      "model": "florence-community/Florence-2-base"
    }
  }
  ```
- **Usage**: `python florence_worker.py --ws-url ws://localhost:3000/ws/florence`

### 5. Tracker Worker → Qwen Worker ✅

- **Protocol**: ZeroMQ PUSH
- **Endpoint**: tcp://127.0.0.1:5580
- **Data Format**: JSON
  ```json
  {
    "type": "tracker_frame",
    "frame_index": 123,
    "video_time_ms": 4100,
    "frame_size": { "w": 640, "h": 480 },
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

### 6. Tracker Worker → NestJS ✅

- **Protocol**: WebSocket
- **Endpoint**: Configurable via --ws_url (e.g., ws://localhost:3000/ws/tracker)
- **Data Format**: JSON (matches NestJS expectations)
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
- **Usage**: `python tracker_worker.py --ws_url ws://localhost:3000/ws/tracker`

### 7. Qwen Worker → NestJS ✅ (NEW!)

- **Protocol**: WebSocket
- **Endpoint**: Configurable via --ws-url (e.g., ws://localhost:3000/ws/qwen)
- **Data Format**: JSON (matches NestJS expectations)
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
- **Usage**: `python qwen_anomaly_worker.py --ws-url ws://localhost:3000/ws/qwen`

## Running the System

### Start All Workers

1. **Start Video Broadcaster**:

   ```bash
   python video_broadcaster.py path/to/video.mp4
   ```

2. **Start Qwen Worker** (must start before Florence/Tracker):

   ```bash
   python qwen_anomaly_worker.py --ws-url ws://localhost:3000/ws/qwen
   ```

3. **Start Florence Worker**:

   ```bash
   python florence_worker.py --ws-url ws://localhost:3000/ws/florence
   ```

4. **Start Tracker Worker**:
   ```bash
   python tracker_worker.py --ws_url ws://localhost:3000/ws/tracker
   ```

### Command Line Arguments

#### Florence Worker

- `--video-endpoint`: ZeroMQ endpoint for video frames (default: tcp://127.0.0.1:5560)
- `--qwen-endpoint`: ZeroMQ endpoint to send data to Qwen (default: tcp://127.0.0.1:5580)
- `--ws-url`: WebSocket URL for NestJS (default: "none")
- `--model`: Florence model name (default: florence-community/Florence-2-base)
- `--device`: cpu or cuda (default: cpu)
- `--process_every_n_frames`: Process 1 frame every N frames (default: 30)

#### Tracker Worker

- `--sub_endpoint`: ZeroMQ endpoint for video frames (default: tcp://127.0.0.1:5560)
- `--ws_url`: WebSocket URL for NestJS (default: "none")
- `--yolo_model`: YOLO model file (default: yolov8n.pt)
- `--conf_th`: Confidence threshold (default: 0.35)
- `--send_every_n_frames`: Send 1 frame every N frames (default: 1)

#### Qwen Worker

- `--ws-url`: WebSocket URL for NestJS (default: "none")
- `--model`: Qwen model name (default: Qwen/Qwen2.5-3B-Instruct)
- `--device`: cpu or cuda (default: cuda)
- `--zmq-endpoint`: ZeroMQ endpoint to receive data (default: tcp://127.0.0.1:5580)

## Summary

✅ **Tracker** → Sends tracking data to **Qwen** (ZMQ) AND **NestJS** (WebSocket)
✅ **Florence** → Sends vision data to **Qwen** (ZMQ) AND **NestJS** (WebSocket)
✅ **Qwen** → Sends anomaly detection results to **NestJS** (WebSocket)

This ensures complete data flow where NestJS receives:

1. Raw tracking data from Tracker
2. Raw vision/caption data from Florence
3. Processed anomaly detection results from Qwen
