# CRIMENO Surveillance System Architecture

## Overview

Clean layered architecture with proper separation of concerns. All workers communicate only through ZeroMQ, WebSocket communication is centralized in the message broker.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│           VIDEO_BROADCASTER (5560 PUB, 5561 REP)          │
│  - Reads video file                                         │
│  - Broadcasts frames via ZMQ PUB (5560)                    │
│  - Listens for control commands from NestJS (5561)         │
└────────────────────┬──────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼ SUB 5560               ▼ SUB 5560
   ┌──────────────┐         ┌──────────────┐
   │  FLORENCE    │         │  TRACKER     │
   │   WORKER     │         │   WORKER     │
   │              │         │              │
   │ - Vision AI  │         │ - YOLOv8+IOU │
   │ - Captions   │         │ - Tracking   │
   │ - OCR        │         │ - Motion     │
   │ - Weapons    │         │              │
   └──────┬───────┘         └──────┬───────┘
          │                        │
          │ PUSH 5580             │ PUSH 5580
          └────────────┬──────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   QWEN_ANOMALY       │
            │   (PULL 5580)        │
            │                      │
            │ - Listens to frames  │
            │ - Buffers & merges   │
            │ - LLM analysis       │
            │ - Anomaly scoring    │
            └──────────┬───────────┘
                       │
                       │ PUSH 5580
                       ▼
            ┌──────────────────────────────┐
            │  MESSAGE_BROKER.py           │
            │  (PULL 5580 - Routes by type)│
            │                              │
            │  Routes:                     │
            │  - florence_frame → /ws/... │
            │  - tracker_frame → /ws/...  │
            │  - qwen_anomaly → /ws/...   │
            └──────┬──────────┬──────────┬─┘
                   │          │          │
                   ▼          ▼          ▼
         /ws/florence   /ws/tracker   /ws/qwen
                   │          │          │
            ┌──────┴──────────┴──────────┘
            │
            ▼
    ┌──────────────────────┐
    │   NESTJS GATEWAYS    │
    │ (3 separate paths)   │
    │                      │
    │ - Florence Gateway   │
    │ - Tracker Gateway    │
    │ - Qwen Gateway       │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   NESTJS FRONTEND    │
    │   (React + API)      │
    │                      │
    │ - Real-time updates  │
    │ - Dashboard          │
    │ - Controls broadcast │
    └──────────────────────┘
```

## Port Configuration

| Port | Protocol      | Use                | Sender                           | Receiver          |
| ---- | ------------- | ------------------ | -------------------------------- | ----------------- |
| 5560 | ZMQ PUB/SUB   | Frame streaming    | Broadcaster                      | Florence, Tracker |
| 5561 | ZMQ REP/REQ   | Control commands   | NestJS                           | Broadcaster       |
| 5580 | ZMQ PUSH/PULL | Results & analysis | Florence, Tracker, Qwen → Broker | Message Broker    |
| 3000 | WebSocket     | Frontend updates   | Message Broker (routes by type)  | NestJS Gateways   |

**WebSocket Routing (Port 3000):**

- `florence_frame` → `/ws/florence`
- `tracker_frame` → `/ws/tracker`
- `qwen_anomaly` → `/ws/qwen`

## Layer Separation

### Layer 1: Data Source

- **video_broadcaster.py** - Frame source, no business logic

### Layer 2: Processing Workers

- **florence_worker.py** - Vision AI tasks (only knows about 5560 input, 5580 output)
- **tracker_worker.py** - Object tracking (only knows about 5560 input, 5580 output)
- **qwen_anomaly_worker.py** - Anomaly detection (only knows about 5580)

### Layer 3: Message Hub

- **message_broker.py** - Central communication (ZMQ ↔ WebSocket bridge)

### Layer 4: Frontend

- **NestJS** - UI, user interaction

## Key Design Principles

✅ **Workers are decoupled**

- Each worker only knows its input and one output port (5580)
- Workers don't know about WebSocket, NestJS, or each other
- Workers don't care where data goes after 5580

✅ **Single point of communication**

- All results flow through ZMQ 5580
- Message Broker handles WebSocket forwarding
- Easy to add new workers or remove them

✅ **Clean configuration**

- All ports and settings in `config.py`
- No hardcoded values scattered across files
- Single source of truth

✅ **Scalable**

- Easy to add new workers (they just push to 5580)
- Easy to add new consumers (broker broadcasts via WS)
- Easy to replace message transport (change broker, not workers)

## Message Format

All messages are JSON with `type` field for routing.

### NestJS WebSocket Integration

**Critical for NestJS Implementation:**

- **WebSocket Endpoint**: `ws://localhost:3000/ws/broker`
- **Protocol**: Standard WebSocket (not Socket.IO)
- **Message Format**: JSON strings
- **All three models send data that arrives at NestJS**

### Data Flow to NestJS

```
Florence Worker → ZMQ 5580 → Message Broker → WebSocket 3000 → NestJS ✅
Tracker Worker → ZMQ 5580 → Message Broker → WebSocket 3000 → NestJS ✅
Qwen Worker    → ZMQ 5580 → Message Broker → WebSocket 3000 → NestJS ✅
```

**Important**: Qwen receives Florence + Tracker data from 5580 (PULL), processes it, then sends its own results back to 5580 (PUSH). All three message types arrive at NestJS.

### Message Type 1: Florence Frame Analysis

**Frequency**: ~1-2 messages per second (depends on `--every` parameter)

```json
{
  "type": "florence_frame",
  "frame_index": 123,
  "video_time_ms": 5000,
  "raw": {
    "more_detailed_caption": "A person in a blue shirt walks through a store aisle past shelves of products",
    "object_detection": "<OD>person<loc_123><loc_456><loc_789><loc_012>shelf<loc_...",
    "ocr": "<OCR>Store Hours: 9:00 - 21:00\nAisle 5\n14:30",
    "open_vocab_weapons": "<OPEN_VOCABULARY_DETECTION>"
  },
  "text_overlay": {
    "datetime_candidates": ["14:30", "9:00", "21:00"]
  },
  "meta": {
    "generated_at_unix_ms": 1709740800000,
    "model": "florence-community/Florence-2-base"
  }
}
```

**NestJS should handle**:

- Display caption in UI
- Show detected objects
- Display OCR text (timestamps, signage)
- Alert on weapon detection (if not empty)

### Message Type 2: Tracker Frame

**Frequency**: ~10-30 messages per second (depends on `--send_every_n_frames`)

```json
{
  "type": "tracker_frame",
  "frame_index": 125,
  "video_time_ms": 5083,
  "frame_size": {
    "w": 1920,
    "h": 1080
  },
  "tracks": [
    {
      "track_id": 1,
      "cls": "person",
      "conf": 0.89,
      "bbox": {
        "x1": 450,
        "y1": 200,
        "x2": 650,
        "y2": 800
      }
    },
    {
      "track_id": 2,
      "cls": "backpack",
      "conf": 0.76,
      "bbox": {
        "x1": 500,
        "y1": 300,
        "x2": 580,
        "y2": 450
      }
    }
  ]
}
```

**NestJS should handle**:

- Draw bounding boxes on video canvas
- Show track IDs (persistent across frames)
- Display object count
- Track movement patterns

### Message Type 3: Qwen Anomaly Result

**Frequency**: ~0.5-1 message per second (depends on scene change detection)

```json
{
  "type": "qwen_anomaly",
  "frame_range": {
    "start": 120,
    "end": 150
  },
  "result": {
    "label": "suspicious",
    "anomaly_score": 0.65,
    "reasoning": "Person loitering near high-value items for extended period, frequent glances toward exit",
    "action": "Increase monitoring of this individual"
  }
}
```

**NestJS should handle**:

- Display anomaly alerts in UI
- Show anomaly score (0.0-1.0)
  - 0.0-0.3: Normal behavior
  - 0.3-0.7: Suspicious activity
  - 0.8-1.0: Criminal activity
- Display reasoning text
- Show recommended action
- Log to anomaly history

### Message Type Routing

NestJS WebSocket client should route by `type` field:

```typescript
// Example NestJS/TypeScript handler
websocket.on("message", (data: string) => {
  const msg = JSON.parse(data);

  switch (msg.type) {
    case "florence_frame":
      handleFlorenceFrame(msg);
      break;
    case "tracker_frame":
      handleTrackerFrame(msg);
      break;
    case "qwen_anomaly":
      handleQwenAnomaly(msg);
      break;
    default:
      console.warn("Unknown message type:", msg.type);
  }
});
```

## Running the System

See `How to run.txt` for execution order.

**Important**: Start processes in this order:

1. Message Broker (waits for connections)
2. Video Broadcaster (waits for frames)
3. All workers (connect to their inputs/outputs)

## Adding a New Worker

1. Create `my_worker.py`
2. Open input socket to port 5580 (PULL)
3. Process data
4. Open output socket to port 5580 (PUSH)
5. No WebSocket code needed - broker handles it!
6. Test: run `python my_worker.py`

## Benefits of This Architecture

- ✅ **Testable**: Each worker can run independently
- ✅ **Maintainable**: No coupling between workers
- ✅ **Scalable**: Add workers without changing code
- ✅ **Reliable**: Message broker buffers and retries
- ✅ **Observable**: All data flows through one point
- ✅ **Flexible**: Easy to change communication (ZMQ → RabbitMQ, etc.)

## NestJS Integration Checklist

**For the NestJS team to verify:**

- [ ] WebSocket client connects to `ws://localhost:3000/ws/broker`
- [ ] Client handles reconnection on disconnect
- [ ] Client parses incoming JSON messages
- [ ] Client routes by `msg.type` field:
  - [ ] `florence_frame` - Vision AI analysis
  - [ ] `tracker_frame` - Object tracking data
  - [ ] `qwen_anomaly` - Anomaly detection results
- [ ] UI displays Florence captions and OCR
- [ ] UI draws tracker bounding boxes on video
- [ ] UI shows anomaly alerts with scores
- [ ] All three data streams are received and displayed
- [ ] Error handling for malformed messages
- [ ] Performance: Can handle 30+ messages/second

**Testing the Integration:**

1. Start message broker: `python message_broker.py`
2. Start video broadcaster: `python video_broadcaster.py videos/shop.mp4`
3. Start workers: `python florence_worker.py`, `python tracker_worker.py`, `python qwen_anomaly_worker.py`
4. Connect NestJS WebSocket client to `ws://localhost:3000/ws/broker`
5. Verify all three message types arrive in NestJS
6. Check console logs in message_broker.py to see connection status

**Expected Behavior:**

- Florence sends ~1-2 messages/second with vision analysis
- Tracker sends ~10-30 messages/second with object positions
- Qwen sends ~0.5-1 messages/second with anomaly scores
- Message broker logs: `📨 Received florence_frame | frame=123`
- NestJS receives all three types through single WebSocket connection
