# NestJS Integration Guide - Python Backend Ports

## Overview

This guide explains the Python backend architecture and port configuration for integrating with NestJS WebSocket gateways.

**IMPORTANT**: All workers now send data **IMMEDIATELY** to NestJS through the Message Broker. No waiting for Qwen processing!

## Architecture Summary (IMMEDIATE DELIVERY)

```
┌─────────────────┐
│ Video Stream    │
│ (Port 5560 PUB) │
└────────┬────────┘
         │
    ┌────┴─────┐
    ↓          ↓
┌──────────┐ ┌──────────┐
│ Florence │ │ Tracker  │
│  Worker  │ │  Worker  │
└────┬─────┘ └────┬─────┘
     │            │
     └──→ 5580 ←──┘  (Message Broker - Central Hub)
            ↓
     ┌─────────────────┐
     │ Message Broker  │
     │ (Python Server) │
     └────┬────────┬───┘
          │        │
    IMMEDIATE    Forward copy
     to NestJS   to Qwen (5581)
          │            ↓
     WebSocket   ┌─────────────┐
     Port 3000   │ Qwen Worker │
          │      └──────┬──────┘
          │             │
          │      Results back (5580)
          │             │
          └─────────────┘
                  ↓
     ┌────────────────┐
     │ NestJS Backend │
     └────────────────┘
```

## Port Configuration

| Port     | Protocol      | Purpose                          | Bound By               | Connected By                                          |
| -------- | ------------- | -------------------------------- | ---------------------- | ----------------------------------------------------- |
| **5560** | ZMQ PUB/SUB   | Video frame broadcast            | video_broadcaster.py   | florence_worker.py, tracker_worker.py                 |
| **5561** | ZMQ REQ/REP   | Video control commands           | video_broadcaster.py   | -                                                     |
| **5580** | ZMQ PUSH/PULL | Message broker (ALL workers hub) | message_broker.py      | florence_worker.py, tracker_worker.py, qwen_worker.py |
| **5581** | ZMQ PUSH/PULL | Qwen input (analysis copy)       | qwen_anomaly_worker.py | message_broker.py                                     |
| **3000** | WebSocket     | Frontend communication           | NestJS                 | message_broker.py (as client)                         |

## Message Flow for NestJS

The **message_broker.py** (Python) connects to your **NestJS WebSocket gateways** as a **client**.

### Critical Architecture Change

**IMMEDIATE DELIVERY**: Florence and Tracker data now reaches NestJS **immediately** without waiting for Qwen processing!

**Flow**:

1. Florence/Tracker process frames → Send to Message Broker (port 5580)
2. Message Broker **IMMEDIATELY** forwards to NestJS
3. Message Broker **ALSO** forwards a copy to Qwen (port 5581) for analysis
4. Qwen processes data → Sends results back to Message Broker (port 5580)
5. Message Broker forwards Qwen results to NestJS

**Result**: NestJS receives Florence/Tracker data in real-time, and Qwen anomaly alerts separately when ready!

### Expected Setup

```
Python message_broker.py → WebSocket Client → NestJS Gateway (Server on port 3000)
```

### NestJS Should Provide

Three WebSocket gateway endpoints:

- `ws://127.0.0.1:3000/ws/florence` - Receives Florence vision data **(IMMEDIATE)**
- `ws://127.0.0.1:3000/ws/tracker` - Receives tracking data **(IMMEDIATE)**
- `ws://127.0.0.1:3000/ws/qwen` - Receives anomaly detection data **(when ready)**

## Message Types

### 1. Florence Worker Messages (Type: `florence_frame`)

**Sent to**: `ws://127.0.0.1:3000/ws/florence`

**Format**:

```json
{
  "type": "florence_frame",
  "frame_index": 123,
  "video_time_ms": 4100,
  "timestamp": 1709712345678,
  "caption": "A person walking in a store",
  "objects": ["person", "chair", "table"],
  "weapons_detected": ["gun"],
  "ocr_text": "STORE HOURS 9AM-5PM",
  "dates_found": ["2024-03-15"],
  "times_found": ["14:30"]
}
```

**Key Fields**:

- `frame_index`: Sequential frame number
- `video_time_ms`: Video timestamp in milliseconds
- `timestamp`: Unix timestamp (ms) when processed
- `caption`: Natural language scene description
- `objects`: List of detected objects (COCO classes)
- `weapons_detected`: Weapons found (gun, knife, etc.)
- `ocr_text`: Text extracted from video frame
- `dates_found`, `times_found`: Temporal information extracted

### 2. Tracker Worker Messages (Type: `tracker_frame`)

**Sent to**: `ws://127.0.0.1:3000/ws/tracker`

**Format**:

```json
{
  "type": "tracker_frame",
  "frame_index": 123,
  "video_time_ms": 4100,
  "timestamp": 1709712345678,
  "tracks": [
    {
      "track_id": 1,
      "class_name": "person",
      "confidence": 0.92,
      "bbox": [100, 150, 200, 400],
      "age": 45
    },
    {
      "track_id": 2,
      "class_name": "backpack",
      "confidence": 0.87,
      "bbox": [120, 180, 180, 250],
      "age": 12
    }
  ],
  "motion_detected": true
}
```

**Key Fields**:

- `tracks`: Array of tracked objects
  - `track_id`: Persistent ID across frames
  - `class_name`: Object type (person, car, backpack, etc.)
  - `confidence`: Detection confidence (0-1)
  - `bbox`: Bounding box `[x1, y1, x2, y2]`
  - `age`: How many frames this track has existed
- `motion_detected`: Boolean indicating motion in frame

### 3. Qwen Anomaly Messages (Type: `qwen_anomaly`)

**Sent to**: `ws://127.0.0.1:3000/ws/qwen`

**Format**:

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
    "reasoning": "Person loitering near cash register for extended period",
    "recommendations": "Alert security personnel to monitor situation"
  }
}
```

**Key Fields**:

- `frame_range`: Frame window analyzed
- `result.label`: Classification - `"normal"`, `"suspicious"`, or `"criminal"`
- `result.anomaly_score`: Score 0.0-1.0
  - `0.0-0.2`: Normal
  - `0.3-0.7`: Suspicious
  - `0.8-1.0`: Criminal
- `result.reasoning`: Natural language explanation
- `result.recommendations`: Suggested actions

## NestJS Gateway Implementation

### Florence Gateway

```typescript
@WebSocketGateway({
  cors: { origin: "*" },
  path: "/ws/florence",
})
export class FlorenceGateway {
  @WebSocketServer()
  server: Server;

  // Message broker connects as client and sends data
  handleConnection(client: Socket) {
    console.log("Florence client connected");
  }

  @SubscribeMessage("florence_frame")
  handleFlorenceData(client: Socket, payload: any) {
    // Process Florence vision data
    // Store in DB, forward to frontend, etc.
  }
}
```

### Tracker Gateway

```typescript
@WebSocketGateway({
  cors: { origin: "*" },
  path: "/ws/tracker",
})
export class TrackerGateway {
  @WebSocketServer()
  server: Server;

  @SubscribeMessage("tracker_frame")
  handleTrackerData(client: Socket, payload: any) {
    // Process tracking data
    // Update object positions, maintain track history
  }
}
```

### Qwen Anomaly Gateway

```typescript
@WebSocketGateway({
  cors: { origin: "*" },
  path: "/ws/qwen",
})
export class QwenGateway {
  @WebSocketServer()
  server: Server;

  @SubscribeMessage("qwen_anomaly")
  handleAnomalyData(client: Socket, payload: any) {
    // Process anomaly detection
    // Trigger alerts if score > threshold
    if (payload.result.anomaly_score > 0.7) {
      // Alert logic
    }
  }
}
```

## Message Broker Behavior

The Python `message_broker.py`:

1. **Binds** to ZMQ port 5582 (receives from all Python workers)
2. **Connects** to NestJS WebSocket gateways as a client
3. Routes messages based on type:
   - `florence_frame` → `ws://127.0.0.1:3000/ws/florence`
   - `tracker_frame` → `ws://127.0.0.1:3000/ws/tracker`
   - `qwen_anomaly` → `ws://127.0.0.1:3000/ws/qwen`
4. Handles reconnection if NestJS is not ready

## Important Notes for NestJS Developer

### 1. Message Broker is a WebSocket Client

The Python `message_broker.py` **connects TO your NestJS server**, not the other way around. Your NestJS gateways should act as servers waiting for connections.

### 2. Message Format

Messages arrive as JSON strings. The `type` field determines routing:

```typescript
interface BaseMessage {
  type: "florence_frame" | "tracker_frame" | "qwen_anomaly";
  frame_index?: number;
  video_time_ms?: number;
  timestamp: number;
}
```

### 3. Timing Expectations

- **Florence**: Processes every ~30 frames (configurable)
- **Tracker**: Sends every frame (configurable, can be throttled)
- **Qwen**: Analyzes windows of frames, sends only when anomaly detected

### 4. Connection Order

Start services in this order:

1. NestJS server (port 3000)
2. Python video_broadcaster.py
3. Python message_broker.py (connects to NestJS)
4. Python workers (florence, tracker, qwen)

### 5. Data Persistence

Consider implementing:

- Frame-level data storage (Florence + Tracker)
- Anomaly event logging (Qwen)
- Track history (Tracker - maintain object trajectories)
- Alert system (Qwen - for high anomaly scores)

## Testing the Integration

### 1. Check NestJS is Ready

```bash
# Test WebSocket endpoint is listening
curl http://localhost:3000/ws/florence
```

### 2. Start Python Message Broker

```bash
python message_broker.py
```

Expected output:

```
✅ ZMQ PULL bound on tcp://127.0.0.1:5582
✅ Connected to NestJS Florence gateway: ws://127.0.0.1:3000/ws/florence
✅ Connected to NestJS Tracker gateway: ws://127.0.0.1:3000/ws/tracker
✅ Connected to NestJS Qwen gateway: ws://127.0.0.1:3000/ws/qwen
```

### 3. Verify Data Flow

Check NestJS console for incoming messages from each worker.

## Troubleshooting

### "Connection Refused" Errors

- Ensure NestJS is running and WebSocket gateways are configured
- Verify port 3000 is not blocked by firewall
- Check CORS settings allow connections from localhost

### No Messages Arriving

- Verify Python workers are running and connected to correct ports
- Check message_broker.py logs for ZMQ connection status
- Ensure all workers are using correct port configuration from `config.py`

### Port Already in Use

- Each Python worker must use its designated port (5560, 5581, 5582)
- Only ONE process can BIND to each port
- Kill stale Python processes: `taskkill /F /IM python.exe`

## Configuration Reference

All Python backend ports are defined in `config.py`:

```python
ZMQ_VIDEO_BROADCASTER_ENDPOINT = "tcp://127.0.0.1:5560"
ZMQ_VIDEO_CMD_ENDPOINT = "tcp://127.0.0.1:5561"
ZMQ_QWEN_INPUT_ENDPOINT = "tcp://127.0.0.1:5581"
ZMQ_MESSAGE_BROKER_ENDPOINT = "tcp://127.0.0.1:5582"

WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 3000
WEBSOCKET_PATH = "/ws/broker"
```

## Summary for NestJS Integration

**What You Need to Do:**

1. Create 3 WebSocket gateways on port 3000:
   - `/ws/florence` - Vision AI data
   - `/ws/tracker` - Object tracking data
   - `/ws/qwen` - Anomaly detection data

2. Handle incoming messages with `type` field for routing

3. Start NestJS **before** Python message_broker.py

4. Implement data persistence and alert logic based on your requirements

**What Python Does:**

- Processes video frames through 3 AI models
- Routes results through ZMQ ports (5560, 5581, 5582)
- Connects to your NestJS gateways and streams data
- Handles reconnection automatically

---

For questions or issues, refer to [PORT_CONFIGURATION.md](PORT_CONFIGURATION.md) for detailed port mapping.
