# Port Configuration

## Problem Solved

**Issue**: ZMQ "already in use" error - multiple workers were trying to bind/connect to the same port (5580).

**Solution**: Each communication channel now has its own dedicated port.

## Port Mapping

| Port     | Type      | Purpose                | Bind (Server)          | Connect (Clients)                     |
| -------- | --------- | ---------------------- | ---------------------- | ------------------------------------- |
| **5560** | PUB/SUB   | Video frames broadcast | video_broadcaster.py   | florence_worker.py, tracker_worker.py |
| **5561** | REQ/REP   | Video control commands | video_broadcaster.py   | (future client commands)              |
| **5581** | PUSH/PULL | Qwen worker input      | qwen_anomaly_worker.py | florence_worker.py, tracker_worker.py |
| **5582** | PUSH/PULL | Message broker input   | message_broker.py      | qwen_anomaly_worker.py                |

## Data Flow

```
Video Broadcaster (5560 PUB)
         ↓
    ┌────────────┐
    ↓            ↓
Florence      Tracker
(SUB 5560)    (SUB 5560)
    ↓            ↓
[PUSH 5581]  [PUSH 5581]  ← Both push to Qwen input port
         ↓
    Qwen Worker
    (PULL 5581)
         ↓
    [PUSH 5582]  ← Qwen pushes to Message Broker
         ↓
   Message Broker
   (PULL 5582)
         ↓
    WebSocket 3000
         ↓
      NestJS
```

## Configuration

All ports are defined in [config.py](config.py):

```python
ZMQ_VIDEO_BROADCASTER_ENDPOINT = "tcp://127.0.0.1:5560"  # Video frames (PUB)
ZMQ_VIDEO_CMD_ENDPOINT = "tcp://127.0.0.1:5561"          # Video commands (REP)
ZMQ_QWEN_INPUT_ENDPOINT = "tcp://127.0.0.1:5581"         # Florence + Tracker → Qwen (PUSH → PULL)
ZMQ_MESSAGE_BROKER_ENDPOINT = "tcp://127.0.0.1:5582"     # All workers → Message Broker (PUSH → PULL)
```

## Key Changes

### Before (Problematic)

- **Port 5580**: Everyone tried to use this
  - Florence PUSH → 5580
  - Tracker PUSH → 5580
  - Qwen PULL (bind) → 5580 ❌ CONFLICT
  - Qwen PUSH → 5580 ❌ CONFLICT
  - Message Broker PULL (bind) → 5580 ❌ CONFLICT

### After (Fixed)

- **Port 5581**: Qwen input only
  - Florence PUSH → 5581
  - Tracker PUSH → 5581
  - Qwen PULL (bind) ← 5581 ✅
- **Port 5582**: Message Broker input only
  - Qwen PUSH → 5582
  - Message Broker PULL (bind) ← 5582 ✅

## Worker Configuration

### Florence Worker

```python
# Reads from: ZMQ_VIDEO_BROADCASTER_ENDPOINT (5560)
# Writes to: ZMQ_QWEN_INPUT_ENDPOINT (5581)
python florence_worker.py --device cuda
```

### Tracker Worker

```python
# Reads from: ZMQ_VIDEO_BROADCASTER_ENDPOINT (5560)
# Writes to: ZMQ_QWEN_INPUT_ENDPOINT (5581)
python tracker_worker.py
```

### Qwen Anomaly Worker

```python
# Reads from: ZMQ_QWEN_INPUT_ENDPOINT (5581) - BIND
# Writes to: ZMQ_MESSAGE_BROKER_ENDPOINT (5582) - CONNECT
python qwen_anomaly_worker.py --device cuda
```

### Message Broker

```python
# Reads from: ZMQ_MESSAGE_BROKER_ENDPOINT (5582) - BIND
# Writes to: WebSocket 3000
python message_broker.py
```

## Important Notes

1. **Only ONE process can BIND** to a port (the server/listener)
2. **Multiple processes can CONNECT** to a bound port (the clients)
3. **Start order matters**:
   - Start servers FIRST (those that BIND): video_broadcaster, message_broker, qwen_worker
   - Start clients AFTER (those that CONNECT): florence_worker, tracker_worker
4. All port configurations are centralized in `config.py` for easy management
5. **CRITICAL**: Message Broker acts as a central hub - receives from ALL workers and:
   - Immediately sends Florence/Tracker data to NestJS (no waiting!)
   - Forwards copies of Florence/Tracker to Qwen for analysis
   - Sends Qwen results to NestJS when ready

## Performance Benefits

✅ **Zero latency for Florence/Tracker data** - NestJS receives data immediately
✅ **Non-blocking architecture** - Qwen processing doesn't delay other workers
✅ **Parallel processing** - All workers can send data simultaneously
✅ **Real-time updates** - Frontend gets data as fast as workers produce it

## Troubleshooting

If you still get "already in use" errors:

1. Check if any process from previous run is still active
2. Kill all Python processes: `taskkill /F /IM python.exe` (Windows)
3. Wait a few seconds for ports to be released
4. Restart in proper order
