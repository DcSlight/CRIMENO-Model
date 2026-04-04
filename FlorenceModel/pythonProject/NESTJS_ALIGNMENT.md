# NestJS Gateway Alignment

qwen json example:

----- prompt output:
```
==================== Anomaly decision ====================
Frames 0–60
{
  "anomaly_score": 0.5,
  "label": "suspicious",
  "reason": "Individuals acting suspiciously near cash register.",
  "key_moments": [
    "Man holding credit card near cash register",
    "Woman looking intensely at cash register"
  ]
}
=========================================================
```
---

## 1. Florence Worker → Florence Gateway

### NestJS Expects (florence.gateway.ts)


```typescript
path: '/ws/florence'
Expected fields:
- type: 'florence_frame'
- frame_index
- video_time_ms
- caption
- weapons_detected (array)
- objects (array)
```

### Python Now Sends ✅

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
  "text_overlay": {...},
  "meta": {...}
}
```

**WebSocket URL**: `ws://localhost:3000/ws/florence`

---

## 2. Tracker Worker → Tracker Gateway

### NestJS Expects (tracker.gateway.ts)

```typescript
path: '/ws/tracker'
Expected fields:
- type: 'tracker_frame'
- frame_index
- video_time_ms
- tracks (array)
- motion_detected (boolean)
```

### Python Now Sends ✅

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

**WebSocket URL**: `ws://localhost:3000/ws/tracker`

---

## 3. Qwen Worker → Qwen Gateway

### NestJS Expects (qwen.gateway.ts)

```typescript
path: '/ws/qwen'
Expected fields:
- type: 'qwen_anomaly'
- frame_range.start
- frame_range.end
- result.label
- result.anomaly_score
```

### Python Now Sends ✅

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

**WebSocket URL**: `ws://localhost:3000/ws/qwen`

---

## What Was Fixed

### Florence Worker Changes:

1. ✅ Added `"type": "florence_frame"` field
2. ✅ Added `"caption"` field (copy of `raw.more_detailed_caption`)
3. ✅ Added `"objects"` array field (parsed from `raw.object_detection`)
4. ✅ Added `"weapons_detected"` array field (parsed from `raw.open_vocab_weapons`)
5. ✅ Kept all `raw` fields for internal processing

### Tracker Worker Changes:

1. ✅ Added `"motion_detected"` boolean field
   - `true` if any tracks have `cls_name == "moving_object"`
   - `false` otherwise

### Qwen Worker Changes:

1. ✅ Restructured payload to use nested objects:
   - `frame_range.start` instead of `frame_start`
   - `frame_range.end` instead of `frame_end`
   - `result.anomaly_score` instead of flat `anomaly_score`
   - `result.label` instead of flat `label`
   - `result.reason` and `result.key_moments` inside `result` object

---

## Testing

### Start Commands (in order):

1. **NestJS Backend**:

   ```bash
   cd path/to/CRIMENO-Backend
   npm run start
   ```

2. **Qwen Worker**:

   ```bash
   python qwen_anomaly_worker.py --ws-url ws://localhost:3000/ws/qwen
   ```

3. **Florence Worker**:

   ```bash
   python florence_worker.py --ws-url ws://localhost:3000/ws/florence
   ```

4. **Tracker Worker**:

   ```bash
   python tracker_worker.py --ws_url ws://localhost:3000/ws/tracker
   ```

5. **Video Broadcaster**:
   ```bash
   python video_broadcaster.py path/to/video.mp4
   ```

### Expected NestJS Logs:

**Florence Gateway**:

```
🧠 florence frame=123 t=4100ms objects=5 weapons=0 | A person standing near counter...
```

**Tracker Gateway**:

```
⚡ tracker frame=123 t=4100ms tracks=3 motion=no
```

**Qwen Gateway**:

```
🚨 qwen range=120-150 label=suspicious score=0.85
```

---

## Compatibility

✅ **Perfect Match** - All Python workers now send data in the exact format expected by NestJS gateways.

The NestJS logging will now work correctly and display meaningful information from all workers!
