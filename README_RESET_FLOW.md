# Pipeline Reset & Sync Flow

## Why reset exists

When a new video is selected, every component in the pipeline holds stale state from the previous video:

- **Florence** may have old-video frames buffered in its ZMQ SUB socket (because it only processes 1 frame every N and can be mid-inference for 10–15s while frames pile up).
- **Tracker** has active tracks, track IDs, and a background subtractor trained on the old video.
- **Qwen/Groq** has a sliding window of captions and tracker events from the old video.

Without a coordinated reset, old detections and bounding boxes will appear over the new video for several seconds.

---

## Components involved

| Component | File | Role in reset |
|---|---|---|
| **Broadcaster** | `video_broadcaster.py` | Orchestrates the whole handshake. Sends `reset` signal, waits for acks, controls when NestJS gets `{ok:true}`. |
| **Tracker** | `tracker_worker.py` | Clears tracks, resets motion detector, drains SUB buffer, emits UI-clear payload to React, sends both acks. |
| **Florence** | `florence_worker.py` | Recreates background subtractor, drains SUB buffer, sends `reset_ack`. |
| **Qwen/Groq** | `qwen_anomaly_worker.py` / `groq_anomaly_worker.py` | Clears event history and buffers. No ack — fire-and-forget. |
| **React UI** | Dashboard | Shows loading spinner while `POST /videos/selection` is pending. Video plays only after `{ok:true}` returns. |

---

## Sockets used

| Port | Type | Direction | Purpose |
|---|---|---|---|
| `5560` | ZMQ PUB/SUB | Broadcaster → Workers | Video frames (`frame` topic) and reset signal (`reset` topic) |
| `5561` | ZMQ REQ/REP | NestJS → Broadcaster | Play commands from NestJS; broadcaster replies with `{ok:true}` |
| `5562` | ZMQ PUSH/PULL | Workers → Broadcaster | Acks: `reset_ack` and `first_frame_ack` |
| `5580` | ZMQ PUSH/PULL | Tracker/Florence → Qwen | Frame data + reset forward |
| `ws://localhost:3000/ws/tracker` | WebSocket | Tracker → NestJS → React | Bounding box payloads and reset clear signal |

---

## Full reset sequence (step by step)

```
NestJS           Broadcaster          Florence            Tracker            Qwen/Groq        React
  │                   │                   │                   │                   │              │
  │──POST /selection──▶                   │                   │                   │              │
  │                   │                   │                   │                   │              │
  │             open cap                  │                   │                   │              │
  │             set frame_index=0         │                   │                   │              │
  │                   │                   │                   │                   │              │
  │             [b"reset"] ──────────────▶│                   │                   │              │
  │                   │                   │                   │                   │              │
  │                   │        watcher receives reset         │                   │              │
  │                   │            │──{"type":"reset"}───────────────────────────▶│              │
  │                   │            │      │                   │                   │ clear queues │
  │                   │            │      │                   │                   │              │
  │             [b"reset"] ──────────────────────────────────▶│                   │              │
  │                   │            │      │       watcher receives reset          │              │
  │                   │            │      │           │──{"type":"reset"}─────────▶              │
  │                   │            │      │           │      clear bboxes──────────────────────▶│
  │                   │            │      │           │       (reset: true WS)                  │
  │                   │            │      │           │                           │              │
  │                   │            │      │       _reset_event.set()              │              │
  │                   │            │      │   _first_frame_pending.set()          │              │
  │                   │            │      │                   │                   │              │
  │                   │        _reset_event.set()             │                   │              │
  │                   │            │      │                   │                   │              │
  │                   │            │  main loop wakes (≤100ms RCVTIMEO)          │              │
  │                   │            │      │               drain SUB               │              │
  │                   │            │      │               clear tracks            │              │
  │                   │            │      │               reset MotionDetector    │              │
  │                   │            │      │           _reset_done_event.set()     │              │
  │                   │            │      │                   │                   │              │
  │                   │        main loop wakes (≤100ms RCVTIMEO)                 │              │
  │                   │        drain SUB buffer               │                   │              │
  │                   │        recreate bg_subtractor         │                   │              │
  │                   │        _reset_done_event.set()        │                   │              │
  │                   │            │      │                   │                   │              │
  │                   │            │  watcher: reset_ack ────────────────────────────────────▶  │
  │              ◀─reset_ack (florence)   │                   │                   │              │
  │                   │            │  watcher: reset_ack ──────────────────────────────────▶    │
  │              ◀─reset_ack (tracker)    │                   │                   │              │
  │                   │                   │                   │                   │              │
  │             publish [b"frame", b"0", b"0", jpg]           │                   │              │
  │                   │──────────────────────────────────────▶│                   │              │
  │                   │                   │                   │                   │              │
  │                   │                   │         run YOLO on frame 0           │              │
  │                   │                   │                   │──tracker_frame────────────────▶│
  │                   │                   │                   │    (bboxes for frame 0)         │
  │                   │                   │                   │                   │              │
  │              ◀─first_frame_ack (tracker)                  │                   │              │
  │                   │                   │                   │                   │              │
  │          send_json({status:"ok"}) ─▶  │                   │                   │              │
  │◀─{ok:true}────────│                   │                   │                   │              │
  │                   │                   │                   │                   │              │
  │             stream_start_time = now   │                   │                   │              │
  │             frame_index = 1          │                   │                   │              │
  │             begin paced streaming ──────────────────────▶│──────────────────▶│              │
  │                   │                   │                   │                   │   hide spinner│
  │                   │                   │                   │                   │   play video ─▶
```

---

## The two acks explained

### Ack 1 — `reset_ack` (Florence + Tracker)

**Meaning:** "My local state is clean. I have no stale frames buffered. You can start streaming the new video."

**When sent:** After the worker's main loop drains its SUB buffer and clears all local state (tracks, background subtractor, etc.).

**Broadcaster waits up to:** `WORKER_ACK_TIMEOUT_S = 20s`

### Ack 2 — `first_frame_ack` (Tracker only)

**Meaning:** "I have run YOLO on frame 0 and sent the first `tracker_frame` payload to React over WebSocket. The bbox is already in transit."

**When sent:** In `tracker_worker.py`, immediately after `ws_send_json(ws, payload)` fires for the first non-clear `tracker_frame` after a reset. Controlled by the `_first_frame_pending` threading.Event.

**Broadcaster waits up to:** `FIRST_FRAME_ACK_TIMEOUT_S = 5s`

**Why only Tracker?** Florence emits once every 30 frames (default) — waiting for its first emit would add 10–15 seconds of latency. YOLO inference is fast (~50–200ms), so tracker can ack quickly.

---

## What each worker does on reset

### Broadcaster (`video_broadcaster.py`)

1. Releases old `cv2.VideoCapture`.
2. Opens new capture, reads `CAP_PROP_FPS`.
3. Publishes `[b"reset"]` on PUB socket (port 5560).
4. Waits for `reset_ack` from both Florence and Tracker (ack socket, port 5562).
5. Reads frame 0, resizes if needed, publishes `[b"frame", b"0", b"0", jpg]` as a warmup frame.
6. Waits for `first_frame_ack` from Tracker.
7. Sends `{status:"ok"}` on REP socket to NestJS.
8. Broadcasts `[b"meta", w, h, fps]` on PUB.
9. Sets `stream_start_time = time.time()`, `frame_index = 1`.
10. Resumes normal paced streaming.

### Florence (`florence_worker.py`)

**Watcher thread** (separate SUB on `reset` topic):
- Immediately forwards `{"type":"reset"}` to Qwen/Groq via PUSH.
- Sets `_reset_event` so the main loop will drain on next iteration.
- Waits up to 15s for `_reset_done_event`, then sends `reset_ack` to broadcaster.

**Main loop:**
- Checks `_reset_event` at the top of every iteration (before `recv_multipart`, with `RCVTIMEO=100ms` to avoid infinite block).
- On reset: recreates `cv2.createBackgroundSubtractorMOG2`, drains SUB buffer with `NOBLOCK`, sets `_reset_done_event`.
- Also checks mid-inference (after all four Florence tasks complete) to discard stale results if reset arrived during long inference.

### Tracker (`tracker_worker.py`)

**Watcher thread** (separate SUB on `reset` topic):
- Immediately forwards `{"type":"reset"}` to Qwen/Groq via PUSH.
- Dispatches `"clear"` to the asyncio `_clear_queue` → `clear_sender` task sends `{type:"tracker_frame", tracks:[], reset:true}` over WebSocket to React (bboxes disappear immediately).
- Sets `_first_frame_pending` and `_reset_event`.
- Waits up to 15s for `_reset_done_event`, then sends `reset_ack` to broadcaster.

**Main loop:**
- Checks `_reset_event` at the top of every iteration (before `recv_multipart`, with `RCVTIMEO=100ms`).
- On reset: clears `tracks`, resets `next_track_id = 1`, creates a new `MotionDetector`, sets `first_frame = True`, drains SUB buffer, sets `_reset_done_event`.
- Also checks mid-inference (after both YOLO model calls) to discard stale detections.
- After the first real `ws_send_json` post-reset: if `_first_frame_pending` is set, sends `first_frame_ack` to broadcaster and clears the flag.

### Qwen/Groq (`qwen_anomaly_worker.py` / `groq_anomaly_worker.py`)

- Receives `{"type":"reset"}` via ZMQ PULL (forwarded by Florence and Tracker watchers — two resets per video switch is normal and safe).
- Clears `raw_queue`, `event_history`, `tracker_buffer`.
- Does **not** ack the broadcaster — it is not in the critical sync path.
- `latest_business_context` is intentionally preserved across resets.

---

## Cold-start behaviour (first play after system start)

Before the fix, the first `/selection` call took ~15 seconds. Here is why and how it is fixed.

### The problem

The watcher waits for `_reset_done_event` (set by the main loop). The main loop only signals after returning from `recv_multipart`. On the very first play, no frames have ever been published, so `recv_multipart` blocks indefinitely. The watcher's 15s self-timeout fires and it acks "anyway". The broadcaster sees the late ack and finally replies to NestJS.

### The fix

Both workers now set `RCVTIMEO = 100ms` on their frame SUB sockets and check `_reset_event` **before** calling `recv_multipart`. If the recv times out with no frame (the cold-start case), the loop immediately cycles back to the reset check — finds `_reset_event` set, applies the reset (drain is a no-op, nothing is buffered), sets `_reset_done_event`. The watcher sees the event in ≤100ms instead of waiting 15s.

| Condition | Before fix | After fix |
|---|---|---|
| First play (nothing buffered) | ~15s (watcher timeout) | ~100ms |
| Subsequent plays (frames buffered) | <100ms (existing frame wakes loop) | <100ms (unchanged) |
| Mid-inference reset (Florence) | Works (mid-inference checkpoint at line 463) | Works (unchanged) |

---

## Timeout safety

All waits have explicit timeouts so the system degrades gracefully rather than hanging:

| Wait | Timeout | Fallback |
|---|---|---|
| `reset_ack` from Florence + Tracker | `WORKER_ACK_TIMEOUT_S = 20s` | Log which workers are missing, proceed anyway |
| `first_frame_ack` from Tracker | `FIRST_FRAME_ACK_TIMEOUT_S = 5s` | Log, reply to NestJS anyway (`first_frame_ready: false`) |
| Watcher waiting for main loop `_reset_done_event` | `ACK_TIMEOUT_S = 15s` | Log, send `reset_ack` anyway |

---

## FPS pacing and why `stream_start_time` matters

The broadcaster paces frame delivery using:

```python
expected_time = stream_start_time + (frame_index / video_fps)
time_diff = expected_time - time.time()
if time_diff > 0:
    time.sleep(time_diff)
```

`stream_start_time` is set **after** the full handshake (after `cmd_socket.send_json`). This means:

- Frame 1's `expected_time` = `stream_start_time + 1/fps` ≈ 33ms from when React received `{ok:true}`.
- The React `<video>` element starts at `t=0` at approximately the same moment.
- Frames are delivered at the natural video FPS from the start — no catch-up burst.

If `stream_start_time` were set before the ~15s ack wait (as it was before the fix), `expected_time` for all early frames would be far in the past, and the broadcaster would sprint frames at full CPU speed until it caught up to wall-clock time.

---

## Sequence of flags in `tracker_worker.py`

```
watcher receives [b"reset"]
    │
    ├── groq_sock.send({"type":"reset"})
    ├── _clear_queue.put("clear")  → React clears bboxes
    ├── _reset_done_event.clear()
    ├── _first_frame_pending.set()
    └── _reset_event.set()
         │
         └── wait _reset_done_event (≤15s)
              │
              └── send reset_ack to broadcaster

main loop (≤100ms later)
    │
    ├── sees _reset_event.is_set()
    ├── clear tracks, reset IDs, new MotionDetector
    ├── drain SUB buffer (NOBLOCK)
    └── _reset_done_event.set()  ← unblocks watcher

main loop processes warmup frame 0
    │
    ├── run_yolo (both models)
    ├── build tracks_payload
    ├── ws_send_json(ws, payload)  ← bbox for frame 0 sent to React
    │
    └── _first_frame_pending.is_set() → True
         ├── _first_frame_pending.clear()
         └── ack_socket.send_json({"worker":"tracker","type":"first_frame_ack"})
              │
              └── broadcaster receives this → replies {ok:true} to NestJS
```
