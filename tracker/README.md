# tracker/ — YOLO Detection & Tracking Worker

This folder contains the real-time detection and tracking layer.
It runs a single YOLO model per frame to detect people and robbery-relevant
objects, then streams bounding-box payloads to both React (via WebSocket)
and the Groq anomaly worker (via ZMQ PUSH).

## Files

| File | Purpose |
|---|---|
| `tracker_worker.py` | Main worker: ZMQ subscriber, IOU tracker, reset handshake, WS + Groq dispatch |
| `detection.py` | Detection utilities: `Track`, `run_yolo`, IOU helpers, frame encode/decode, debug visualisation |
| `config.py` | Detection constants — object allowlist. **Edit here to tune without touching code.** |
| `yolo26s.pt` | General object detection (COCO, 26s) |
| `yolov8s.pt` | Legacy weights (kept for reference) |
| `logs_output.jsonl` | Auto-generated log of every tracker payload sent to NestJS (one JSON object per line; `overlay_jpg_b64` omitted) |

## How it works

1. **YOLO26** — general object detection. Only robbery-relevant classes are kept (defined in `config.py → ROBBERY_OBJECT_CLASSES`), plus `person` which is always kept.
2. A simple **IOU tracker** assigns stable `track_id`s across frames.

Weapon detection for the system as a whole is handled independently by a
separate VLM worker — it is not part of this tracker.

## Tuning — edit `config.py` only

| What to change | Where in `config.py` |
|---|---|
| Robbery-relevant COCO classes to keep | `ROBBERY_OBJECT_CLASSES` |

No changes to `detection.py` or `tracker_worker.py` needed.

## Reset handshake

On video switch the broadcaster publishes a `reset` message. A dedicated watcher thread (never blocked by YOLO inference) receives it, immediately forwards a reset to the Groq worker and clears React's bboxes, then waits for the main loop to drain stale frames and clear track state before sending `reset_ack` to the broadcaster. The main loop also sends `first_frame_ack` after the first real bbox is on its way to React, which lets the broadcaster reply `{ok:true}` to NestJS and unblock the loading spinner.

See [project root README](../README.md) for the full reset flow.

## Launch command

```bash
python tracker/tracker_worker.py \
  --device cuda \
  --ws_url ws://127.0.0.1:3000/ws/tracker \
  --send_overlay 0 \
  --send_every_n_frames 5 \
  --anomaly-endpoint tcp://127.0.0.1:5581
```
