# tracker/ — YOLO Detection & Tracking Worker

This folder contains the real-time multi-model detection and tracking layer.
It runs three YOLO models per frame to detect people, objects, weapons, and
appearance cues, then streams bounding-box payloads to both React (via WebSocket)
and the Groq anomaly worker (via ZMQ PUSH).

## Files

| File | Purpose |
|---|---|
| `tracker_worker.py` | Main worker: ZMQ subscriber, IOU tracker, reset handshake, WS + Groq dispatch |
| `detection.py` | Detection utilities: `Track`, `MotionDetector`, `run_yolo`, IOU helpers, frame encode/decode, debug visualisation |
| `config.py` | All detection constants — class thresholds, object allowlist, open-vocab prompts. **Edit here to tune without touching code.** |
| `yolo26s.pt` | General object detection (COCO, 26s) |
| `yoloe-26s-seg.pt` | Open-vocabulary appearance + weapon detection (YOLOE-26) |
| `yolov8s.pt` | Legacy weights (kept for reference) |
| `Suspicious_Activities_nano.pt` | Custom classifier: Fighting, Man_With_Gun, Man_with_Knife, Theaf_Robbery |

## How it works

Each frame goes through three models in sequence:

1. **YOLO26** — general object detection. Only robbery-relevant classes are kept (defined in `config.py → ROBBERY_OBJECT_CLASSES`).
2. **Suspicious nano model** — custom Fighting/robbery classifier. Per-class confidence floors in `config.py → SUSPICIOUS_CLASS_THRESHOLDS`. Weapon classes are additionally gated to detections that overlap a person bbox.
3. **YOLOE-26 open-vocab** — runs the prompts in `config.py → YOLOE_PROMPTS` to detect appearance tags (hood, mask…) and weapons (gun, knife…). Weapon hits go through the same person-gate + temporal confirmation as the nano model. Appearance hits are attached to the nearest person track after `--appearance_confirm` frames.

A simple **IOU tracker** assigns stable `track_id`s across frames. Motion fallback (MOG2 background subtraction) fires when YOLO finds nothing.

## Tuning — edit `config.py` only

| What to change | Where in `config.py` |
|---|---|
| Robbery-relevant COCO classes to keep | `ROBBERY_OBJECT_CLASSES` |
| Confidence floor per suspicious class | `SUSPICIOUS_CLASS_THRESHOLDS` |
| Appearance prompts (what YOLOE looks for) | `APPEARANCE_PROMPTS` |
| Weapon prompts (open-vocab) | `WEAPON_PROMPTS` |
| Classes that require person-overlap gating | `WEAPON_SUSPICIOUS_CLASSES` |

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
