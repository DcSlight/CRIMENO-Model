# Criminal Activity Detection – Layer 1 (Evidence Collection)

## Purpose
Layer 1 is responsible for **collecting raw evidence from video streams**.
It does **not** perform reasoning or classification of criminal activity.
Instead, it produces structured, time-aligned evidence that can later be consumed by Layer 2 (Inference & Reasoning).

The goal is to preserve **as much useful information as possible**:
- What appears in the scene
- How objects move over time
- How the scene description evolves

All outputs are stored as **JSONL** for offline processing.

---

## High-Level Architecture

```
Video File
   │
   ▼
video_broadcaster.py
(ZeroMQ PUB)
   │
   ├──▶ florence_worker.py   (every 60 frames)
   │        └── florence.jsonl
   │
   └──▶ tracker_worker.py    (every 5 frames)
            └── tracker.jsonl
```

Each worker is independent and subscribes to the same video stream.

---

## video_broadcaster.py

### Responsibility
- Reads a video file
- Streams frames via ZeroMQ (PUB)
- Maintains frame index and video timestamp

### Frame Message Format (ZeroMQ multipart)

```
[
  topic="frame",
  frame_index (string),
  video_time_ms (string),
  jpg_bytes
]
```

This format is shared by **all workers**.

---

## florence_worker.py

### Responsibility
- Subscribes to the video stream
- Processes one frame every N frames (default: 60)
- Runs a Vision→Text model (Florence)
- Produces a semantic description of the scene

### Output File
`florence.jsonl`

Each line represents **one processed frame**.

### Florence JSON Structure

```json
{
  "frame_index": 60,
  "video_time_ms": 2002,

  "raw": {
    "more_detailed_caption": "...",
    "object_detection": "...",
    "ocr": "..."
  },

  "text_overlay": {
    "datetime_candidates": ["2024-01-12 18:32"]
  },

  "meta": {
    "generated_at_unix_ms": 1730000000000,
    "worker": "florence_worker",
    "device": "cuda",
    "every": 60
  }
}
```

### Why this is useful
- Provides a **human-readable narrative** of the scene
- Captures **semantic changes** over time
- OCR adds environmental context (time, signage, labels)

---

## tracker_worker.py

### Responsibility
- Subscribes to the video stream
- Processes frames at high frequency (default: every 5 frames)
- Detects and tracks objects over time
- Collects motion evidence even when classification fails

### Detection Sources
1. **YOLO (multi-class, COCO)**
   - Provides class, confidence, bounding box
   - Assigns `track_id` for temporal identity
2. **Motion fallback (background subtraction)**
   - Captures moving objects without classification
   - Assigns `motion_track_id`

### Output File
`tracker.jsonl`

Each line represents **one processed frame**.

---

### Tracker JSON Structure

```json
{
  "frame_index": 65,
  "video_time_ms": 2170,

  "meta": {
    "generated_at_unix_ms": 1730000000123,
    "worker": "tracker_worker",
    "device": "cuda",
    "yolo_model": "yolov8n.pt",
    "every": 5
  },

  "detections": {
    "yolo": [
      {
        "bbox_xyxy": [412, 180, 512, 420],
        "cls_id": 0,
        "cls_name": "person",
        "conf": 0.87,
        "track_id": 3,
        "source": "yolo"
      }
    ],

    "motion": [
      {
        "bbox_xyxy": [500, 300, 560, 360],
        "area_px": 1320,
        "cls_name": "moving_object",
        "motion_track_id": 18,
        "source": "motion"
      }
    ]
  }
}
```

---

## Types of Evidence Collected

### Identity Evidence
- `track_id` (YOLO)
- `motion_track_id` (motion fallback)

### Geometric Evidence
- Bounding boxes
- Object size (area)
- Position over time

### Semantic Evidence
- Object class (`person`, `book`, etc.)
- Detection confidence

### Motion Evidence
- Moving regions even without classification
- Useful for detecting interactions or unknown objects

---

## Time Alignment Between Florence and Tracker

Both outputs share:
- `frame_index`
- `video_time_ms`

This allows Layer 2 to:
- Match semantic descriptions to tracked objects
- Detect changes over time for the **same entity**

Example:
> Frame 0: “person wearing a black shirt”  
> Frame 60: “person holding a gun”  
→ same `track_id` ⇒ same individual

---

## Notes on Video Loops (Debug Only)

During debugging, videos may run in loops.
In the final system:
- Videos are processed once
- Or trackers are reset between segments

Chronological order can still be inferred using:
- `generated_at_unix_ms`
- `video_time_ms`

---

## Current Status

✔ Layer 1 is stable  
✔ JSON schemas are consistent  
✔ Evidence is meaningful and reusable  
✔ Ready for Layer 2 (Inference & Reasoning)

---

## Next Possible Steps
- Define Layer 2 reasoning logic
- Add appearance embeddings (ReID)
- Add interaction / event extraction
- Formalize schemas using Pydantic or dataclasses
