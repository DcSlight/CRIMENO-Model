# CRIMENO — Criminal Activity Detection ML Pipeline

> **Component deep-dives:** [Groq anomaly worker](groq/README.md) · [VLM worker](vlm/README.md) · [Tracker](tracker/README.md)

---

## How to Run

### Prerequisites (one-time setup)

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root (use `.env.example` as a template):

```
GROQ_API_KEY=gsk_YOUR_KEY_HERE
```

> `GROQ_API_KEY` is shared by the anomaly worker (Step 2) and the VLM worker (Step 4).

### Launch order

Run each command in a **separate terminal**, from the project root:

**Step 1 — Video Broadcaster**
```bash
python video_broadcaster.py videos/shop.mp4
```

**Step 2 — Groq Anomaly Worker** (API-based reasoning, requires `GROQ_API_KEY`)
```bash
python groq/groq_anomaly_worker.py \
  --ws_url ws://127.0.0.1:3000/ws/groq \
  --decision-frames 60 \
  --tracker-every 5
```

`--decision-frames` — minimum video frames between two Groq API calls (cost knob, default 60).
Groq fires **at most** once per this many frames regardless of how fast VLM runs.
Only lower this to increase decision frequency (= higher spend).

**Step 3 — YOLO Tracker**
```bash
python tracker/tracker_worker.py \
  --device cuda \
  --ws_url ws://127.0.0.1:3000/ws/tracker \
  --send_overlay 0 \
  --send_every_n_frames 5 \
  --anomaly-endpoint tcp://127.0.0.1:5581
```

**Step 4 — VLM Scene Analyser** (Groq vision, uses `GROQ_API_KEY` — same key as Step 2)
```bash
python vlm/vlm_worker.py \
  --every 60 \
  --ws-url none \
  --anomaly-endpoint tcp://127.0.0.1:5581
```

> **VLM is the decision anchor.** Each VLM frame triggers a potential Groq call
> (subject to `--decision-frames` throttle). Cost is fully decoupled from VLM speed:
> running VLM faster gives fresher context but never increases API spend.
> Note: the VLM's own Groq vision call and the anomaly worker's Groq text call
> share the same `GROQ_API_KEY` quota.

**Model options:**

| Flag | Model | Notes |
|---|---|---|
| *(default)* | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq's primary vision model |
| `--vlm_model qwen/qwen3.6-27b` | newer 27B multimodal alternate |

> **`--ws-url none`** — use this unless the NestJS backend exposes a `/ws/vlm` route
> (it does **not** by default). A missing route causes a hang on retry.

---

## Eval harness

```bash
py eval/score_logs.py                   # compare real logs against the mock ground truth
py eval/score_logs.py --json out.json   # also write the summary metrics to JSON
```

Pure stdlib, no Groq/API calls. Compares a real pipeline output log (default:
`groq/logs_output.jsonl`) against a hand-authored mock (default: the jewelry-store mock in
`CRIMENO-Backend/mocks/`), aligning entries by frame range. Reports a normal/suspicious/criminal
confusion matrix, accuracy, under-calls (real less severe than mock — missed events, the
dangerous direction) vs. over-calls (false alarms), score error, and a text-similarity score
between the mock's and the real `reason` narrative.

---

## Architecture

### Pipeline overview

```
Video File
   │
   ▼
video_broadcaster.py
(ZMQ PUB  tcp://127.0.0.1:5560)
   │
   ├──▶ tracker/tracker_worker.py    (every N frames)
   │        YOLO26 + YOLOE-26 + Suspicious nano model
   │        → ZMQ PUSH tcp://127.0.0.1:5581
   │
   └──▶ vlm/vlm_worker.py           (every 60 frames, default)
            Llama 4 Scout via Groq (API) — full-frame scene analysis
            → ZMQ PUSH tcp://127.0.0.1:5581
                     │
                     ▼
           groq/groq_anomaly_worker.py
           Llama 3.3 70B via Groq API
           ZMQ PULL on tcp://127.0.0.1:5581
                     │
                     ▼
           WebSocket → NestJS (ws://localhost:3000/ws/groq)
                     │
                     ▼
           React Dashboard (GroqWidget)
```

VLM is the **decision anchor**: each VLM frame triggers a potential Groq call.
The nearest tracker frame is automatically attached as enrichment context.

### ZMQ socket map

| Port | Type | Direction | Purpose |
|---|---|---|---|
| `5560` | PUB / SUB | Broadcaster → Workers | Video frames (`frame` topic) + reset signal (`reset` topic) |
| `5561` | REQ / REP | NestJS → Broadcaster | Play/select commands; broadcaster replies `{ok:true}` |
| `5562` | PUSH / PULL | Workers → Broadcaster | `reset_ack` + `first_frame_ack` handshake |
| `5581` | PUSH / PULL | Tracker + VLM → Groq worker | Frame data + reset forwarding |

---

### Components

#### `video_broadcaster.py`
- Reads a video file; streams frames via ZMQ PUB.
- Maintains `frame_index` and `video_time_ms` in each message.
- Orchestrates the video-switch reset handshake.

**Frame message format (ZMQ multipart):**
```
[ topic="frame", frame_index (str), video_time_ms (str), jpg_bytes ]
```

---

#### `tracker/tracker_worker.py`
- Subscribes to the video stream.
- Runs three detection models per frame:
  1. **YOLO26** — general multi-class detection (COCO), with IOU tracker for `track_id`.
  2. **YOLOE-26-seg** (open-vocab) — appearance tags (hood, mask, dark clothing) + weapon detection (gun, knife) via text prompt.
  3. **Suspicious_Activities_nano** — custom model for fighting / robbery cues.
- Motion fallback (MOG2 background subtraction) captures moving regions when classification fails.
- Sends `tracker_frame` payloads to NestJS (bounding boxes for React UI) and enrichment records to the Groq worker.

**Model weights** (in `tracker/`):

| File | Purpose |
|---|---|
| `yolo26s.pt` | General object detection |
| `yoloe-26s-seg.pt` | Open-vocab appearance + weapon detection |
| `yolov8s.pt` | Fallback / legacy weights |
| `Suspicious_Activities_nano.pt` | Custom fighting/robbery classifier |

**Tracker JSON payload:**
```json
{
  "frame_index": 65,
  "video_time_ms": 2170,
  "meta": {
    "generated_at_unix_ms": 1730000000123,
    "worker": "tracker_worker",
    "device": "cuda",
    "yolo_model": "yolo26s.pt",
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

#### `vlm/vlm_worker.py`
- Processes one full frame every N frames (default: 60).
- Calls **Groq vision** (default `meta-llama/llama-4-scout-17b-16e-instruct`) via API — uses the same `GROQ_API_KEY` as the anomaly worker.
- Returns a single structured JSON per frame:
  - Scene description, people actions, appearance, weapon description.
  - Binary cues (yes/no/unclear): `gun`, `knife`, `reaching_display_case`, `reaching_behind_counter`, `hands_up`, `face_concealed`, `aggression`.
- Pushes each result to the Groq anomaly worker via ZMQ.

---

#### `groq/groq_anomaly_worker.py`
- ZMQ PULL on port `5581` — receives VLM frames (primary) and tracker enrichment.
- Buffers every VLM frame it sees; fires a Groq API call at most once per `--decision-frames`
  frames, but narrates over the last `--window-frames` (default 3) buffered observations —
  a real temporal span, not a single instant, so `frame_range` in the output is a span too.
- **Hard-signal bypass:** `gun`, `knife`, `hands_up`, `aggression` cues always trigger a Groq call, regardless of the throttle.
- Optionally prepends **business context** (store name, sensitivity, forbidden behaviours) to every prompt when NestJS sends a `business_context` ZMQ message.
- **Groq narrates, code scores:** Groq's API call returns only `reason`/`key_moments`/an advisory
  `concern` tag — the `anomaly_score`/`label` are computed deterministically from the VLM cue
  history by `groq/scoring.py` (tiered evidence weights + a persistence/corroboration gate that
  requires a confirmed weapon or a sustained, corroborated forbidden action before ever labeling
  "criminal" — never a single loosely-matched cue on one frame).
- Sends final anomaly verdict to NestJS via WebSocket.
- Appends full context (prompt, Groq's raw narrative, and the final code-scored result) to
  `groq/groq_context_log.txt` for debugging.

**Anomaly output payload:**
```json
{
  "type": "groq_anomaly",
  "frame_range": { "start": 0, "end": 3 },
  "result": {
    "anomaly_score": 0.85,
    "label": "criminal",
    "reason": "Person concealing item under jacket near exit",
    "key_moments": ["item hidden in jacket", "rapid movement to door"]
  }
}
```

---

### Reset & sync flow

When a new video is selected every component holds stale state from the previous video. A coordinated reset handshake prevents old bounding boxes and detections bleeding into the new stream.

#### Components involved

| Component | File | Role |
|---|---|---|
| **Broadcaster** | `video_broadcaster.py` | Sends `reset` signal, waits for acks, controls when NestJS gets `{ok:true}` |
| **Tracker** | `tracker/tracker_worker.py` | Clears tracks, resets motion detector, drains SUB buffer, sends React a clear payload, sends both acks |
| **VLM** | `vlm/vlm_worker.py` | Drains SUB buffer, clears internal state, forwards reset to Groq |
| **Groq** | `groq/groq_anomaly_worker.py` | Clears event history and buffers — no ack (fire-and-forget) |
| **React UI** | Dashboard | Shows loading spinner until `{ok:true}` returns; video plays only then |

#### The two acks

**`reset_ack`** — sent by Tracker (and VLM in place of Florence) after draining their SUB buffer and clearing all local state. Meaning: *"I'm clean, you can start streaming."* Broadcaster waits up to `WORKER_ACK_TIMEOUT_S = 20s`.

**`first_frame_ack`** — sent by Tracker only, immediately after `ws_send_json` fires for the first real `tracker_frame` post-reset (frame 0 bounding boxes are already on their way to React). Meaning: *"Frame 0 is in transit, React won't show a black box."* Broadcaster waits up to `FIRST_FRAME_ACK_TIMEOUT_S = 5s`.

#### Cold-start fix

On the very first play, workers' SUB sockets have never received a frame so `recv_multipart` would block. Both workers set `RCVTIMEO = 100ms` and check `_reset_event` **before** calling `recv_multipart` — if recv times out with nothing buffered the loop immediately cycles back, applies the reset (drain is a no-op), and sets `_reset_done_event`. The watcher unblocks in ≤100ms instead of waiting the 15s self-timeout.

| Condition | Latency |
|---|---|
| First play (nothing buffered) | ~100ms |
| Subsequent plays (frames buffered) | <100ms |

#### FPS pacing

`stream_start_time` is set **after** the full handshake completes (after `{ok:true}` is sent to NestJS). This means frame 1's `expected_time = stream_start_time + 1/fps` ≈ 33ms after React receives the response — frames are delivered at natural video FPS from the very first frame, no catch-up burst.
