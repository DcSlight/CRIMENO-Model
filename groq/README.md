# groq/ — Anomaly Reasoning Worker

This folder contains the Groq-based anomaly detection layer.
It receives VLM + tracker evidence over ZMQ and uses **Llama 3.3 70B** (via Groq API)
to classify each scene window as `normal`, `suspicious`, or `criminal`.

## Files

| File | Purpose |
|---|---|
| `groq_anomaly_worker.py` | Main worker: ZMQ receiver, change detection, Groq API call, WebSocket output |
| `event_builder.py` | Rules-based text processing — builds event sentences from VLM + tracker records |
| `prompt.txt` | System prompt template sent to Llama 3.3 70B — edit here to tune model behaviour |
| `groq_context_log.txt` | Auto-generated log of every prompt sent to Groq (for debugging) |

## How it works

1. **Receives** `vlm_frame` records (decision anchor) and `tracker_frame` records (enrichment) on ZMQ PULL `tcp://127.0.0.1:5581`.
2. **Throttles** Groq calls to at most once per `--decision-frames` video frames (default 60) — cost is fully decoupled from VLM speed.
3. **Hard-signal bypass** — `gun`, `knife`, `hands_up`, `aggression` cues always trigger a Groq call, overriding the throttle.
4. **Builds a prompt** by combining recent event history + the current VLM frame into `prompt.txt` and sends it to Groq.
5. **Clamps scores** to enforce label consistency (`normal ≤ 0.2`, `suspicious 0.3–0.7`, `criminal ≥ 0.8`).
6. **Sends** the anomaly verdict to NestJS via WebSocket.

## Tuning

- **Model behaviour / scoring rules** → edit `prompt.txt` directly. No code change needed.
- **Decision frequency** → `--decision-frames` CLI flag (lower = more calls = higher cost).
- **Model** → `--groq-model` CLI flag (default: `llama-3.3-70b-versatile`).

## Launch command

```bash
python groq/groq_anomaly_worker.py \
  --ws_url ws://127.0.0.1:3000/ws/groq \
  --decision-frames 60 \
  --tracker-every 5
```

Requires `GROQ_API_KEY` in `.env` (see `.env.example` in the project root).
