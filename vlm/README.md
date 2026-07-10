# vlm/ — Vision-Language Model Worker

This folder contains the VLM scene-analysis layer.
It calls the **Google Gemini API** (default `gemini-3.5-flash`) to produce structured
scene analysis for each video frame, which the Groq anomaly worker uses as its
primary decision anchor. Requires a `GEMINI_API_KEY` (see `.env.example`).

## Files

| File | Purpose |
|---|---|
| `vlm_worker.py` | Main worker: ZMQ subscriber, frame throttling, result dispatch to Groq + NestJS |
| `vlm_model.py` | Gemini client, inference call, JSON parsing, output sanitization, summary building |
| `prompt.txt` | Instruction sent to Gemini — edit here to change what the model analyses |
| `logs_output.jsonl` | Auto-generated log of every VLM record sent to NestJS (one JSON object per line) |

## How it works

1. **Subscribes** to the broadcaster PUB socket (`tcp://127.0.0.1:5560`) for `frame` and `reset` topics.
2. **Throttles** — processes one frame every `--every` frames (default 60) to bound API request rate.
3. **Runs inference** via a single structured-JSON Gemini vision call (prompt from `prompt.txt`), sending the raw JPEG bytes directly.
4. **Parses** the JSON response into a structured QA dict + a one-line `summary` string.
5. **PUSHes** a `vlm_frame` record to the Groq anomaly worker (`tcp://127.0.0.1:5581`).
6. **Optionally** forwards the same record to NestJS via WebSocket (disabled by default with `--ws-url none`).

## Output record (`vlm_frame`)

```json
{
  "type": "vlm_frame",
  "frame_index": 120,
  "video_time_ms": 4000,
  "qa": {
    "description": "A person stands at the store counter.",
    "people_actions": "Person leans over counter towards cashier.",
    "appearance": "Dark hoodie, face partially covered.",
    "weapon": "none",
    "gun": "no",
    "knife": "no",
    "reaching_counter": "yes",
    "hands_up": "no",
    "face_concealed": "yes",
    "aggression": "no"
  },
  "summary": "Scene: A person stands at the store counter. People: ...",
  "meta": { "generated_at_unix_ms": 1730000000000, "model": "gemini-3.5-flash" }
}
```

## Tuning

### Adding or changing output fields — edit `prompt.txt` only, zero code changes

`vlm_model.py` parses the JSON schema block inside `prompt.txt` at startup and derives everything from it automatically:

| What's detected | How |
|---|---|
| **Binary fields** (yes/no) | Description starts with `"yes or no"` |
| **Optional-text fields** | Description contains `"or 'none'"` |
| **Raw-fallback field** | First remaining field (gets raw model output on JSON parse failure) |
| **All other fields** | Default to `"-"` on fallback |

So to add a new field (e.g. `"loitering"`), just add it to the JSON block in `prompt.txt`:
```json
"loitering": "yes or no — is anyone standing idle for an unusually long time"
```
The fallback dict, binary-key detection, sanitizer, and summary builder all update automatically.

- **Frame rate** → `--every` CLI flag (lower = more API requests + fresher context for Groq, but higher chance of hitting free-tier rate limits).
- **Model** → `--vlm_model` flag:

| Model | Notes |
|---|---|
| `gemini-3.5-flash` *(default)* | fast, free tier |
| `gemini-2.5-flash` | higher quality, slower, lower free-tier RPM |
| `gemini-2.5-flash-lite` | fastest/cheapest, lower quality |

> `gemini-2.0-flash` was shut down by Google on 2026-06-01 — do not use it.

## Launch command

```bash
python vlm/vlm_worker.py \
  --every 60 \
  --ws-url none \
  --anomaly-endpoint tcp://127.0.0.1:5581
```
