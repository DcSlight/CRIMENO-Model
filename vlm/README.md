# vlm/ — Vision-Language Model Worker

This folder contains the VLM scene-analysis layer.
It calls a **local vision model via [Ollama](https://ollama.com/download)** (default
`minicpm-v4.5`) to produce structured scene analysis for each video frame, which the
Groq anomaly worker uses as its primary decision anchor. Runs entirely on your GPU —
no API key, no quota, no per-request cost.

**One-time setup:** install Ollama, then `ollama pull minicpm-v4.5` (~6.1 GB). The Ollama
server runs in the background on `http://localhost:11434` and must be running before you
start `vlm_worker.py`.

## Files

| File | Purpose |
|---|---|
| `vlm_worker.py` | Main worker: ZMQ subscriber, frame throttling, result dispatch to Groq + NestJS |
| `vlm_model.py` | Ollama client, inference call (with retry/backoff), JSON parsing, output sanitization, summary building |
| `prompt.txt` | Instruction sent to the vision model — edit here to change what the model analyses |
| `logs_output.jsonl` | Auto-generated log of every VLM record sent to NestJS (one JSON object per line) |

## How it works

1. **Subscribes** to the broadcaster PUB socket (`tcp://127.0.0.1:5560`) for `frame` and `reset` topics.
2. **Throttles** — processes one frame every `--every` frames (default 60) to bound inference rate.
3. **Runs inference** via a single structured-JSON Ollama chat call (prompt from `prompt.txt`), sending the JPEG bytes directly and passing the schema (derived from `prompt.txt`) as a `format` constraint so the model is forced to emit valid JSON with every key; retries up to 3 times with backoff on transient errors or empty completions before falling back.
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
  "meta": { "generated_at_unix_ms": 1730000000000, "model": "minicpm-v4.5" }
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

- **Frame rate** → `--every` CLI flag (lower = more inference calls + fresher context for Groq, but higher load on your GPU — throughput is bounded by local inference speed now, not an external quota).
- **Server** → `--ollama-host` flag or `OLLAMA_HOST` env var (default `http://localhost:11434`).
- **Model** → `--vlm_model` flag (must be pulled first via `ollama pull <tag>`):

| Model | Notes |
|---|---|
| `minicpm-v4.5` *(default)* | 8B, "GPT-4o-level" vision, ~6.1 GB — good quality/speed balance |
| `minicpm-v4.6` | lighter/faster decoder — try this if per-frame latency is too high |

> Vision analysis moved off every hosted free tier after each failed in practice: Groq
> deprecated `meta-llama/llama-4-scout-17b-16e-instruct` on the free/dev tier (Jun 2026),
> following Llama 4 Maverick (Feb 2026, now text-only); Groq's remaining free vision model,
> `qwen/qwen3.6-27b`, is a preview reasoning model that burned its whole token budget on
> internal `<think>` tokens before writing any JSON; and Gemini's free tier caps out at
> **5 requests/minute** per key, far below what `--every 60` demands. Running locally via
> Ollama removes the quota problem entirely — throughput is now bounded only by your GPU.

## Launch command

```bash
python vlm/vlm_worker.py \
  --every 60 \
  --ws-url none \
  --anomaly-endpoint tcp://127.0.0.1:5581
```
