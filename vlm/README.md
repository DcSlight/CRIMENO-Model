# vlm/ — Vision-Language Model Worker

This folder contains the VLM scene-analysis layer.
It calls a **local vision model via [Ollama](https://ollama.com/download)** (default
`gemma3:4b`) to produce structured scene analysis for each video frame, which the
Groq anomaly worker uses as its primary decision anchor. Runs entirely on your GPU —
no API key, no quota, no per-request cost.

**One-time setup:** install Ollama, then `ollama pull gemma3:4b` (~3 GB). The
Ollama server runs in the background on `http://localhost:11434` and must be running before
you start `vlm_worker.py`.

## Files

| File | Purpose |
|---|---|
| `vlm_worker.py` | Main worker: ZMQ subscriber, model warm-up, frame throttling, result dispatch to Groq + NestJS |
| `vlm_model.py` | Ollama client, inference call (with retry/backoff), JSON parsing, output sanitization, summary building |
| `prompt.txt` | Instruction sent to the vision model — edit here to change what the model analyses |
| `logs_output.jsonl` | Auto-generated log of every VLM record sent to NestJS (one JSON object per line) |

## How it works

0. **Warms up** the model with one dummy frame right after startup — loads it into VRAM and
   surfaces any load/compatibility crash immediately, before the rest of the pipeline
   (broadcaster/tracker/anomaly worker) is running. Mirrors `tracker/tracker_worker.py`'s
   YOLO warm-up. Watch for the `[VLM] Warm-up result: ...` line: a real scene description
   means the model is working; "Scene analysis unavailable" means something's wrong (wrong
   model tag, Ollama not running, or the model doesn't support this Ollama build — see below).
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
  "meta": { "generated_at_unix_ms": 1730000000000, "model": "gemma3:4b" }
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
| `gemma3:4b` *(default)* | ~3 GB — natively supported by Ollama's current engine; light enough to run alongside `tracker/tracker_worker.py`'s YOLO models on the same GPU without contention |
| `gemma3:12b` | stronger at structured/OCR-style output, but too heavy to run concurrently with the tracker on this GPU — use for VLM-only runs or if you free up GPU headroom |
| `qwen2.5vl:7b` | tried as the default first — loaded fine but was both slow and weak in practice on this project's Pascal/Vulkan GPU |
| `llava` | last resort — loads reliably but weakest at structured JSON of the group |

> **Do not confuse `qwen2.5vl` with `qwen2-vl`** (no `.5`) — the latter is an older model with a
> known broken vision-projector (`mmproj`) bug in Ollama.

> Vision analysis moved off every hosted free tier after each failed in practice: Groq
> deprecated `meta-llama/llama-4-scout-17b-16e-instruct` on the free/dev tier (Jun 2026),
> following Llama 4 Maverick (Feb 2026, now text-only); Groq's remaining free vision model,
> `qwen/qwen3.6-27b`, is a preview reasoning model that burned its whole token budget on
> internal `<think>` tokens before writing any JSON; and Gemini's free tier caps out at
> **5 requests/minute** per key, far below what `--every 60` demands. Running locally via
> Ollama removes the quota problem entirely — throughput is now bounded only by your GPU.
>
> Several local models were tried before landing on `gemma3:4b`:
> - **`minicpm-v4.5`/`minicpm-v4.6`** crash official Ollama's `llama-server` backend on
>   load/inference (`exit status 0xc0000005`, an access violation). That architecture was never
>   mainlined into Ollama at all; running it requires an unofficial fork
>   ([tc-mb/ollama](https://github.com/tc-mb/ollama)) that isn't merged upstream.
> - **`llama3.2-vision:11b`** fails to load (`unknown model architecture: 'mllama'`). Ollama's
>   **new inference engine dropped `mllama` support** in its rewrite — it never existed in
>   mainline llama.cpp, only ran on Ollama's own private patches, and there's no fix/ETA
>   ([ollama/ollama#16490](https://github.com/ollama/ollama/issues/16490), open).
> - **`qwen2.5vl:7b`** loaded and ran, but proved both slow and weak in practice on this
>   project's GPU (Pascal/Vulkan — see below).
> - **`gemma3:12b`** gave the strongest structured JSON output of the group, but is too heavy
>   to run at the same time as `tracker/tracker_worker.py`'s YOLO models — both processes
>   compete for the same GPU, and running VLM + tracker together was the actual requirement.
>
> The pattern: Ollama's current engine only **natively** supports a specific architecture set —
> **Llama 4, Gemma 3, Qwen 2.5 VL, Mistral Small 3.1**. `gemma3:4b` is in that set, small enough
> to share the GPU with the tracker, and still uses the same `format=` structured-JSON
> mechanism this worker relies on (see `_RESPONSE_SCHEMA` in `vlm_model.py`). If quality is
> insufficient at 4b, `gemma3:12b` is the fallback for VLM-only runs.

### Before trying a different vision model

Avoid repeating the cycle above — check *before* pulling a multi-GB model:

1. Prefer one of the current engine's supported families: **Llama 4, Gemma 3, Qwen 2.5 VL,
   Mistral Small 3.1**. Pull from the [official Ollama vision library](https://ollama.com/search?c=vision)
   only — community-imported/fused GGUFs have their own `mmproj` wiring bugs.
2. If unsure, check the model's Hugging Face `config.json` → `architectures` field first.
   `mllama` and `MiniCPMV` are red flags; `Gemma3ForConditionalGeneration`,
   `Qwen2_5_VLForConditionalGeneration`, `Mistral3ForConditionalGeneration` are green.
3. On this project's GPU (P40/Pascal): stick to the default GGUF quant Ollama pulls (`Q4_K_M`) —
   never a `-fp16`/`bf16` variant (Pascal has no BF16 and crippled FP16 throughput). Don't set
   `OLLAMA_FLASH_ATTENTION=1` (Pascal can't use it; the default auto-fallback is correct). Watch
   the `[VLM] Warm-up result: ...` startup log — the whole model should fit in 24 GB VRAM, so any
   "offloaded N/M layers to CPU" line signals a problem.

## Launch command

```bash
python vlm/vlm_worker.py \
  --every 60 \
  --ws-url none \
  --anomaly-endpoint tcp://127.0.0.1:5581
```
