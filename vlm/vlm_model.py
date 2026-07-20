import io
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image
import ollama
from google import genai
from google.genai import types

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

_HERE = Path(__file__).resolve().parent
_REFUSAL_PREFIXES = ("sorry", "unanswerable", "i cannot", "i can't", "i am not")


# ============================================================
# Schema parsing — drives ALL field-level behaviour
# ============================================================

def _parse_schema(prompt_text: str) -> Tuple[List[str], Dict[str, str], Set[str]]:
    """Extract the JSON schema block from prompt.txt and derive:
      - fields:      ordered list of field names
      - defaults:    field → fallback value ('no', 'none', '-', or '' for raw-text field)
      - binary_keys: set of fields whose description starts with 'yes or no'

    Detection rules (applied to each field's description string):
      'yes or no' prefix  → binary, default 'no'
      "or 'none'"         → optional text, default 'none'
      first remaining key → raw-text field, default '' (sentinel for raw model output)
      other remaining     → default '-'
    """
    match = re.search(r"\{([^{}]+)\}", prompt_text, re.DOTALL)
    if not match:
        raise ValueError("prompt.txt must contain a JSON schema block { ... } with the output fields.")

    schema: Dict[str, str] = json.loads("{" + match.group(1) + "}")

    fields: List[str] = list(schema.keys())
    defaults: Dict[str, str] = {}
    binary_keys: Set[str] = set()
    raw_text_field_seen = False

    for key, desc in schema.items():
        desc_l = str(desc).lower()
        if desc_l.startswith("yes or no"):
            defaults[key] = "no"
            binary_keys.add(key)
        elif "or 'none'" in desc_l:
            defaults[key] = "none"
        elif not raw_text_field_seen:
            defaults[key] = ""          # sentinel: replace with raw model output on fallback
            raw_text_field_seen = True
        else:
            defaults[key] = "-"

    return fields, defaults, binary_keys


_PROMPT_TEXT    = (_HERE / "prompt.txt").read_text(encoding="utf-8").strip()
_SCHEMA_FIELDS, _SCHEMA_DEFAULTS, _BINARY_KEYS = _parse_schema(_PROMPT_TEXT)

print(f"[VLM] Schema loaded — {len(_SCHEMA_FIELDS)} fields, "
      f"binary: {sorted(_BINARY_KEYS)}")

# JSON schema built straight from prompt.txt's field list — passed to Ollama's `format`
# for constrained decoding, so the model is forced to emit valid JSON with every key
# instead of relying on it to follow the prompt's instructions unprompted.
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {key: {"type": "string"} for key in _SCHEMA_FIELDS},
    "required": list(_SCHEMA_FIELDS),
}


# ============================================================
# Utilities
# ============================================================

def now_unix_ms() -> int:
    return int(time.time() * 1000)


def pil_from_jpg(jpg_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpg_bytes)).convert("RGB")


def _fallback_dict(raw_text: str) -> Dict[str, str]:
    """Build a safe fallback response dict from the schema defaults.
    The sentinel ('') field receives the raw model output (truncated), or a
    fixed message if the output looks like a refusal."""
    result = {}
    for key, default in _SCHEMA_DEFAULTS.items():
        if default == "":
            result[key] = (
                raw_text[:300]
                if raw_text and not raw_text.lower().startswith("sorry")
                else "Scene analysis unavailable."
            )
        else:
            result[key] = default
    return result


# ============================================================
# Model loading
# ============================================================

_DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Bounded retry for transient failures (server mid-load, momentary connection drop) and
# for empty completions — previously a single hiccup dropped straight to the
# "Scene analysis unavailable" fallback for that frame. Both backends can hiccup: local
# Ollama doesn't have quota/rate-limit failures but can still stall mid-load; the online
# Gemini backend is a network call and can transiently fail, or return empty on a
# safety-filter false-positive.
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_S = (0.5, 1.0, 2.0)

# Surveillance frames legitimately contain weapons/aggression cues — Gemini's default
# safety filters can block or empty out the response on exactly the frames that matter,
# so the harm categories relevant to this analysis are relaxed to BLOCK_NONE. Online
# backend only; Ollama has no equivalent filtering layer.
_SAFETY_SETTINGS = [
    types.SafetySetting(category=cat, threshold=types.HarmBlockThreshold.BLOCK_NONE)
    for cat in (
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    )
]


def load_vlm(backend: str, model_id: str, host_or_key: str = ""):
    """backend='local'  → Ollama client. host_or_key is the Ollama server URL (optional).
    backend='online' → Gemini client. host_or_key is the API key (or GEMINI_API_KEY env var)."""
    if backend == "online":
        api_key = host_or_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Pass --gemini-api-key or set it in .env.")
        client = genai.Client(api_key=api_key)
        print(f"✅ [VLM] Gemini client ready — model={model_id}")
        return client

    host = host_or_key or os.environ.get("OLLAMA_HOST", "") or _DEFAULT_OLLAMA_HOST
    client = ollama.Client(host=host)

    try:
        available = {m.model for m in client.list().models}
        # Ollama tags are commonly reported with an implicit ":latest" suffix.
        if model_id not in available and f"{model_id}:latest" not in available:
            print(f"[VLM] ⚠️ Model '{model_id}' not found on {host}. "
                  f"Run: ollama pull {model_id}")
    except Exception as e:
        print(f"[VLM] ⚠️ Could not reach Ollama at {host} to verify model: {e}")

    print(f"✅ [VLM] Ollama client ready — model={model_id} @ {host}")
    return client


# ============================================================
# Inference
# ============================================================

def _call_local(client: "ollama.Client", model_id: str, jpg_bytes: bytes, max_new_tokens: int) -> str:
    resp = client.chat(
        model=model_id,
        messages=[{
            "role": "user",
            "content": _PROMPT_TEXT,
            "images": [jpg_bytes],
        }],
        format=_RESPONSE_SCHEMA,
        options={"temperature": 0.0, "num_predict": max_new_tokens},
    )
    return (resp.message.content or "").strip()


def _call_online(client: "genai.Client", model_id: str, jpg_bytes: bytes, max_new_tokens: int) -> str:
    resp = client.models.generate_content(
        model=model_id,
        contents=[
            types.Part.from_bytes(data=jpg_bytes, mime_type="image/jpeg"),
            _PROMPT_TEXT,
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=max_new_tokens,
            safety_settings=_SAFETY_SETTINGS,
        ),
    )
    return (resp.text or "").strip()


def analyze_frame(backend: str, client, model_id: str, jpg_bytes: bytes,
                  max_new_tokens: int = 256) -> Dict[str, str]:
    """Single vision call — local Ollama or online Gemini, per `backend` — → structured dict.
    Retries transient failures / empty completions a few times before falling back to the
    schema default."""
    call = _call_online if backend == "online" else _call_local
    label = "Gemini" if backend == "online" else "Ollama"

    raw_text = ""
    last_err: Optional[Exception] = None

    for attempt in range(_MAX_ATTEMPTS):
        try:
            raw_text = call(client, model_id, jpg_bytes, max_new_tokens)
            if raw_text:
                break
            last_err = None
            print(f"[VLM] ⚠️ {label} returned an empty completion (attempt {attempt + 1}/{_MAX_ATTEMPTS})")
        except Exception as e:
            last_err = e
            print(f"[VLM] ⚠️ {label} call failed (attempt {attempt + 1}/{_MAX_ATTEMPTS}): {e}")

        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(_RETRY_BACKOFF_S[attempt])

    if not raw_text:
        if last_err is not None:
            print(f"[VLM] ⚠️ {label} call failed after {_MAX_ATTEMPTS} attempts: {last_err}")
        else:
            print(f"[VLM] ⚠️ {label} returned no content after {_MAX_ATTEMPTS} attempts")
        return _fallback_dict("")

    return _parse_vlm_output(raw_text)


def _parse_vlm_output(text: str) -> Dict[str, str]:
    """Strip markdown fences and parse JSON. Returns schema-driven fallback on failure."""
    cleaned = re.sub(r"```json", "", text, flags=re.IGNORECASE).replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return _sanitize(parsed)
    except Exception:
        pass

    match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return _sanitize(parsed)
        except Exception:
            pass

    print(f"[VLM] ⚠️ JSON parse failed. Raw output: {text[:200]!r}")
    return _fallback_dict(text)


def _sanitize(d: Dict) -> Dict[str, str]:
    """Ensure all values are strings; reset refusal phrases in binary fields to 'no'."""
    result = {}
    for k, v in d.items():
        s = str(v).strip()
        if k in _BINARY_KEYS and any(s.lower().startswith(p) for p in _REFUSAL_PREFIXES):
            s = "no"
        result[k] = s
    return result


def build_summary(qa: Dict[str, str]) -> str:
    """Flatten the QA dict into a single-line summary consumed by the Groq worker.
    Field order and labels are derived from the schema in prompt.txt."""
    parts = []
    for key in _SCHEMA_FIELDS:
        default = _SCHEMA_DEFAULTS.get(key, "-") or "-"
        label   = key.replace("_", " ").title()
        value   = qa.get(key, default)
        parts.append(f"{label}: {value}")
    return ". ".join(parts) + "."
