import io
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

from PIL import Image

try:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import transformers
    transformers.logging.set_verbosity_error()
except Exception:
    torch = None
    Qwen2_5_VLForConditionalGeneration = None
    AutoProcessor = None

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

def load_vlm(model_id: str, device_str: str):
    if Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError(
            "Qwen2.5-VL requires transformers>=4.49 and accelerate. "
            'Run: pip install "transformers>=4.49" accelerate'
        )

    use_cuda = device_str == "cuda" and torch.cuda.is_available()
    dtype  = torch.bfloat16 if use_cuda else torch.float32
    device = "cuda:0" if use_cuda else "cpu"

    print(f"[VLM] Loading {model_id} on {device} ({dtype})...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"✅ [VLM] {model_id} ready on {device}")
    return model, processor, device, dtype


# ============================================================
# Inference
# ============================================================

def analyze_frame(model, processor, device, dtype, image: Image.Image,
                  max_new_tokens: int = 256) -> Dict[str, str]:
    """Single Qwen2.5-VL call → structured dict. Safe fallback on parse failure."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": _PROMPT_TEXT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    raw_text = processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
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
