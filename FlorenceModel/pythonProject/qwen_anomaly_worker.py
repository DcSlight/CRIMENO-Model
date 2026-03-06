import json
import re
import zmq
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda"  # or "cpu"
ZMQ_ENDPOINT = "tcp://127.0.0.1:5580"

BASE_WINDOW_SIZE = 3
MAX_QUEUE_SIZE = 30
MAX_EVENT_HISTORY = 10

CONTEXT_LOG_FILE = "qwen_context_log.txt"


# ============================================================
# Load Qwen model
# ============================================================

def load_qwen_pipeline(model_name: str, device_str: str):
    if device_str == "cuda" and torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.float16
        print("Qwen device: cuda")
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
        print("Qwen device: cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
    )

    print(f"✅ Qwen pipeline loaded ({model_name}) on {device_str}")
    return text_gen


# ============================================================
# Rule-based caption summarization
# ============================================================

GENERIC_START_PATTERNS = [
    "the image shows",
    "this image shows",
    "the image is a still from",
    "the image appears to be",
    "it shows",
    "in the image",
]

GENERIC_NOISE_FRAGMENTS = [
    "the overall atmosphere",
    "the overall ambience",
    "the store appears to be",
    "the space appears to be",
    "overall atmosphere",
    "cluttered and disorganized",
    "well-stocked with various items",
    "various items for sale",
]


def is_generic_sentence(sent: str) -> bool:
    s = sent.strip().lower()
    if not s:
        return True
    for p in GENERIC_START_PATTERNS:
        if s.startswith(p):
            return True
    for frag in GENERIC_NOISE_FRAGMENTS:
        if frag in s:
            return True
    return False


def clean_caption(raw_caption: str, max_sentences: int = 3) -> str:
    if not raw_caption:
        return ""

    text = raw_caption.replace("<MORE_DETAILED_CAPTION>", "").strip()
    text = re.sub(r"\s+", " ", text)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return ""

    meaningful: List[str] = []
    for s in sentences:
        if not is_generic_sentence(s):
            meaningful.append(s)

    if not meaningful:
        meaningful = sentences

    meaningful = meaningful[:max_sentences]
    cleaned = " ".join(meaningful)

    if len(cleaned) > 400:
        cleaned = cleaned[:400].rstrip() + "..."

    return cleaned


def build_tracker_sentence(rec: Dict[str, Any]) -> str:
    """
    Builds a detailed textual summary from YOLO tracker data.
    Includes track_id, class, confidence, and bbox.
    This helps Qwen understand continuity of objects across frames.
    """
    tracker = rec.get("tracker")
    if not tracker:
        return ""

    tracks = tracker.get("tracks", [])
    if not tracks:
        return ""

    parts = []
    for t in tracks:
        tid = t.get("track_id")
        cls = t.get("cls", "object")
        conf = t.get("conf", 0.0)
        bbox = t.get("bbox", {})

        x1 = bbox.get("x1")
        y1 = bbox.get("y1")
        x2 = bbox.get("x2")
        y2 = bbox.get("y2")

        parts.append(
            f"ID {tid}: {cls} (confidence {conf:.2f}) at [{x1},{y1},{x2},{y2}]"
        )

    if not parts:
        return ""

    return "YOLO tracker detected: " + "; ".join(parts) + "."



def build_event_sentence(rec: Dict[str, Any]) -> str:
    """
    Builds a short event sentence from Florence caption + YOLO tracker,
    using only rule-based logic.
    """
    caption = rec.get("raw", {}).get("more_detailed_caption", "")
    cleaned = clean_caption(caption)

    tracker_text = build_tracker_sentence(rec)

    if cleaned and tracker_text:
        return f"{cleaned} {tracker_text}"
    elif cleaned:
        return cleaned
    elif tracker_text:
        return tracker_text

    return "No significant visual change."


# ============================================================
# Change detection
# ============================================================

def simple_similarity(a: str, b: str) -> float:
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union


def is_significant_change(new_event: str, last_event: Optional[str], threshold: float = 0.9) -> bool:
    if not last_event:
        return True
    sim = simple_similarity(new_event, last_event)
    return sim < threshold


# ============================================================
# Scene description builder
# ============================================================

def build_scene_description(event_history: List[str], current_window_events: List[str]) -> str:
    lines = []

    if event_history:
        lines.append("Recent context:")
        for ev in event_history[-MAX_EVENT_HISTORY:]:
            lines.append(f"- {ev}")

    if current_window_events:
        lines.append("\nCurrent window:")
        for ev in current_window_events:
            lines.append(f"- {ev}")

    return "\n".join(lines)


# ============================================================
# Prompt builder
# ============================================================

def build_prompt(scene_description: str) -> str:
    prompt = f"""
    You are an expert system for video surveillance anomaly detection.

    You receive short textual descriptions of what happens in a surveillance video over time.
    These descriptions come from two sources:
    1. Florence (semantic captions, OCR, object descriptions, behaviors, interactions)
    2. YOLO tracker (object IDs, classes, bounding boxes, continuity across frames)

    Your task is to determine whether the described situation represents:
    - normal everyday behavior,
    - suspicious behavior, or
    - criminal/dangerous behavior.

    ==================== CRITICAL INSTRUCTIONS ====================

    ### 1. JSON OUTPUT FORMAT (STRICT)
    You must output EXACTLY one JSON object with the following fields:
    - "anomaly_score": a float in [0, 1]
    - "label": one of ["normal", "suspicious", "criminal"]
    - "reason": a short explanation (max 30 words)
    - "key_moments": a list of short phrases describing meaningful events

    ### 2. SCORING CONSISTENCY RULES
    - If label == "normal"     → anomaly_score MUST be <= 0.2
    - If label == "suspicious" → anomaly_score MUST be between 0.3 and 0.7
    - If label == "criminal"   → anomaly_score MUST be >= 0.8
    - The score MUST always match the label category.

    ### 3. HOW TO USE THE INPUTS
    - Florence text is the PRIMARY source for understanding actions, behaviors, and context.
    - YOLO tracker data (IDs, classes, bounding boxes) is SECONDARY and should be used ONLY to:
      * understand continuity of people/objects across frames
      * detect repeated presence or movement patterns
      * identify that the same person appears in multiple frames

    ### 4. PROHIBITED USE OF YOLO DATA
    You MUST NOT use YOLO tracker data as evidence of anomaly.
    Specifically:
    - DO NOT use confidence scores as reasons.
    - DO NOT use bounding boxes as reasons.
    - DO NOT use “ID 1”, “ID 2”, etc. as key moments.
    - DO NOT treat “multiple people detected” as suspicious by itself.

    ### 5. KEY MOMENTS RULES
    "key_moments" MUST:
    - be based ONLY on semantic content from Florence (captions, OCR, behaviors)
    - describe meaningful actions, interactions, or unusual events
    - NOT include YOLO technical data (IDs, confidence, bbox)

    Examples of GOOD key moments:
    - "man holding phone instead of payment method"
    - "customer leaning over cash register"
    - "person reaching into backpack"
    - "individual looking around nervously"

    Examples of BAD key moments (FORBIDDEN):
    - "ID 1: person (confidence 0.91)"
    - "three people detected"
    - "bounding box moved left"

    ### 6. REASON FIELD RULES
    The "reason" MUST:
    - describe behavioral or contextual anomalies
    - be based on Florence semantic content
    - NOT mention YOLO IDs, confidence, or bounding boxes
    - be short and human‑interpretable

    ==============================================================

    Here is the scene description in chronological order:

    {scene_description}

    Respond with exactly one JSON object and nothing else.
    """.strip()

    return prompt

# ============================================================
# JSON extraction
# ============================================================

def parse_qwen_output(text: str) -> Dict[str, Any]:
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    json_candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for cand in json_candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue

    if "anomaly_score" in text:
        try:
            fixed = "{" + text.strip().strip(",") + "}"
            return json.loads(fixed)
        except Exception:
            pass

        lines = []
        for line in text.splitlines():
            if ":" in line:
                lines.append(line.strip().rstrip(","))
        try:
            fixed = "{ " + ", ".join(lines) + " }"
            return json.loads(fixed)
        except Exception:
            pass

    return {
        "anomaly_score": 0.0,
        "label": "unknown",
        "reason": "Failed to parse model JSON output",
        "raw_output": text[:500],
    }


def call_qwen_for_anomaly(text_gen, prompt: str) -> Dict[str, Any]:
    out = text_gen(prompt, max_new_tokens=256, do_sample=False, temperature=0.0)
    if isinstance(out, list) and out:
        generated = out[0].get("generated_text", "")
    else:
        generated = str(out)

    if generated.startswith(prompt):
        generated = generated[len(prompt):].strip()

    return parse_qwen_output(generated)


# ============================================================
# Main worker
# ============================================================

def main():
    text_gen = load_qwen_pipeline(MODEL_NAME, DEVICE)

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(ZMQ_ENDPOINT)  # now both Florence + tracker connect here
    print(f"🔗 Qwen worker bound on {ZMQ_ENDPOINT}")

    raw_queue: List[Dict[str, Any]] = []
    event_history: List[str] = []

    # ✨ NEW: buffer for YOLO tracker frames by frame_index
    tracker_buffer: Dict[int, Dict[str, Any]] = {}

    window_size = BASE_WINDOW_SIZE
    jump_size = 2

    try:
        while True:
            msg = socket.recv()
            rec = json.loads(msg.decode("utf-8"))

            # ✨ NEW: if this is a tracker frame → store and continue
            if rec.get("type") == "tracker_frame":
                frame_idx = rec.get("frame_index")
                if isinstance(frame_idx, int):
                    tracker_buffer[frame_idx] = rec
                continue

            # From here: assume this is a Florence record (as before)
            frame_idx = rec.get("frame_index")

            # Attach tracker info if available for same frame_index
            if isinstance(frame_idx, int) and frame_idx in tracker_buffer:
                rec["tracker"] = tracker_buffer[frame_idx]

            raw_queue.append(rec)

            if len(raw_queue) > MAX_QUEUE_SIZE:
                drop_count = len(raw_queue) - MAX_QUEUE_SIZE
                raw_queue = raw_queue[drop_count:]
                print(f"⚠️ Dropping {drop_count} old records to avoid backlog.")

            if len(raw_queue) < window_size:
                continue

            current_window = raw_queue[:window_size]

            current_window_events = []
            for r in current_window:
                ev = build_event_sentence(r)
                current_window_events.append(ev)

            new_event = current_window_events[-1]
            print(f"[DEBUG] New event: {new_event}")

            last_event = event_history[-1] if event_history else None
            if last_event:
                sim = simple_similarity(new_event, last_event)
                print(f"[DEBUG] Similarity to last event: {sim:.3f}")

                if sim < 0.20:
                    print("[DEBUG] Scene reset triggered → clearing event history.")
                    event_history = []
            else:
                print("[DEBUG] No last event, treating as significant change.")

            if last_event and simple_similarity(new_event, last_event) >= 0.90:
                print("[DEBUG] No significant change → SKIP sending to Qwen.")
                raw_queue = raw_queue[jump_size:]
                continue

            print("[DEBUG] Significant change detected → SEND to Qwen.")

            for ev in current_window_events:
                if all(simple_similarity(ev, old) < 0.90 for old in event_history):
                    event_history.append(ev)

            if len(event_history) > MAX_EVENT_HISTORY:
                event_history = event_history[-MAX_EVENT_HISTORY:]

            scene_description = build_scene_description(event_history, current_window_events)

            scene_lines = scene_description.split("\n")
            scene_lines = list(dict.fromkeys(scene_lines))
            scene_description = "\n".join(scene_lines)

            prompt = build_prompt(scene_description)

            with open(CONTEXT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write("\n==================== NEW PROMPT ====================\n")
                f.write(prompt)
                f.write("\n====================================================\n")

            result = call_qwen_for_anomaly(text_gen, prompt)

            label = result.get("label", "")
            score = float(result.get("anomaly_score", 0.0))

            if label == "normal":
                score = min(score, 0.2)
            elif label == "suspicious":
                score = max(0.3, min(score, 0.7))
            elif label == "criminal":
                score = max(score, 0.8)

            result["anomaly_score"] = score

            frame_start = current_window[0].get("frame_index")
            frame_end = current_window[-1].get("frame_index")

            print("\n==================== Anomaly decision ====================")
            print(f"Frames {frame_start}–{frame_end}")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("=========================================================\n")

            raw_queue = raw_queue[jump_size:]

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Qwen anomaly worker).")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()