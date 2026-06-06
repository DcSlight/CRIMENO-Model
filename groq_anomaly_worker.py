import json
import os
import re
import zmq
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ============================================================
# Configuration
# ============================================================

GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
ZMQ_ENDPOINT = "tcp://127.0.0.1:5581"

BASE_WINDOW_SIZE = 3
MAX_QUEUE_SIZE = 30
MAX_EVENT_HISTORY = 10

CONTEXT_LOG_FILE = "groq_context_log.txt"


# ============================================================
# Load Groq client
# ============================================================

def load_groq_client(api_key: str) -> Groq:
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Pass --groq-api-key or set env var GROQ_API_KEY.")
    client = Groq(api_key=api_key)
    print("✅ Groq client initialized")
    return client


# ============================================================
# Rule-based caption summarization
# ============================================================

GENERIC_START_PATTERNS = [
    "the image shows",
    "this image shows",
    "the image is a still from",
    "the image appears to be",
    "the image is taken from",
    "it shows",
    "in the image",
]

ALERT_KEYWORDS = [
    "robber", "thief", "robbery", "gun", "knife", "weapon", "pistol", "firearm",
    "lying on the floor", "lying on floor", "on the ground", "victim",
    "assault", "attack", "threatening", "threat", "crime", "criminal",
    "fleeing", "fleeing the scene", "running away", "hands up", "hands raised",
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

    # Prioritize alert sentences so they survive the max_sentences cutoff
    def _is_alert(s: str) -> bool:
        sl = s.lower()
        return any(kw in sl for kw in ALERT_KEYWORDS)

    alert_sents = [s for s in meaningful if _is_alert(s)]
    other_sents = [s for s in meaningful if not _is_alert(s)]
    meaningful = alert_sents + other_sents

    meaningful = meaningful[:max_sentences]
    cleaned = " ".join(meaningful)

    if len(cleaned) > 500:
        cleaned = cleaned[:500].rstrip() + "..."

    return cleaned


def build_tracker_sentence(rec: Dict[str, Any]) -> str:
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

        pose_str = ""
        pose = t.get("pose")
        if pose and isinstance(pose, dict):
            active = [k for k, v in pose.items() if v]
            if active:
                pose_str = ", pose: " + ", ".join(active)

        parts.append(
            f"ID {tid}: {cls} (confidence {conf:.2f}) at [{x1},{y1},{x2},{y2}]{pose_str}"
        )

    if not parts:
        return ""

    return "YOLO tracker detected: " + "; ".join(parts) + "."


def build_event_sentence(rec: Dict[str, Any]) -> str:
    caption = rec.get("raw", {}).get("more_detailed_caption", "")
    cleaned = clean_caption(caption)

    # Inject Florence weapon detections as a hard alert prefix
    weapon_prefix = ""
    weapons_raw = rec.get("raw", {}).get("open_vocab_weapons", "")
    if weapons_raw and not str(weapons_raw).startswith("[ERROR"):
        labels = re.findall(r"'([^']+)'", str(weapons_raw))
        labels = [l for l in labels if l not in ("bboxes", "bboxes_labels")]
        if labels:
            weapon_prefix = f"WEAPON DETECTED by Florence: {', '.join(sorted(set(labels)))}. "

    tracker_text = build_tracker_sentence(rec)

    body = (weapon_prefix + cleaned).strip()
    if body and tracker_text:
        return f"{body} {tracker_text}"
    elif body:
        return body
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


def normalize_business_context(context_body: Any) -> str:
    if context_body is None:
        return ""

    if isinstance(context_body, str):
        return context_body.strip()

    try:
        return json.dumps(context_body, ensure_ascii=False)
    except Exception:
        return str(context_body).strip()


# ============================================================
# Scene description builder
# ============================================================

def build_scene_description(
    event_history: List[str],
    current_window_events: List[str],
    business_context: str = "",
) -> str:
    lines = []

    if business_context:
        lines.append("Business context (from NestJS):")
        lines.append(f"- {business_context}")

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
    These descriptions come from three sources:
    1. Florence (semantic captions, OCR, object descriptions, behaviors, interactions)
    2. YOLO tracker (object IDs, classes, bounding boxes, continuity across frames)
    3. Pose estimator (per-person posture flags indicating intent: arm_extended_aim, hands_above_head, torso_lean_forward, crouching)

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

    ### 3. HOW TO USE THE INPUTS — STRICT HIERARCHY

    Florence is the PRIMARY source. It tells you WHAT is happening: who is in the scene, what objects are present (gun, register, phone, bag), what behaviors are visible (paying, browsing, threatening, fleeing), and the overall situational context (normal transaction, argument, robbery in progress).

    Pose flags are SUPPORTING evidence. They tell you HOW a person is posturing — they confirm or contradict the intent Florence describes. Pose flags CANNOT by themselves establish that a crime is occurring. A crouching person is only criminal evidence if Florence also describes a threatening context (weapon visible, victim posture, register being emptied, etc.).

    YOLO tracker data (IDs, classes, bounding boxes) is REFERENCE-ONLY: continuity of people/objects across frames. Never evidence on its own.

    DECISION RULE:
    - "criminal" requires BOTH (a) Florence describing threatening/criminal behavior or objects (gun, weapon, robbery, threat, force, fleeing, victim) AND (b) pose flags that are consistent with that interpretation.
    - "suspicious" requires Florence to describe unusual or ambiguous behavior, optionally reinforced by pose flags.
    - If Florence describes only normal activity (working, shopping, paying, talking), the label is "normal" regardless of which pose flags are active. A cashier crouching to open a register is normal. A customer leaning over a counter is normal.
    - If Florence and pose disagree, trust Florence for the label and note the contradiction in "reason".

    ### 4. PROHIBITED USE OF YOLO DATA
    You MUST NOT use raw YOLO tracker data as primary evidence of anomaly.
    Specifically:
    - DO NOT use confidence scores as reasons.
    - DO NOT use bounding boxes as reasons.
    - DO NOT use "ID 1", "ID 2", etc. as key moments.
    - DO NOT treat "multiple people detected" as suspicious by itself.
    NOTE: Pose flags (arm_extended_aim, hands_above_head, torso_lean_forward, crouching) are PERMITTED and ENCOURAGED as direct evidence of intent. They appear in the tracker sentence as ", pose: <flag_name>" and describe body mechanics, not bounding boxes.

    ### 4b. POSE FLAG INTERPRETATION (CRITICAL — read before deciding)

    Pose flags are BODY MECHANICS. They never describe intent on their own. Always cross-check against Florence:
    - "torso_lean_forward" → reaching/leaning. Normal at desks, counters, registers, laptops. Only meaningful if Florence describes the person leaning into a restricted area or over a victim.
    - "crouching" → bent legs. Completely ambiguous in isolation: opening a drawer, picking up an item, restocking, tying shoes, hiding. NEVER use crouching alone as criminal evidence. A cashier crouching at the register is normal work behavior.
    - "arm_extended_aim" → raised straight arm. Could be pointing, reaching for a high shelf, or aiming. Only escalate if Florence describes a weapon, threat, or victim in the scene.
    - "hands_above_head" → both arms raised. Strong victim posture, but still requires Florence to confirm a threatening context (otherwise could be stretching or reaching).
    - "arm_extended_aim" + "crouching" on the same person → strong robbery indicator, but STILL requires Florence to confirm the scene is threatening (weapon, multiple people, register, victim). Do not escalate if Florence describes a single person working alone.
    - No pose flags active → neutral. Decide entirely from Florence.

    ### 5. KEY MOMENTS RULES
    "key_moments" MUST:
    - be based on semantic content from Florence AND pose estimator flags
    - describe meaningful actions, interactions, or unusual events
    - NOT include YOLO technical data (IDs, confidence, bbox)

    Examples of GOOD key moments:
    - "man holding phone instead of payment method"
    - "customer leaning over cash register"
    - "person reaching into backpack"
    - "individual looking around nervously"
    - "individual extended arm with weapon aimed forward"
    - "person crouched behind display counter"
    - "subject leaning over restricted area"
    - "multiple individuals with hands raised above head"

    Examples of BAD key moments (FORBIDDEN):
    - "ID 1: person (confidence 0.91)"
    - "three people detected"
    - "bounding box moved left"

    ### 6. REASON FIELD RULES (STRICT)

    The "reason" MUST:
    - Reference what Florence describes: the actual scene content (people, objects, actions, location in the store).
    - If the label is "criminal" or "suspicious", the reason MUST name the Florence-derived threat (e.g., "armed individual aiming at cashier", "person fleeing with merchandise", "customer reaching into employee area"). Pose flags may be mentioned as supporting detail but must NOT be the entire reason.
    - A reason that mentions ONLY pose mechanics (e.g., "person crouching", "arm extended") with no Florence content is INVALID — re-derive the label from Florence first.
    - Be short, human-interpretable, and free of YOLO IDs/confidence/bbox.

    ==============================================================

    Below is the scene description.

    "Recent context" lists older events for background only.
    "Current window" lists the most recent events — THIS is what you must decide on.
    Your label, score, reason, and key_moments MUST be derived from "Current window".
    Use "Recent context" only to disambiguate the current window (e.g., to know whether
    a person already entered the store with a weapon). Never decide solely from "Recent
    context"; never ignore "Current window".

    {scene_description}

    Respond with exactly one JSON object and nothing else.
    """.strip()

    return prompt


# ============================================================
# JSON extraction
# ============================================================

def parse_groq_output(text: str) -> Dict[str, Any]:
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


def call_groq_for_anomaly(client: Groq, model_name: str, prompt: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    generated = response.choices[0].message.content or ""
    return parse_groq_output(generated)


# ============================================================
# WebSocket connection
# ============================================================

async def ws_connect_loop(ws_url: str):
    import websockets

    if ws_url.lower() == "none":
        print("[WS] Disabled (ws_url=none)")
        return None

    backoff = 0.25
    while True:
        try:
            ws = await websockets.connect(ws_url, max_size=16 * 1024 * 1024)
            print(f"[WS] Connected: {ws_url}")
            return ws
        except Exception as e:
            print(f"[WS] Connect failed: {e} (retry in {backoff:.2f}s)")
            await asyncio.sleep(backoff)
            backoff = min(5.0, backoff * 1.7)


async def ws_send_json(ws, payload: Dict[str, Any]):
    if ws is not None:
        await ws.send(json.dumps(payload, ensure_ascii=False))


# ============================================================
# Main worker
# ============================================================

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws-url", "--ws_url", dest="ws_url", default="none",
                        help="WebSocket URL for forwarding anomaly results to NestJS (or 'none' to disable).")
    parser.add_argument("--groq-api-key", "--groq_api_key", dest="groq_api_key", default="",
                        help="Groq API key (or set env var GROQ_API_KEY).")
    parser.add_argument("--groq-model", "--groq_model", dest="groq_model", default=GROQ_MODEL_NAME,
                        help="Groq model to use for anomaly detection.")
    parser.add_argument("--zmq-endpoint", default=ZMQ_ENDPOINT,
                        help="ZMQ PULL endpoint to receive frames from Florence/Tracker.")
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(args.zmq_endpoint)
    print(f"🔗 Groq worker bound on {args.zmq_endpoint}")

    print("[INFO] Initializing Groq client. ZMQ receiver is already bound.")
    client = load_groq_client(args.groq_api_key)

    ws = await ws_connect_loop(args.ws_url)

    raw_queue: List[Dict[str, Any]] = []
    event_history: List[str] = []
    latest_business_context = ""

    tracker_buffer: Dict[int, Dict[str, Any]] = {}

    window_size = BASE_WINDOW_SIZE
    jump_size = 2

    try:
        while True:
            msg = await asyncio.to_thread(socket.recv)
            rec = json.loads(msg.decode("utf-8"))

            if rec.get("type") == "reset":
                raw_queue.clear()
                event_history.clear()
                tracker_buffer.clear()
                print("[GROQ] Reset received — cleared raw_queue, event_history, tracker_buffer")
                continue

            if rec.get("type") == "business_context":
                context_body = rec.get("context")
                latest_business_context = normalize_business_context(context_body)
                print("[CTX] Received business context: " + latest_business_context)
                continue

            if rec.get("type") == "tracker_frame":
                frame_idx = rec.get("frame_index")
                if isinstance(frame_idx, int):
                    tracker_buffer[frame_idx] = rec
                continue

            frame_idx = rec.get("frame_index")

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
                print("[DEBUG] No significant change → SKIP sending to Groq.")
                raw_queue = raw_queue[jump_size:]
                continue

            print("[DEBUG] Significant change detected → SEND to Groq.")

            # Snapshot history BEFORE merging current window so "Recent context"
            # and "Current window" don't share bullets (dedup would wipe the window).
            history_for_prompt = list(event_history)

            if latest_business_context:
                print(f"[CTX] Injecting business context into prompt:\n{latest_business_context}")
            else:
                print("[CTX] No business context — sending prompt without it.")

            scene_description = build_scene_description(
                history_for_prompt,
                current_window_events,
                latest_business_context,
            )

            prompt = build_prompt(scene_description)

            with open(CONTEXT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write("\n==================== NEW PROMPT ====================\n")
                f.write(prompt)
                f.write("\n====================================================\n")

            result = await asyncio.to_thread(call_groq_for_anomaly, client, args.groq_model, prompt)

            # Update history AFTER sending so the next iteration's Recent context
            # is accurate but the current window stays separate in this prompt.
            for ev in current_window_events:
                if all(simple_similarity(ev, old) < 0.90 for old in event_history):
                    event_history.append(ev)
            if len(event_history) > MAX_EVENT_HISTORY:
                event_history = event_history[-MAX_EVENT_HISTORY:]

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

            anomaly_payload = {
                "type": "groq_anomaly",
                "frame_range": {
                    "start": frame_start,
                    "end": frame_end,
                },
                "result": {
                    "anomaly_score": result["anomaly_score"],
                    "label": result.get("label", "unknown"),
                    "reason": result.get("reason", ""),
                    "key_moments": result.get("key_moments", []),
                }
            }

            try:
                await ws_send_json(ws, anomaly_payload)
                print(f"[WS] Sent anomaly result to NestJS")
            except Exception as e:
                print(f"[WS] Send failed: {e}. Reconnecting...")
                ws = await ws_connect_loop(args.ws_url)

            raw_queue = raw_queue[jump_size:]

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Groq anomaly worker).")
    finally:
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        socket.close()
        context.term()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
