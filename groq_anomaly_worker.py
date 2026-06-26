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


# Semantic phrases for the tracker's robbery-focused signals. Unlike raw YOLO
# IDs/bboxes, these are reliable evidence: weapon classes have already passed
# person-overlap + multi-frame temporal confirmation in the tracker, and
# appearance attributes come from the open-vocab model.
SUSPICIOUS_CLASS_PHRASES = {
    # custom nano model classes
    "Man_With_Gun": "a person appears to be holding a gun",
    "Man_with_Knife": "a person appears to be holding a knife",
    "Theaf_Robbery": "possible robbery/theft behavior",
    "Fighting": "physical fighting between people",
    # open-vocab YOLOE weapon classes (person-gated + temporally confirmed)
    "gun": "a person appears to be holding a gun",
    "pistol": "a person appears to be holding a gun",
    "handgun": "a person appears to be holding a gun",
    "rifle": "a person appears to be holding a rifle",
    "knife": "a person appears to be holding a knife",
}


def build_appearance_weapon_sentence(rec: Dict[str, Any]) -> str:
    """Two clearly-separated lines from the tracker so the LLM can weight them
    differently: confirmed WEAPONS (strong evidence) vs APPEARANCE (context only)."""
    tracker = rec.get("tracker")
    if not tracker:
        return ""

    weapons: List[str] = []
    appearance: List[str] = []
    for t in tracker.get("tracks", []):
        cls = t.get("cls", "")
        phrase = SUSPICIOUS_CLASS_PHRASES.get(cls)
        if phrase and phrase not in weapons:
            weapons.append(phrase)

        attrs = t.get("attributes") or []
        if attrs and cls == "person":
            appearance.append("person wearing " + ", ".join(attrs))

    lines: List[str] = []
    if weapons:
        lines.append("WEAPON ALERT: " + "; ".join(weapons) + ".")
    if appearance:
        lines.append("Appearance (context only, NOT proof of crime): " + "; ".join(appearance) + ".")

    return " ".join(lines)


def build_vlm_sentence(rec: Dict[str, Any]) -> str:
    """The local VLM's full-frame Q&A summary — the most action-aware source.
    Passed through verbatim (NOT clean_caption, which would truncate it)."""
    vlm = rec.get("vlm")
    if not vlm:
        return ""
    summary = (vlm.get("summary") or "").strip()
    return ("VLM observation: " + summary) if summary else ""


def build_event_sentence(rec: Dict[str, Any]) -> str:
    vlm_text = build_vlm_sentence(rec)
    appearance_text = build_appearance_weapon_sentence(rec)
    tracker_text = build_tracker_sentence(rec)

    # VLM behavior first (most action-aware), then tracker/appearance cues.
    parts = [p for p in (vlm_text, appearance_text, tracker_text) if p]
    if parts:
        return " ".join(parts)

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
    These descriptions come from two sources:
    1. VLM observation (full-frame behavioral Q&A: actions, weapons, theft cues, concealment)
    2. YOLO tracker (object IDs, classes, bounding boxes, weapon alerts, continuity across frames)

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
    - "criminal" REQUIRES one of:
        a) a WEAPON ALERT from the tracker (strong, person-gated confirmation), OR
        b) a VLM observation that explicitly states a weapon is visible/held (e.g.
           "Gun visible: yes", "Weapon described: person holding a pistol"), OR
        c) a clearly described forbidden ACTION (e.g. theft / taking items, forcing a
           display case, physical aggression, reaching behind the counter, leaving
           without payment).
      Appearance ALONE (clothing, hood, mask, hat) can NEVER be "criminal".

    ### 3. HOW TO USE THE INPUTS — BEHAVIOR FIRST
    - The PRIMARY question is: what are people DOING, and does it match the store's
      Allowed behaviors or the Forbidden behaviors in the business context above?
      Base your decision mainly on the ACTIONS described vs that list.
    - "VLM observation:" lines come from a vision model that looked directly at the
      full video frame and answered specific questions (actions, appearance, weapons,
      theft/forbidden actions, posture, aggression, concealed faces). This is the
      PRIMARY behavioral evidence. A VLM weapon answer (e.g. "Gun visible: yes" or
      a weapon description like "person holding a pistol") is STRONG evidence and
      can justify "criminal" on its own, just like a WEAPON ALERT.
    - "WEAPON ALERT:" lines from the tracker are STRONG evidence (weapons are
      person-gated + temporally confirmed) and can justify "criminal" on their own.
    - "Appearance (context only...)" lines are WEAK, SUPPORTING context. Clothing,
      hoods, masks and hats are frequently benign (hard hats, fashion, weather).
      Appearance may RAISE concern only when combined with suspicious behavior. It
      MUST NOT be the sole reason and MUST NOT push the label above "suspicious".
    - Raw "YOLO tracker detected:" lines (IDs, classes, bounding boxes) are SECONDARY
      and should be used ONLY to:
      * understand continuity of people/objects across frames
      * detect repeated presence or movement patterns
      * identify that the same person appears in multiple frames

    ### 4. PROHIBITED USE OF RAW YOLO DATA
    You MUST NOT use the raw "YOLO tracker detected:" technical data as evidence.
    Specifically:
    - DO NOT use confidence scores as reasons.
    - DO NOT use bounding boxes as reasons.
    - DO NOT use "ID 1", "ID 2", etc. as key moments.
    - DO NOT treat "multiple people detected" as suspicious by itself.
    (This prohibition does NOT apply to WEAPON ALERT / Appearance lines.)

    ### 5. KEY MOMENTS RULES
    "key_moments" MUST:
    - describe ACTIONS / behaviors first (what people do), based on VLM observations and the
      store's forbidden-behaviors list; a confirmed weapon is also a valid key moment
    - mention appearance only as a MODIFIER of an action, never on its own
    - NOT include raw YOLO technical data (IDs, confidence, bbox)

    Examples of GOOD key moments:
    - "person reaching behind the cashier counter"
    - "customer leaning over a display case and taking an item"
    - "individual hiding jewelry inside their jacket"
    - "person appears to be holding a gun"
    - "hooded person forcing open a display case"   (appearance + action)

    Examples of BAD key moments (FORBIDDEN):
    - "person wearing a hood"          (appearance with no action)
    - "multiple people wearing hats"   (appearance with no action)
    - "ID 1: person (confidence 0.91)"
    - "three people detected"

    ### 6. REASON FIELD RULES
    The "reason" MUST:
    - describe the behavioral/contextual anomaly (the ACTION), or the weapon
    - be based on VLM observation content (and the forbidden-behaviors list)
    - NOT be about appearance alone (e.g. "person wearing a hood" is NOT acceptable)
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
                        help="ZMQ PULL endpoint to receive frames from Tracker/VLM.")
    parser.add_argument("--decision-frames", dest="decision_frames", type=int, default=60,
                        help="Number of video frames per Groq decision window.")
    parser.add_argument("--tracker-every", dest="tracker_every", type=int, default=5,
                        help="Must match tracker's --send_every_n_frames; used to size the decision window.")
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(args.zmq_endpoint)
    print(f"🔗 Groq worker bound on {args.zmq_endpoint}")

    print("[INFO] Initializing Groq client. ZMQ receiver is already bound.")
    client = load_groq_client(args.groq_api_key)

    ws = await ws_connect_loop(args.ws_url)

    event_history: List[str] = []
    latest_business_context = ""

    # Tracker is the fast side buffer; VLM is the decision anchor.
    tracker_buffer: Dict[int, Dict[str, Any]] = {}

    # Throttle clock: video frame index of the last Groq decision.
    # Groq fires at most once per --decision-frames video frames regardless of VLM speed.
    last_decision_frame: Optional[int] = None

    try:
        while True:
            msg = await asyncio.to_thread(socket.recv)
            rec = json.loads(msg.decode("utf-8"))

            if rec.get("type") == "reset":
                event_history.clear()
                tracker_buffer.clear()
                last_decision_frame = None
                print("[GROQ] Reset received — cleared event_history, tracker_buffer, last_decision_frame")
                continue

            if rec.get("type") == "business_context":
                context_body = rec.get("context")
                latest_business_context = normalize_business_context(context_body)
                print("[CTX] Received business context: " + latest_business_context)
                continue

            if rec.get("type") == "tracker_frame":
                # Tracker is the side buffer. Store every tracker frame keyed by frame index.
                frame_idx = rec.get("frame_index")
                if isinstance(frame_idx, int):
                    tracker_buffer[frame_idx] = rec
                    if len(tracker_buffer) > MAX_QUEUE_SIZE:
                        for k in sorted(tracker_buffer)[:-MAX_QUEUE_SIZE]:
                            tracker_buffer.pop(k, None)
                continue

            if rec.get("type") != "vlm_frame":
                # florence_frame and any unknown types — ignore.
                continue

            # --- VLM anchor: each vlm_frame is a candidate Groq decision point ---
            frame_idx = rec.get("frame_index")

            # Throttle: fire Groq at most once per --decision-frames video frames.
            # This decouples cost from VLM velocity — running VLM faster never raises API spend.
            if (last_decision_frame is not None and isinstance(frame_idx, int)
                    and frame_idx - last_decision_frame < args.decision_frames):
                continue

            # Advance the throttle clock now (before the similarity skip) so even a
            # skipped Groq call counts against the budget — avoids burst on scene change.
            if isinstance(frame_idx, int):
                last_decision_frame = frame_idx

            # Attach the nearest tracker record. Tracker is dense (every 5 frames),
            # so the nearest is almost always within --tracker-every * 3 frames.
            if isinstance(frame_idx, int) and tracker_buffer:
                nearest = min(tracker_buffer, key=lambda k: abs(k - frame_idx))
                if abs(nearest - frame_idx) <= args.tracker_every * 3:
                    rec["tracker"] = tracker_buffer[nearest]

            # Self-reference so build_vlm_sentence reads rec["vlm"]["summary"].
            rec["vlm"] = rec

            new_event = build_event_sentence(rec)
            print(f"[DEBUG] New event (VLM frame {frame_idx}): {new_event}")

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
                continue

            print("[DEBUG] Significant change detected → SEND to Groq.")

            if all(simple_similarity(new_event, old) < 0.90 for old in event_history):
                event_history.append(new_event)

            if len(event_history) > MAX_EVENT_HISTORY:
                event_history = event_history[-MAX_EVENT_HISTORY:]

            if latest_business_context:
                print(f"[CTX] Injecting business context into prompt:\n{latest_business_context}")
            else:
                print("[CTX] No business context — sending prompt without it.")

            scene_description = build_scene_description(
                event_history,
                [new_event],
                latest_business_context,
            )

            scene_lines = scene_description.split("\n")
            scene_lines = list(dict.fromkeys(scene_lines))
            scene_description = "\n".join(scene_lines)

            prompt = build_prompt(scene_description)

            with open(CONTEXT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write("\n==================== NEW PROMPT ====================\n")
                f.write(prompt)
                f.write("\n====================================================\n")

            result = await asyncio.to_thread(call_groq_for_anomaly, client, args.groq_model, prompt)

            label = result.get("label", "")
            score = float(result.get("anomaly_score", 0.0))

            if label == "normal":
                score = min(score, 0.2)
            elif label == "suspicious":
                score = max(0.3, min(score, 0.7))
            elif label == "criminal":
                score = max(score, 0.8)

            result["anomaly_score"] = score

            print("\n==================== Anomaly decision ====================")
            print(f"VLM frame {frame_idx}")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("=========================================================\n")

            anomaly_payload = {
                "type": "groq_anomaly",
                "frame_range": {
                    "start": frame_idx,
                    "end": frame_idx,
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
