import json
import os
import re
import sys
import zmq
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq

from event_builder import build_event_sentence, CUE_LABELS
from scoring import (
    WEAPON_CONFIDENCE_CUE,
    apply_scoring,
    grade_weapon_text,
    new_threat_state,
    score_from_cues,
)

_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent

sys.path.insert(0, str(_PROJECT_ROOT))
import session_log

load_dotenv(_PROJECT_ROOT / ".env")

# ============================================================
# Configuration
# ============================================================

GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
ZMQ_ENDPOINT    = "tcp://127.0.0.1:5581"

MAX_QUEUE_SIZE    = 30
MAX_EVENT_HISTORY = 10

# Log files live next to this script (groq/ folder) regardless of CWD. These are the
# legacy fallback paths, used only when no session_log session exists yet.
CONTEXT_LOG_FILE = str(_HERE / "groq_context_log.txt")
OUTPUT_LOG_FILE  = _HERE / "logs_output.jsonl"


def _resolve_log_paths():
    """Resolve the current session's jsonl + plaintext-context log paths. Falls back
    to the legacy flat files (CONTEXT_LOG_FILE/OUTPUT_LOG_FILE) if no session exists
    yet — e.g. the broadcaster hasn't played a video, or session_log hiccuped."""
    jsonl_log = session_log.resolve_log_path("groq")
    if jsonl_log is None:
        return OUTPUT_LOG_FILE, CONTEXT_LOG_FILE
    context_name = jsonl_log.name.replace("groq_v", "groq_context_v").replace(".jsonl", ".txt")
    return jsonl_log, jsonl_log.parent / context_name

# Cue keys from the VLM's structured QA dict (vlm_worker.py → rec["qa"]). Sourced from
# event_builder.CUE_LABELS so the worker, the prompt-text builder, and scoring.py can
# never drift out of sync with each other or with the VLM schema (vlm/prompt.txt).
# HARD_SIGNAL_KEYS: if ANY of these is "yes", the frame is ALWAYS sent to Groq.
BINARY_CUE_KEYS  = list(CUE_LABELS.keys())
HARD_SIGNAL_KEYS = ["gun", "knife", "hands_up", "aggression"]

# The VLM's 3-state answers are free-form strings: neither backend enforces an enum (Ollama
# types every field as a bare "string", the Gemini path sends no response schema at all), and
# vlm_model only rewrites outright refusals. So "yes, a gun", "YES - handgun", "unclear
# (partially obscured)" and "no." all reach us verbatim — and scoring._cue_weight does an
# EXACT dict lookup, so every one of them silently scored 0.0 while the change-detector below
# (v.startswith("y")) still treated them as hard signals. That is a silent weapon-evidence
# dropout, so normalize to the {yes, no, unclear} vocabulary at ingest.
#
# The aliases also cover CRIMENO-Backend/mocks' hand-labeled vocabulary ("uncertain",
# "possible", "partial", "victims controlled", and the older `reaching_counter` key), so the
# ground-truth VLM mocks can be replayed straight through the scorer as a regression fixture.
_CUE_KEY_ALIASES = {"reaching_counter": "reaching_behind_counter"}
_CUE_VALUE_ALIASES = {
    "uncertain": "unclear",
    "possible": "unclear",
    "possibly": "unclear",
    "partial": "unclear",
    "partially": "unclear",
    "maybe": "unclear",
    "victims controlled": "yes",
    "true": "yes",
    "false": "no",
    "none": "no",
}


def normalize_cue_value(value: Any) -> str:
    """Map a raw VLM cue answer onto the {yes, no, unclear} vocabulary scoring.py expects."""
    text = str(value or "").strip().lower().rstrip(".!,;:")
    if not text:
        return "no"
    if text in _CUE_VALUE_ALIASES:
        return _CUE_VALUE_ALIASES[text]
    if text in ("yes", "no", "unclear"):
        return text
    # Prefixed / qualified answers: "yes, a gun", "unclear (partially obscured)", "no visible…"
    for prefix in ("unclear", "yes", "no"):
        if text.startswith(prefix):
            return prefix
    for alias, canonical in _CUE_VALUE_ALIASES.items():
        if text.startswith(alias):
            return canonical
    return "unclear"


def extract_raw_cues(qa: Dict[str, Any]) -> Dict[str, str]:
    """Build the normalized cue dict scoring.score_from_cues consumes.

    Includes the derived `weapon_confidence` cue graded from the free-text `weapon` field —
    that field never reached the score before, even though it is where the VLM's actual
    confidence lives (vlm/prompt.txt forbids answering gun/knife "yes" on hedged wording, so
    on real footage the structured fields are almost always "unclear" regardless of whether
    the frame shows a handgun aimed at a cashier or an unidentifiable dark object).
    """
    cues: Dict[str, str] = {}
    for key in BINARY_CUE_KEYS:
        value = qa.get(key)
        if value is None:
            for alias, canonical in _CUE_KEY_ALIASES.items():
                if canonical == key and alias in qa:
                    value = qa.get(alias)
                    break
        cues[key] = normalize_cue_value(value)
    cues[WEAPON_CONFIDENCE_CUE] = grade_weapon_text(qa.get("weapon", ""))
    return cues

_PROMPT_TEMPLATE = (_HERE / "prompt.txt").read_text(encoding="utf-8")


# ============================================================
# Groq client
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


def normalize_business_context(context_body: Any) -> str:
    if context_body is None:
        return ""
    if isinstance(context_body, str):
        return context_body.strip()
    try:
        return json.dumps(context_body, ensure_ascii=False)
    except Exception:
        return str(context_body).strip()


_SCORING_LEVEL_RE = re.compile(r"scoring:\s*(\w+)", flags=re.IGNORECASE)


def extract_scoring_level(business_context: str) -> str:
    """Pull `scoring_level` (conservative/balanced/aggressive) back out of the flattened
    business-context text NestJS sends today (e.g. "Sensitivity: high; scoring: aggressive;
    interaction: high"). Lightweight regex extraction — avoids requiring the backend to send
    structured JSON (that's the separate, not-yet-started business-context-contract phase);
    defaults to "balanced" if absent or unrecognized.
    """
    if not business_context:
        return "balanced"
    m = _SCORING_LEVEL_RE.search(business_context)
    if not m:
        return "balanced"
    level = m.group(1).strip().lower()
    return level if level in ("conservative", "balanced", "aggressive") else "balanced"


def multi_person_converging(rec: Dict[str, Any], min_people: int = 3) -> bool:
    """Heuristic corroborating signal for the scoring gate: does the attached tracker
    frame show several people present at once (e.g. multiple suspects positioning)?
    """
    tracker = rec.get("tracker")
    if not tracker:
        return False
    tracks = tracker.get("tracks", [])
    person_count = sum(1 for t in tracks if t.get("cls") == "person")
    return person_count >= min_people


# ============================================================
# Prompt builder
# ============================================================

def build_scene_description(
    earlier_events: List[Dict[str, Any]],
    now_events: List[Dict[str, Any]],
    business_context: str = "",
) -> str:
    """`earlier_events`/`now_events` are lists of {"frame_index": int, "text": str},
    oldest -> newest. Explicitly labeling EARLIER vs NOW (instead of the old single
    "Recent context" / "Current window" split, where "Current window" ended up empty
    at runtime due to a dedup collision) gives Groq a clear now-vs-then to narrate a
    trajectory from, and lets it cite which frame a change appeared in.
    """
    lines = []
    if business_context:
        lines.append("Business context (from NestJS):")
        lines.append(f"- {business_context}")
    if earlier_events:
        lines.append("EARLIER observations (oldest → most recent):")
        for ev in earlier_events[-MAX_EVENT_HISTORY:]:
            lines.append(f"- (frame {ev['frame_index']}) {ev['text']}")
    if now_events:
        lines.append("\nNOW — current window (oldest → most recent; compare against EARLIER):")
        for ev in now_events:
            lines.append(f"- (frame {ev['frame_index']}) {ev['text']}")
    return "\n".join(lines)


def build_prompt(scene_description: str) -> str:
    return _PROMPT_TEMPLATE.format(scene_description=scene_description).strip()


# ============================================================
# Groq API call + JSON parsing
# ============================================================

def parse_groq_output(text: str) -> Dict[str, Any]:
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    for cand in re.findall(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            return json.loads(cand)
        except Exception:
            continue

    if "reason" in text:
        try:
            return json.loads("{" + text.strip().strip(",") + "}")
        except Exception:
            pass
        try:
            lines = [l.strip().rstrip(",") for l in text.splitlines() if ":" in l]
            return json.loads("{ " + ", ".join(lines) + " }")
        except Exception:
            pass

    return {
        "reason": "Failed to parse model JSON output",
        "key_moments": [],
        "concern": "",
        "raw_output": text[:500],
    }


def call_groq_for_anomaly(client: Groq, model_name: str, prompt: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.0,
    )
    return parse_groq_output(response.choices[0].message.content or "")


# ============================================================
# Scoring
# ============================================================
# The anomaly_score/label are no longer decided by Groq — see groq/scoring.py.
# apply_scoring / score_from_cues are imported at the top of this file and re-exported
# here (via that import) for any external caller that does
# `from groq_anomaly_worker import apply_scoring`.


# ============================================================
# WebSocket helpers
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
# Main worker loop
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
                        help="Minimum video frames between two Groq API calls (cost knob).")
    parser.add_argument("--tracker-every", dest="tracker_every", type=int, default=5,
                        help="Must match tracker's --send_every_n_frames; sizes the decision window.")
    parser.add_argument("--window-frames", dest="window_frames", type=int, default=3,
                        help="How many of the most recent VLM observations to show Groq as the "
                             "'NOW' window (gives it a real temporal arc to narrate instead of one "
                             "instant). Independent of --decision-frames, which only controls how "
                             "often a Groq call fires.")
    args = parser.parse_args()

    context = zmq.Context()
    socket  = context.socket(zmq.PULL)
    socket.bind(args.zmq_endpoint)
    print(f"🔗 Groq worker bound on {args.zmq_endpoint}")

    print("[INFO] Initializing Groq client. ZMQ receiver is already bound.")
    client = load_groq_client(args.groq_api_key)

    ws = await ws_connect_loop(args.ws_url)

    current_jsonl_log, current_context_log = _resolve_log_paths()

    # Long-term memory: one entry per past DECISION (a Groq call actually made), shown to
    # Groq as "EARLIER observations". Entries: {"frame_index": int, "text": str}.
    event_history: List[Dict[str, Any]] = []
    # Short-term buffer: one entry per VLM frame RECEIVED, regardless of whether it was
    # throttled/skipped from becoming a decision. Feeds both the sliding "NOW" window
    # (real temporal arc) and the scoring module's cue-persistence tracking (real
    # frame-level resolution, not just once-per-decision resolution).
    # Entries: {"frame_index": int, "text": str, "raw_cues": Dict[str, str]}.
    vlm_frame_buffer: List[Dict[str, Any]] = []
    latest_business_context: str      = ""
    tracker_buffer: Dict[int, Dict]   = {}
    last_decision_frame: Optional[int] = None
    last_cues: Optional[Dict[str, str]] = None  # raw 3-state (yes/no/unclear) values, not booleans
    threat_state: Dict[str, float]    = new_threat_state()

    buffer_cap = max(MAX_EVENT_HISTORY, args.window_frames) + 5

    try:
        while True:
            try:
                msg = await asyncio.to_thread(socket.recv)
                rec = json.loads(msg.decode("utf-8"))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[ERROR] Failed to receive/decode message: {exc}")
                continue

            msg_type = rec.get("type")

            if msg_type == "reset":
                event_history.clear()
                vlm_frame_buffer.clear()
                tracker_buffer.clear()
                last_decision_frame = None
                last_cues = None
                # A latched threat must NOT survive a video switch — otherwise the next clip
                # opens at "criminal" because the previous one ended mid-robbery.
                threat_state = new_threat_state()
                # Broadcaster may have started a new business/session before this reset.
                current_jsonl_log, current_context_log = _resolve_log_paths()
                print("[GROQ] Reset — cleared event_history, vlm_frame_buffer, tracker_buffer, "
                      "last_decision_frame, last_cues, threat_state")
                continue

            if msg_type == "business_context":
                latest_business_context = normalize_business_context(rec.get("context"))
                print("[CTX] Received business context: " + latest_business_context)
                continue

            if msg_type == "tracker_frame":
                frame_idx = rec.get("frame_index")
                if isinstance(frame_idx, int):
                    tracker_buffer[frame_idx] = rec
                    if len(tracker_buffer) > MAX_QUEUE_SIZE:
                        for k in sorted(tracker_buffer)[:-MAX_QUEUE_SIZE]:
                            tracker_buffer.pop(k, None)
                continue

            if msg_type != "vlm_frame":
                continue

            # --- VLM anchor: each vlm_frame is a candidate Groq decision point ---
            frame_idx = rec.get("frame_index")

            # Attach nearest tracker record.
            if isinstance(frame_idx, int) and tracker_buffer:
                nearest = min(tracker_buffer, key=lambda k: abs(k - frame_idx))
                if abs(nearest - frame_idx) <= args.tracker_every * 3:
                    rec["tracker"] = tracker_buffer[nearest]

            rec["vlm"] = rec
            new_event = build_event_sentence(rec)
            qa       = rec.get("qa") or {}
            raw_cues = extract_raw_cues(qa)
            print(f"[DEBUG] New event (VLM frame {frame_idx}): {new_event}")

            # Buffer EVERY vlm_frame seen (before the throttle below) so the sliding "NOW"
            # window and the scoring module's persistence tracking both get real frame-level
            # resolution, not just once-per-decision resolution.
            if isinstance(frame_idx, int):
                vlm_frame_buffer.append({"frame_index": frame_idx, "text": new_event, "raw_cues": raw_cues})
                if len(vlm_frame_buffer) > buffer_cap:
                    vlm_frame_buffer = vlm_frame_buffer[-buffer_cap:]

            # Throttle: fire Groq at most once per --decision-frames video frames.
            if (last_decision_frame is not None and isinstance(frame_idx, int)
                    and frame_idx - last_decision_frame < args.decision_frames):
                continue

            if isinstance(frame_idx, int):
                last_decision_frame = frame_idx

            # Cue-based change detection.
            # HARD SIGNALS (gun/knife/hands-up/aggression): always send on a definitive "yes".
            # Any cue flip: send. Identical benign cues: skip.
            # NOTE: raw (3-state: yes/no/unclear) values are compared for change detection —
            # not just the yes/no-ish boolean — so a "no" -> "unclear" escalation still
            # triggers a Groq call even though it isn't a hard signal on its own.
            cues         = {k: v.startswith("y") for k, v in raw_cues.items()}
            hard_now     = (any(cues[k] for k in HARD_SIGNAL_KEYS)
                            or raw_cues.get(WEAPON_CONFIDENCE_CUE) == "high")
            cues_changed = (last_cues is None) or (raw_cues != last_cues)
            print(f"[DEBUG] Cues: {raw_cues} | hard={hard_now} | changed={cues_changed}")

            # Scene-reset: major textual divergence clears stale history.
            if event_history and simple_similarity(new_event, event_history[-1]["text"]) < 0.20:
                print("[DEBUG] Scene reset (textual divergence) → clearing event history.")
                event_history = []

            if not hard_now and not cues_changed:
                print("[DEBUG] All cues benign + unchanged → SKIP.")
                continue

            last_cues = raw_cues
            print(f"[DEBUG] Sending to Groq (hard={hard_now}, changed={cues_changed}).")

            # Sliding window: the last --window-frames buffered VLM observations become the
            # "NOW" span Groq narrates over — frame_range below becomes a real span instead
            # of a single instant, and Groq gets an actual trajectory to compare against.
            now_window = vlm_frame_buffer[-args.window_frames:]
            now_frame  = now_window[-1]

            if all(simple_similarity(now_frame["text"], old["text"]) < 0.90 for old in event_history):
                event_history.append({"frame_index": now_frame["frame_index"], "text": now_frame["text"]})
            if len(event_history) > MAX_EVENT_HISTORY:
                event_history = event_history[-MAX_EVENT_HISTORY:]

            if latest_business_context:
                print(f"[CTX] Injecting business context:\n{latest_business_context}")
            else:
                print("[CTX] No business context.")

            # Exclude the just-appended now_frame from "EARLIER" — it's already shown as
            # part of now_window. This is the fix for the old bug where the current frame's
            # line ended up duplicated into both sections and then silently deduped away,
            # leaving "Current window" always empty.
            scene_description = build_scene_description(
                event_history[:-1], now_window, latest_business_context,
            )
            prompt = build_prompt(scene_description)

            try:
                groq_result = await asyncio.to_thread(call_groq_for_anomaly, client, args.groq_model, prompt)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[ERROR] Groq API call failed (frame {frame_idx}): {exc}")
                continue

            # Code, not Groq, decides the number (see groq/scoring.py) — this is what stops a
            # single loosely-matched cue (e.g. one frame of "reaching behind counter") from
            # snapping straight to "criminal". Persistence is measured over the buffered
            # per-frame cue history; scoring_level comes from the business-context text;
            # multi-person convergence is a corroborating signal from the tracker; Groq's own
            # "concern" tag is only a small bounded tiebreak, never the deciding factor.
            # threat_state latches an established incident across decisions so the score
            # doesn't collapse back to "normal" the moment the weapon leaves frame while the
            # suspects are still emptying the display cases — see scoring.LATCH_* .
            cue_history   = [entry["raw_cues"] for entry in vlm_frame_buffer]
            scoring_level = extract_scoring_level(latest_business_context)
            converging    = multi_person_converging(rec)
            score, label, threat_state = score_from_cues(
                cue_history,
                scoring_level=scoring_level,
                multi_person_converge=converging,
                concern=groq_result.get("concern", ""),
                prior_state=threat_state,
            )

            result = {
                "anomaly_score": score,
                "label": label,
                "reason": groq_result.get("reason", ""),
                "key_moments": groq_result.get("key_moments", []),
            }

            # Coherence canary: Groq's advisory concern and the code-computed label are
            # produced from different (though overlapping) evidence views, so they CAN
            # legitimately disagree — but a "high"/"normal" or "low"/"criminal" pairing is
            # worth a look, since it either means the narrative is overreacting or the
            # scoring weights need retuning against CRIMENO-Backend/mocks' ground truth.
            groq_concern = str(groq_result.get("concern", "")).strip().lower()
            if (groq_concern == "high" and label == "normal") or (groq_concern == "low" and label == "criminal"):
                print(f"[WARN] Concern/label mismatch at frame {frame_idx}: "
                      f"concern={groq_concern!r} but label={label!r} (score={score:.2f}) — "
                      f"worth a look via eval/score_logs.py.")

            with open(current_context_log, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*54}\n")
                f.write(f"VLM frame {frame_idx} "
                        f"(window {now_window[0]['frame_index']}-{now_window[-1]['frame_index']})\n")
                f.write(f"{'='*54}\n")
                f.write("INPUT TO GROQ:\n")
                f.write(prompt)
                f.write("\n\nRAW OUTPUT FROM GROQ (narrative only — score/label are code-computed):\n")
                f.write(json.dumps(groq_result, ensure_ascii=False, indent=2))
                f.write("\n\nFINAL RESULT (code-scored):\n")
                f.write(json.dumps(result, ensure_ascii=False, indent=2))
                f.write("\n")

            print("\n==================== Anomaly decision ====================")
            print(f"VLM frame {frame_idx} | scoring_level={scoring_level} | "
                  f"multi_person_converge={converging}")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("=========================================================\n")

            anomaly_payload = {
                "type": "groq_anomaly",
                "frame_range": {
                    "start": now_window[0]["frame_index"],
                    "end":   now_window[-1]["frame_index"],
                },
                "result": result,
            }

            with open(current_jsonl_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(anomaly_payload, ensure_ascii=False) + "\n")

            try:
                await ws_send_json(ws, anomaly_payload)
                print("[WS] Sent anomaly result to NestJS")
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
