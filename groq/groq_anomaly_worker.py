import json
import os
import re
import zmq
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq

from event_builder import build_event_sentence

_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent

load_dotenv(_PROJECT_ROOT / ".env")

# ============================================================
# Configuration
# ============================================================

GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
ZMQ_ENDPOINT    = "tcp://127.0.0.1:5581"

MAX_QUEUE_SIZE    = 30
MAX_EVENT_HISTORY = 10

# Log file lives next to this script (groq/ folder) regardless of CWD.
CONTEXT_LOG_FILE = str(_HERE / "groq_context_log.txt")

# Cue keys from the VLM's structured QA dict (vlm_worker.py → rec["qa"]).
# HARD_SIGNAL_KEYS: if ANY of these is "yes", the frame is ALWAYS sent to Groq.
BINARY_CUE_KEYS  = ["gun", "knife", "reaching_counter", "hands_up", "face_concealed", "aggression"]
HARD_SIGNAL_KEYS = ["gun", "knife", "hands_up", "aggression"]

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


# ============================================================
# Prompt builder
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

    if "anomaly_score" in text:
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
    return parse_groq_output(response.choices[0].message.content or "")


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
    args = parser.parse_args()

    context = zmq.Context()
    socket  = context.socket(zmq.PULL)
    socket.bind(args.zmq_endpoint)
    print(f"🔗 Groq worker bound on {args.zmq_endpoint}")

    print("[INFO] Initializing Groq client. ZMQ receiver is already bound.")
    client = load_groq_client(args.groq_api_key)

    ws = await ws_connect_loop(args.ws_url)

    event_history: List[str]          = []
    latest_business_context: str      = ""
    tracker_buffer: Dict[int, Dict]   = {}
    last_decision_frame: Optional[int] = None
    last_cues: Optional[Dict[str, bool]] = None

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
                tracker_buffer.clear()
                last_decision_frame = None
                last_cues = None
                print("[GROQ] Reset — cleared event_history, tracker_buffer, last_decision_frame, last_cues")
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

            # Throttle: fire Groq at most once per --decision-frames video frames.
            if (last_decision_frame is not None and isinstance(frame_idx, int)
                    and frame_idx - last_decision_frame < args.decision_frames):
                continue

            if isinstance(frame_idx, int):
                last_decision_frame = frame_idx

            # Attach nearest tracker record.
            if isinstance(frame_idx, int) and tracker_buffer:
                nearest = min(tracker_buffer, key=lambda k: abs(k - frame_idx))
                if abs(nearest - frame_idx) <= args.tracker_every * 3:
                    rec["tracker"] = tracker_buffer[nearest]

            rec["vlm"] = rec
            new_event = build_event_sentence(rec)
            print(f"[DEBUG] New event (VLM frame {frame_idx}): {new_event}")

            # Cue-based change detection.
            # HARD SIGNALS (gun/knife/hands-up/aggression): always send.
            # Any cue flip: send. Identical benign cues: skip.
            qa   = rec.get("qa") or {}
            cues = {k: str(qa.get(k, "")).strip().lower().startswith("y")
                    for k in BINARY_CUE_KEYS}
            hard_now     = any(cues[k] for k in HARD_SIGNAL_KEYS)
            cues_changed = (last_cues is None) or (cues != last_cues)
            print(f"[DEBUG] Cues: {cues} | hard={hard_now} | changed={cues_changed}")

            # Scene-reset: major textual divergence clears stale history.
            if event_history and simple_similarity(new_event, event_history[-1]) < 0.20:
                print("[DEBUG] Scene reset (textual divergence) → clearing event history.")
                event_history = []

            if not hard_now and not cues_changed:
                print("[DEBUG] All cues benign + unchanged → SKIP.")
                continue

            last_cues = cues
            print(f"[DEBUG] Sending to Groq (hard={hard_now}, changed={cues_changed}).")

            if all(simple_similarity(new_event, old) < 0.90 for old in event_history):
                event_history.append(new_event)
            if len(event_history) > MAX_EVENT_HISTORY:
                event_history = event_history[-MAX_EVENT_HISTORY:]

            if latest_business_context:
                print(f"[CTX] Injecting business context:\n{latest_business_context}")
            else:
                print("[CTX] No business context.")

            scene_description = build_scene_description(
                event_history, [new_event], latest_business_context,
            )
            scene_description = "\n".join(dict.fromkeys(scene_description.split("\n")))

            prompt = build_prompt(scene_description)

            try:
                result = await asyncio.to_thread(call_groq_for_anomaly, client, args.groq_model, prompt)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[ERROR] Groq API call failed (frame {frame_idx}): {exc}")
                continue

            with open(CONTEXT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*54}\n")
                f.write(f"VLM frame {frame_idx}\n")
                f.write(f"{'='*54}\n")
                f.write("INPUT TO GROQ:\n")
                f.write(prompt)
                f.write("\n\nOUTPUT FROM GROQ:\n")
                f.write(json.dumps(result, ensure_ascii=False, indent=2))
                f.write("\n")

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
                "frame_range": {"start": frame_idx, "end": frame_idx},
                "result": {
                    "anomaly_score": result["anomaly_score"],
                    "label":         result.get("label", "unknown"),
                    "reason":        result.get("reason", ""),
                    "key_moments":   result.get("key_moments", []),
                },
            }

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
