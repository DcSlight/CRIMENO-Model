import io
import json
import re
import time
import zmq
import torch
from PIL import Image
from transformers import pipeline


def load_florence_pipeline():
    """Load Florence-2 as an image-text-to-text pipeline."""
    has_cuda = torch.cuda.is_available()

    if has_cuda:
        vision_pipe = pipeline(
            "image-text-to-text",
            model="florence-community/Florence-2-base",
            device=0,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("✅ Florence-2 pipeline loaded on GPU")
    else:
        vision_pipe = pipeline(
            "image-text-to-text",
            model="florence-community/Florence-2-base",
            device=-1,
            trust_remote_code=True,
        )
        print("✅ Florence-2 pipeline loaded on CPU")

    return vision_pipe


def _extract_generated_text(result):
    """Normalize pipeline output into a string."""
    if isinstance(result, list) and len(result) > 0:
        first = result[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]
        return str(first)
    return str(result)


def _safe_run_task(vision_pipe, image, task_token):
    """
    Try to run a Florence task token.
    If the model/task fails, return None (and keep pipeline alive).
    """
    try:
        result = vision_pipe(image, text=task_token)
        return _extract_generated_text(result)
    except Exception:
        return None


def _guess_indoor_outdoor(text):
    if not text:
        return None
    t = text.lower()
    indoor_hints = ["indoors", "inside", "store", "shop", "supermarket", "mall", "aisle", "counter", "checkout"]
    outdoor_hints = ["outdoors", "outside", "street", "sidewalk", "road", "parking", "sky", "trees"]
    indoor_score = sum(1 for w in indoor_hints if w in t)
    outdoor_score = sum(1 for w in outdoor_hints if w in t)
    if indoor_score == 0 and outdoor_score == 0:
        return None
    if indoor_score >= outdoor_score:
        return "INDOOR"
    return "OUTDOOR"


def _extract_people_count(text):
    """
    Heuristic people count extraction from caption text.
    Not perfect, but better than nothing when we only have captions.
    """
    if not text:
        return None

    t = text.lower()

    # Common phrases
    patterns = [
        r"\b(\d+)\s+(people|persons|men|women|kids|children|customers|shoppers)\b",
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(people|persons|men|women|kids|children|customers|shoppers)\b",
        r"\b(a|an)\s+(man|woman|person|customer|shopper)\b",
    ]

    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }

    for p in patterns:
        m = re.search(p, t)
        if m:
            val = m.group(1)
            if val.isdigit():
                return int(val)
            if val in word_to_num:
                return word_to_num[val]
            if val in ["a", "an"]:
                return 1

    # Fallback: if it contains "a man"/"a woman" etc
    if "a man" in t or "a woman" in t or "a person" in t:
        return 1

    return None


def _extract_location_guess(text):
    if not text:
        return None
    t = text.lower()

    # Simple location categories (extend as needed)
    mapping = [
        ("supermarket", ["supermarket", "grocery", "aisle", "checkout", "shopping cart", "shelves"]),
        ("convenience_store", ["convenience", "corner store", "counter", "cashier", "register"]),
        ("shop", ["shop", "store", "retail", "boutique"]),
        ("street", ["street", "sidewalk", "crosswalk", "traffic", "road"]),
        ("parking_lot", ["parking lot", "parked cars", "parking"]),
        ("home", ["living room", "kitchen", "bedroom", "house", "apartment"]),
        ("office", ["office", "desk", "computer", "meeting room"]),
    ]

    for label, hints in mapping:
        score = sum(1 for h in hints if h in t)
        if score >= 2:
            return label

    return None


def _extract_datetime_candidates(text):
    """
    Look for timestamp-like patterns that may appear in overlays described by caption.
    If you later add real OCR, you can replace this with OCR output parsing.
    """
    if not text:
        return []

    patterns = [
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",          # YYYY-MM-DD
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",        # DD/MM/YYYY
        r"\b\d{1,2}:\d{2}(:\d{2})?\b",               # HH:MM(:SS)
    ]

    found = []
    for p in patterns:
        for m in re.finditer(p, text):
            found.append(m.group(0))

    # Deduplicate while keeping order
    unique = []
    seen = set()
    for x in found:
        if x not in seen:
            unique.append(x)
            seen.add(x)
    return unique


def _classify_activity(text):
    """
    Very rough activity labeling based on caption keywords.
    This is a placeholder until you add a dedicated classifier.
    """
    if not text:
        return {"label": None, "confidence": 0.0, "is_suspicious": False, "reasons": []}

    t = text.lower()

    rules = [
        ("shopping_checkout", ["checkout", "cashier", "paying", "payment", "register", "counter"]),
        ("shopping_browsing", ["shopping", "browsing", "aisle", "shelves", "cart", "basket"]),
        ("walking", ["walking", "walking down", "walks", "pedestrian"]),
        ("running", ["running", "sprinting"]),
        ("fighting", ["fight", "fighting", "punch", "kicking", "assault"]),
        ("stealing", ["steal", "stealing", "shoplifting", "hiding an item", "conceal"]),
        ("weapon_present", ["gun", "knife", "rifle", "weapon"]),
        ("loitering", ["loiter", "lingering", "standing around"]),
    ]

    best = None
    best_hits = 0
    for label, keywords in rules:
        hits = sum(1 for k in keywords if k in t)
        if hits > best_hits:
            best_hits = hits
            best = label

    confidence = 0.0
    if best_hits > 0:
        confidence = min(0.95, 0.25 + 0.15 * best_hits)

    suspicious_labels = {"fighting", "stealing", "weapon_present"}
    is_suspicious = best in suspicious_labels

    reasons = []
    if is_suspicious:
        reasons.append(f"Matched activity label: {best}")

    return {
        "label": best,
        "confidence": round(confidence, 2),
        "is_suspicious": is_suspicious,
        "reasons": reasons,
    }


def build_analysis(frame_idx, video_time_ms, detailed_caption, od_text, ocr_text):
    indoor_outdoor = _guess_indoor_outdoor(detailed_caption)
    location_guess = _extract_location_guess(detailed_caption)
    people_count = _extract_people_count(detailed_caption)
    datetime_candidates = _extract_datetime_candidates((ocr_text or "") + "\n" + (detailed_caption or ""))

    activity = _classify_activity(detailed_caption)

    analysis = {
        "frame_index": frame_idx,
        "video_time_ms": video_time_ms,
        "scene": {
            "indoor_outdoor": indoor_outdoor,
            "location_guess": location_guess,
        },
        "people": {
            "count_guess": people_count,
        },
        "objects": {
            "raw_detection_text": od_text,  # may be None if task unsupported
        },
        "text_overlay": {
            "raw_ocr_text": ocr_text,       # may be None if task unsupported
            "datetime_candidates": datetime_candidates,
        },
        "activity": activity,
        "raw": {
            "detailed_caption": detailed_caption,
        },
        "meta": {
            "generated_at_unix_ms": int(time.time() * 1000),
        },
    }

    return analysis


def render_text_report(analysis):
    frame_idx = analysis.get("frame_index")
    video_time_ms = analysis.get("video_time_ms")
    scene = analysis.get("scene", {})
    people = analysis.get("people", {})
    activity = analysis.get("activity", {})
    overlay = analysis.get("text_overlay", {})
    raw = analysis.get("raw", {})

    lines = []
    lines.append(f"Frame {frame_idx} | t={video_time_ms}ms")
    lines.append(f"Scene: indoor_outdoor={scene.get('indoor_outdoor')} | location_guess={scene.get('location_guess')}")
    lines.append(f"People: count_guess={people.get('count_guess')}")
    lines.append(f"Activity: label={activity.get('label')} | confidence={activity.get('confidence')} | suspicious={activity.get('is_suspicious')}")
    if activity.get("reasons"):
        lines.append(f"Suspicion reasons: {activity.get('reasons')}")

    if overlay.get("datetime_candidates"):
        lines.append(f"Datetime candidates: {overlay.get('datetime_candidates')}")

    lines.append("Caption:")
    lines.append(raw.get("detailed_caption") or "")
    return "\n".join(lines)


def main():
    vision_pipe = load_florence_pipeline()

    # ZeroMQ PULL socket
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://127.0.0.1:5560")
    print("🔗 Connected to video broadcaster on tcp://127.0.0.1:5560")

    # Try multiple Florence tasks.
    # Some may not be supported by your Florence-2 variant; we handle None gracefully.
    TASK_DETAILED = "<MORE_DETAILED_CAPTION>"
    TASK_OD = "<OD>"
    TASK_OCR = "<OCR>"

    # Optional output file
    out_path = "analysis.jsonl"

    try:
        with open(out_path, "a", encoding="utf-8") as f:
            while True:
                parts = socket.recv_multipart()

                # Backward compatible:
                # Old format: [frame_idx, jpg_bytes]
                # New format: [frame_idx, pos_msec, jpg_bytes]
                if len(parts) == 2:
                    frame_idx_bytes, jpg_bytes = parts
                    video_time_ms = -1
                else:
                    frame_idx_bytes, pos_msec_bytes, jpg_bytes = parts
                    try:
                        video_time_ms = int(pos_msec_bytes.decode("utf-8"))
                    except Exception:
                        video_time_ms = -1

                frame_idx = int(frame_idx_bytes.decode("utf-8"))

                # Decode JPEG to PIL image
                image = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")

                # Run Florence tasks
                detailed_caption = _safe_run_task(vision_pipe, image, TASK_DETAILED)
                od_text = _safe_run_task(vision_pipe, image, TASK_OD)
                ocr_text = _safe_run_task(vision_pipe, image, TASK_OCR)

                analysis = build_analysis(
                    frame_idx=frame_idx,
                    video_time_ms=video_time_ms,
                    detailed_caption=detailed_caption,
                    od_text=od_text,
                    ocr_text=ocr_text,
                )

                # Print TEXT report + JSON
                print("\n" + "=" * 80)
                print(render_text_report(analysis))
                print("-" * 80)
                print("JSON:")
                print(json.dumps(analysis, ensure_ascii=False, indent=2))

                # Persist JSONL
                f.write(json.dumps(analysis, ensure_ascii=False) + "\n")
                f.flush()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (worker).")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
