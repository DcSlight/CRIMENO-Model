import re
from typing import Any, Dict, List


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

# Semantic phrases for tracker robbery-focused signals. Weapon classes have already
# passed person-overlap + multi-frame temporal confirmation in the tracker.
SUSPICIOUS_CLASS_PHRASES = {
    "Man_With_Gun":   "a person appears to be holding a gun",
    "Man_with_Knife": "a person appears to be holding a knife",
    "Theaf_Robbery":  "possible robbery/theft behavior",
    "Fighting":       "physical fighting between people",
    "gun":            "a person appears to be holding a gun",
    "pistol":         "a person appears to be holding a gun",
    "handgun":        "a person appears to be holding a gun",
    "rifle":          "a person appears to be holding a rifle",
    "knife":          "a person appears to be holding a knife",
}


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

    meaningful = [s for s in sentences if not is_generic_sentence(s)]
    if not meaningful:
        meaningful = sentences

    cleaned = " ".join(meaningful[:max_sentences])
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
        tid  = t.get("track_id")
        cls  = t.get("cls", "object")
        conf = t.get("conf", 0.0)
        bbox = t.get("bbox", {})
        x1, y1, x2, y2 = bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")
        parts.append(f"ID {tid}: {cls} (confidence {conf:.2f}) at [{x1},{y1},{x2},{y2}]")

    return ("YOLO tracker detected: " + "; ".join(parts) + ".") if parts else ""


def build_appearance_weapon_sentence(rec: Dict[str, Any]) -> str:
    """Two clearly-separated lines: confirmed WEAPONS (strong) vs APPEARANCE (context only)."""
    tracker = rec.get("tracker")
    if not tracker:
        return ""

    weapons: List[str] = []
    appearance: List[str] = []
    for t in tracker.get("tracks", []):
        cls    = t.get("cls", "")
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
    """VLM full-frame Q&A summary — passed through verbatim, not truncated."""
    vlm = rec.get("vlm")
    if not vlm:
        return ""
    summary = (vlm.get("summary") or "").strip()
    return ("VLM observation: " + summary) if summary else ""


def build_event_sentence(rec: Dict[str, Any]) -> str:
    vlm_text        = build_vlm_sentence(rec)
    appearance_text = build_appearance_weapon_sentence(rec)
    tracker_text    = build_tracker_sentence(rec)

    parts = [p for p in (vlm_text, appearance_text, tracker_text) if p]
    return " ".join(parts) if parts else "No significant visual change."
