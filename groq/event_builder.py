import re
from typing import Any, Dict, List


# Canonical VLM binary-cue key -> human-readable label, used to build the condensed
# "flags" line (see build_vlm_sentence) and shared with groq_anomaly_worker.py /
# scoring.py so all three stay in lock-step with the VLM schema (vlm/prompt.txt).
CUE_LABELS: Dict[str, str] = {
    "gun":                     "gun",
    "knife":                   "knife",
    "reaching_display_case":   "reaching display case",
    "reaching_behind_counter": "reaching behind counter",
    "hands_up":                "hands up",
    "face_concealed":          "face concealed",
    "aggression":              "aggression",
}

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
    """Person/object COUNTS only — no raw track IDs, confidence scores, or bounding boxes.

    Raw bbox/ID text is noise to an LLM narrator (it can't reason about pixel coordinates)
    and was diluting/competing with the actual behavioral signal. Counts still let Groq
    reason about "multiple people converging" without the technical soup.
    """
    tracker = rec.get("tracker")
    if not tracker:
        return ""

    tracks = tracker.get("tracks", [])
    if not tracks:
        return ""

    counts: Dict[str, int] = {}
    for t in tracks:
        cls = t.get("cls", "object")
        counts[cls] = counts.get(cls, 0) + 1

    parts = [f"{n} {cls}{'s' if n != 1 else ''}" for cls, n in counts.items()]
    return ("Tracker detected: " + ", ".join(parts) + ".") if parts else ""


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
    """VLM observation — the Description sentence plus a condensed flag line.

    Previously this passed through vlm.summary verbatim, which is a flat dump of every
    schema field including repeated "Gun: no. Knife: no." boilerplate. That register
    taught Groq to echo cues ("Reaching display case") instead of narrating behavior.
    Now: keep the (already security-led) Description sentence, and only surface cues
    that are NOT "no" — the true, security-relevant signal — as a short flag line.
    """
    vlm = rec.get("vlm")
    if not vlm:
        return ""
    qa = vlm.get("qa") or {}
    description = (qa.get("description") or "").strip()

    if not description:
        # Fallback for older log/mock formats that only have the flat summary string.
        summary = (vlm.get("summary") or "").strip()
        return ("VLM observation: " + summary) if summary else ""

    # "weapon" is free text (e.g. "long, thin object, possibly a rifle or shotgun"), not a
    # yes/no/unclear cue like gun/knife — it previously wasn't surfaced here at all, so
    # anything the VLM could describe but not confidently classify as gun-or-knife (a bat,
    # an ambiguous long object) was invisible to Groq's narrative. Surface it explicitly
    # whenever it's not "none" (it still never feeds the deterministic score — only
    # gun/knife do that — this is narrative-only, same as the description sentence).
    weapon_note = (qa.get("weapon") or "").strip()
    if weapon_note and weapon_note.lower() not in ("none", "n/a", "none visible"):
        description = f"{description} Weapon note: {weapon_note}."

    flags = []
    for key, label in CUE_LABELS.items():
        value = str(qa.get(key, "")).strip().lower()
        if value and value != "no":
            flags.append(f"{label}: {value}")
    flag_text = ("Flags — " + "; ".join(flags) + ".") if flags else "Flags — none."

    return f"VLM observation: {description} {flag_text}"


def build_event_sentence(rec: Dict[str, Any]) -> str:
    vlm_text        = build_vlm_sentence(rec)
    appearance_text = build_appearance_weapon_sentence(rec)
    tracker_text    = build_tracker_sentence(rec)

    parts = [p for p in (vlm_text, appearance_text, tracker_text) if p]
    return " ".join(parts) if parts else "No significant visual change."
