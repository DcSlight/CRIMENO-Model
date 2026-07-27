"""
Deterministic, code-owned anomaly scoring for the Groq worker.

Per the Fable-5 architecture review (see plan doc), Groq's job is NARRATION — describing
what changed across the recent frames and why it matters — not deciding the anomaly number.
Letting the LLM invent both the story AND the score is what produced a false "criminal 0.9"
verdict off a single unconfirmed frame of `reaching_behind_counter: yes` (a shopkeeper
reaching under his own counter, no weapon, not sustained, no corroborating cue).

This module owns the number instead. It converts a short HISTORY of VLM cue observations
(raw 3-state yes/no/unclear values, oldest -> newest) into an (anomaly_score, label,
threat_state) triple using:
  - tiered evidence weights (a confirmed weapon outweighs an ambiguous cue),
  - recency-decayed evidence (a cue that WAS true 1-2 observations ago but isn't in the
    current frame still counts, just less — see EVIDENCE_DECAY below),
  - graded weapon confidence parsed from the VLM's free-text `weapon` field (see
    grade_weapon_text) rather than the near-unreachable literal `gun: "yes"`,
  - a "criminal" gate that requires EITHER a high-confidence weapon OR a forbidden action,
    in both cases corroborated by a SUSTAINED second cue — and critically, the gate is
    computed from RAW per-frame cues only, never from the decayed evidence sum above, so a
    single stale/decayed signal can never manufacture a fake multi-frame streak (see
    score_from_cues for why this separation matters),
  - a latching THREAT STATE so an established armed robbery stays "criminal" while the
    suspects loot the store, instead of decaying back to "normal" a few seconds after the
    weapon leaves frame (see ThreatState / LATCH_* below).

Pure functions only — no I/O, no ZMQ, no LLM calls — so this is directly unit-testable in
isolation, without pulling in the live worker's dependencies. The threat state is threaded
through explicitly (in as `prior_state`, out as the third return value) rather than held in
module globals, so that purity survives the addition of memory.
"""

from typing import Dict, List, Optional, Tuple

# ------------------------------------------------------------------
# Evidence tiers & weights (tuned against CRIMENO-Backend/mocks' hand-labeled ground truth;
# see eval/score_logs.py to compare real pipeline output against it).
# ------------------------------------------------------------------

# Cues that alone confirm a weapon and can (with sustained corroboration) justify "criminal".
CONFIRMED_WEAPON_CUES = ("gun", "knife")

# Derived cue holding the graded confidence of the VLM's free-text weapon description.
# Produced by grade_weapon_text() and injected into the cue dict by the worker.
WEAPON_CONFIDENCE_CUE = "weapon_confidence"

# Forbidden-action cues: can only push the label to "criminal" if sustained + corroborated
# (see the gate in score_from_cues) — never off one loosely-matched frame.
FORBIDDEN_ACTION_CUES = ("reaching_behind_counter",)

# Cues that corroborate a sustained forbidden action (strengthen the criminal case).
CORROBORATING_CUES = ("aggression", "hands_up")

# Weight applied per cue value. A cue/value combination not listed here contributes 0.
CUE_WEIGHTS: Dict[str, Dict[str, float]] = {
    # gun/knife "unclear" is weighted much closer to "yes" than the other cues: real VLM
    # footage shows hedged weapon language ("possibly a rifle", "appears to be a weapon")
    # is the NORM, not the exception — a VLM confidently saying "yes, a gun" on a partially
    # obscured or awkwardly-angled weapon is rare. Treating weapon "unclear" as barely-worth-
    # mentioning (as an earlier low weight did) meant the ONLY weapon signal seen in a real
    # robbery video was almost invisible to the score.
    "gun":                     {"yes": 1.00, "unclear": 0.50},
    "knife":                   {"yes": 1.00, "unclear": 0.50},
    # Graded from the free-text weapon description. This is the cue that actually carries
    # weapon evidence in practice, because vlm/prompt.txt explicitly forbids the VLM from
    # answering gun/knife "yes" whenever its own wording is hedged — and on real footage it
    # essentially always is. See grade_weapon_text.
    WEAPON_CONFIDENCE_CUE:     {"high": 0.85, "medium": 0.45, "low": 0.10},
    # "unclear" sits near half of "yes" for these too, for the same reason it does for
    # gun/knife above: vlm/prompt.txt tells the VLM to answer "unclear" whenever its own
    # wording is hedged, and on real footage hedging is the norm. Weighting hedged aggression
    # at a third of confirmed aggression (as an earlier 0.12 did) meant two simultaneously
    # hedged cues — "face partially concealed" AND "aggression possible" — summed to less
    # than a single confirmed one, and scored below the alert threshold on a frame ground
    # truth calls suspicious 0.42.
    "aggression":              {"yes": 0.35, "unclear": 0.18},
    "hands_up":                {"yes": 0.25, "unclear": 0.12},
    "reaching_behind_counter": {"yes": 0.50, "unclear": 0.20},
    "face_concealed":          {"yes": 0.15, "unclear": 0.08},
    # Browsing motion — innocent on its own, see REACHING_DISPLAY_CASE_LOOTING_WEIGHT for
    # why it stops being innocent once a robbery is already established.
    "reaching_display_case":   {"yes": 0.00},
}

# `reaching_display_case` is the single most common cue during the back half of a jewelry
# robbery — because that is exactly what looting a display case looks like to the VLM. Scored
# as flat 0.00 it made the system go BLIND precisely when the crime was in progress (16/16
# consecutive under-calls against ground truth on jewerly_store_short). Reaching into a case
# is genuinely innocent while the scene is calm and genuinely damning once a weapon or a
# restraint has already been established, so the weight is conditioned on the threat state
# rather than fixed.
REACHING_DISPLAY_CASE_LOOTING_WEIGHT = 0.30

# Optional bonus when the tracker shows several people converging on the counter/register.
MULTI_PERSON_CONVERGE_WEIGHT = 0.10

# scoring_level (from business context) shifts the DECISION THRESHOLDS rather than adding a
# constant to the raw score.
#
# Why not an additive bias: an additive bias is applied unconditionally, so it raises the
# FLOOR of every window including completely idle ones. With a +0.25 "aggressive" bias an
# empty scene started at raw 0.25, and a single `face_concealed: yes` (0.15) reached 0.40 —
# over the 0.30 suspicious line — which labeled quiet browsing at the very first frame of a
# video "suspicious". Business context should change how READILY we act on real evidence, not
# manufacture evidence where there is none. With thresholds, an empty scene stays at raw 0.0
# and reads "normal" at every sensitivity level.
SCORING_LEVEL_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "conservative": {"suspicious": 0.38, "criminal": 0.58},
    "balanced":     {"suspicious": 0.30, "criminal": 0.50},
    "aggressive":   {"suspicious": 0.24, "criminal": 0.42},
}
DEFAULT_SCORING_LEVEL = "balanced"

# Recency decay for evidence that is no longer visible in the current frame but appeared
# recently. A cue's effective weight for a given past frame is base_weight * DECAY^age
# (age=0 is the current/most recent frame in the buffer, age increases going back in time),
# and each cue's contribution to the raw score is the MAX decayed weight seen across the
# buffer — not just whatever the single most-recent frame shows.
#
# Why: a real production bug showed a weapon flagged ("gun: unclear") in one observation,
# then NOT flagged in the very next one (VLM occlusion / angle change / person tucked it
# away) — and the old "only look at the current frame" design scored that window "normal",
# even though the very next VLM observation showed the person reaching into a display case
# (i.e. the robbery narrative kept escalating; the weapon evidence didn't retract). Weapons
# don't evaporate in a few seconds — a cue that WAS true recently is still meaningful
# evidence now, just fading. DECAY is tuned so evidence 1-2 observations old still visibly
# raises the score into "suspicious" but — combined with the criminal gate below, which
# deliberately does NOT use this decayed value — cannot alone reach "criminal".
#
# NOTE this decay governs SECONDS, not minutes: past age ~4 a cue contributes <5%. That is
# the right horizon for "is this cue still visible", and the wrong horizon for "is a robbery
# still underway" — the latter is what ThreatState below exists to carry.
EVIDENCE_DECAY = 0.55

# A forbidden-action cue must hold "yes" for at least this many consecutive decision
# points before it can (with corroboration) open the criminal gate.
PERSISTENCE_MIN_STREAK_FOR_CRIMINAL = 2

# A weapon reading only opens the criminal gate when a corroborating cue has held for at
# least this many consecutive observations. One isolated frame of `aggression: yes` next to a
# hallucinated weapon is not a robbery — on jewerly_store_short the VLM reported "a handgun
# pointed toward the seated employee" during a stretch that ground truth labels `normal`, and
# the single-frame aggression spike beside it is exactly what this streak requirement rejects.
WEAPON_CORROBORATION_MIN_STREAK = 2

# Weapon cues flicker badly on real footage (unclear -> no -> unclear across adjacent frames
# as the object is occluded or the angle shifts). A weapon seen in only ONE frame of the
# recent window is damped rather than trusted at full weight; behaviour streaks, not weapon
# flicker, should be what carries a criminal call.
WEAPON_PERSISTENCE_FRAMES = 2
WEAPON_PERSISTENCE_WINDOW = 4
WEAPON_EVIDENCE_CUES = CONFIRMED_WEAPON_CUES + (WEAPON_CONFIDENCE_CUE,)

# Bounded nudge from Groq's advisory "concern" tag (low/medium/high). Small on purpose —
# it can break a tie near a label boundary, it can never open the criminal gate by itself.
CONCERN_TIEBREAK_NUDGE = 0.05

# A lone weak cue must not be able to raise an alert by itself. `hands_up: yes` fires on a
# customer gesturing over a counter, `face_concealed: yes` fires on ordinary headwear despite
# vlm/prompt.txt explicitly excluding it — either one alone is noise, not evidence. Weapon and
# forbidden-action cues are exempt (listed in STRONG_SOLO_CUES): those ARE meaningful alone,
# and they stay exempt even once decayed, so the "weapon flickers out of view for a frame"
# case EVIDENCE_DECAY exists to cover is not undone by this rule.
MIN_CUES_FOR_SUSPICIOUS = 2
STRONG_SOLO_CUES = CONFIRMED_WEAPON_CUES + (WEAPON_CONFIDENCE_CUE,) + FORBIDDEN_ACTION_CUES

LABELS = ("normal", "suspicious", "criminal")

# ------------------------------------------------------------------
# Threat latching
# ------------------------------------------------------------------
# Ground truth for a real armed robbery holds "criminal" for ~60 CONSECUTIVE SECONDS, while
# the evidence decay above has a horizon of ~8. Without memory the score collapses back to
# "normal" the moment the weapon and the struggle leave frame — even though the robbery is
# still in progress and the suspects are visibly emptying the cases.
#
# So: entering the criminal gate LATCHES a threat level. The gate itself is unchanged (it
# still needs raw, sustained, corroborated evidence), so latching only governs how the state
# is EXITED, never how it is entered — the anti-false-positive guard this module was built
# around is fully preserved.
LATCH_HOLD_DECISIONS = 20      # decisions held at full threat after the gate fires
LATCH_DECAY_DECISIONS = 20     # decisions over which the threat then fades to zero
LATCH_CALM_EXIT_STREAK = 6     # consecutive calm decisions that release the latch early
LATCH_FLOOR_SCORE = 0.90       # raw-score floor at full threat (matches ground-truth plateau)
LATCH_CRIMINAL_MIN_LEVEL = 0.5  # threat level above which "criminal" stays available


def new_threat_state() -> Dict[str, float]:
    """A fresh, un-latched threat state. Callers thread this through score_from_cues."""
    return {"level": 0.0, "hold_left": 0, "calm_streak": 0}


def _cue_weight(key: str, value: str, threat_level: float = 0.0) -> float:
    if key == "reaching_display_case" and value == "yes" and threat_level > 0.0:
        # Looting, not browsing — see REACHING_DISPLAY_CASE_LOOTING_WEIGHT.
        return REACHING_DISPLAY_CASE_LOOTING_WEIGHT * threat_level
    return CUE_WEIGHTS.get(key, {}).get(value, 0.0)


def _streak_length(cue_history: List[Dict[str, str]], key: str, value: str) -> int:
    """How many of the most recent consecutive decision points had `key` == `value`."""
    streak = 0
    for cues in reversed(cue_history):
        if str(cues.get(key, "")).strip().lower() == value:
            streak += 1
        else:
            break
    return streak


# ------------------------------------------------------------------
# Weapon free-text grading
# ------------------------------------------------------------------
# vlm/prompt.txt instructs the VLM that hedged wording MUST map to gun/knife "unclear",
# NEVER "yes". On real footage the wording is essentially always hedged, so the literal
# `gun == "yes"` the criminal gate used to require has never once fired. Meanwhile "unclear"
# collapses genuinely different observations onto one weight:
#
#   "appears to be a handgun ... pointed toward the seated employee"   -> unclear
#   "might be a tool or weapon, but it is not clearly identifiable"    -> unclear
#
# Both scored 0.50. The free text is where the confidence actually lives, so we grade it.
# Order matters: an explicit disclaimer of identifiability outranks any weapon noun that
# happens to appear in the same sentence.
_WEAPON_LOW_CONFIDENCE_MARKERS = (
    "not clearly identifiable",
    "not identifiable",
    "cannot be identified",
    "unidentified",
    "hard to tell",
    "unclear what",
    "might be",
    "may be",
    "could be",
    "possibly",
    "perhaps",
)

_WEAPON_HIGH_CONFIDENCE_MARKERS = (
    "handgun",
    "firearm",
    "pistol",
    "revolver",
    "rifle",
    "shotgun",
    "pointed at",
    "pointed toward",
    "pointing at",
    "aimed at",
    "brandish",
    "holding a gun",
    "holding a knife",
    "blade",
)

_WEAPON_NONE_VALUES = ("", "none", "n/a", "na", "no", "none visible", "no weapon")


def grade_weapon_text(weapon_text: str) -> str:
    """Grade the VLM's free-text `weapon` field into "high" / "medium" / "low" / "none".

    Pure string classification — no model call. Returns "none" when the field is empty or
    explicitly negative, so the caller can inject the result as an ordinary cue value.

    "low" is deliberately reachable: a description that disclaims its own identifiability
    ("might be a tool or weapon, but it is not clearly identifiable") should contribute
    almost nothing, where the old flat `unclear` weight scored it the same as a handgun
    aimed at a cashier.
    """
    text = (weapon_text or "").strip().lower()
    if text in _WEAPON_NONE_VALUES:
        return "none"

    # A disclaimer of identifiability wins over any weapon noun in the same sentence.
    if any(marker in text for marker in _WEAPON_LOW_CONFIDENCE_MARKERS):
        return "low"
    if any(marker in text for marker in _WEAPON_HIGH_CONFIDENCE_MARKERS):
        return "high"
    return "medium"


def _weapon_persistence_factor(cue_history: List[Dict[str, str]], key: str) -> float:
    """Damping factor for a weapon cue seen in only a frame or two of the recent window.

    Returns 1.0 once the cue has appeared in WEAPON_PERSISTENCE_FRAMES or more of the last
    WEAPON_PERSISTENCE_WINDOW observations, scaling down linearly below that.
    """
    if key not in WEAPON_EVIDENCE_CUES:
        return 1.0
    window = cue_history[-WEAPON_PERSISTENCE_WINDOW:]
    seen = sum(
        1
        for cues in window
        if CUE_WEIGHTS.get(key, {}).get(str(cues.get(key, "")).strip().lower(), 0.0) > 0.0
    )
    if seen >= WEAPON_PERSISTENCE_FRAMES:
        return 1.0
    return seen / float(WEAPON_PERSISTENCE_FRAMES)


def apply_concern_tiebreak(raw: float, concern: str) -> float:
    """Small bounded nudge from Groq's advisory concern tag. Never enough on its own to
    cross the criminal gate — only to break a tie near a normal/suspicious/criminal
    boundary. `concern` is expected to be "low" / "medium" / "high" (case-insensitive);
    anything else is a no-op.
    """
    concern = (concern or "").strip().lower()
    if concern == "high":
        raw += CONCERN_TIEBREAK_NUDGE
    elif concern == "low":
        raw -= CONCERN_TIEBREAK_NUDGE
    return max(0.0, min(1.0, raw))


def apply_scoring(label: str, raw_score: float, scoring_level: str = "balanced") -> float:
    """Clamp a raw score into the fixed band for its label:
      normal <= 0.2, suspicious in [0.3, 0.7], criminal >= 0.8.

    Pure function — kept independently importable for backward compatibility (the eval
    harness and any external caller). `scoring_level` is accepted but unused here; the
    scoring_level sensitivity is applied earlier, as a threshold shift, inside
    score_from_cues.
    """
    score = raw_score
    if label == "normal":
        score = min(score, 0.2)
    elif label == "suspicious":
        score = max(0.3, min(score, 0.7))
    elif label == "criminal":
        score = max(score, 0.8)
    return round(score, 2)


def _is_calm(current: Dict[str, str]) -> bool:
    """True when the current observation shows nothing consistent with an ongoing incident.

    Used only to release the threat latch early — a robbery that visibly ends (suspects gone,
    staff upright, nothing in anyone's hands) shouldn't keep the store at "criminal" for the
    full hold window.

    `reaching_display_case` counts as NOT calm here, which is the opposite of how it is
    treated in a cold scene. That asymmetry is the point: once an incident is established,
    people at the display cases are the suspects emptying them. Scoring that as calm is what
    made the latch release ~20s into the looting phase and hand back 12 consecutive
    under-calls against ground truth.
    """
    for key in WEAPON_EVIDENCE_CUES:
        value = str(current.get(key, "")).strip().lower()
        if CUE_WEIGHTS.get(key, {}).get(value, 0.0) > 0.0:
            return False
    watched = ("aggression", "hands_up", "reaching_display_case") + FORBIDDEN_ACTION_CUES
    for key in watched:
        if str(current.get(key, "")).strip().lower() in ("yes", "unclear"):
            return False
    return True


def _advance_threat_state(
    state: Dict[str, float], gate_open: bool, calm: bool,
) -> Dict[str, float]:
    """Advance the latching threat state by one decision point.

    Entry is driven purely by `gate_open` (the raw-cue criminal gate), so latching can never
    manufacture a criminal verdict that the gate itself wouldn't have allowed.
    """
    level = float(state.get("level", 0.0))
    hold_left = int(state.get("hold_left", 0))
    calm_streak = int(state.get("calm_streak", 0))

    if gate_open:
        return {"level": 1.0, "hold_left": LATCH_HOLD_DECISIONS, "calm_streak": 0}

    calm_streak = calm_streak + 1 if calm else 0

    if level <= 0.0:
        return {"level": 0.0, "hold_left": 0, "calm_streak": calm_streak}

    if calm_streak >= LATCH_CALM_EXIT_STREAK:
        # Sustained calm — the incident is over, drop the latch rather than idling it out.
        return {"level": 0.0, "hold_left": 0, "calm_streak": calm_streak}

    if hold_left > 0:
        return {"level": level, "hold_left": hold_left - 1, "calm_streak": calm_streak}

    level = max(0.0, level - 1.0 / float(LATCH_DECAY_DECISIONS))
    return {"level": level, "hold_left": 0, "calm_streak": calm_streak}


def score_from_cues(
    cue_history: List[Dict[str, str]],
    scoring_level: str = DEFAULT_SCORING_LEVEL,
    multi_person_converge: bool = False,
    concern: str = "",
    prior_state: Optional[Dict[str, float]] = None,
) -> Tuple[float, str, Dict[str, float]]:
    """Compute (anomaly_score, label, threat_state) from a history of raw cue dicts.

    `cue_history` is ordered oldest -> newest; the LAST entry is the current decision
    point. It is used TWICE, for two deliberately DIFFERENT purposes that must never be
    conflated:
      1. Raw-score magnitude: each cue's contribution is the MAX of (weight * DECAY^age)
         across the whole buffer — recent-but-not-current evidence still counts, decayed.
      2. Criminal-gate eligibility: computed from RAW per-frame values only (the current
         frame's own value, plus a streak count over undecayed history) — never from the
         decayed sum above. If the gate used decayed/aggregated values, a single stale
         misfire smeared across several decisions by the decay could look like a genuine
         multi-frame streak and reopen the "one loose cue -> criminal" bug this module
         was originally built to close.

    Each dict maps cue key -> "yes" / "no" / "unclear" (raw VLM values), plus the derived
    `weapon_confidence` key ("high" / "medium" / "low" / "none") from grade_weapon_text.

    `prior_state` is the threat state returned by the previous call (or None / a fresh
    new_threat_state() to start). The updated state is returned as the third element and
    MUST be threaded back in on the next call for latching to work.
    """
    state = dict(prior_state) if prior_state else new_threat_state()

    if not cue_history:
        return 0.0, "normal", state

    current = cue_history[-1]
    prior_level = float(state.get("level", 0.0))

    # 1. Decayed-max raw score: recent evidence that has since dropped out of view still
    # contributes, just fading with age, instead of being invisible the moment it's not in
    # the single most-recent frame (see EVIDENCE_DECAY docstring for the bug this fixes).
    all_keys = set()
    for cues in cue_history:
        all_keys.update(cues.keys())

    raw = 0.0
    has_strong_solo_cue = False
    for key in all_keys:
        best = 0.0
        for age, cues in enumerate(reversed(cue_history)):
            value = str(cues.get(key, "")).strip().lower()
            weight = _cue_weight(key, value, prior_level)
            if weight <= 0:
                continue
            decayed = weight * (EVIDENCE_DECAY ** age)
            if decayed > best:
                best = decayed
        contribution = best * _weapon_persistence_factor(cue_history, key)
        raw += contribution
        if contribution > 0 and key in STRONG_SOLO_CUES:
            has_strong_solo_cue = True

    # Corroboration is counted on the CURRENT frame only. Counting decayed contributions
    # instead would let one weak cue corroborate itself across consecutive frames.
    current_cues = sum(
        1 for key in all_keys
        if _cue_weight(key, str(current.get(key, "")).strip().lower(), prior_level) > 0
    )

    if multi_person_converge:
        raw += MULTI_PERSON_CONVERGE_WEIGHT

    raw = apply_concern_tiebreak(raw, concern)
    raw = max(0.0, min(1.0, raw))

    # 2. Criminal gate: deliberately re-reads `current` and raw `cue_history` directly —
    # NOT the decayed `raw` sum above — so it never opens on one loosely-matched cue,
    # decayed or not.
    def _sustained(keys, min_streak: int) -> bool:
        for key in keys:
            if str(current.get(key, "")).strip().lower() != "yes":
                continue
            if _streak_length(cue_history, key, "yes") >= min_streak:
                return True
        return False

    # Weapon path. A literal gun/knife "yes" still counts, but so does a HIGH-confidence
    # free-text reading — otherwise this path is unreachable, since vlm/prompt.txt forbids
    # the VLM from saying "yes" whenever its own wording is hedged. Either way the weapon
    # must be corroborated by a cue that has HELD for several observations, so a hallucinated
    # weapon beside a one-frame aggression spike cannot open the gate.
    weapon_confident = (
        any(str(current.get(k, "")).strip().lower() == "yes" for k in CONFIRMED_WEAPON_CUES)
        or str(current.get(WEAPON_CONFIDENCE_CUE, "")).strip().lower() == "high"
    )
    weapon_gate = weapon_confident and _sustained(
        CORROBORATING_CUES + FORBIDDEN_ACTION_CUES, WEAPON_CORROBORATION_MIN_STREAK,
    )

    # Forbidden-action path, unchanged: sustained intrusion behind the counter, corroborated
    # by aggression or hands-up in the current frame.
    sustained_and_corroborated = False
    for key in FORBIDDEN_ACTION_CUES:
        if str(current.get(key, "")).strip().lower() != "yes":
            continue
        if _streak_length(cue_history, key, "yes") < PERSISTENCE_MIN_STREAK_FOR_CRIMINAL:
            continue
        if any(str(current.get(k, "")).strip().lower() == "yes" for k in CORROBORATING_CUES):
            sustained_and_corroborated = True
            break

    criminal_gate_open = weapon_gate or sustained_and_corroborated

    # 3. Latch: an established incident keeps the score elevated while the suspects work,
    # instead of decaying to "normal" the moment the weapon leaves frame.
    state = _advance_threat_state(state, criminal_gate_open, _is_calm(current))
    level = float(state["level"])
    if level > 0.0:
        raw = max(raw, LATCH_FLOOR_SCORE * level)

    thresholds = SCORING_LEVEL_THRESHOLDS.get(
        (scoring_level or "").strip().lower(), SCORING_LEVEL_THRESHOLDS[DEFAULT_SCORING_LEVEL],
    )

    # A single weak cue can't raise an alert on its own — it needs either corroboration from a
    # second cue or to be a strong cue in its own right. Latched incidents bypass this: the
    # corroboration already happened when the gate fired.
    corroborated = (
        current_cues >= MIN_CUES_FOR_SUSPICIOUS
        or has_strong_solo_cue
        or level > 0.0
    )

    criminal_available = criminal_gate_open or level >= LATCH_CRIMINAL_MIN_LEVEL
    if criminal_available and raw >= thresholds["criminal"]:
        label = "criminal"
    elif raw >= thresholds["suspicious"] and corroborated:
        label = "suspicious"
    else:
        label = "normal"

    score = apply_scoring(label, raw, scoring_level)
    return score, label, state
