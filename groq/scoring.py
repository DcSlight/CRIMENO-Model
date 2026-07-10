"""
Deterministic, code-owned anomaly scoring for the Groq worker.

Per the Fable-5 architecture review (see plan doc), Groq's job is NARRATION — describing
what changed across the recent frames and why it matters — not deciding the anomaly number.
Letting the LLM invent both the story AND the score is what produced a false "criminal 0.9"
verdict off a single unconfirmed frame of `reaching_behind_counter: yes` (a shopkeeper
reaching under his own counter, no weapon, not sustained, no corroborating cue).

This module owns the number instead. It converts a short HISTORY of VLM binary-cue
observations (raw 3-state yes/no/unclear values, oldest -> newest) into an
(anomaly_score, label) pair using:
  - tiered evidence weights (a confirmed weapon outweighs an ambiguous cue),
  - a persistence discount (a cue's first appearance counts for less — kills one-frame spikes),
  - a "criminal" gate that requires EITHER a confirmed weapon OR a forbidden action that is
    both sustained across multiple decision points AND corroborated by a second cue.

Pure functions only — no I/O, no ZMQ, no LLM calls — so this is directly unit-testable and
importable by eval/run_eval.py without pulling in the live worker's dependencies.
"""

from typing import Dict, List, Tuple

# ------------------------------------------------------------------
# Evidence tiers & weights (starting values — tune against CRIMENO-Backend/mocks
# via eval/run_eval.py rather than guessing further).
# ------------------------------------------------------------------

# Cues that alone confirm a weapon and can justify "criminal" on a single frame.
CONFIRMED_WEAPON_CUES = ("gun", "knife")

# Forbidden-action cues: can only push the label to "criminal" if sustained + corroborated
# (see the gate in score_from_cues) — never off one loosely-matched frame.
FORBIDDEN_ACTION_CUES = ("reaching_behind_counter",)

# Cues that corroborate a sustained forbidden action (strengthen the criminal case).
CORROBORATING_CUES = ("aggression", "hands_up")

# Weight applied per cue value. A cue/value combination not listed here contributes 0.
CUE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "gun":                     {"yes": 1.00, "unclear": 0.15},
    "knife":                   {"yes": 1.00, "unclear": 0.15},
    "aggression":              {"yes": 0.35, "unclear": 0.12},
    "hands_up":                {"yes": 0.25, "unclear": 0.08},
    "reaching_behind_counter": {"yes": 0.50, "unclear": 0.20},
    "face_concealed":          {"yes": 0.15, "unclear": 0.05},
    "reaching_display_case":   {"yes": 0.00},  # allowed behavior — never adds concern
}

# Optional bonus when the tracker shows several people converging on the counter/register.
MULTI_PERSON_CONVERGE_WEIGHT = 0.10

# scoring_level (from business context) biases the raw score before the label/gate decision.
SCORING_LEVEL_BIAS = {"conservative": -0.10, "balanced": 0.0, "aggressive": 0.10}

# A cue's FIRST appearance in the history counts at this fraction of its full weight;
# it reaches full weight once it has persisted for >1 consecutive decision point. Gentle
# on purpose (0.8, not 0.5) — a single confirmed "yes" on a forbidden action is still real
# evidence and should land solidly in "suspicious" on its own; the CRIMINAL gate below
# (which requires either a confirmed weapon or a sustained + corroborated streak) is what
# actually stops a one-frame spike from being labeled criminal, not this discount.
PERSISTENCE_DISCOUNT_FIRST_SEEN = 0.8

# A forbidden-action cue must hold "yes" for at least this many consecutive decision
# points before it can (with corroboration) open the criminal gate.
PERSISTENCE_MIN_STREAK_FOR_CRIMINAL = 2

# Bounded nudge from Groq's advisory "concern" tag (low/medium/high). Small on purpose —
# it can break a tie near a label boundary, it can never open the criminal gate by itself.
CONCERN_TIEBREAK_NUDGE = 0.05

LABELS = ("normal", "suspicious", "criminal")


def _cue_weight(key: str, value: str) -> float:
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
    scoring_level bias is applied earlier, to the raw score, inside score_from_cues.
    """
    score = raw_score
    if label == "normal":
        score = min(score, 0.2)
    elif label == "suspicious":
        score = max(0.3, min(score, 0.7))
    elif label == "criminal":
        score = max(score, 0.8)
    return score


def score_from_cues(
    cue_history: List[Dict[str, str]],
    scoring_level: str = "balanced",
    multi_person_converge: bool = False,
    concern: str = "",
) -> Tuple[float, str]:
    """Compute (anomaly_score, label) from a history of raw 3-state cue dicts.

    `cue_history` is ordered oldest -> newest; the LAST entry is the current decision
    point (used both for the raw-score sum and as the point being gated). Earlier entries
    are used only to measure persistence (how long a cue has held its current value).
    Each dict maps cue key -> "yes" / "no" / "unclear" (raw VLM values).
    """
    if not cue_history:
        return 0.0, "normal"

    current = cue_history[-1]

    raw = 0.0
    for key, value in current.items():
        value = str(value).strip().lower()
        weight = _cue_weight(key, value)
        if weight <= 0:
            continue
        streak = _streak_length(cue_history, key, value)
        discount = 1.0 if streak > 1 else PERSISTENCE_DISCOUNT_FIRST_SEEN
        raw += weight * discount

    if multi_person_converge:
        raw += MULTI_PERSON_CONVERGE_WEIGHT

    raw += SCORING_LEVEL_BIAS.get(scoring_level, 0.0)
    raw = apply_concern_tiebreak(raw, concern)
    raw = max(0.0, min(1.0, raw))

    # --- Criminal gate: never open on one loosely-matched cue ---
    confirmed_weapon = any(
        str(current.get(k, "")).strip().lower() == "yes" for k in CONFIRMED_WEAPON_CUES
    )
    sustained_and_corroborated = False
    for key in FORBIDDEN_ACTION_CUES:
        if str(current.get(key, "")).strip().lower() != "yes":
            continue
        streak = _streak_length(cue_history, key, "yes")
        if streak < PERSISTENCE_MIN_STREAK_FOR_CRIMINAL:
            continue
        if any(str(current.get(k, "")).strip().lower() == "yes" for k in CORROBORATING_CUES):
            sustained_and_corroborated = True
            break

    criminal_gate_open = confirmed_weapon or sustained_and_corroborated

    if criminal_gate_open and raw >= 0.5:
        label = "criminal"
    elif raw >= 0.3:
        label = "suspicious"
    else:
        label = "normal"

    score = apply_scoring(label, raw, scoring_level)
    return score, label
