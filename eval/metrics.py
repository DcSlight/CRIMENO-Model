"""
Shared alignment + metrics library for the eval layer. Both entry points
(build_analytics.py and score_logs.py) import this instead of each re-implementing
JSONL loading, latest-version resolution, and accuracy/MAE -  which used to exist in
two or three copies that quietly disagreed with each other.

Pure stdlib + a read-only, by-path load of groq/scoring.py (never modified, never
duplicated -  imported exactly like the live worker does it, because scoring.py and the
installed `groq` SDK package share a name).

Two independent sources of predictions, both reduced to the same shape (a list of
(frame, label, score, reason) points, oldest -> newest) so they can share one evaluate():

 - `points_from_windows()` - what the live pipeline actually wrote (groq_vN.jsonl).
    Only covers however much of the video that run reached.
 - `points_from_replay()` - replays the VLM cue stream (vlm_vN.jsonl) straight through
    groq/scoring.py. Covers the WHOLE video regardless of whether the groq worker was
    stopped early, which is what lets the confusion matrix's "criminal" row be populated
    even from a partial live run.

Grading happens on a UNIFORM TIME GRID (one sample every `step` frames, default 60 = 2s,
matching the VLM's own cadence) rather than one row per emitted window. Sliding windows
overlap and land unevenly across the video; grading per-window over-weights whatever
region the pipeline happened to emit most densely and under-weights everything else.
"""

import importlib.util
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
MOCKS_ROOT = REPO_ROOT.parent / "CRIMENO-Backend" / "mocks"
LOGS_ROOT = REPO_ROOT / "logs"

# scoring.py lives in groq/, whose package name collides with the installed `groq` SDK, so
# it's loaded by path rather than imported normally -  same trick eval/replay_cues.py used.
# Read-only: nothing here ever calls back into scoring.py's internals to change them.
_spec = importlib.util.spec_from_file_location("crimeno_scoring", REPO_ROOT / "groq" / "scoring.py")
scoring = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scoring)

LABELS = scoring.LABELS  # ("normal", "suspicious", "criminal")
LABEL_ORDER = {label: i for i, label in enumerate(LABELS)}
SCORING_LEVELS = sorted(scoring.SCORING_LEVEL_THRESHOLDS)
DEFAULT_SCORING_LEVEL = scoring.DEFAULT_SCORING_LEVEL

# Grid resolution: 60 frames = 2s at the pipeline's 30fps, matching how often the VLM
# actually produces a new cue observation (see `every 60` in vlm_worker.py's How-to-run
# invocation) -  one grid sample per real decision point, not an arbitrary finer/coarser rate.
DEFAULT_STEP_FRAMES = 60

# One entry per business the dashboard knows about. `log_folder` is the on-disk folder name
# under logs/ (historical, from when videos were named directly); `key` is also the mock
# folder name under CRIMENO-Backend/mocks/ and the dashboard's business key. This single map
# replaces build_analytics.BUSINESS_MAP, score_logs.BUSINESS_LOG_FOLDERS, and
# replay_cues.BUSINESSES, which were three hand-kept copies of the same data.
BUSINESSES: Dict[str, Dict[str, str]] = {
    "jewelry":   {"log_folder": "jewerly_store_short", "display_name": "Jewelry Store"},
    "market":    {"log_folder": "market",               "display_name": "Market"},
    "gun_store": {"log_folder": "gun_store_robbery",     "display_name": "Gun Store"},
}

_VERSION_RE = re.compile(r"_v(\d+)\.jsonl$")


def key_for_log_folder(log_folder: str) -> Optional[str]:
    """Reverse lookup: logs/<folder> -> our business key. Used to map
    logs/current_session.json's `business` pointer back to a key for --business's default."""
    for key, info in BUSINESSES.items():
        if info["log_folder"] == log_folder:
            return key
    return None


def latest_log(business_key: str, worker: str) -> Optional[Path]:
    """Highest-numbered <worker>_vN.jsonl for this business, or None if it has none yet.
    worker is one of "vlm", "groq", "tracker". Replaces build_analytics.resolve_latest_version,
    score_logs.resolve_latest_for_business, and replay_cues._latest_versioned_log."""
    info = BUSINESSES.get(business_key)
    if info is None:
        return None
    log_dir = LOGS_ROOT / info["log_folder"] / worker
    if not log_dir.is_dir():
        return None
    best_version, best_path = 0, None
    for p in log_dir.glob(f"{worker}_v*.jsonl"):
        m = _VERSION_RE.search(p.name)
        if m and int(m.group(1)) >= best_version:
            best_version, best_path = int(m.group(1)), p
    return best_path


def mock_path_for(business_key: str) -> Path:
    return MOCKS_ROOT / business_key / "groq_mock.jsonl"


# ------------------------------------------------------------------
# JSONL loading
# ------------------------------------------------------------------

def load_windows(path: Path) -> List[Dict]:
    """Parse a groq_vN.jsonl (or a groq_mock.jsonl ground-truth file -  same shape) into
    normalized {start, end, label, score, reason, key_moments} dicts, file order preserved."""
    windows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fr = obj.get("frame_range", {}) or {}
            res = obj.get("result", {}) or {}
            windows.append({
                "start": fr.get("start"),
                "end": fr.get("end"),
                "label": res.get("label", "unknown"),
                "score": float(res.get("anomaly_score", 0.0)),
                "reason": res.get("reason", "") or "",
                "key_moments": res.get("key_moments", []) or [],
            })
    return windows


def load_vlm_records(path: Path) -> List[Dict]:
    """Parse a vlm_vN.jsonl cue stream, oldest -> newest (file order)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_tracker_frames(path: Optional[Path]) -> Dict[int, Dict]:
    """frame_index -> tracker record, for the multi-person-convergence replay signal.
    Skips reset markers. Returns {} if path is None/missing (replay still works, just
    without the tracker corroboration bonus)."""
    frames: Dict[int, Dict] = {}
    if path is None or not path.is_file():
        return frames
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("reset") or rec.get("frame_index", 0) == -1:
                continue
            fi = rec.get("frame_index")
            if isinstance(fi, int):
                frames[fi] = rec
    return frames


def non_degenerate(windows: List[Dict]) -> List[Dict]:
    """Drops zero-length warm-up windows (start == end) -  e.g. a single "0-0" ping some
    worker runs emit before the first real sliding window. These aren't a real observation
    of the video and inflate event counts (totalEvents, trend buckets) if left in."""
    return [w for w in windows if w["start"] != w["end"]]


# ------------------------------------------------------------------
# Ground truth lookup
# ------------------------------------------------------------------

def gt_at(gt_windows: List[Dict], frame: int) -> Tuple[Optional[str], Optional[float]]:
    """Ground truth is contiguous and non-overlapping by construction, so a direct
    containment check is exact -  no nearest-span fallback needed (unlike the old
    find_mock_span, which dragged out-of-range frames back onto whatever span was
    closest; that's no longer needed because callers only ever query frames inside
    [mock_start, mock_end])."""
    for w in gt_windows:
        if w["start"] <= frame <= w["end"]:
            return w["label"], w["score"]
    return None, None


def gt_range(gt_windows: List[Dict]) -> Tuple[int, int]:
    return min(w["start"] for w in gt_windows), max(w["end"] for w in gt_windows)


# ------------------------------------------------------------------
# Predictions as points: (frame, label, score, reason) - both sources reduce to this
# ------------------------------------------------------------------

Point = Tuple[int, str, float, str]


def points_from_windows(windows: List[Dict]) -> List[Point]:
    """A window's verdict becomes the standing prediction as of its END frame. Sorted and
    de-duplicated by end frame (later windows in the file win ties)."""
    by_end: Dict[int, Point] = {}
    for w in windows:
        by_end[w["end"]] = (w["end"], w["label"], w["score"], w.get("reason", ""))
    return [by_end[k] for k in sorted(by_end)]


def points_range(points: List[Point]) -> Tuple[Optional[int], Optional[int]]:
    if not points:
        return None, None
    return points[0][0], points[-1][0]


def covered_range_from_windows(windows: List[Dict]) -> Tuple[Optional[int], Optional[int]]:
    """Unlike points_range (which anchors each window's verdict to its END frame -  see
    points_from_windows), the reported/graded COVERAGE of a run should span each window's
    full [start, end], not just where its last window's decision landed. This only affects
    whether early frames are reported as covered; predicted_at still returns None for them
    either way (no point exists at or before them), so grading is unaffected -  this is
    purely so `coverage()` reports the actual frame span the run observed."""
    if not windows:
        return None, None
    return min(w["start"] for w in windows), max(w["end"] for w in windows)


def predicted_at(points: List[Point], frame: int,
                  covered_start: Optional[int], covered_end: Optional[int]) -> Optional[Tuple[str, float, str]]:
    """The standing prediction at `frame`: the most recent point at or before it. Returns
    None outside [covered_start, covered_end] -  a run that stopped early has NO opinion
    about frames past what it covered, and that must show up as "not covered", not get
    silently paired with whatever window happens to be nearest (that silent reshuffling is
    exactly what made a partial run's confusion matrix misleading before this refactor)."""
    if covered_start is None or frame < covered_start or frame > covered_end:
        return None
    result = None
    for pf, label, score, reason in points:
        if pf <= frame:
            result = (label, score, reason)
        else:
            break
    return result


# ------------------------------------------------------------------
# Cue replay (moved from eval/replay_cues.py, unchanged logic)
# ------------------------------------------------------------------

CUE_KEYS = [
    "gun", "knife", "reaching_display_case", "reaching_behind_counter",
    "hands_up", "face_concealed", "aggression",
]

# Keep in sync with groq_anomaly_worker._CUE_KEY_ALIASES / _CUE_VALUE_ALIASES.
KEY_ALIASES = {"reaching_behind_counter": "reaching_counter"}
VALUE_ALIASES = {
    "uncertain": "unclear", "possible": "unclear", "possibly": "unclear",
    "partial": "unclear", "partially": "unclear", "maybe": "unclear",
    "victims controlled": "yes", "true": "yes", "false": "no", "none": "no",
}


def normalize_cue_value(value) -> str:
    text = str(value or "").strip().lower().rstrip(".!,;:")
    if not text:
        return "no"
    if text in VALUE_ALIASES:
        return VALUE_ALIASES[text]
    if text in ("yes", "no", "unclear"):
        return text
    for prefix in ("unclear", "yes", "no"):
        if text.startswith(prefix):
            return prefix
    for alias, canonical in VALUE_ALIASES.items():
        if text.startswith(alias):
            return canonical
    return "unclear"


def cues_from_qa(qa: Dict) -> Dict[str, str]:
    cues = {}
    for key in CUE_KEYS:
        value = qa.get(key)
        if value is None and key in KEY_ALIASES:
            value = qa.get(KEY_ALIASES[key])
        cues[key] = normalize_cue_value(value)
    cues[scoring.WEAPON_CONFIDENCE_CUE] = scoring.grade_weapon_text(qa.get("weapon", ""))
    return cues


def multi_person_converging(tracker_frames: Dict[int, Dict], frame_idx: int, min_people: int = 3) -> bool:
    """Mirrors groq_anomaly_worker.multi_person_converging: does the nearest tracker frame
    show several people present at once? Read-only against tracker_*.jsonl."""
    if not tracker_frames:
        return False
    nearest = min(tracker_frames, key=lambda k: abs(k - frame_idx))
    tracks = tracker_frames[nearest].get("tracks", [])
    return sum(1 for t in tracks if t.get("cls") == "person") >= min_people


def points_from_replay(vlm_records: List[Dict], tracker_frames: Dict[int, Dict],
                        scoring_level: str, history: int = 15) -> List[Point]:
    """Replay a VLM cue stream through groq/scoring.py, threading threat_state exactly as
    the live worker does. Groq's own narrative `concern` tiebreak is NOT reproduced here (it
    isn't persisted in any log) -  this is a cue+tracker replay of the deterministic scorer,
    not a byte-for-byte reproduction of the live decision."""
    state = scoring.new_threat_state()
    cue_history: List[Dict[str, str]] = []
    points: List[Point] = []
    for rec in vlm_records:
        frame = rec.get("frame_index", 0)
        qa = rec.get("qa") or {}
        cue_history.append(cues_from_qa(qa))
        converging = multi_person_converging(tracker_frames, frame)
        score, label, state = scoring.score_from_cues(
            cue_history[-history:], scoring_level=scoring_level,
            multi_person_converge=converging, prior_state=state,
        )
        points.append((frame, label, score, ""))
    return points


# ------------------------------------------------------------------
# Uniform time grid - used for the trend/severity CHARTS only (build_analytics.py).
# NOT used for accuracy/MAE grading anymore - see evaluate() below for why.
# ------------------------------------------------------------------

def sample_grid(start: int, end: int, step: int = DEFAULT_STEP_FRAMES) -> List[int]:
    """Frames start, start+step, start+2*step, ..., always including `end` exactly even if
    it doesn't land on a step boundary."""
    if end <= start:
        return [start]
    frames = list(range(start, end + 1, step))
    if frames[-1] != end:
        frames.append(end)
    return frames


def predicted_intervals(points: List[Point],
                         covered_start: Optional[int],
                         covered_end: Optional[int]) -> List[Tuple[int, int, str, float, str]]:
    """Turn a sequence of decision points into non-overlapping, frame-exact
    [lo, hi] (both inclusive) intervals, each holding one point's verdict.

    A point's own frame is the LAST frame it's known to hold (see points_from_windows:
    a window's verdict is anchored to its END frame). So point i's interval runs from
    (point[i-1].frame + 1) through point[i].frame; frames before the very FIRST point are
    NOT covered (the model hadn't rendered a verdict yet), matching predicted_at's tested
    semantics exactly. The last point's interval is extended out to `covered_end` (e.g. the
    real run's last window's own END, which can be later than that window's anchor if it
    was a degenerate/zero-width case - in practice these coincide).
    """
    if not points or covered_start is None:
        return []
    intervals = []
    prev_frame = None
    for frame, label, score, reason in points:
        lo = prev_frame + 1 if prev_frame is not None else frame
        if lo <= frame:
            intervals.append((lo, frame, label, score, reason))
        prev_frame = frame
    if covered_end is not None and covered_end > points[-1][0]:
        last_frame, last_label, last_score, last_reason = points[-1]
        intervals.append((last_frame + 1, covered_end, last_label, last_score, last_reason))
    return intervals


def evaluate(gt_windows: List[Dict], points: List[Point],
             covered_start: Optional[int], covered_end: Optional[int]) -> Optional[Dict]:
    """The one accuracy/MAE/confusion-matrix computation, replacing score_logs.build_summary
    and the inline duplicate that used to live in replay_cues.replay (the two disagreed:
    37.5% vs 47% accuracy on the same jewelry run, because they paired windows differently).

    Grades by EXACT FRAME OVERLAP between each predicted interval and every ground-truth
    row it touches - not a single point sample. This matters because the two window
    schemes are out of phase: groq_vN.jsonl's sliding windows are 120 frames wide on a
    60-frame step, while the mock's ground-truth rows are a different, uneven width (jewelry:
    120-180 frames). A single real row like frame_range 660-780 genuinely straddles TWO
    mock rows (660-779 "suspicious" and 780-899 "suspicious") - pairing it to only one of
    them (whichever contains its end frame) silently discards the other. Every overlapping
    frame is counted against the label that predicted it, weighted by how many frames of
    overlap there actually were, so a row spanning 2-3 mock rows contributes proportionally
    to each rather than being arbitrarily assigned to just one.

    Ground-truth frames with NO overlapping prediction (a run stopped early, or a gap
    between windows) are tallied as explicit `not_covered` counts, never folded into the
    confusion matrix as a silent zero.
    """
    if not gt_windows:
        return None

    mock_start, mock_end = gt_range(gt_windows)
    pred_intervals = predicted_intervals(points, covered_start, covered_end)

    matrix = {gt: {pred: 0 for pred in LABELS} for gt in LABELS}
    not_covered = {gt: 0 for gt in LABELS}
    correct = over = under = compared = total = 0
    abs_error = 0.0
    under_examples = []
    over_examples = []

    for g in gt_windows:
        g_lo, g_hi = max(g["start"], mock_start), min(g["end"], mock_end)
        if g_hi < g_lo:
            continue
        span = g_hi - g_lo + 1
        total += span
        gt_label, gt_score = g["label"], g["score"]
        gt_rank = LABEL_ORDER.get(gt_label)
        covered_in_row = 0

        for lo, hi, pred_label, pred_score, _reason in pred_intervals:
            ov_lo, ov_hi = max(lo, g_lo), min(hi, g_hi)
            if ov_hi < ov_lo:
                continue
            overlap = ov_hi - ov_lo + 1
            covered_in_row += overlap
            compared += overlap

            if gt_label in matrix and pred_label in matrix[gt_label]:
                matrix[gt_label][pred_label] += overlap
            if pred_label == gt_label:
                correct += overlap

            pred_rank = LABEL_ORDER.get(pred_label)
            if gt_rank is not None and pred_rank is not None:
                if pred_rank < gt_rank:
                    under += overlap
                    under_examples.append((ov_lo, ov_hi, gt_label, pred_label, overlap))
                elif pred_rank > gt_rank:
                    over += overlap
                    over_examples.append((ov_lo, ov_hi, gt_label, pred_label, overlap))

            abs_error += abs(gt_score - pred_score) * overlap

        if covered_in_row < span and gt_label in not_covered:
            not_covered[gt_label] += span - covered_in_row

    return {
        "total_frames": total,
        "compared_frames": compared,
        "confusion_matrix": matrix,
        "not_covered": not_covered,
        "accuracy": correct / compared if compared else 0.0,
        "score_mae": abs_error / compared if compared else 0.0,
        "under_call_count": under,
        "under_calls": under_examples,
        "over_call_count": over,
        "over_calls": over_examples,
    }


def row_overlap_table(real_windows: List[Dict], gt_windows: List[Dict]) -> List[Dict]:
    """Diagnostic breakdown for --verbose: for each individual REAL row (as emitted, before
    being folded into intervals), which mock row(s) it overlaps and by how many frames.
    This is what makes "one real row spans 2-3 mock rows" visible directly, rather than
    only showing up as a number inside the aggregate confusion matrix."""
    rows = []
    for w in real_windows:
        touches = []
        for g in gt_windows:
            lo, hi = max(w["start"], g["start"]), min(w["end"], g["end"])
            if hi >= lo:
                touches.append({"gt_start": g["start"], "gt_end": g["end"],
                                 "gt_label": g["label"], "overlap_frames": hi - lo + 1})
        rows.append({"start": w["start"], "end": w["end"], "label": w["label"],
                     "score": w["score"], "mock_rows_touched": touches})
    return rows


def coverage(covered_start: Optional[int], covered_end: Optional[int],
             mock_start: int, mock_end: int) -> Dict:
    """How much of the ground truth's full frame range a prediction source actually
    reached. A run stopped early (or crashed) only produces windows over a PREFIX of the
    video -  this surfaces that fact explicitly instead of leaving a reader to infer it from
    an oddly-shaped confusion matrix."""
    if covered_start is None or covered_end is None:
        return {"covered_range": None, "mock_full_range": [mock_start, mock_end], "coverage_pct": 0.0}
    mock_span = mock_end - mock_start
    covered_span = max(0, min(covered_end, mock_end) - max(covered_start, mock_start))
    pct = (covered_span / mock_span) if mock_span > 0 else 1.0
    return {
        "covered_range": [covered_start, covered_end],
        "mock_full_range": [mock_start, mock_end],
        "coverage_pct": round(min(1.0, max(0.0, pct)), 4),
    }


def text_similarity_stats(gt_windows: List[Dict], real_windows: List[Dict]) -> Dict:
    """Optional diagnostic, log-source only: how similar is the pipeline's narrated `reason`
    text to the mock's, and how often did Groq's own JSON parse fail. Not part of
    accuracy/MAE -  replay has no narrative text to compare (groq/scoring.py never invents
    one), so this is only meaningful for the shipped groq_vN.jsonl."""
    import difflib
    if not gt_windows or not real_windows:
        return {"avg_similarity": None, "parse_failure_count": None}
    sims = []
    parse_failures = 0
    for real in real_windows:
        gt_label, _ = gt_at(gt_windows, real["end"])
        if gt_label is None:
            continue
        gt_window = next((w for w in gt_windows if w["end"] == real["end"] or
                           (w["start"] <= real["end"] <= w["end"])), None)
        if gt_window is not None:
            sims.append(difflib.SequenceMatcher(None, gt_window["reason"], real["reason"]).ratio())
        if real["reason"] == "Failed to parse model JSON output":
            parse_failures += 1
    return {
        "avg_similarity": (sum(sims) / len(sims)) if sims else None,
        "parse_failure_count": parse_failures,
    }
