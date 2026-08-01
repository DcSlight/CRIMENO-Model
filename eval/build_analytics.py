"""
Build a single analytics.json snapshot from the REAL running-history logs of every business
that has log data on disk, in the shapes the CRIMENO dashboard's analytics widgets expect
(see CRIMENO-Backend's analytics.types.ts / analytics.mock.ts for the TS side).

Each business is computed independently from its own latest groq_vN.jsonl / vlm_vN.jsonl /
tracker_vN.jsonl, using its own ground-truth mock (CRIMENO-Backend/mocks/<key>/groq_mock.jsonl)
for the confusion matrix and eval summary. A business with no log files yet is simply absent
from the per-business dicts - the NestJS backend falls back to mock data for any key missing
there.

Pure stdlib (json, argparse, pathlib, re, time, collections) plus eval/metrics.py (itself
stdlib-only) - no groq/ or vlm/ file is imported or modified; groq/scoring.py is loaded
read-only, by path, exactly as the live worker loads it.

`scoring_level` (conservative/balanced/aggressive) is NOT auto-detected - it isn't persisted
anywhere eval can read (the live worker pulls it from a websocket business-context string at
runtime). Pass --scoring-level if you know what a business actually runs at; it defaults to
"balanced" and applies the same to every business in this snapshot.

Usage:
  py eval/build_analytics.py
  py eval/build_analytics.py --scoring-level aggressive
  py eval/build_analytics.py --out PATH
  py eval/build_analytics.py --mock PATH   # override ground-truth mock for ALL businesses (testing only)
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import metrics  # noqa: E402

REPO_ROOT = _HERE.parent
LOGS_ROOT = metrics.LOGS_ROOT
SESSION_FILE = LOGS_ROOT / "current_session.json"
DEFAULT_OUT = REPO_ROOT / "analytics.json"

ALL_BUSINESSES = [{"key": key, "name": info["display_name"]} for key, info in metrics.BUSINESSES.items()]

# confusionMatrixByBusiness ships to the dashboard in SECONDS, not raw frames - metrics.evaluate()
# grades frame-by-frame (see its docstring for why), but a reader expects a confusion matrix
# cell to be a plain count, not "2220" with no unit. Accuracy/MAE/etc in the `eval` object stay
# frame-weighted and untouched - only this display matrix is converted.
FPS = 30

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "at",
    "and", "or", "with", "for", "near", "that", "this", "it", "as", "be", "by",
    "from", "has", "have",
}

_WORD_RE = re.compile(r"[a-z]+")

_TYPE_KEYWORDS = {
    "Armed Robbery": ("robbery", "armed"),
    "Weapon Detected": ("weapon", "firearm", "gun", "knife"),
    "Theft / Stolen Goods": ("steal", "theft", "stolen"),
    "Restricted Area Breach": ("restricted", "climb", "breach"),
}


def resolve_active_run():
    """Return (business_key, log_folder, version) for the run to analyze. Prefers
    logs/current_session.json's pointer if it names a business we know; otherwise falls
    back to jewelry's highest available version."""
    default_key = "jewelry"
    session = None
    try:
        session = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except Exception:
        session = None

    key = default_key
    if session:
        mapped = metrics.key_for_log_folder(session.get("business", ""))
        if mapped:
            key = mapped

    log_path = metrics.latest_log(key, "groq")
    version = None
    if log_path is not None:
        m = re.search(r"_v(\d+)\.jsonl$", log_path.name)
        version = int(m.group(1)) if m else 1
    elif session and session.get("version"):
        version = session["version"]
    else:
        version = 1

    return key, metrics.BUSINESSES[key]["log_folder"], version


def replay_points_for(vlm_path, tracker_path, scoring_level: str):
    """Full-video predictions via cue-replay through groq/scoring.py - the SAME source the
    confusion matrix already prefers over the live groq log, used here so KPIs/trend/severity
    describe the real video's actual length instead of whatever prefix a live run happened to
    reach before being stopped early. Returns ([], None, None) if there's no vlm cue stream to
    replay - callers should fall back to the live log's own (possibly partial) coverage."""
    if vlm_path is None:
        return [], None, None
    vlm_records = metrics.load_vlm_records(vlm_path)
    tracker_frames = metrics.load_tracker_frames(tracker_path)
    points = metrics.points_from_replay(vlm_records, tracker_frames, scoring_level)
    covered_start, covered_end = metrics.points_range(points)
    return points, covered_start, covered_end


def compute_kpis(points: list, covered_start, covered_end,
                  step: int = metrics.DEFAULT_STEP_FRAMES) -> dict:
    if not points or covered_start is None:
        return {
            "totalEvents": 0, "criminalEvents": 0, "avgAnomalyScore": 0.0,
            "activeBusinesses": len(ALL_BUSINESSES), "alertsToday": 0,
        }
    labels, scores = [], []
    for frame in metrics.sample_grid(covered_start, covered_end, step):
        pred = metrics.predicted_at(points, frame, covered_start, covered_end)
        if pred is not None:
            label, score, _reason = pred
            labels.append(label)
            scores.append(score)
    total = len(labels)
    criminal = sum(1 for label in labels if label == "criminal")
    avg_score = sum(scores) / total if total else 0.0
    return {
        "totalEvents": total,
        "criminalEvents": criminal,
        "avgAnomalyScore": avg_score,
        "activeBusinesses": len(ALL_BUSINESSES),
        "alertsToday": criminal,
    }


def compute_anomaly_trend(points: list, covered_start, covered_end,
                           step: int = metrics.DEFAULT_STEP_FRAMES) -> list:
    """One point per fixed `step`-frame bucket (default 2s) across the whole run, with the
    anomaly score placed into whichever series matches that bucket's standing label and 0 in
    the other two. Replaces the old per-window trend, which keyed each point on a window's
    START frame - since sliding windows overlap and several can share start=0, that produced
    duplicate x-axis keys (three "0s" points) and gaps wherever no window happened to start
    (frames 180-300 on the jewelry run), crushing the "normal" series into a stub at the
    origin instead of a real timeline."""
    if not points or covered_start is None:
        return []
    trend = []
    for frame in metrics.sample_grid(covered_start, covered_end, step):
        point = {"time": f"{round(frame / 30)}s", "normal": 0, "suspicious": 0, "criminal": 0}
        pred = metrics.predicted_at(points, frame, covered_start, covered_end)
        if pred is not None:
            label, score, _reason = pred
            if label in point:
                point[label] = round(score * 100)
        trend.append(point)
    return trend


def compute_anomaly_type(segments: list) -> list:
    counts = Counter()
    for s in segments:
        text = (s["reason"] + " " + " ".join(s["key_moments"])).lower()
        for type_name, keywords in _TYPE_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                counts[type_name] += 1
        if s["label"] == "suspicious":
            counts["Suspicious Behavior"] += 1

    return [{"type": t, "count": c} for t, c in counts.items() if c > 0]


def compute_severity(points: list, covered_start, covered_end,
                      step: int = metrics.DEFAULT_STEP_FRAMES) -> list:
    """Grid-sampled distribution ("how much of the run's covered video sat at each label"),
    not a raw count of overlapping windows - a window emitted every 60 frames but spanning
    120 would otherwise be counted once per emission regardless of how long it actually held."""
    counts = {label: 0 for label in metrics.LABELS}
    if points and covered_start is not None:
        for frame in metrics.sample_grid(covered_start, covered_end, step):
            pred = metrics.predicted_at(points, frame, covered_start, covered_end)
            if pred is not None and pred[0] in counts:
                counts[pred[0]] += 1
    return [
        {"name": "Normal", "value": counts["normal"]},
        {"name": "Suspicious", "value": counts["suspicious"]},
        {"name": "Criminal", "value": counts["criminal"]},
    ]


def compute_word_frequencies(segments: list, top_n: int = 24) -> list:
    counts = Counter()
    for s in segments:
        text = (s["reason"] + " " + " ".join(s["key_moments"])).lower()
        for word in _WORD_RE.findall(text):
            if len(word) < 3 or word in STOPWORDS:
                continue
            counts[word] += 1

    if not counts:
        return []

    top = counts.most_common(top_n)
    max_count = top[0][1]
    result = []
    for word, count in top:
        value = max(1, round(count / max_count * 100))
        result.append({"text": word.capitalize(), "value": value})
    return result


def compute_people_count(frames: dict) -> int:
    """`frames` is metrics.load_tracker_frames's frame_index -> record map."""
    if not frames:
        return 0
    return max(len(f.get("tracks", [])) for f in frames.values())


def compute_eval_summary(business_key: str, mock_path: Path, groq_path, vlm_path, tracker_path,
                          scoring_level: str) -> dict:
    """Both the replay headline (full video) and the shipped-log secondary (whatever that
    run covered), each with an explicit coverage %. See eval/metrics.py's module docstring
    for why two numbers instead of one."""
    if not mock_path.exists():
        print(f"warning: mock file not found at {mock_path}; eval will be null", file=sys.stderr)
        return None

    gt_windows = metrics.load_windows(mock_path)
    mock_start, mock_end = metrics.gt_range(gt_windows)
    out = {"scoring_level": scoring_level}

    if vlm_path is not None:
        vlm_records = metrics.load_vlm_records(vlm_path)
        tracker_frames = metrics.load_tracker_frames(tracker_path)
        points = metrics.points_from_replay(vlm_records, tracker_frames, scoring_level)
        covered_start, covered_end = metrics.points_range(points)
        result = metrics.evaluate(gt_windows, points, covered_start, covered_end)
        cov = metrics.coverage(covered_start, covered_end, mock_start, mock_end)
        out["replay"] = {**{k: v for k, v in result.items() if k not in ("under_calls", "over_calls")},
                          "coverage": cov} if result else None
    else:
        out["replay"] = None

    if groq_path is not None and groq_path.exists():
        real_windows = metrics.non_degenerate(metrics.load_windows(groq_path))
        points = metrics.points_from_windows(real_windows)
        covered_start, covered_end = metrics.covered_range_from_windows(real_windows)
        result = metrics.evaluate(gt_windows, points, covered_start, covered_end)
        cov = metrics.coverage(covered_start, covered_end, mock_start, mock_end)
        out["log"] = {**{k: v for k, v in result.items() if k not in ("under_calls", "over_calls")},
                       "coverage": cov} if result else None
    else:
        out["log"] = None

    # Back-compat top-level confusion matrix (rows=gt, cols=pred, order Normal/Suspicious/
    # Criminal) - this exact shape is read directly by CRIMENO-Backend's analytics.service.ts.
    # Prefer the full-video replay matrix (always complete); fall back to the log's.
    source_result = out["replay"] or out["log"]
    matrix = None
    if source_result:
        m = source_result["confusion_matrix"]
        matrix = [[round(m[gt][pred] / FPS) for pred in metrics.LABELS] for gt in metrics.LABELS]
    out["confusion_matrix_source"] = "replay" if out["replay"] else ("log" if out["log"] else None)
    return out, matrix


def build_analytics(mock_override, scoring_level: str = metrics.DEFAULT_SCORING_LEVEL) -> dict:
    active_key, active_log_folder, active_version = resolve_active_run()

    kpis_by_business = {}
    trend_by_business = {}
    type_by_business = {}
    severity_by_business = {}
    words_by_business = {}
    people_by_business = {}
    confusion_by_business = {}
    narrative_coverage_by_business = {}
    active_eval_summary = None

    for key in metrics.BUSINESSES:
        groq_path = metrics.latest_log(key, "groq")
        if groq_path is None:
            continue  # no real logs for this business yet - leave it to mock fallback

        vlm_path = metrics.latest_log(key, "vlm")
        tracker_path = metrics.latest_log(key, "tracker")

        segments = metrics.non_degenerate(metrics.load_windows(groq_path))
        frames = metrics.load_tracker_frames(tracker_path)

        # KPIs/trend/severity are normalized to the FULL video (cue-replay), matching what
        # the confusion matrix already prefers - otherwise a live run stopped early (or a
        # groq worker that lagged the vlm/tracker workers) makes these widgets silently
        # describe a shorter span than the confusion matrix and the real video length.
        points, covered_start, covered_end = replay_points_for(vlm_path, tracker_path, scoring_level)
        if not points:
            # No vlm cue stream to replay - degrade to whatever the live log itself covers
            # rather than going blank.
            points = metrics.points_from_windows(segments)
            covered_start, covered_end = metrics.covered_range_from_windows(segments)

        kpis_by_business[key] = compute_kpis(points, covered_start, covered_end)
        trend_by_business[key] = compute_anomaly_trend(points, covered_start, covered_end)
        severity_by_business[key] = compute_severity(points, covered_start, covered_end)

        # Anomaly type / word frequencies keyword-match Groq's own narrated `reason` +
        # `key_moments` text, which only exists in the live groq log - cue-replay never
        # invents narration, so these two stay scoped to whatever that log actually covered
        # (see narrativeCoverageByBusiness below for how much of the video that is).
        type_by_business[key] = compute_anomaly_type(segments)
        words_by_business[key] = compute_word_frequencies(segments)

        people_by_business[key] = compute_people_count(frames)

        mock_path = mock_override if mock_override is not None else metrics.mock_path_for(key)
        summary, matrix = compute_eval_summary(
            key, mock_path, groq_path, vlm_path, tracker_path, scoring_level,
        )
        confusion_by_business[key] = matrix
        if summary and summary.get("log"):
            narrative_coverage_by_business[key] = summary["log"]["coverage"]

        if key == active_key:
            active_eval_summary = summary

    return {
        "schema_version": 2,
        "generated_at_unix_ms": int(time.time() * 1000),
        "source": {"business": active_log_folder, "version": active_version},
        "businesses": ALL_BUSINESSES,
        "kpisByBusiness": kpis_by_business,
        "anomalyTrendByBusiness": trend_by_business,
        "anomalyTypeByBusiness": type_by_business,
        "severityByBusiness": severity_by_business,
        "wordFrequenciesByBusiness": words_by_business,
        "peopleByBusiness": people_by_business,
        "confusionMatrixByBusiness": confusion_by_business,
        # Diagnostic only (nothing currently reads this): how much of the real video
        # anomalyType/wordFrequencies actually cover, since they're stuck on the live log.
        "narrativeCoverageByBusiness": narrative_coverage_by_business,
        "eval": active_eval_summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"Output JSON path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--mock", type=Path, default=None,
        help=(
            "Ground-truth mock JSONL for confusion matrix. If omitted (default), each "
            "business resolves its own mock at CRIMENO-Backend/mocks/<key>/groq_mock.jsonl. "
            "If given, this single path overrides ALL businesses (mainly for testing)."
        ),
    )
    parser.add_argument(
        "--scoring-level", choices=metrics.SCORING_LEVELS, default=metrics.DEFAULT_SCORING_LEVEL,
        help=f"Applied uniformly to every business's replay eval (not auto-detected - it "
             f"isn't persisted anywhere). Default: {metrics.DEFAULT_SCORING_LEVEL}",
    )
    args = parser.parse_args()

    analytics = build_analytics(args.mock, args.scoring_level)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(analytics, f, indent=2)

    print(f"Wrote analytics JSON to {args.out}")


if __name__ == "__main__":
    main()
