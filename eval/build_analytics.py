"""
Build a single analytics.json snapshot from the REAL running-history logs of
every business in BUSINESS_MAP that has log data on disk, in the shapes the
CRIMENO dashboard's analytics widgets expect (see CRIMENO-Backend's
analytics.types.ts / analytics.mock.ts for the TS side).

Each business is computed independently from its own latest groq_vN.jsonl /
tracker_vN.jsonl, using its own ground-truth mock (CRIMENO-Backend/mocks/<key>/
groq_mock.jsonl by convention) for the confusion matrix. A business with no log
files yet is simply absent from the per-business dicts - the NestJS backend
falls back to mock data for any key missing there.

Pure stdlib only (json, argparse, pathlib, re, time, collections) plus a library
import of eval/score_logs.py (never modified/duplicated - imported as-is for
load_jsonl / find_mock_span / build_summary / LABELS).

Usage:
  py eval/build_analytics.py
  py eval/build_analytics.py --out PATH
  py eval/build_analytics.py --logs PATH --out PATH
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
import score_logs  # noqa: E402  (load_jsonl, find_mock_span, build_summary, LABELS)

REPO_ROOT = _HERE.parent
LOGS_ROOT = REPO_ROOT / "logs"
SESSION_FILE = LOGS_ROOT / "current_session.json"
DEFAULT_OUT = REPO_ROOT / "analytics.json"
MOCKS_ROOT = REPO_ROOT.parent / "CRIMENO-Backend" / "mocks"

# Single source of truth for business log-folder-name -> {key, name} used across
# the dashboard. Every business listed here gets real computed numbers IF it has
# log files on disk (checked at run time) - businesses with no logs yet are simply
# absent from the per-business dicts in the output, and NestJS falls back to mock
# data for any key missing there.
BUSINESS_MAP = {
    "jewerly_store_short": {"key": "jewelry", "name": "Jewelry Store"},
    "market": {"key": "market", "name": "Market"},
    # "shop": {"key": "market", "name": "Market"},
    # "supermarket_b": {"key": "gun_store", "name": "Gun Store"},
}

ALL_BUSINESSES = [
    {"key": "jewelry", "name": "Jewelry Store"},
    {"key": "market", "name": "Market"},
    {"key": "gun_store", "name": "Gun Store"},
]

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


def find_business_groq_dir(business: str) -> Path:
    return LOGS_ROOT / business / "groq"


def find_business_tracker_dir(business: str) -> Path:
    return LOGS_ROOT / business / "tracker"


def resolve_active_run(default_business: str = "jewerly_store_short"):
    """Return (business, version) for the run to analyze. Prefers
    logs/current_session.json's pointer if it names a business we know how to
    compute real data for; otherwise falls back to the highest vN found among
    that business's groq_vN.jsonl files."""
    business = default_business
    version = None

    session = None
    try:
        session = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except Exception:
        session = None

    if session and session.get("business") in BUSINESS_MAP:
        business = session["business"]
        version = session.get("version")

    if version is None:
        groq_dir = find_business_groq_dir(business)
        best = 0
        if groq_dir.is_dir():
            for p in groq_dir.glob("groq_v*.jsonl"):
                m = re.search(r"_v(\d+)\.jsonl$", p.name)
                if m:
                    best = max(best, int(m.group(1)))
        version = best if best else 1

    return business, version


def load_groq_segments(path: Path) -> list:
    """Parse a groq_vN.jsonl file into a list of raw dicts with the fields we
    need for analytics (kept separate from score_logs.load_jsonl, which
    normalizes to a different, narrower shape for scoring only)."""
    segments = []
    if not path.exists():
        return segments
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fr = obj.get("frame_range", {}) or {}
            res = obj.get("result", {}) or {}
            segments.append({
                "start": fr.get("start", 0),
                "end": fr.get("end", 0),
                "label": res.get("label", "unknown"),
                "score": float(res.get("anomaly_score", 0.0)),
                "reason": res.get("reason", "") or "",
                "key_moments": res.get("key_moments", []) or [],
            })
    return segments


def load_tracker_frames(path: Path) -> list:
    """Parse a tracker_vN.jsonl file, skipping reset markers
    (frame_index == -1 / reset: true)."""
    frames = []
    if not path.exists():
        return frames
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("reset") or obj.get("frame_index", 0) == -1:
                continue
            frames.append(obj)
    return frames


def compute_kpis(segments: list) -> dict:
    total = len(segments)
    criminal = sum(1 for s in segments if s["label"] == "criminal")
    avg_score = sum(s["score"] for s in segments) / total if total else 0.0
    return {
        "totalEvents": total,
        "criminalEvents": criminal,
        "avgAnomalyScore": avg_score,
        "activeBusinesses": 3,
        "alertsToday": criminal,
    }


def compute_anomaly_trend(segments: list) -> list:
    trend = []
    for s in segments:
        time_s = round(s["start"] / 30)
        point = {"time": f"{time_s}s", "normal": 0, "suspicious": 0, "criminal": 0}
        label = s["label"]
        if label in point:
            point[label] = round(s["score"] * 100)
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


def compute_severity(segments: list) -> list:
    counts = {label: 0 for label in score_logs.LABELS}
    for s in segments:
        if s["label"] in counts:
            counts[s["label"]] += 1
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


def compute_people_count(frames: list) -> int:
    if not frames:
        return 0
    return max(len(f.get("tracks", [])) for f in frames)


def compute_confusion_and_eval(mock_path: Path, real_path: Path):
    """Returns (confusion_matrix_3x3_or_None, eval_summary_dict_or_None)."""
    if not mock_path.exists():
        print(
            f"warning: mock file not found at {mock_path}; "
            "confusionMatrix and eval will be null",
            file=sys.stderr,
        )
        return None, None

    if not real_path.exists():
        print(
            f"warning: real groq log not found at {real_path}; "
            "confusionMatrix and eval will be null",
            file=sys.stderr,
        )
        return None, None

    mock_entries = score_logs.load_jsonl(mock_path)
    real_entries = score_logs.load_jsonl(real_path)

    if not mock_entries or not real_entries:
        print(
            "warning: mock or real log parsed empty; confusionMatrix and eval will be null",
            file=sys.stderr,
        )
        return None, None

    pairs = [
        (score_logs.find_mock_span(mock_entries, real["end"]), real)
        for real in real_entries
    ]
    summary = score_logs.build_summary(pairs)

    matrix_dict = summary["confusion_matrix"]
    matrix = [
        [matrix_dict[gt][pred] for pred in score_logs.LABELS]
        for gt in score_logs.LABELS
    ]

    eval_summary = {k: v for k, v in summary.items() if k != "similarities"}

    return matrix, eval_summary


def resolve_latest_version(log_business: str, logs_root: Path):
    """Highest available groq_vN.jsonl version for this business's logs, or None
    if that business has no log files at all yet."""
    groq_dir = logs_root / log_business / "groq"
    best = 0
    if groq_dir.is_dir():
        for p in groq_dir.glob("groq_v*.jsonl"):
            m = re.search(r"_v(\d+)\.jsonl$", p.name)
            if m:
                best = max(best, int(m.group(1)))
    return best if best else None


def build_analytics(logs_root: Path, mock_override) -> dict:
    active_business, active_version = resolve_active_run()

    kpis_by_business = {}
    trend_by_business = {}
    type_by_business = {}
    severity_by_business = {}
    words_by_business = {}
    people_by_business = {}
    confusion_by_business = {}
    active_eval_summary = None

    for log_business, info in BUSINESS_MAP.items():
        key = info["key"]
        version = resolve_latest_version(log_business, logs_root)
        if version is None:
            continue  # no real logs for this business yet - leave it to mock fallback

        groq_path = logs_root / log_business / "groq" / f"groq_v{version}.jsonl"
        tracker_path = logs_root / log_business / "tracker" / f"tracker_v{version}.jsonl"

        segments = load_groq_segments(groq_path)
        frames = load_tracker_frames(tracker_path)

        kpis_by_business[key] = compute_kpis(segments)
        trend_by_business[key] = compute_anomaly_trend(segments)
        type_by_business[key] = compute_anomaly_type(segments)
        severity_by_business[key] = compute_severity(segments)
        words_by_business[key] = compute_word_frequencies(segments)
        people_by_business[key] = compute_people_count(frames)

        mock_path = mock_override if mock_override is not None else (MOCKS_ROOT / key / "groq_mock.jsonl")
        matrix, summary = compute_confusion_and_eval(mock_path, groq_path)
        confusion_by_business[key] = matrix

        if log_business == active_business:
            active_eval_summary = summary

    return {
        "schema_version": 2,
        "generated_at_unix_ms": int(time.time() * 1000),
        "source": {"business": active_business, "version": active_version},
        "businesses": ALL_BUSINESSES,
        "kpisByBusiness": kpis_by_business,
        "anomalyTrendByBusiness": trend_by_business,
        "anomalyTypeByBusiness": type_by_business,
        "severityByBusiness": severity_by_business,
        "wordFrequenciesByBusiness": words_by_business,
        "peopleByBusiness": people_by_business,
        "confusionMatrixByBusiness": confusion_by_business,
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
        "--logs", type=Path, default=LOGS_ROOT,
        help=f"Logs root directory (default: {LOGS_ROOT})",
    )
    parser.add_argument(
        "--mock", type=Path, default=None,
        help=(
            "Ground-truth mock JSONL for confusion matrix. If omitted (default), "
            f"each business resolves its own mock at {MOCKS_ROOT}/<key>/groq_mock.jsonl. "
            "If given, this single path overrides ALL businesses (mainly for testing)."
        ),
    )
    args = parser.parse_args()

    analytics = build_analytics(args.logs, args.mock)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(analytics, f, indent=2)

    print(f"Wrote analytics JSON to {args.out}")


if __name__ == "__main__":
    main()
