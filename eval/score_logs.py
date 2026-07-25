"""
Compare the real Groq/scorer pipeline's output log against a hand-authored mock
("ground truth") log and print a simple scoring report.

Pure stdlib only (json, difflib, pathlib, argparse) - no groq/ imports, no API calls.

Both files are JSONL with one JSON object per line, shaped like:
  {"type": "groq_anomaly", "frame_range": {"start": 0, "end": 60},
   "result": {"anomaly_score": 0.05, "label": "normal", "reason": "...", "key_moments": [...]}}

The mock's frame_range spans are contiguous and non-overlapping (ground truth).
The real log's frame_range spans are overlapping sliding windows. Both use the
same frame-index axis for the same video, so alignment is a plain integer
comparison: for each real entry, find the mock span containing its `end` frame.

Usage:
  py eval/score_logs.py
  py eval/score_logs.py --mock PATH --logs PATH --json out.json
"""

import argparse
import difflib
import json
import sys
from pathlib import Path

LABELS = ["normal", "suspicious", "criminal"]
LABEL_ORDER = {"normal": 0, "suspicious": 1, "criminal": 2}
PARSE_FAILURE_REASON = "Failed to parse model JSON output"

_HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(_HERE.parent))
import session_log

DEFAULT_MOCK = _HERE.parent.parent / "CRIMENO-Backend" / "mocks" / "jewelry" / "groq_mock.jsonl"
# Prefer the most recent session-scoped groq log; fall back to the legacy flat file if
# no session has ever been recorded (e.g. session_log.py wasn't wired up yet, or no
# video has ever been played through the broadcaster).
DEFAULT_LOGS = session_log.latest_groq_log() or (_HERE.parent / "groq" / "logs_output.jsonl")


def load_jsonl(path: Path) -> list:
    """Parse a JSONL file into a list of {start, end, label, score, reason} dicts."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fr = obj.get("frame_range", {})
            res = obj.get("result", {})
            entries.append({
                "start": fr.get("start"),
                "end": fr.get("end"),
                "label": res.get("label", "unknown"),
                "score": float(res.get("anomaly_score", 0.0)),
                "reason": res.get("reason", ""),
            })
    return entries


def find_mock_span(mock_entries: list, frame: int) -> dict:
    """Find the mock entry whose [start, end] contains `frame`. If frame falls in a
    gap, fall back to the nearest mock span by distance to its closer edge."""
    for m in mock_entries:
        if m["start"] <= frame <= m["end"]:
            return m
    return min(mock_entries, key=lambda m: min(abs(frame - m["start"]), abs(frame - m["end"])))


def build_summary(pairs: list) -> dict:
    matrix = {gt: {pred: 0 for pred in LABELS} for gt in LABELS}
    under_calls = []
    over_calls = 0
    correct = 0
    score_errors = []
    similarities = []
    parse_failures = 0

    for mock, real in pairs:
        if mock["label"] in matrix and real["label"] in matrix[mock["label"]]:
            matrix[mock["label"]][real["label"]] += 1
        if mock["label"] == real["label"]:
            correct += 1

        mock_rank = LABEL_ORDER.get(mock["label"])
        real_rank = LABEL_ORDER.get(real["label"])
        if mock_rank is not None and real_rank is not None:
            if real_rank < mock_rank:
                under_calls.append((real["start"], real["end"], mock["label"], real["label"]))
            elif real_rank > mock_rank:
                over_calls += 1

        score_errors.append(abs(mock["score"] - real["score"]))
        similarities.append(difflib.SequenceMatcher(None, mock["reason"], real["reason"]).ratio())

        if real["reason"] == PARSE_FAILURE_REASON:
            parse_failures += 1

    total = len(pairs)
    return {
        "confusion_matrix": matrix,
        "accuracy": correct / total if total else 0.0,
        "under_call_count": len(under_calls),
        "under_calls": under_calls,
        "over_call_count": over_calls,
        "score_mae": sum(score_errors) / total if total else 0.0,
        "avg_similarity": sum(similarities) / total if total else 0.0,
        "parse_failure_count": parse_failures,
        "similarities": similarities,
    }


def print_report(pairs: list, summary: dict) -> None:
    print(f"{'frame_range':<16}{'mock_label':<14}{'mock_score':<12}"
          f"{'real_label':<14}{'real_score':<12}{'similarity':<10}")
    for (mock, real), sim in zip(pairs, summary["similarities"]):
        frame_range = f"{real['start']}-{real['end']}"
        print(f"{frame_range:<16}{mock['label']:<14}{mock['score']:<12.2f}"
              f"{real['label']:<14}{real['score']:<12.2f}{sim:<10.2f}")

    print("\n=== Summary ===")
    print("Confusion matrix (rows=mock, cols=real):")
    header = "".ljust(14) + "".join(l.ljust(12) for l in LABELS)
    print(header)
    for gt in LABELS:
        row = gt.ljust(14) + "".join(str(summary["confusion_matrix"][gt][pred]).ljust(12) for pred in LABELS)
        print(row)

    print(f"\nAccuracy: {summary['accuracy']:.2%} ({sum(1 for m, r in pairs if m['label'] == r['label'])}/{len(pairs)})")
    print(f"Under-calls (real less severe than mock): {summary['under_call_count']}")
    if summary["under_calls"]:
        for start, end, mock_label, real_label in summary["under_calls"]:
            print(f"    frame {start}-{end}: mock={mock_label} real={real_label}")
    else:
        print("    (none)")
    print(f"Over-calls (real more severe than mock): {summary['over_call_count']}")
    print(f"Score MAE: {summary['score_mae']:.3f}")
    print(f"Average reason similarity: {summary['avg_similarity']:.3f}")
    print(f"Parse-failure count: {summary['parse_failure_count']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", type=Path, default=DEFAULT_MOCK,
                         help=f"Ground-truth mock JSONL (default: {DEFAULT_MOCK})")
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS,
                         help=f"Real pipeline output JSONL (default: {DEFAULT_LOGS})")
    parser.add_argument("--json", type=Path, default=None,
                         help="Optional path to also write summary metrics as JSON")
    args = parser.parse_args()

    mock_entries = load_jsonl(args.mock)
    real_entries = load_jsonl(args.logs)

    pairs = [(find_mock_span(mock_entries, real["end"]), real) for real in real_entries]

    summary = build_summary(pairs)
    print_report(pairs, summary)

    if args.json:
        out = {k: v for k, v in summary.items() if k != "similarities"}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote JSON summary to {args.json}")


if __name__ == "__main__":
    main()
