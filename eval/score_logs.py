"""
Score the CRIMENO pipeline against hand-labeled ground truth: accuracy, MAE, and the
confusion matrix, graded by EXACT FRAME OVERLAP against the mock's ground-truth rows (not
a single point sample per row) - see eval/metrics.py's evaluate() docstring for why: the
mock's rows and groq_vN.jsonl's sliding windows are out of phase, so one real row commonly
straddles 2-3 mock rows, and every overlapping frame needs to count against whichever label
predicted it.

Two things are reported for a business, back to back:

  1. REPLAY (headline) - the VLM cue stream (vlm_vN.jsonl) replayed straight through
     groq/scoring.py. Covers the WHOLE video, so ground truth outside whatever the live
     groq worker reached still gets a real prediction instead of reading as a silent zero.
  2. GROQ LOG (secondary) - the shipped groq_vN.jsonl exactly as produced, with an explicit
     coverage % - this is "what actually ran", at whatever fraction of the video that run
     reached.

`scoring_level` (conservative/balanced/aggressive) is NOT looked up anywhere automatically -
it isn't persisted in any log (the live worker pulls it from a websocket business-context
string at runtime). Pass --scoring-level yourself if you know what the business runs at;
it defaults to "balanced" otherwise. Use --sweep to see all three at once.

Neither groq/scoring.py nor any vlm/ file is modified or duplicated by this module - the
scorer is loaded read-only, by path, exactly as the live worker loads it.

Usage:
  py eval/score_logs.py --business jewelry
  py eval/score_logs.py --business jewelry --source log
  py eval/score_logs.py --business jewelry --scoring-level aggressive
  py eval/score_logs.py --business jewelry --sweep
  py eval/score_logs.py --business jewelry --verbose
  py eval/score_logs.py --business jewelry --json out.json
  py eval/score_logs.py --mock PATH --logs PATH        # explicit paths, log-source only
"""

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
import metrics  # noqa: E402
import session_log  # noqa: E402

FPS = 30


def _secs(frames: int) -> float:
    return round(frames / FPS, 1)


def print_matrix(matrix) -> None:
    print("Confusion matrix (rows=ground truth, cols=predicted, cell = overlapping seconds):")
    header = "".ljust(14) + "".join(l.ljust(12) for l in metrics.LABELS)
    print(header)
    for gt in metrics.LABELS:
        row = gt.ljust(14) + "".join(f"{_secs(matrix[gt][pred])}s".ljust(12) for pred in metrics.LABELS)
        print(row)


def print_row_overlaps(real_windows, gt_windows) -> None:
    """--verbose diagnostic: for each real row, exactly which mock row(s) it straddles."""
    print("\nPer-row overlap (one real row can touch 2-3 mock rows when the window schemes "
          "are out of phase):")
    for row in metrics.row_overlap_table(real_windows, gt_windows):
        touches = ", ".join(
            f"[{t['gt_start']}-{t['gt_end']}]={t['gt_label']} ({t['overlap_frames']}f)"
            for t in row["mock_rows_touched"]
        )
        print(f"  real [{row['start']}-{row['end']}]={row['label']}  ->  {touches or '(no overlap)'}")


def print_block(title: str, result, cov, extra=None, verbose_calls=False) -> None:
    print(f"\n--- {title} ---")
    if result is None:
        print("  (no ground truth to grade against)")
        return
    print_matrix(result["confusion_matrix"])
    if any(result["not_covered"].values()):
        parts = ", ".join(f"{k}={v}f ({_secs(v)}s)" for k, v in result["not_covered"].items() if v)
        print(f"Not covered (ground truth this source never reached): {parts}")
    if cov and cov["covered_range"] is not None:
        c0, c1 = cov["covered_range"]
        m0, m1 = cov["mock_full_range"]
        note = "" if cov["coverage_pct"] >= 0.999 else " - partial run"
        print(f"Coverage: frames {c0}-{c1} of {m0}-{m1} ({cov['coverage_pct']:.0%}){note}")
    print(f"Accuracy: {result['accuracy']:.2%}  "
          f"({result['compared_frames']}/{result['total_frames']} frames graded, "
          f"{_secs(result['compared_frames'])}/{_secs(result['total_frames'])}s)")
    print(f"Score MAE: {result['score_mae']:.3f}")
    print(f"Over-called frames (predicted more severe than ground truth): "
          f"{result['over_call_count']} ({_secs(result['over_call_count'])}s)")
    print(f"Under-called frames (predicted less severe than ground truth): "
          f"{result['under_call_count']} ({_secs(result['under_call_count'])}s)")
    if verbose_calls and result["under_calls"]:
        for lo, hi, gt_label, pred_label, overlap in result["under_calls"]:
            print(f"    frames {lo}-{hi} ({overlap}f): gt={gt_label} pred={pred_label}")
    if extra and extra.get("avg_similarity") is not None:
        print(f"Average reason similarity: {extra['avg_similarity']:.3f}")
        print(f"Parse-failure count: {extra['parse_failure_count']}")


def run_business(business_key: str, source: str, levels, verbose: bool) -> dict:
    mock_path = metrics.mock_path_for(business_key)
    if not mock_path.is_file():
        print(f"{business_key}: no ground truth at {mock_path} - skipped")
        return {}

    gt_windows = metrics.load_windows(mock_path)
    mock_start, mock_end = metrics.gt_range(gt_windows)

    vlm_path = metrics.latest_log(business_key, "vlm")
    tracker_path = metrics.latest_log(business_key, "tracker")
    groq_path = metrics.latest_log(business_key, "groq")
    tracker_frames = metrics.load_tracker_frames(tracker_path)

    json_out = {"business": business_key, "levels": {}}

    for level in levels:
        print(f"\n=== {business_key}  scoring_level={level} ===")
        level_out = {}

        if source in ("replay", "both"):
            if vlm_path is None:
                print("\n--- REPLAY ---\n  no vlm cue stream - skipped")
            else:
                vlm_records = metrics.load_vlm_records(vlm_path)
                points = metrics.points_from_replay(vlm_records, tracker_frames, level)
                covered_start, covered_end = metrics.points_range(points)
                result = metrics.evaluate(gt_windows, points, covered_start, covered_end)
                cov = metrics.coverage(covered_start, covered_end, mock_start, mock_end)
                print_block(f"REPLAY  ({vlm_path.name})", result, cov, verbose_calls=verbose)
                level_out["replay"] = {k: v for k, v in (result or {}).items()
                                        if k not in ("under_calls", "over_calls")}
                level_out["replay"]["coverage"] = cov

        if source in ("log", "both"):
            if groq_path is None:
                print("\n--- GROQ LOG ---\n  no groq_vN.jsonl - skipped")
            else:
                real_windows = metrics.non_degenerate(metrics.load_windows(groq_path))
                points = metrics.points_from_windows(real_windows)
                covered_start, covered_end = metrics.covered_range_from_windows(real_windows)
                result = metrics.evaluate(gt_windows, points, covered_start, covered_end)
                cov = metrics.coverage(covered_start, covered_end, mock_start, mock_end)
                sim = metrics.text_similarity_stats(gt_windows, real_windows)
                print_block(f"GROQ LOG  ({groq_path.name})", result, cov, extra=sim, verbose_calls=verbose)
                if verbose:
                    print_row_overlaps(real_windows, gt_windows)
                level_out["log"] = {k: v for k, v in (result or {}).items()
                                     if k not in ("under_calls", "over_calls")}
                level_out["log"]["coverage"] = cov
                level_out["log"]["text_similarity"] = sim

        json_out["levels"][level] = level_out

    return json_out


def run_explicit(mock_path: Path, logs_path: Path, verbose: bool) -> dict:
    """Legacy power-user path: explicit --mock/--logs, no --business, log-source only."""
    gt_windows = metrics.load_windows(mock_path)
    mock_start, mock_end = metrics.gt_range(gt_windows)
    real_windows = metrics.non_degenerate(metrics.load_windows(logs_path))
    points = metrics.points_from_windows(real_windows)
    covered_start, covered_end = metrics.covered_range_from_windows(real_windows)
    result = metrics.evaluate(gt_windows, points, covered_start, covered_end)
    cov = metrics.coverage(covered_start, covered_end, mock_start, mock_end)
    sim = metrics.text_similarity_stats(gt_windows, real_windows)
    print(f"\n=== mock={mock_path}  logs={logs_path} ===")
    print_block("GROQ LOG", result, cov, extra=sim, verbose_calls=verbose)
    if verbose:
        print_row_overlaps(real_windows, gt_windows)
    out = {k: v for k, v in (result or {}).items() if k not in ("under_calls", "over_calls")}
    out["coverage"] = cov
    out["text_similarity"] = sim
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--business", choices=sorted(metrics.BUSINESSES), default=None,
                         help="Business to score (default: all, or the current session's business "
                              "if --mock/--logs aren't given either)")
    parser.add_argument("--scoring-level", choices=metrics.SCORING_LEVELS,
                         default=metrics.DEFAULT_SCORING_LEVEL,
                         help=f"conservative/balanced/aggressive - not auto-detected (it isn't "
                              f"persisted anywhere), pass what the business actually runs at. "
                              f"Default: {metrics.DEFAULT_SCORING_LEVEL}")
    parser.add_argument("--sweep", action="store_true",
                         help="Report all three scoring levels instead of just --scoring-level")
    parser.add_argument("--source", choices=("replay", "log", "both"), default="replay",
                         help="'replay' (default): one accuracy number, from replaying the "
                              "full video through the scorer. 'log': grade only the shipped "
                              "groq_vN.jsonl instead. 'both': show both side by side.")
    parser.add_argument("--verbose", action="store_true",
                         help="List under-call frames and, for the log source, the per-row "
                              "mock-overlap breakdown")
    parser.add_argument("--json", type=Path, default=None, help="Also write the summary as JSON")
    parser.add_argument("--mock", type=Path, default=None, help="Explicit ground-truth mock JSONL")
    parser.add_argument("--logs", type=Path, default=None, help="Explicit real pipeline output JSONL")
    args = parser.parse_args()

    if args.mock or args.logs:
        mock_path = args.mock or metrics.mock_path_for("jewelry")
        logs_path = args.logs or session_log.latest_groq_log() or (_HERE.parent / "groq" / "logs_output.jsonl")
        out = run_explicit(mock_path, logs_path, args.verbose)
        if args.json:
            args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"\nWrote JSON summary to {args.json}")
        return

    if args.business:
        keys = [args.business]
    else:
        session = session_log.read_session()
        default_key = metrics.key_for_log_folder(session["business"]) if session else None
        keys = [default_key] if default_key else sorted(metrics.BUSINESSES)

    levels = metrics.SCORING_LEVELS if args.sweep else [args.scoring_level]

    all_out = {}
    for key in keys:
        print(f"\n########## {key} ##########")
        all_out[key] = run_business(key, args.source, levels, args.verbose)

    if args.json:
        args.json.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
        print(f"\nWrote JSON summary to {args.json}")


if __name__ == "__main__":
    main()
