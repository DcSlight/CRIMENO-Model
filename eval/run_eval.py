"""
Evaluate the Groq anomaly-scoring pipeline against hand-labeled mock ground truth.

Ground truth source: CRIMENO-Backend/mocks/<store>/{vlm_mock.jsonl, groq_mock.jsonl}.
Each groq_mock.jsonl entry's {label, anomaly_score} for a given frame_range is treated
as the "correct" answer a well-tuned pipeline should produce.

Two modes:
  --fast (default): replays each mock's OWN label/score through the real apply_scoring()
                     clamp/bias function. Zero API calls, zero cost. This is a sanity/
                     regression check on the scoring math (Phase 1/4), NOT a measure of
                     prompt quality — it can't be, since it never calls the LLM.
  --live:            rebuilds the real prompt from each vlm_mock.jsonl entry and calls the
                     REAL Groq anomaly model, then compares its label/score against the
                     mock's ground truth. This is the only mode that measures whether a
                     PROMPT change actually improved accuracy. Costs Groq API tokens —
                     use deliberately, not as a default/repeated check.

Does NOT validate the VLM's prompt/quality — that requires real video + a real VLM call,
which is nondeterministic and not replayable from these static mocks. This harness only
validates the Groq reasoning + scoring layer.

Usage:
  py eval/run_eval.py                     # fast mode, no API calls
  py eval/run_eval.py --live              # calls real Groq API (costs tokens)
  py eval/run_eval.py --mocks-dir PATH    # override the default sibling-repo mocks path
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_GROQ_DIR = _REPO_ROOT / "groq"

# groq_anomaly_worker.py / event_builder.py are plain scripts (no package __init__.py),
# same import pattern they use for each other — add groq/ to sys.path before importing.
sys.path.insert(0, str(_GROQ_DIR))

from event_builder import build_event_sentence  # noqa: E402
from groq_anomaly_worker import (  # noqa: E402
    apply_scoring,
    build_prompt,
    build_scene_description,
    call_groq_for_anomaly,
    load_groq_client,
)

LABELS = ["normal", "suspicious", "criminal"]

DEFAULT_MOCKS_DIR = _REPO_ROOT.parent / "CRIMENO-Backend" / "mocks"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def discover_stores(mocks_dir: Path) -> List[Path]:
    if not mocks_dir.is_dir():
        raise FileNotFoundError(
            f"Mocks directory not found: {mocks_dir}\n"
            f"Pass --mocks-dir to point at CRIMENO-Backend/mocks."
        )
    stores = []
    for child in sorted(mocks_dir.iterdir()):
        if child.is_dir() and (child / "vlm_mock.jsonl").exists() and (child / "groq_mock.jsonl").exists():
            stores.append(child)
    return stores


def eval_store_fast(groq_mock: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Replay each mock's own label/score through apply_scoring(). No API calls."""
    rows = []
    for entry in groq_mock:
        result = entry.get("result", {})
        label = result.get("label", "unknown")
        raw_score = float(result.get("anomaly_score", 0.0))
        rescored = apply_scoring(label, raw_score)
        rows.append({
            "frame_range": entry.get("frame_range"),
            "label": label,
            "mock_score": raw_score,
            "rescored": rescored,
            "score_delta": abs(rescored - raw_score),
        })
    return {"mode": "fast", "rows": rows}


def eval_store_live(vlm_mock: List[Dict[str, Any]], groq_mock: List[Dict[str, Any]],
                    client, model_name: str) -> Dict[str, Any]:
    """Call the real Groq anomaly model on each mock VLM frame, compare vs. ground truth."""
    rows = []
    n = min(len(vlm_mock), len(groq_mock))
    if len(vlm_mock) != len(groq_mock):
        print(f"  [WARN] vlm_mock ({len(vlm_mock)}) and groq_mock ({len(groq_mock)}) "
              f"entry counts differ — comparing first {n} by position.")

    for i in range(n):
        vlm_entry = vlm_mock[i]
        gt = groq_mock[i].get("result", {})
        gt_label = gt.get("label", "unknown")
        gt_score = float(gt.get("anomaly_score", 0.0))

        rec = {"vlm": vlm_entry, "qa": vlm_entry.get("qa", {})}
        event_sentence = build_event_sentence(rec)
        scene_description = build_scene_description([], [event_sentence], "")
        prompt = build_prompt(scene_description)

        try:
            result = call_groq_for_anomaly(client, model_name, prompt)
        except Exception as e:
            print(f"  [ERROR] Groq call failed for frame {vlm_entry.get('frame_index')}: {e}")
            continue

        pred_label = result.get("label", "unknown")
        pred_score = apply_scoring(pred_label, float(result.get("anomaly_score", 0.0)))

        rows.append({
            "frame_index": vlm_entry.get("frame_index"),
            "gt_label": gt_label, "gt_score": gt_score,
            "pred_label": pred_label, "pred_score": pred_score,
            "correct": pred_label == gt_label,
            "score_abs_error": abs(pred_score - gt_score),
        })
    return {"mode": "live", "rows": rows}


def confusion_matrix(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    matrix = {gt: {pred: 0 for pred in LABELS + ["unknown"]} for gt in LABELS + ["unknown"]}
    for r in rows:
        gt, pred = r["gt_label"], r["pred_label"]
        matrix.setdefault(gt, {}).setdefault(pred, 0)
        matrix[gt][pred] += 1
    return matrix


def print_fast_report(store_name: str, result: Dict[str, Any]) -> None:
    rows = result["rows"]
    if not rows:
        print(f"  (no entries)")
        return
    max_delta = max(r["score_delta"] for r in rows)
    print(f"  {len(rows)} entries scored | max |rescored - mock_score| = {max_delta:.3f}")
    for r in rows:
        flag = "  " if r["score_delta"] < 1e-6 else " !"
        print(f"  {flag} frame_range={r['frame_range']} label={r['label']:<10} "
              f"mock={r['mock_score']:.2f} rescored={r['rescored']:.2f}")


def print_live_report(store_name: str, result: Dict[str, Any]) -> Tuple[int, int]:
    rows = result["rows"]
    if not rows:
        print("  (no entries)")
        return 0, 0
    correct = sum(1 for r in rows if r["correct"])
    total = len(rows)
    mae = sum(r["score_abs_error"] for r in rows) / total
    print(f"  Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)  "
          f"Mean abs score error: {mae:.3f}")

    matrix = confusion_matrix(rows)
    used_labels = [l for l in LABELS if any(r["gt_label"] == l or r["pred_label"] == l for r in rows)]
    header = "  gt\\pred".ljust(14) + "".join(l[:4].ljust(8) for l in used_labels)
    print(header)
    for gt in used_labels:
        line = f"  {gt}".ljust(14)
        for pred in used_labels:
            line += str(matrix.get(gt, {}).get(pred, 0)).ljust(8)
        print(line)

    for r in rows:
        if not r["correct"]:
            print(f"    [MISS] frame={r['frame_index']} gt={r['gt_label']} "
                  f"pred={r['pred_label']} (score gt={r['gt_score']:.2f} pred={r['pred_score']:.2f})")
    return correct, total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mocks-dir", type=Path, default=DEFAULT_MOCKS_DIR,
                        help=f"Directory containing per-store mock folders (default: {DEFAULT_MOCKS_DIR})")
    parser.add_argument("--live", action="store_true",
                        help="Call the real Groq API and measure label accuracy against ground truth. "
                             "Costs API tokens — omit for the free, deterministic clamp-only check.")
    parser.add_argument("--groq-api-key", default="", help="Groq API key (or set GROQ_API_KEY env var).")
    parser.add_argument("--groq-model", default="llama-3.3-70b-versatile",
                        help="Groq model to use in --live mode.")
    args = parser.parse_args()

    stores = discover_stores(args.mocks_dir)
    if not stores:
        print(f"No mock store folders found under {args.mocks_dir}")
        return

    print(f"Mode: {'LIVE (calls real Groq API — costs tokens)' if args.live else 'FAST (no API calls)'}")
    print(f"Stores found: {[s.name for s in stores]}\n")

    client = load_groq_client(args.groq_api_key) if args.live else None

    total_correct, total_n = 0, 0
    for store_dir in stores:
        print(f"=== {store_dir.name} ===")
        groq_mock = load_jsonl(store_dir / "groq_mock.jsonl")

        if args.live:
            vlm_mock = load_jsonl(store_dir / "vlm_mock.jsonl")
            result = eval_store_live(vlm_mock, groq_mock, client, args.groq_model)
            c, n = print_live_report(store_dir.name, result)
            total_correct += c
            total_n += n
        else:
            result = eval_store_fast(groq_mock)
            print_fast_report(store_dir.name, result)
        print()

    if args.live and total_n:
        print(f"=== Overall ===\nAccuracy: {total_correct}/{total_n} "
              f"({100 * total_correct / total_n:.1f}%)")


if __name__ == "__main__":
    main()
