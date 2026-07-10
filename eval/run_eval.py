"""
Evaluate the Groq anomaly-scoring pipeline against hand-labeled mock ground truth.

Ground truth source: CRIMENO-Backend/mocks/<store>/{vlm_mock.jsonl, groq_mock.jsonl}.
Each groq_mock.jsonl entry's {label, anomaly_score} for a given frame_range is treated
as the "correct" answer a well-tuned pipeline should produce.

As of the "make Groq narrate, code scores" rework, anomaly_score/label are no longer
decided by the LLM — they're computed deterministically from the VLM's cue history by
groq/scoring.py (score_from_cues). Groq's only remaining job is the `reason`/`key_moments`
narrative. So the two modes below now test two different things:

  --fast (default): replays each store's vlm_mock.jsonl cue sequence through the REAL
                     score_from_cues() (the same code path the live worker uses) and
                     compares the resulting label/score against groq_mock.jsonl's ground
                     truth. Zero API calls, zero cost — this is the real accuracy check
                     for the scoring logic (weights/persistence/criminal-gate), since
                     scoring is deterministic code, not an LLM call.
  --live:            does everything --fast does, PLUS calls the real Groq model for each
                     step to fetch its narrative `reason`, printed next to the mock's
                     ground-truth reason so you can eyeball whether the prose reads like
                     an analyst narrating a trajectory or like a robot echoing a cue.
                     There is no "accuracy" number for prose — it's a manual spot-check.
                     Costs Groq API tokens — use deliberately, not as a default/repeated
                     check.

Does NOT validate the VLM's own prompt/quality — that requires real video + a real VLM
call, which is nondeterministic and not replayable from these static mocks.

Usage:
  py eval/run_eval.py                     # fast mode, no API calls
  py eval/run_eval.py --live              # calls real Groq API (costs tokens)
  py eval/run_eval.py --mocks-dir PATH    # override the default sibling-repo mocks path
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_GROQ_DIR = _REPO_ROOT / "groq"

# groq_anomaly_worker.py / event_builder.py / scoring.py are plain scripts (no package
# __init__.py), same import pattern they use for each other — add groq/ to sys.path.
sys.path.insert(0, str(_GROQ_DIR))

from event_builder import build_event_sentence, CUE_LABELS  # noqa: E402
from scoring import score_from_cues  # noqa: E402
from groq_anomaly_worker import (  # noqa: E402
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


# Some older mocks (gun_store, market) predate the strict yes/no/unclear 3-state VLM
# schema (vlm/prompt.txt) and used informal free-text-ish values instead. Map the ones
# that are unambiguously equivalent to a real 3-state value. Deliberately NOT mapped:
# "yes_contextual" (gun_store's "firearms visible as expected store inventory" — a real,
# intentional distinction from an actively-wielded/stolen weapon; collapsing it to "yes"
# would wrongly fire criminal on frame 0 of a gun store just for having guns on the wall).
_LEGACY_VALUE_ALIASES = {
    "partial": "unclear",
    "possible": "unclear",
    "yes_stolen": "yes",
    "yes_stolen_multiple": "yes",
}


def _extract_raw_cues(qa: Dict[str, Any]) -> Dict[str, str]:
    """Raw 3-state cue dict for one VLM mock entry, with legacy-format shims (see
    _LEGACY_VALUE_ALIASES above) plus a field-rename shim: some mocks (e.g.
    CRIMENO-Backend/mocks/jewelry) predate the reaching_counter ->
    reaching_display_case/reaching_behind_counter split and only have a single
    "reaching_counter" field. Since that field's "yes" in the jewelry mock specifically
    narrates suspects AT the counter (not customers browsing a display case), alias it
    to the more security-relevant reaching_behind_counter for evaluation purposes only —
    these shims live here, not in scoring.py, so production scoring stays schema-pure
    (the real VLM only ever emits strict yes/no/unclear per the current prompt.txt).
    """
    raw_cues = {k: str(qa.get(k, "")).strip().lower() for k in CUE_LABELS}
    raw_cues = {k: _LEGACY_VALUE_ALIASES.get(v, v) for k, v in raw_cues.items()}
    if "reaching_counter" in qa and not qa.get("reaching_behind_counter"):
        legacy_val = str(qa.get("reaching_counter", "")).strip().lower()
        raw_cues["reaching_behind_counter"] = _LEGACY_VALUE_ALIASES.get(legacy_val, legacy_val)
    return raw_cues


def eval_store(
    vlm_mock: List[Dict[str, Any]],
    groq_mock: List[Dict[str, Any]],
    live: bool = False,
    client=None,
    model_name: str = "",
) -> Dict[str, Any]:
    """Replay a store's vlm_mock.jsonl cue sequence through the real score_from_cues(),
    simulating the live worker's growing cue history, and compare against groq_mock.jsonl
    ground truth. In --live mode, also call the real Groq model for the narrative `reason`
    at each step (extra, costs tokens) purely for a side-by-side prose spot-check.
    """
    rows = []
    n = min(len(vlm_mock), len(groq_mock))
    if len(vlm_mock) != len(groq_mock):
        print(f"  [WARN] vlm_mock ({len(vlm_mock)}) and groq_mock ({len(groq_mock)}) "
              f"entry counts differ — comparing first {n} by position.")

    cue_history: List[Dict[str, str]] = []
    earlier_events: List[Dict[str, Any]] = []

    for i in range(n):
        vlm_entry = vlm_mock[i]
        gt = groq_mock[i].get("result", {})
        gt_label = gt.get("label", "unknown")
        gt_score = float(gt.get("anomaly_score", 0.0))
        frame_index = vlm_entry.get("frame_index")

        raw_cues = _extract_raw_cues(vlm_entry.get("qa", {}))
        cue_history.append(raw_cues)

        pred_score, pred_label = score_from_cues(cue_history)

        row: Dict[str, Any] = {
            "frame_index": frame_index,
            "gt_label": gt_label, "gt_score": gt_score, "gt_reason": gt.get("reason", ""),
            "pred_label": pred_label, "pred_score": pred_score,
            "correct": pred_label == gt_label,
            "score_abs_error": abs(pred_score - gt_score),
        }

        if live and client is not None:
            rec = {"vlm": vlm_entry, "qa": vlm_entry.get("qa", {})}
            event_sentence = build_event_sentence(rec)
            now_event = {"frame_index": frame_index, "text": event_sentence}
            scene_description = build_scene_description(earlier_events, [now_event], "")
            prompt = build_prompt(scene_description)
            try:
                groq_result = call_groq_for_anomaly(client, model_name, prompt)
                row["groq_reason"] = groq_result.get("reason", "")
            except Exception as e:
                print(f"  [ERROR] Groq call failed for frame {frame_index}: {e}")
                row["groq_reason"] = ""
            earlier_events.append(now_event)
            if len(earlier_events) > 10:
                earlier_events = earlier_events[-10:]

        rows.append(row)

    return {"mode": "live" if live else "fast", "rows": rows}


def confusion_matrix(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    matrix = {gt: {pred: 0 for pred in LABELS + ["unknown"]} for gt in LABELS + ["unknown"]}
    for r in rows:
        gt, pred = r["gt_label"], r["pred_label"]
        matrix.setdefault(gt, {}).setdefault(pred, 0)
        matrix[gt][pred] += 1
    return matrix


def print_report(store_name: str, result: Dict[str, Any]) -> "tuple[int, int]":
    rows = result["rows"]
    live = result["mode"] == "live"
    if not rows:
        print("  (no entries)")
        return 0, 0

    correct = sum(1 for r in rows if r["correct"])
    total = len(rows)
    mae = sum(r["score_abs_error"] for r in rows) / total
    print(f"  Deterministic scoring accuracy: {correct}/{total} ({100 * correct / total:.1f}%)  "
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

    if live:
        print("\n  --- Narrative spot-check (Groq reason vs. mock ground-truth reason) ---")
        for r in rows:
            print(f"    frame={r['frame_index']}")
            print(f"      mock : {r['gt_reason']}")
            print(f"      groq : {r.get('groq_reason', '')}")

    return correct, total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mocks-dir", type=Path, default=DEFAULT_MOCKS_DIR,
                        help=f"Directory containing per-store mock folders (default: {DEFAULT_MOCKS_DIR})")
    parser.add_argument("--live", action="store_true",
                        help="Also call the real Groq API for each step to spot-check the "
                             "narrative `reason` against the mock's. Costs API tokens — omit "
                             "for the free, deterministic scoring-only check.")
    parser.add_argument("--groq-api-key", default="", help="Groq API key (or set GROQ_API_KEY env var).")
    parser.add_argument("--groq-model", default="llama-3.3-70b-versatile",
                        help="Groq model to use in --live mode.")
    args = parser.parse_args()

    stores = discover_stores(args.mocks_dir)
    if not stores:
        print(f"No mock store folders found under {args.mocks_dir}")
        return

    print(f"Mode: {'LIVE (scoring + real Groq narrative — costs tokens)' if args.live else 'FAST (deterministic scoring only, no API calls)'}")
    print(f"Stores found: {[s.name for s in stores]}\n")

    client = load_groq_client(args.groq_api_key) if args.live else None

    total_correct, total_n = 0, 0
    for store_dir in stores:
        print(f"=== {store_dir.name} ===")
        vlm_mock = load_jsonl(store_dir / "vlm_mock.jsonl")
        groq_mock = load_jsonl(store_dir / "groq_mock.jsonl")

        result = eval_store(vlm_mock, groq_mock, live=args.live, client=client, model_name=args.groq_model)
        c, n = print_report(store_dir.name, result)
        total_correct += c
        total_n += n
        print()

    if total_n:
        print(f"=== Overall ===\nDeterministic scoring accuracy: {total_correct}/{total_n} "
              f"({100 * total_correct / total_n:.1f}%)")


if __name__ == "__main__":
    main()
