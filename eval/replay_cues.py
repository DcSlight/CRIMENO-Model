"""Replay a VLM cue stream through groq/scoring.py and score it against ground truth.

Complements eval/score_logs.py. That harness compares an already-produced groq_vN.jsonl
against the mocks, so it can only measure a run that actually happened (and only over the
frames that run covered). This one drives the scorer directly from a vlm_*.jsonl cue stream,
which makes it possible to:

  - evaluate a scoring change WITHOUT re-running the whole live pipeline,
  - evaluate the full video even when the live run was stopped early,
  - replay CRIMENO-Backend/mocks' own vlm_mock.jsonl as a cue-level regression fixture.

Usage:
    py eval/replay_cues.py                          # all businesses, both cue sources
    py eval/replay_cues.py --business jewelry
    py eval/replay_cues.py --business jewelry --source real --verbose
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_MOCKS = _REPO.parent / "CRIMENO-Backend" / "mocks"

# scoring.py lives in groq/, whose package name collides with the installed `groq` SDK, so
# load it by path rather than importing it.
_spec = importlib.util.spec_from_file_location("crimeno_scoring", _REPO / "groq" / "scoring.py")
scoring = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scoring)

# Log folder -> mock folder. Mirrors eval/build_analytics.BUSINESS_MAP.
BUSINESSES = {
    "jewelry": "jewerly_store_short",
    "market": "market",
    "gun_store": "gun_store_robbery",
}

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

LABEL_ORDER = {"normal": 0, "suspicious": 1, "criminal": 2}


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


def load_ground_truth(mock_path: Path) -> List[Dict]:
    return [json.loads(l) for l in mock_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def gt_at(gt: List[Dict], frame: int) -> Tuple[Optional[str], Optional[float]]:
    for span in gt:
        if span["frame_range"]["start"] <= frame <= span["frame_range"]["end"]:
            return span["result"]["label"], span["result"]["anomaly_score"]
    return None, None


def latest_vlm_log(business_folder: str) -> Optional[Path]:
    vlm_dir = _REPO / "logs" / business_folder / "vlm"
    if not vlm_dir.is_dir():
        return None
    logs = sorted(vlm_dir.glob("vlm_v*.jsonl"),
                  key=lambda p: int("".join(c for c in p.stem.split("_v")[-1] if c.isdigit()) or 0))
    return logs[-1] if logs else None


def replay(vlm_path: Path, mock_path: Path, scoring_level: str,
           history: int = 15, verbose: bool = False) -> Dict:
    gt = load_ground_truth(mock_path)
    state = scoring.new_threat_state()
    cue_history: List[Dict[str, str]] = []
    exact = over = under = 0
    abs_error = 0.0
    rows = []

    for line in vlm_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        qa = rec.get("qa") or {}
        cue_history.append(cues_from_qa(qa))
        score, label, state = scoring.score_from_cues(
            cue_history[-history:], scoring_level=scoring_level, prior_state=state,
        )
        frame = rec.get("frame_index", 0)
        gt_label, gt_score = gt_at(gt, frame)
        if gt_label is None:
            continue
        delta = LABEL_ORDER[label] - LABEL_ORDER[gt_label]
        abs_error += abs(gt_score - score)
        if delta == 0:
            exact += 1
        elif delta > 0:
            over += 1
        else:
            under += 1
        rows.append((rec.get("video_time_ms", 0) / 1000, gt_label, gt_score,
                     label, score, state["level"], delta))

    total = max(1, exact + over + under)
    if verbose:
        print(f'{"t":>5} {"GROUND TRUTH":>19} {"PIPELINE":>19} {"latch":>6}  verdict')
        for t, gl, gs, l, s, lv, d in rows:
            verdict = "OK" if d == 0 else ("OVER" if d > 0 else "*** UNDER ***")
            print(f"{t:>5.0f} {gl:>12}{gs:>7.2f} {l:>12}{s:>7.2f} {lv:>6.2f}  {verdict}")
    return {
        "accuracy": exact / total, "over_call_count": over, "under_call_count": under,
        "score_mae": abs_error / total, "n": total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--business", choices=sorted(BUSINESSES), help="default: all")
    ap.add_argument("--source", choices=("mock", "real", "both"), default="both",
                    help="'mock' replays the hand-labeled vlm_mock.jsonl, 'real' the newest live log")
    ap.add_argument("--scoring-level", choices=sorted(scoring.SCORING_LEVEL_THRESHOLDS),
                    help="default: report every level")
    ap.add_argument("--history", type=int, default=15, help="cue-buffer depth (worker uses 15)")
    ap.add_argument("--verbose", action="store_true", help="print the per-frame table")
    args = ap.parse_args()

    keys = [args.business] if args.business else sorted(BUSINESSES)
    levels = [args.scoring_level] if args.scoring_level else sorted(scoring.SCORING_LEVEL_THRESHOLDS)
    sources = ["mock", "real"] if args.source == "both" else [args.source]

    for key in keys:
        mock_path = _MOCKS / key / "groq_mock.jsonl"
        if not mock_path.is_file():
            print(f"{key}: no ground truth at {mock_path} — skipped")
            continue
        for source in sources:
            if source == "mock":
                vlm_path = _MOCKS / key / "vlm_mock.jsonl"
            else:
                vlm_path = latest_vlm_log(BUSINESSES[key])
            if vlm_path is None or not vlm_path.is_file():
                print(f"{key:>10} {source:>5}: no cue stream — skipped")
                continue
            for level in levels:
                r = replay(vlm_path, mock_path, level, args.history, args.verbose)
                print(f"{key:>10} {source:>5} {level:>12}: "
                      f"acc={r['accuracy']:>4.0%} over={r['over_call_count']:>2} "
                      f"UNDER={r['under_call_count']:>2} mae={r['score_mae']:.3f} "
                      f"(n={r['n']}, {vlm_path.name})")


if __name__ == "__main__":
    main()
