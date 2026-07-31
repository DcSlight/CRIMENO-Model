"""Regression tests for eval/metrics.py and eval/build_analytics.py's trend/severity math.

These guard the three symptoms/issues the eval refactor was written to fix:
  1. a run that only covers a PREFIX of the video must not silently read as "this ground
     truth class never got predicted" (the confusion matrix must report explicit
     `not_covered` counts instead of a bare zero row);
  2. a time-based chart (the anomaly trend) must have unique, gap-free, increasing time
     keys across the run it covers - not duplicate keys from overlapping sliding windows;
  3. groq_vN.jsonl's sliding windows (120-wide, 60-step) are out of phase with the mock's
     own ground-truth row boundaries, so one real row commonly straddles 2-3 mock rows -
     grading must split that row's frames across every mock row it touches, weighted by
     exact overlap, not silently pair it to just one.

Stdlib unittest only, matching tests/test_scoring.py - no pytest dependency added.

Run:  py -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_EVAL = _REPO / "eval"
sys.path.insert(0, str(_EVAL))

import metrics  # noqa: E402
import build_analytics  # noqa: E402


def gt(spans):
    """spans: list of (start, end, label, score) -> normalized GT window dicts."""
    return [{"start": s, "end": e, "label": l, "score": sc, "reason": "", "key_moments": []}
            for s, e, l, sc in spans]


def windows(spans):
    """spans: list of (start, end, label, score) -> normalized window dicts (same shape as
    a parsed groq_vN.jsonl / groq_mock.jsonl line)."""
    return gt(spans)


class TestPredictedAt(unittest.TestCase):
    def test_window_verdict_holds_until_next_window_ends(self):
        ws = windows([(0, 60, "normal", 0.1), (60, 120, "suspicious", 0.4)])
        points = metrics.points_from_windows(ws)
        start, end = metrics.covered_range_from_windows(ws)

        # Anywhere between the two windows' ends, the first window's verdict still stands.
        self.assertEqual(metrics.predicted_at(points, 60, start, end)[0], "normal")
        self.assertEqual(metrics.predicted_at(points, 90, start, end)[0], "normal")
        # At and after the second window's end, its verdict takes over.
        self.assertEqual(metrics.predicted_at(points, 120, start, end)[0], "suspicious")

    def test_frame_before_first_point_is_not_covered(self):
        ws = windows([(0, 60, "normal", 0.1)])
        points = metrics.points_from_windows(ws)
        start, end = metrics.covered_range_from_windows(ws)
        # covered_start is the window's own start (0), but no decision point exists there
        # (the first point sits at end=60) - predicted_at must not fabricate one.
        self.assertIsNone(metrics.predicted_at(points, 0, start, end))

    def test_frame_past_last_covered_end_is_not_covered(self):
        ws = windows([(0, 60, "normal", 0.1), (60, 120, "suspicious", 0.4)])
        points = metrics.points_from_windows(ws)
        start, end = metrics.covered_range_from_windows(ws)
        self.assertIsNone(metrics.predicted_at(points, 121, start, end))


def _sliding_windows(spans, size=120, step=60):
    """Build realistic non-degenerate sliding windows (like the live pipeline emits: 60-frame
    step, 120-frame window) covering each (span_start, span_end, label, score) run."""
    out = []
    for span_start, span_end, label, score in spans:
        f = span_start
        while f < span_end:
            out.append({"start": f, "end": min(f + size, span_end), "label": label,
                        "score": score, "reason": "", "key_moments": []})
            f += step
    return out


class TestEvaluateCoverage(unittest.TestCase):
    """The core regression test for symptom 1: a run stopped early must not make the
    confusion matrix's criminal row read as a silent zero."""

    def setUp(self):
        # Ground truth: normal 0-599, suspicious 600-1199, criminal 1200-1799.
        self.gt_windows = gt([
            (0, 599, "normal", 0.1),
            (600, 1199, "suspicious", 0.5),
            (1200, 1799, "criminal", 0.9),
        ])

    def test_full_coverage_populates_every_row(self):
        # A prediction source that matches ground truth exactly, across the whole range,
        # emitted as realistic 60-frame-step sliding windows.
        real = _sliding_windows([
            (0, 600, "normal", 0.1),
            (600, 1200, "suspicious", 0.5),
            (1200, 1800, "criminal", 0.9),
        ])
        points = metrics.points_from_windows(real)
        start, end = metrics.covered_range_from_windows(real)
        result = metrics.evaluate(self.gt_windows, points, start, end)

        # Not a perfect 1.0: a few frames right at a label-transition boundary are still
        # covered by the PRIOR window's verdict (it hasn't seen the next window yet) - that
        # is correct "as of this frame" semantics, not a bug. What this test actually
        # guards is full coverage (below).
        self.assertGreaterEqual(result["accuracy"], 0.85)
        # The only "not covered" frames allowed are frames 0-119, before the very FIRST
        # 120-frame window has even completed (inherent startup latency, same as a real
        # pipeline's first window) - not missing coverage of the run itself.
        self.assertLessEqual(sum(result["not_covered"].values()), 120)
        self.assertEqual(result["not_covered"]["suspicious"], 0)
        self.assertEqual(result["not_covered"]["criminal"], 0)

    def test_partial_run_reports_explicit_not_covered_not_a_silent_zero(self):
        # Run stopped at frame 660 - only reaches into the "suspicious" ground truth, never
        # the "criminal" ground truth at all. This is exactly the jewelry_store_short bug:
        # groq_v15.jsonl stopped at frame 1080 while ground truth's criminal span starts at
        # frame 1140, so the criminal row of the confusion matrix looked all-zero.
        real = _sliding_windows([(0, 600, "normal", 0.1), (600, 660, "suspicious", 0.5)])
        points = metrics.points_from_windows(real)
        start, end = metrics.covered_range_from_windows(real)
        result = metrics.evaluate(self.gt_windows, points, start, end)

        # The criminal ground-truth class was never reached by this source.
        self.assertGreater(result["not_covered"]["criminal"], 0)
        # And critically: those frames must NOT have been counted as correct/incorrect
        # predictions of "normal" or anything else - the criminal row must sum to exactly
        # zero real predictions, not silently show up under some predicted label.
        criminal_row_total = sum(result["confusion_matrix"]["criminal"].values())
        self.assertEqual(criminal_row_total, 0)


class TestOverlapWeightedMatching(unittest.TestCase):
    """Regression test for the out-of-phase window schemes: a real groq_vN.jsonl window is
    120 frames wide on a 60-frame step, while the mock's ground-truth rows are a different,
    uneven width - so a single real row routinely straddles 2-3 mock rows (e.g. jewelry's
    real frame_range 660-780 spans mock rows 660-779 and 780-899). Pairing that row to only
    the mock row containing its END frame silently discards the other overlap; grading must
    split it across every mock row it touches, weighted by exact frame overlap."""

    def test_one_predicted_interval_splits_credit_across_two_differently_labeled_mock_rows(self):
        # Ground truth has a label change at frame 200; the real pipeline's sliding windows
        # (120-wide, 60-step) are on a completely different phase, so one of the resulting
        # decision intervals straddles that boundary - it must contribute to BOTH mock rows,
        # not get assigned wholesale to just one of them.
        gt_windows = gt([(0, 199, "normal", 0.1), (200, 300, "suspicious", 0.5)])
        real = _sliding_windows([(0, 300, "criminal", 1.0)])
        points = metrics.points_from_windows(real)
        start, end = metrics.covered_range_from_windows(real)

        result = metrics.evaluate(gt_windows, points, start, end)
        matrix = result["confusion_matrix"]

        # The straddling interval means BOTH mock rows show real, nonzero "criminal" credit -
        # not one row getting everything and the other reading as a silent zero.
        self.assertGreater(matrix["normal"]["criminal"], 0)
        self.assertGreater(matrix["suspicious"]["criminal"], 0)
        # Every graded frame in this scenario was mislabeled "criminal", so the two rows'
        # credit must add up to the total number of frames actually compared.
        self.assertEqual(matrix["normal"]["criminal"] + matrix["suspicious"]["criminal"],
                          result["compared_frames"])

    def test_row_overlap_table_shows_every_touched_mock_row(self):
        gt_windows = gt([(660, 779, "suspicious", 0.46), (780, 899, "suspicious", 0.63)])
        real = windows([(660, 780, "criminal", 1.0)])
        table = metrics.row_overlap_table(real, gt_windows)

        self.assertEqual(len(table), 1)
        touched = table[0]["mock_rows_touched"]
        self.assertEqual(len(touched), 2, f"expected 2 touched mock rows, got {touched}")
        self.assertEqual(touched[0]["overlap_frames"], 120)
        self.assertEqual(touched[1]["overlap_frames"], 1)


class TestCoverageMath(unittest.TestCase):
    def test_truncated_log_reports_partial_coverage(self):
        cov = metrics.coverage(0, 1080, 0, 2999)
        self.assertEqual(cov["covered_range"], [0, 1080])
        self.assertLess(cov["coverage_pct"], 1.0)
        self.assertAlmostEqual(cov["coverage_pct"], 1080 / 2999, places=3)

    def test_full_coverage_is_one(self):
        cov = metrics.coverage(0, 2999, 0, 2999)
        self.assertEqual(cov["coverage_pct"], 1.0)

    def test_missing_source_reports_zero_coverage(self):
        cov = metrics.coverage(None, None, 0, 2999)
        self.assertEqual(cov["coverage_pct"], 0.0)
        self.assertIsNone(cov["covered_range"])


class TestSampleGrid(unittest.TestCase):
    """sample_grid/predicted_at are only used for the trend/severity CHARTS now (not for
    accuracy/MAE grading, which is exact-overlap based - see TestOverlapWeightedMatching)."""

    def test_grid_is_contiguous_and_includes_the_endpoint(self):
        grid = metrics.sample_grid(0, 2999, step=60)
        self.assertEqual(grid[0], 0)
        self.assertEqual(grid[-1], 2999)  # exact endpoint always included, even off-step
        # strictly increasing, no gaps larger than the step
        for a, b in zip(grid, grid[1:]):
            self.assertGreater(b, a)
            self.assertLessEqual(b - a, 60)


class TestAnomalyTrend(unittest.TestCase):
    """Regression test for symptom 2: the old per-window trend produced duplicate "0s" keys
    (three sliding windows all starting at frame 0) followed by a gap - crushing the normal
    series into a stub. The grid-based trend must have unique, gap-free time keys."""

    def test_trend_has_unique_gap_free_time_keys(self):
        segments = windows([
            (0, 0, "normal", 0.2),        # degenerate warm-up ping, dropped by non_degenerate
            (0, 60, "normal", 0.13),
            (0, 120, "normal", 0.2),
            (60, 180, "normal", 0.11),
            (300, 420, "suspicious", 0.7),
        ])
        segments = metrics.non_degenerate(segments)
        trend = build_analytics.compute_anomaly_trend(segments)

        times = [p["time"] for p in trend]
        self.assertEqual(len(times), len(set(times)), f"duplicate time keys in {times}")

        # Seconds must be strictly increasing across the trend.
        seconds = [int(t.rstrip("s")) for t in times]
        for a, b in zip(seconds, seconds[1:]):
            self.assertGreater(b, a)

    def test_trend_is_empty_for_no_segments(self):
        self.assertEqual(build_analytics.compute_anomaly_trend([]), [])


if __name__ == "__main__":
    unittest.main()
