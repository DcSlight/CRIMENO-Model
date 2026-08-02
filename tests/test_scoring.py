"""Unit tests for groq/scoring.py.

Stdlib unittest rather than pytest — the repo has no test dependency and requirements.txt
should not grow one just for this.

Run:  py -m unittest discover -s tests -v
"""

import importlib.util
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# groq/ shadows the installed `groq` SDK package name, so load the module by path.
_spec = importlib.util.spec_from_file_location("crimeno_scoring", _REPO / "groq" / "scoring.py")
scoring = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scoring)

CALM = {
    "gun": "no", "knife": "no", "reaching_display_case": "no",
    "reaching_behind_counter": "no", "hands_up": "no", "face_concealed": "no",
    "aggression": "no", scoring.WEAPON_CONFIDENCE_CUE: "none",
}


def cues(**overrides):
    frame = dict(CALM)
    frame.update(overrides)
    return frame


def run(history, level="balanced", state=None):
    """Score a whole history in one call (the worker passes the full buffer each time)."""
    return scoring.score_from_cues(history, scoring_level=level, prior_state=state)


def run_sequence(frames, level="balanced", history=15):
    """Feed frames one at a time, threading threat state — mirrors the live worker loop."""
    state = scoring.new_threat_state()
    out = []
    buffer = []
    for frame in frames:
        buffer.append(frame)
        score, label, state = scoring.score_from_cues(
            buffer[-history:], scoring_level=level, prior_state=state,
        )
        out.append((score, label, state["level"]))
    return out


class TestCalmScenes(unittest.TestCase):
    def test_empty_history_is_normal(self):
        score, label, _ = run([])
        self.assertEqual(label, "normal")
        self.assertEqual(score, 0.0)

    def test_calm_scene_is_normal_at_every_level(self):
        for level in scoring.SCORING_LEVEL_THRESHOLDS:
            with self.subTest(level=level):
                _, label, _ = run([cues()] * 5, level=level)
                self.assertEqual(label, "normal")

    def test_browsing_is_normal_even_when_aggressive(self):
        """reaching_display_case is ordinary shopping while nothing else is wrong."""
        history = [cues(reaching_display_case="yes")] * 6
        _, label, _ = run(history, level="aggressive")
        self.assertEqual(label, "normal")

    def test_business_context_never_manufactures_evidence(self):
        """The regression that made frame 0 of a quiet video read 'suspicious'.

        An additive scoring_level bias raised the floor of every window; thresholds must not.
        """
        for level in scoring.SCORING_LEVEL_THRESHOLDS:
            with self.subTest(level=level):
                score, label, _ = run([cues()], level=level)
                self.assertEqual(label, "normal")
                self.assertLessEqual(score, 0.2)


class TestSingleWeakCue(unittest.TestCase):
    def test_lone_face_concealed_is_not_suspicious(self):
        _, label, _ = run([cues(face_concealed="yes")], level="aggressive")
        self.assertEqual(label, "normal")

    def test_lone_hands_up_is_not_suspicious(self):
        """A customer gesturing over a counter is not an alert."""
        _, label, _ = run([cues(hands_up="yes")], level="aggressive")
        self.assertEqual(label, "normal")

    def test_two_weak_cues_together_do_alert(self):
        """Corroboration is the whole difference — one cue is noise, two is evidence."""
        _, label, _ = run([cues(face_concealed="unclear", aggression="unclear")],
                          level="aggressive")
        self.assertEqual(label, "suspicious")

    def test_strong_cue_alerts_alone(self):
        """Weapon and forbidden-action cues are exempt from the corroboration rule."""
        _, label, _ = run([cues(reaching_behind_counter="yes")])
        self.assertEqual(label, "suspicious")


class TestWeaponGrading(unittest.TestCase):
    def test_grades_confident_descriptions_high(self):
        for text in (
            "The man in the blue jacket is holding an object that appears to be a handgun.",
            "The person in the aisle is holding a dark, handgun-shaped object.",
            "A rifle is visible, pointed toward the cashier.",
        ):
            with self.subTest(text=text):
                self.assertEqual(scoring.grade_weapon_text(text), "high")

    def test_grades_disclaimed_descriptions_low(self):
        for text in (
            "The dark object held by the person on the right might be a tool or weapon, "
            "but it is not clearly identifiable.",
            "The person is holding a dark, possibly metallic object that could be a weapon.",
        ):
            with self.subTest(text=text):
                self.assertEqual(scoring.grade_weapon_text(text), "low")

    def test_disclaimer_outranks_weapon_noun(self):
        """'handgun' appearing next to 'not clearly identifiable' must not grade high."""
        text = "Possibly a handgun, but it is not clearly identifiable from this angle."
        self.assertEqual(scoring.grade_weapon_text(text), "low")

    def test_empty_and_none_grade_none(self):
        for text in ("", "none", "None", "n/a", "none visible", None):
            with self.subTest(text=text):
                self.assertEqual(scoring.grade_weapon_text(text), "none")

    def test_low_confidence_weapon_does_not_open_gate(self):
        history = [cues(**{scoring.WEAPON_CONFIDENCE_CUE: "low"}, aggression="yes")] * 4
        _, label, _ = run(history, level="aggressive")
        self.assertNotEqual(label, "criminal")

    def test_high_confidence_weapon_needs_sustained_corroboration(self):
        """A hallucinated weapon beside a ONE-frame aggression spike is not a robbery.

        This is the jewelry-store 14s case: the VLM reported 'a handgun pointed toward the
        seated employee' over a stretch ground truth labels normal.
        """
        history = [
            cues(),
            cues(),
            cues(**{scoring.WEAPON_CONFIDENCE_CUE: "high"}, aggression="yes"),
        ]
        _, label, _ = run(history, level="aggressive")
        self.assertEqual(label, "suspicious")

    def test_high_confidence_weapon_with_sustained_aggression_is_criminal(self):
        history = [
            cues(aggression="yes"),
            cues(aggression="yes"),
            cues(**{scoring.WEAPON_CONFIDENCE_CUE: "high"}, aggression="yes"),
        ]
        _, label, _ = run(history)
        self.assertEqual(label, "criminal")

    def test_isolated_weapon_sighting_is_damped(self):
        """One flickering frame of weapon evidence scores below two sustained frames."""
        one = [cues(), cues(), cues(**{scoring.WEAPON_CONFIDENCE_CUE: "high"})]
        two = [cues(), cues(**{scoring.WEAPON_CONFIDENCE_CUE: "high"}),
               cues(**{scoring.WEAPON_CONFIDENCE_CUE: "high"})]
        self.assertLess(run(one)[0], run(two)[0])


class TestCriminalGate(unittest.TestCase):
    def test_single_frame_behind_counter_is_not_criminal(self):
        """The original false-positive this module was built to close."""
        history = [cues(), cues(reaching_behind_counter="yes", aggression="yes")]
        _, label, _ = run(history, level="aggressive")
        self.assertEqual(label, "suspicious")

    def test_sustained_and_corroborated_is_criminal(self):
        history = [
            cues(reaching_behind_counter="yes"),
            cues(reaching_behind_counter="yes", aggression="yes"),
        ]
        _, label, _ = run(history)
        self.assertEqual(label, "criminal")

    def test_sustained_without_corroboration_is_not_criminal(self):
        history = [cues(reaching_behind_counter="yes")] * 4
        _, label, _ = run(history)
        self.assertNotEqual(label, "criminal")


class TestThreatLatching(unittest.TestCase):
    def _robbery_then_looting(self, looting_frames):
        """Gate opens, then the suspects just move around the display cases."""
        return [
            cues(reaching_behind_counter="yes"),
            cues(reaching_behind_counter="yes", aggression="yes"),   # gate opens here
        ] + [cues(reaching_display_case="yes")] * looting_frames

    def test_latch_engages_when_gate_opens(self):
        out = run_sequence(self._robbery_then_looting(0))
        self.assertEqual(out[-1][1], "criminal")
        self.assertEqual(out[-1][2], 1.0)

    def test_latch_holds_through_looting(self):
        """The 16-consecutive-under-call bug: score used to collapse to normal here."""
        out = run_sequence(self._robbery_then_looting(12))
        for score, label, level in out[2:]:
            self.assertEqual(label, "criminal")
            self.assertGreaterEqual(score, 0.8)
            self.assertGreater(level, 0.0)

    def test_latch_releases_on_sustained_calm(self):
        frames = self._robbery_then_looting(0) + [cues()] * (scoring.LATCH_CALM_EXIT_STREAK + 1)
        out = run_sequence(frames)
        self.assertEqual(out[-1][1], "normal")
        self.assertEqual(out[-1][2], 0.0)

    def test_looting_does_not_count_as_calm(self):
        """People at the display cases after a robbery are suspects, not shoppers."""
        self.assertTrue(scoring._is_calm(cues()))
        self.assertFalse(scoring._is_calm(cues(reaching_display_case="yes")))

    def test_latch_decays_to_zero_eventually(self):
        """Bounded memory — a latch must not pin a camera to 'criminal' forever."""
        total = scoring.LATCH_HOLD_DECISIONS + scoring.LATCH_DECAY_DECISIONS + 5
        out = run_sequence(self._robbery_then_looting(total))
        self.assertEqual(out[-1][2], 0.0)

    def test_state_is_not_shared_between_calls(self):
        """score_from_cues must stay pure — no module-level state."""
        history = self._robbery_then_looting(0)
        _, _, state = run(history)
        self.assertEqual(state["level"], 1.0)
        _, label, _ = run([cues()])          # fresh call, no prior_state
        self.assertEqual(label, "normal")

    def test_prior_state_is_not_mutated(self):
        state = scoring.new_threat_state()
        snapshot = dict(state)
        run(self._robbery_then_looting(0), state=state)
        self.assertEqual(state, snapshot)


class TestDisplayCaseContext(unittest.TestCase):
    def test_display_case_is_inert_when_calm(self):
        self.assertEqual(scoring._cue_weight("reaching_display_case", "yes", 0.0), 0.0)

    def test_display_case_counts_once_threat_established(self):
        self.assertGreater(scoring._cue_weight("reaching_display_case", "yes", 1.0), 0.0)


class TestScoringLevels(unittest.TestCase):
    def test_aggressive_alerts_no_later_than_conservative(self):
        history = [cues(aggression="unclear", face_concealed="unclear")]
        order = {"normal": 0, "suspicious": 1, "criminal": 2}
        ranks = [order[run(history, level=lvl)[1]]
                 for lvl in ("conservative", "balanced", "aggressive")]
        self.assertLessEqual(ranks[0], ranks[1])
        self.assertLessEqual(ranks[1], ranks[2])

    def test_unknown_level_falls_back_to_balanced(self):
        history = [cues(aggression="yes", face_concealed="yes")]
        self.assertEqual(run(history, level="nonsense")[1], run(history, level="balanced")[1])


class TestApplyScoring(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(scoring.apply_scoring("normal", 0.9), 0.2)
        self.assertEqual(scoring.apply_scoring("suspicious", 0.05), 0.3)
        self.assertEqual(scoring.apply_scoring("suspicious", 0.95), 0.7)
        self.assertEqual(scoring.apply_scoring("criminal", 0.1), 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
