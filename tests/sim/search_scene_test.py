"""Controlled-scene invariants, independent of a candidate route's performance."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.search_scene import SearchDeadline, SearchScenario, SearchScene


class SearchSceneTest(unittest.TestCase):
    def scene(self, **kwargs):
        return SearchScene(SearchScenario(start_pose=(1850, 500), false_positive_p=0, **kwargs))

    def test_trajectory_and_detection_do_not_depend_on_look_count(self):
        config = dict(seed=81, target_start=(1900, 500), target_end=(2000, 550),
                      move_start_s=1, move_duration_s=10, detection_p=0.43, box_jitter=0.01)
        many, few = self.scene(**config), self.scene(**config)
        for _ in range(40):
            many.advance(0.1)
            many.frame()
        few.advance(4.0)
        self.assertEqual(many.target_at(4), few.target_at(4))
        self.assertAlmostEqual(many.px, few.px)
        self.assertAlmostEqual(many.py, few.py)
        a, b = many.frame(), few.frame()
        self.assertEqual(a, b)
        self.assertEqual(many.last_frame_evidence.frame_index, few.last_frame_evidence.frame_index)
        self.assertEqual(many.frame(), a)  # querying one frame again draws nothing new

    def test_same_world_across_different_camera_actions(self):
        a, b = self.scene(target_end=(2500, 700), move_duration_s=20), self.scene(target_end=(2500, 700), move_duration_s=20)
        a.goto(1000, 600)
        a.advance(3.3)
        b.advance(3.3)
        self.assertEqual(a.target_at(a.t), b.target_at(b.t))
        self.assertEqual((a.px, a.py, a.dog_state), (b.px, b.py, b.dog_state))
        self.assertEqual(a._draw(10, "detect"), b._draw(10, "detect"))

    def test_deadline_is_exact_and_cancels_motor(self):
        r = self.scene(budget_s=0.73)
        r.goto(3000, 800)
        with self.assertRaises(SearchDeadline):
            r.advance(5)
        self.assertAlmostEqual(r.t, 0.73)
        self.assertFalse(r.busy())
        self.assertGreater(r.actual_travel_steps, 0)
        with self.assertRaises(SearchDeadline):
            r.advance(0.1)
        self.assertAlmostEqual(r.t, 0.73)

    def test_travel_integrates_actual_partial_motion_and_excludes_noop(self):
        r = self.scene()
        r.goto(1850, 500)
        r.wait_idle()
        self.assertEqual(r.actual_move_count, 0)
        r.goto(3000, 800)
        r.advance(0.8)
        r.stop()
        expected = abs(r.C - 1850) + abs(r.Y - 500)
        self.assertAlmostEqual(r.actual_travel_steps, expected)
        self.assertLess(r.actual_travel_steps, 1450)
        self.assertEqual(r.actual_move_count, 1)
        self.assertEqual(len(r.pose_trace), len(r.world_trace))
        r.goto(1850, 500)
        r.wait_idle()
        self.assertAlmostEqual(r.actual_travel_steps, 2 * expected)
        self.assertEqual(r.actual_move_count, 2)

    def test_lagged_truth_survives_when_target_now_out_of_view(self):
        r = self.scene(target_start=(1850, 500), target_end=(3000, 500),
                       move_start_s=0.2, move_duration_s=0.5, lag_s=1.0)
        r.advance(1.0)
        self.assertFalse(r.in_view())
        box = r.find_now(-1)
        self.assertIsNotNone(box)
        self.assertTrue(box.evidence.true_detection)
        self.assertEqual(box.evidence.target_pose, (1850, 500))
        self.assertAlmostEqual(box.captured, 0)
        self.assertTrue(r.last_detection_true)

    def test_stopped_find_preserves_evidence_after_boundary(self):
        r = self.scene(lag_s=0.9)
        boundary = r.t
        box = r.find_pet(boundary)
        self.assertIsNotNone(box)
        self.assertGreater(box.captured, boundary)
        self.assertTrue(box.evidence.true_detection)
        self.assertLessEqual(box.evidence.arrived_at, r.t)

    def test_lag_uses_exposure_camera_pose_and_untrimmed_history(self):
        r = self.scene(lag_s=25)
        r.goto(3000, 500)
        r.advance(25)
        self.assertFalse(r.in_view())
        box = r.find_now(-1)
        self.assertIsNotNone(box)
        self.assertEqual(box.evidence.camera_pose, (1850, 500))
        self.assertTrue(box.evidence.true_detection)
        self.assertGreater(len(r.pose_trace), 400)
        self.assertEqual(r.pose_trace[0], (0, 1850, 500, 1850))

    def test_absence_occlusion_and_false_positive_labels(self):
        for key in ("absent", "occluded"):
            r = self.scene(**{key: True})
            r.advance(1)
            self.assertIsNone(r.frame()[0])
            self.assertFalse(r.last_frame_evidence.true_detection)
        r = SearchScene(SearchScenario(absent=True, false_positive_p=1))
        box = r.frame()[0]
        self.assertFalse(box.evidence.true_detection)
        self.assertTrue(box.evidence.false_detection)
        self.assertFalse(r.last_detection_true)

    def test_real_tracker_finds_with_exposure_evidence(self):
        from sim.gate1 import load_tracker
        T = load_tracker()
        r = self.scene()
        captured = []

        class EvidenceTracker(T.Tracker):
            def _found(self, box, *args, **kwargs):
                captured.append(box.evidence)
                return super()._found(box, *args, **kwargs)

        tr = EvidenceTracker(r, r.find_pet, clock=lambda: r.t, sleep=r.sleep,
                             log=lambda _: None, find_now=r.find_now)
        result = tr.search("controlled-test")
        self.assertIn("found her", result)
        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0].true_detection)
        self.assertEqual(r.actual_travel_steps, 0)


if __name__ == "__main__":
    unittest.main()
