"""Conservative visual-memory invariants without a camera or household data."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.view_obstructions import Config, Observation, ObstructionMemory


class ViewObstructionsTest(unittest.TestCase):
    def config(self, **changes):
        values = dict(min_gap_s=1, min_samples=3, min_span_s=4, evidence_window_s=20,
                      exclusion_ttl_s=30, probe_interval_s=10, max_frame_age_s=2)
        values.update(changes)
        return Config(**values)

    def obs(self, at, pose=(100, 200), probability=.99, **changes):
        values = dict(pose=pose, captured_at=at, observed_at=at, context='mount1/map1/model1',
                      obstruction_probability=probability)
        values.update(changes)
        return Observation(**values)

    def excluded(self, **config):
        memory = ObstructionMemory(self.config(**config))
        for at in (0, 2, 4):
            result = memory.observe(self.obs(at))
        self.assertTrue(result['skip'])
        return memory

    def test_repeated_independent_visual_evidence_and_minimum_span_are_both_required(self):
        memory = ObstructionMemory(self.config())
        for at in (0, 1, 2):
            result = memory.observe(self.obs(at))
            self.assertEqual(result['state'], 'suspected')
            self.assertFalse(result['skip'])
        result = memory.observe(self.obs(4))
        self.assertEqual(result['state'], 'temporarily_excluded')
        self.assertTrue(result['skip'])
        self.assertEqual(result['excluded_until'], 34)
        self.assertEqual(result['probe_deadline'], 14)

    def test_detector_misses_alone_never_learn_an_obstruction(self):
        memory = ObstructionMemory(self.config())
        for at in range(50):
            result = memory.observe(self.obs(at, probability=None, pet_detected=False))
            self.assertFalse(result['accepted'])
            self.assertEqual(result['ignored_reason'], 'no_visual_obstruction_evidence')
            self.assertEqual(result['state'], 'unknown')
            self.assertFalse(result['skip'])
        self.assertEqual(memory.to_dict()['records'], [])

    def test_cached_frame_repeats_do_not_count_or_refresh_inspection(self):
        memory = ObstructionMemory(self.config())
        memory.observe(self.obs(0))
        for observed in (.5, 1, 1.5):
            result = memory.observe(self.obs(0, observed_at=observed))
            self.assertEqual(result['ignored_reason'], 'duplicate_or_older_frame')
            self.assertEqual(result['high_observations'], 1)
            self.assertEqual(result['last_valid_observed_at'], 0)
            self.assertFalse(result['skip'])

    def test_neighboring_frames_need_the_declared_independence_gap(self):
        memory = ObstructionMemory(self.config())
        memory.observe(self.obs(0))
        result = memory.observe(self.obs(.5))
        self.assertEqual(result['ignored_reason'], 'observations_not_independent')
        self.assertEqual(result['high_observations'], 1)
        self.assertTrue(memory.observe(self.obs(1))['accepted'])
        self.assertEqual(memory.observe(self.obs(2))['high_observations'], 3)
        self.assertFalse(memory.decision((100, 200), 2, memory.context)['skip'])
        self.assertTrue(memory.observe(self.obs(4))['skip'])

    def test_invalid_stale_future_moving_and_low_quality_frames_do_not_learn(self):
        changes = [dict(image_valid=False), dict(quality=.2), dict(settled=False),
                   dict(captured_at=0, observed_at=3), dict(captured_at=4, observed_at=3)]
        for change in changes:
            memory = ObstructionMemory(self.config())
            with self.subTest(change=change):
                result = memory.observe(self.obs(3, **change))
                self.assertFalse(result['accepted'])
                self.assertFalse(result['skip'])
                self.assertEqual(memory.to_dict()['records'], [])

    def test_pet_or_clear_visual_evidence_immediately_revokes_temporary_exclusion(self):
        for probability, pet in [(0, False), (.99, True), (None, True)]:
            memory = self.excluded()
            result = memory.observe(self.obs(5, probability=probability, pet_detected=pet))
            with self.subTest(probability=probability, pet=pet):
                self.assertTrue(result['accepted'])
                self.assertEqual(result['state'], 'unknown')
                self.assertFalse(result['skip'])
                self.assertIsNone(result['excluded_until'])
                self.assertEqual(result['high_observations'], 0)

    def test_ambiguous_probe_fails_open_and_cannot_reuse_old_confirmation(self):
        memory = self.excluded()
        result = memory.observe(self.obs(5, probability=.6))
        self.assertEqual(result['reason'], 'ambiguous_visual_evidence_fail_open')
        self.assertFalse(result['skip'])
        for at in (6, 8):
            self.assertFalse(memory.observe(self.obs(at))['skip'])
        self.assertTrue(memory.observe(self.obs(10))['skip'])

    def test_mandatory_probe_opens_view_before_ttl_expires(self):
        memory = self.excluded()
        self.assertTrue(memory.decision((100, 200), 13.9, memory.context)['skip'])
        result = memory.decision((100, 200), 14, memory.context)
        self.assertEqual(result['state'], 'reinspection_due')
        self.assertTrue(result['probe_due'])
        self.assertFalse(result['skip'])
        self.assertEqual(result['reason'], 'mandatory_probe_due')

    def test_invalid_probe_does_not_renew_exclusion_or_probe_deadline(self):
        for change in [dict(quality=.1), dict(settled=False), dict(image_valid=False),
                       dict(captured_at=4), dict(probability=None)]:
            memory = self.excluded()
            result = memory.observe(self.obs(14, **change))
            with self.subTest(change=change):
                self.assertFalse(result['accepted'])
                self.assertEqual(result['probe_deadline'], 14)
                self.assertEqual(result['excluded_until'], 34)
                self.assertFalse(result['skip'])
                self.assertTrue(memory.decision((100, 200), 15, memory.context)['probe_due'])

    def test_ttl_forces_reinspection_even_if_probe_interval_is_longer(self):
        memory = self.excluded(exclusion_ttl_s=3)
        result = memory.decision((100, 200), 7, memory.context)
        self.assertFalse(result['skip'])
        self.assertEqual(result['state'], 'reinspection_due')
        self.assertEqual(result['reason'], 'exclusion_expired')
        self.assertFalse(memory.decision((100, 200), 100000, memory.context)['skip'])

    def test_old_high_evidence_does_not_confirm_a_later_single_revisit(self):
        memory = ObstructionMemory(self.config())
        memory.observe(self.obs(0)); memory.observe(self.obs(2))
        result = memory.observe(self.obs(25))
        self.assertEqual(result['high_observations'], 1)
        self.assertFalse(result['skip'])

    def test_current_context_change_resets_all_poses_and_old_frames_cannot_roll_it_back(self):
        memory = self.excluded()
        result = memory.decision((100, 200), 5, 'mount2/map1/model1')
        self.assertEqual(result['state'], 'unknown')
        self.assertEqual(memory.to_dict()['records'], [])
        delayed = memory.observe(self.obs(6))
        self.assertEqual(delayed['ignored_reason'], 'context_mismatch')
        self.assertEqual(memory.context, 'mount2/map1/model1')
        self.assertEqual(memory.to_dict()['records'], [])
        self.assertEqual(memory.to_dict()['now'], 5)
        new = self.obs(7, context='mount2/map1/model1')
        self.assertEqual(memory.observe(new)['high_observations'], 1)

    def test_explicit_scene_change_invalidates_even_the_same_fingerprint(self):
        memory = self.excluded()
        memory.setup_changed(memory.context, now=5)
        self.assertFalse(memory.decision((100, 200), 5, memory.context)['skip'])
        old = memory.observe(self.obs(4, observed_at=5.5))
        self.assertEqual(old['ignored_reason'], 'frame_predates_setup')
        self.assertEqual(memory.observe(self.obs(6))['high_observations'], 1)

    def test_older_observation_cannot_restore_a_cleared_belief(self):
        memory = self.excluded()
        memory.observe(self.obs(6, probability=0))
        result = memory.observe(self.obs(5))
        self.assertEqual(result['ignored_reason'], 'out_of_order_observation')
        self.assertFalse(result['skip'])
        self.assertEqual(result['high_observations'], 0)

    def test_probe_budget_is_limited_deterministic_and_consumed_only_by_valid_evidence(self):
        memory = ObstructionMemory(self.config())
        a, b, unseen = (100, 200), (300, 200), (500, 200)
        for at in (0, 2, 4):
            memory.observe(self.obs(at, pose=a))
            memory.observe(self.obs(at, pose=b))
        poses = [unseen, b, a, a]
        due = memory.due(poses, 14, memory.context, limit=2)
        self.assertEqual([x['pose'] for x in due], [list(a), list(b)])
        self.assertTrue(all(not x['skip'] and x['probe_due'] for x in due))
        snapshot = memory.to_dict()
        self.assertEqual(due, memory.due(poses, 14, memory.context, limit=2))
        self.assertEqual(snapshot, memory.to_dict())
        self.assertEqual(memory.due(poses, 14, memory.context, limit=0), [])
        memory.observe(self.obs(14, pose=a))
        memory.observe(self.obs(14, pose=b, image_valid=False))
        self.assertEqual([x['pose'] for x in memory.due(poses, 14, memory.context, limit=2)],
                         [list(b), list(unseen)])

    def test_bounded_storage_evicts_oldest_pose_to_unknown(self):
        memory = ObstructionMemory(self.config(max_poses=2))
        for at, pose in enumerate([(1, 2), (2, 3), (3, 4)]):
            memory.observe(self.obs(at, pose=pose))
        self.assertEqual(len(memory.to_dict()['records']), 2)
        self.assertEqual(memory.decision((1, 2), 2, memory.context)['state'], 'unknown')

    def test_invalid_probe_rotates_mandatory_views_without_renewing_any_deadline(self):
        memory = ObstructionMemory(self.config())
        a, b = (100, 200), (300, 200)
        for at in (0, 2, 4):
            for pose in (a, b):
                memory.observe(self.obs(at, pose=pose))
        self.assertEqual(memory.due([a, b], 14, memory.context, 1)[0]['pose'], list(a))
        result = memory.observe(self.obs(14, pose=a, image_valid=False))
        self.assertFalse(result['accepted'])
        self.assertEqual(result['last_probe_attempt_at'], 14)
        self.assertEqual(result['probe_deadline'], 14)
        self.assertEqual(result['excluded_until'], 34)
        self.assertFalse(result['skip'])
        self.assertEqual(memory.due([a, b], 14, memory.context, 1)[0]['pose'], list(b))
        for pose in (a, b):
            self.assertFalse(memory.decision(pose, 14, memory.context)['skip'])
        snapshot = memory.to_dict()
        memory.due([a, b], 14, memory.context, 1)
        self.assertEqual(memory.to_dict(), snapshot)

    def test_stale_or_duplicate_frames_do_not_count_as_new_probe_attempts(self):
        memory = ObstructionMemory(self.config())
        a, b = (100, 200), (300, 200)
        for at in (0, 2, 4):
            for pose in (a, b):
                memory.observe(self.obs(at, pose=pose))
        stale = memory.observe(self.obs(4, pose=a, observed_at=14, image_valid=False))
        self.assertEqual(stale['last_probe_attempt_at'], 4)
        self.assertEqual(memory.due([a, b], 14, memory.context, 1)[0]['pose'], list(a))
        memory.observe(self.obs(14, pose=a, image_valid=False))
        memory.observe(self.obs(14.5, pose=b, image_valid=False))
        repeated = memory.observe(self.obs(14, pose=a, observed_at=15, image_valid=False))
        self.assertEqual(repeated['ignored_reason'], 'duplicate_or_older_frame')
        self.assertEqual(repeated['last_probe_attempt_at'], 14)
        self.assertEqual(memory.due([a, b], 15, memory.context, 1)[0]['pose'], list(a))

    def test_invalid_unknown_view_attempts_are_bounded_persisted_and_setup_scoped(self):
        memory = ObstructionMemory(self.config(max_poses=2))
        for at, pose in enumerate([(1, 2), (2, 3), (3, 4)]):
            memory.observe(self.obs(at, pose=pose, image_valid=False))
        snapshot = memory.to_dict()
        self.assertEqual(snapshot['records'], [])
        self.assertEqual(len(snapshot['probe_attempts']), 2)
        restored = ObstructionMemory.from_dict(snapshot)
        self.assertEqual(restored.to_dict(), snapshot)
        self.assertEqual(restored.due([(2, 3), (3, 4), (5, 6)], 2, restored.context, 1)[0]['pose'], [5, 6])
        restored.setup_changed('new-setup', 3)
        before = restored.to_dict()
        restored.observe(self.obs(4, pose=(1, 2), image_valid=False))
        self.assertEqual(restored.to_dict(), before)

    def test_persisted_probe_attempts_reject_stale_unbounded_or_duplicate_entries(self):
        snapshot = self.excluded().to_dict()
        malformed = [dict(snapshot, probe_attempts=[{'pose': [1, 2], 'captured_at': 0, 'observed_at': 4}]),
                     dict(snapshot, probe_attempts=snapshot['probe_attempts'] * 2)]
        for saved in malformed:
            with self.subTest(saved=saved), self.assertRaises(ValueError):
                ObstructionMemory.from_dict(saved)

    def test_input_and_serialized_snapshots_are_detached_and_immutable(self):
        pose = [100, 200]
        observation = self.obs(0, pose=pose)
        pose[0] = 999
        self.assertEqual(observation.pose, (100, 200))
        with self.assertRaises(FrozenInstanceError):
            observation.pose = (0, 0)
        memory = self.excluded()
        saved = memory.to_dict()
        restored = ObstructionMemory.from_dict(saved)
        saved['records'][0]['high_captures'].clear()
        saved['records'][0]['pose'][0] = 999
        self.assertTrue(memory.decision((100, 200), 4, memory.context)['skip'])
        self.assertEqual(memory.to_dict(), restored.to_dict())

    def test_explicit_persistence_roundtrip_keeps_ttl_and_requires_fresh_context(self):
        memory = self.excluded()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'obstructions.json'
            memory.save(path)
            self.assertEqual(json.loads(path.read_text()), memory.to_dict())
            restored = ObstructionMemory.load(path)
            self.assertEqual(restored.to_dict(), memory.to_dict())
            self.assertFalse(restored.decision((100, 200), 14, memory.context)['skip'])
            self.assertEqual(restored.decision((100, 200), 15, 'new-mount')['state'], 'unknown')

    def test_invalid_config_observation_and_snapshot_are_rejected(self):
        for changes in [dict(high_threshold=.1, low_threshold=.2), dict(probe_interval_s=float('inf')),
                        dict(min_samples=1), dict(min_samples=True), dict(min_gap_s=0),
                        dict(max_poses=0), dict(evidence_window_s=1), dict(min_quality=-1)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.config(**changes)
        for changes in [dict(pose=[1]), dict(pose=[float('nan'), 1]), dict(context=''),
                        dict(quality=2), dict(obstruction_probability=-1), dict(settled=1)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.obs(0), **changes)
        snapshot = self.excluded().to_dict()
        malformed = []
        for key, value in [('version', 2), ('context', None), ('now', 0)]:
            malformed.append(dict(snapshot, **{key: value}))
        record = snapshot['records'][0]
        for values in [dict(high_captures=[0]), dict(excluded_until=999), dict(high_captures=[0, 0, 4])]:
            malformed.append(dict(snapshot, records=[dict(record, **values)]))
        malformed.append(dict(snapshot, now=30, records=[dict(record, last_captured_at=30,
                                                            last_observed_at=30, next_probe_at=40,
                                                            excluded_until=60)]))
        for saved in malformed:
            with self.subTest(saved=saved), self.assertRaises(ValueError):
                ObstructionMemory.from_dict(saved)

    def test_backwards_queries_and_invalid_budget_are_rejected(self):
        memory = self.excluded()
        with self.assertRaises(ValueError):
            memory.decision((100, 200), 3, memory.context)
        for limit in (-1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                memory.due([(100, 200)], 4, memory.context, limit)
        snapshot = memory.to_dict()
        for query in (lambda: memory.decision((100, 200), 50, ''),
                      lambda: memory.due([(100, 200)], 50, '', 1)):
            with self.assertRaises(ValueError):
                query()
            self.assertEqual(memory.to_dict(), snapshot)


if __name__ == '__main__':
    unittest.main()
