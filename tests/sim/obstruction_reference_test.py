"""Pixel-pair contract and fail-open reference gating, without scene truth."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.obstruction_reference import (FEATURE_COUNT, NORMALIZED_DIFF_SLICE, ReferenceModel,
                                       feature_contract, fit, pair_features)
from sim.obstruction_scene import image_features


class ObstructionReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pairs, labels = [], []
        for gray in range(70, 191, 10):
            current = np.full((12, 16, 3), gray, dtype=np.uint8)
            changed = current.copy()
            changed[::2, ::2] = gray - 45
            changed[1::2, 1::2] = gray + 45
            for reference, label in ((current, 0), (changed, 1)):
                pairs.append(pair_features(current, reference))
                labels.append(label)
        cls.model = fit(pairs, labels, seed=13, n_trees=8, max_depth=4, min_leaf=2)

    def image(self, seed=1):
        return np.random.default_rng(seed).integers(35, 150, (24, 32, 3), dtype=np.uint8)

    def test_identical_rgb_pairs_have_zero_change_and_frozen_descriptor_layout(self):
        image = self.image()
        features = pair_features(image, image)
        self.assertEqual(features.shape, (FEATURE_COUNT,))
        self.assertEqual(FEATURE_COUNT, 241)
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(features[:129], image_features(image))
        np.testing.assert_array_equal(features[129:146], image_features(image)[:17])
        np.testing.assert_allclose(features[146:233], 0, atol=1e-14)
        self.assertEqual(features[233], features[234])
        np.testing.assert_array_equal(features[235:237], [0, 0])

    def test_normalized_difference_is_approximately_invariant_to_global_exposure_and_channels(self):
        reference = self.image(9)
        current = np.rint(reference.astype(float) * np.array([1.1, 1.2, .9]) + [8, 10, 15]).astype(np.uint8)
        features = pair_features(current, reference)
        # Raw appearance is deliberately retained; only the normalized change
        # part has the declared exposure invariance.
        self.assertGreater(features[146:153].max(), .05)
        self.assertLess(features[NORMALIZED_DIFF_SLICE].max(), .003)
        np.testing.assert_array_equal(features[162:165], [0, 0, 0])

    def test_small_registered_shift_is_aligned_but_larger_shift_is_not_forced_perfect(self):
        reference = self.image(12)
        current = np.roll(reference, shift=(-1, 1), axis=(0, 1))
        features = pair_features(current, reference)
        np.testing.assert_allclose(features[NORMALIZED_DIFF_SLICE], 0, atol=1e-14)
        np.testing.assert_array_equal(features[235:237], [.5, -.5])
        outside_limit = pair_features(np.roll(reference, shift=4, axis=1), reference)
        self.assertGreater(outside_limit[NORMALIZED_DIFF_SLICE][3:9].mean(), .03)
        self.assertTrue(np.all(np.abs(outside_limit[235:237]) <= 1))

    def test_missing_unregistered_and_single_frame_prediction_fail_open(self):
        image, other_pose = self.image(2), self.image(3)
        self.assertTrue(self.model.requires_reference)
        for current, reference, registered in [(image, None, True), (None, image, True),
                                              (image, other_pose, False), (image, other_pose, None)]:
            with self.subTest(registered=registered):
                self.assertIsNone(pair_features(current, reference, registered=registered))
                self.assertIsNone(self.model.predict_pair(current, reference, registered=registered))
        self.assertIsNone(self.model.predict(image))
        self.assertIsNone(self.model.predict(None))

    def test_invalid_reference_cannot_silently_become_single_image_prediction(self):
        image = self.image()
        for reference in [image.astype(float), image[:4], image[:, :-1], np.zeros((8, 8))]:
            with self.subTest(shape=reference.shape):
                with self.assertRaises(ValueError):
                    pair_features(image, reference)
                self.assertIsNone(self.model.predict_pair(image, reference))

    def test_same_current_pixels_can_receive_different_scores_from_pixel_references(self):
        current = np.full((12, 16, 3), 125, dtype=np.uint8)
        changed_reference = current.copy()
        changed_reference[::2, ::2] = 80
        changed_reference[1::2, 1::2] = 170
        clear_score = self.model.predict_pair(current, current)
        change_score = self.model.predict_pair(current, changed_reference)
        self.assertLess(clear_score, .3)
        self.assertGreater(change_score, .7)
        self.assertEqual(clear_score, float(self.model.scores([pair_features(current, current)])[0]))
        # Pixels and explicit acquisition validity are the whole input API.
        with self.assertRaises(TypeError):
            self.model.predict_pair(current, current, blocked=True)

    def test_features_are_deterministic_and_do_not_mutate_images(self):
        current, reference = self.image(4), self.image(5)
        before = current.copy(), reference.copy()
        np.testing.assert_array_equal(pair_features(current, reference), pair_features(current, reference))
        np.testing.assert_array_equal(current, before[0])
        np.testing.assert_array_equal(reference, before[1])

    def test_serialization_preserves_reference_requirement_and_is_detached(self):
        doc = self.model.to_dict()
        self.assertIs(doc['requires_registered_reference'], True)
        self.assertIn('unverified on real camera', doc['pair_contract']['reference_assumption'])
        restored = ReferenceModel.from_dict(json.loads(json.dumps(doc, allow_nan=False)))
        image = self.image()
        self.assertEqual(restored.predict_pair(image, image), self.model.predict_pair(image, image))
        self.assertIsNone(restored.predict(image))
        doc['pair_contract']['layout'][0]['count'] = 1
        self.assertEqual(feature_contract()['layout'][0]['count'], 129)
        self.assertEqual(self.model.to_dict()['pair_contract'], feature_contract())

    def test_corrupt_contract_or_inner_schema_cannot_restore_a_reference_model(self):
        doc = self.model.to_dict()
        malformed = [dict(doc, version=2), dict(doc, model='single_rgb'),
                     dict(doc, requires_registered_reference=False), dict(doc, requires_registered_reference=1),
                     dict(doc, pair_contract={}), dict(doc, inner_forest={})]
        changed = copy.deepcopy(doc)
        changed['pair_contract']['alignment'] = 'unbounded oracle alignment'
        malformed.append(changed)
        changed = copy.deepcopy(doc)
        changed['inner_forest']['n_features'] = 240
        malformed.append(changed)
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(ValueError):
                ReferenceModel.from_dict(value)

    def test_training_requires_paired_feature_schema_and_labels_only_enter_fit(self):
        for features in [np.zeros((4, 129)), np.zeros((4, FEATURE_COUNT - 1)), [], [1, 2]]:
            with self.subTest(shape=np.shape(features)), self.assertRaises(ValueError):
                fit(features, [0, 0, 1, 1])
        image = self.image()
        features = pair_features(image, image)
        with self.assertRaises(ValueError):
            fit([features, features], [0, 1], feature_schema='other')
        with self.assertRaises(ValueError):
            fit([features, features], [0, 0])


if __name__ == '__main__':
    unittest.main()
