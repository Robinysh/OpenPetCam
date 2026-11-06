"""Independent classifier contracts using generated data and RGB, never room labels."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.obstruction_model import (PixelModel, classifier_metrics, fit, pixel_quality,
                                   select_threshold)
from sim.obstruction_scene import image_features


class ObstructionModelTest(unittest.TestCase):
    def separable(self):
        axis = np.array([-3, -2.5, -2, -1.5, -1, 1, 1.5, 2, 2.5, 3])
        nuisance = np.array([.1, -.2, .2, -.1, 0, .2, -.2, .1, 0, -.1])
        return np.column_stack((axis, nuisance, np.full(10, 7))), (axis > 0).astype(int)

    def rgb_model(self):
        # An explicit fixed classifier, not a fit to the images tested below.
        count = len(image_features(np.full((8, 12, 3), 100, dtype=np.uint8)))
        return PixelModel(np.zeros(count), np.ones(count), np.linspace(-.05, .05, count), -.2)

    def test_fit_reduces_independent_log_loss_and_generalizes_separable_signal(self):
        features, labels = self.separable()
        model = fit(features, labels)
        scores = model.scores(features)
        loss = -np.mean(labels * np.log(np.clip(scores, 1e-15, 1)) +
                        (1 - labels) * np.log(np.clip(1 - scores, 1e-15, 1)))
        self.assertLess(loss, .2)
        self.assertLess(loss, -np.log(.5) / 2)
        held_out = np.array([[-4, -.3, 7], [-.7, .1, 7], [.7, -.1, 7], [4, .3, 7]])
        np.testing.assert_array_equal(model.scores(held_out) >= .5, [False, False, True, True])
        self.assertTrue(np.isfinite(model.scores(held_out)).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())

    def test_fit_is_deterministic_and_does_not_modify_training_arrays(self):
        features, labels = self.separable()
        before = features.copy(), labels.copy()
        a, b = fit(features, labels), fit(features, labels)
        self.assertEqual(a.to_dict(), b.to_dict())
        np.testing.assert_array_equal(features, before[0])
        np.testing.assert_array_equal(labels, before[1])

    def test_fit_is_invariant_to_positive_feature_unit_changes(self):
        features, labels = self.separable()
        scales, offsets = np.array([100, .1, 3]), np.array([12, -40, 2])
        a = fit(features, labels)
        b = fit(features * scales + offsets, labels)
        np.testing.assert_allclose(a.scores(features), b.scores(features * scales + offsets), atol=1e-10)

    def test_json_roundtrip_preserves_rgb_and_feature_predictions(self):
        model = self.rgb_model()
        yy, xx = np.mgrid[:8, :12]
        image = np.stack((20 + xx * 10, 30 + yy * 20, 70 + (xx + yy) * 5), axis=-1).astype(np.uint8)
        document = json.loads(json.dumps(model.to_dict(), allow_nan=False))
        restored = PixelModel.from_dict(document)
        self.assertEqual(restored.to_dict(), document)
        self.assertEqual(restored.predict(image), model.predict(image))
        np.testing.assert_array_equal(restored.scores([image_features(image)]), model.scores([image_features(image)]))
        self.assertIn('unvalidated', document['scope'])

    def test_model_owns_immutable_parameter_copies(self):
        mean, scale, coefficients = np.zeros(3), np.ones(3), np.array([1., 2, 3])
        model = PixelModel(mean, scale, coefficients, 0)
        before = model.scores([[1, 2, 3]]).copy()
        mean[:] = 100
        scale[:] = 100
        coefficients[:] = -100
        np.testing.assert_array_equal(model.scores([[1, 2, 3]]), before)
        for array in (model.mean, model.scale, model.coefficients):
            with self.assertRaises(ValueError):
                array[0] = 99
        document = model.to_dict()
        document['coefficients'][0] = -99
        np.testing.assert_array_equal(model.scores([[1, 2, 3]]), before)

    def test_identical_rgb_is_independent_of_hidden_truth_and_pet_misses(self):
        model = self.rgb_model()
        image = np.full((8, 12, 3), [80, 110, 140], dtype=np.uint8)
        records = [{'rgb': image.copy(), 'oracle_blocked': blocked,
                    'pet_present': present, 'empty_detector': missing}
                   for blocked, present, missing in [(False, True, False), (True, False, True), (False, False, True)]]
        scores = [model.predict(record['rgb']) for record in records]
        self.assertEqual(scores, [scores[0]] * len(records))
        for record in records[1:]:
            np.testing.assert_array_equal(image_features(record['rgb']), image_features(records[0]['rgb']))
        # A detector verdict or an empty-observation record is not an RGB/feature substitute.
        for wrong in [None, [], {'empty_detector': True}, {'score': 0, 'pet_present': False}]:
            with self.subTest(value=wrong), self.assertRaises((ValueError, TypeError)):
                model.predict(wrong)
        with self.assertRaises((ValueError, TypeError)):
            fit([{'empty_detector': True}, {'empty_detector': False}], [0, 1])

    def test_confusion_counts_include_threshold_boundary_and_both_classes(self):
        result = classifier_metrics([1, 0, 1, 0, 1], [.9, .8, .4, .1, .5], .5)
        self.assertEqual({k: result[k] for k in ('n', 'tp', 'fp', 'tn', 'fn')},
                         {'n': 5, 'tp': 2, 'fp': 1, 'tn': 1, 'fn': 1})
        self.assertAlmostEqual(result['recall'], 2 / 3)
        self.assertAlmostEqual(result['precision'], 2 / 3)
        self.assertEqual(result['false_positive_rate'], .5)
        empty_positive = classifier_metrics([0, 0], [.1, .2], .9)
        self.assertEqual(empty_positive['precision'], 0)
        self.assertEqual(empty_positive['recall'], 0)
        self.assertEqual(empty_positive['false_positive_rate'], 0)

    def test_threshold_is_frozen_from_development_before_contradictory_test_labels(self):
        development_labels, development_scores = [0, 0, 1, 1], [.1, .79, .82, .99]
        selected = select_threshold(development_labels, development_scores)
        self.assertEqual(selected['threshold'], .8)
        before = copy.deepcopy(selected)
        # A test-label oracle would pick a different threshold. Evaluation must not refit it.
        test_labels = [1, 1, 0, 0]
        oracle = select_threshold(test_labels, development_scores)
        self.assertNotEqual(oracle['threshold'], selected['threshold'])
        measured = classifier_metrics(test_labels, development_scores, selected['threshold'])
        self.assertEqual(measured['threshold'], .8)
        self.assertEqual(measured['fp'], 2)
        self.assertEqual(selected, before)
        self.assertEqual(selected['development']['fp'], 0)
        self.assertFalse(selected['real_camera_calibrated'])

    def test_threshold_tie_is_stricter_and_unsafe_calibration_is_rejected(self):
        selected = select_threshold([0, 1], [.1, .99])
        self.assertEqual(selected['threshold'], .99)
        self.assertEqual(selected['development']['tp'], 1)
        with self.assertRaisesRegex(ValueError, 'disable the classifier'):
            select_threshold([0, 1], [1, 1])
        for labels in ([0, 0], [1, 1]):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, 'both positive and negative'):
                select_threshold(labels, [.1, .99])

    def test_quality_uses_usable_brightness_not_texture_or_entropy(self):
        for rgb in ([50, 50, 50], [100, 130, 160], [220, 220, 220]):
            uniform = np.full((12, 16, 3), rgb, dtype=np.uint8)
            self.assertEqual(float(np.var(uniform, axis=(0, 1)).sum()), 0)
            self.assertEqual(pixel_quality(uniform), 1)
        for brightness, expected in [(7, 0), (8, 1), (249, 1), (250, 0)]:
            with self.subTest(brightness=brightness):
                self.assertEqual(pixel_quality(np.full((8, 8, 3), brightness)), expected)
        for invalid in [np.empty((0, 4, 3)), np.ones((8, 8)), np.full((8, 8, 3), np.nan),
                        np.full((8, 8, 3), np.inf), np.full((8, 8, 3), -1),
                        np.full((8, 8, 3), [256, 100, 100])]:
            self.assertEqual(pixel_quality(invalid), 0)

    def test_fit_rejects_invalid_features_labels_and_hyperparameters(self):
        features, labels = self.separable()
        for invalid in [[], [1, 2], np.ones((2, 2, 2)), [[0, np.nan], [1, 2]], [[0, np.inf], [1, 2]]]:
            with self.subTest(features=repr(invalid)), self.assertRaises(ValueError):
                fit(invalid, [0, 1])
        for invalid in [labels[:-1], labels[:, None], np.zeros(10), np.ones(10),
                        [0] * 9 + [2], [0] * 9 + [np.nan], [0] * 9 + [-1]]:
            with self.subTest(labels=repr(invalid)), self.assertRaises(ValueError):
                fit(features, invalid)
        for kwargs in [{'l2': 0}, {'l2': -.1}, {'l2': np.nan}, {'l2': np.inf},
                       {'iterations': 0}, {'iterations': 201}, {'iterations': 2.5}, {'iterations': True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit(features, labels, **kwargs)

    def test_model_artifact_schema_dimensions_and_finiteness_are_validated(self):
        valid = PixelModel(np.zeros(2), np.ones(2), np.array([1., -1]), 0).to_dict()
        invalid_docs = [None, [], {}, dict(valid, version=2), dict(valid, features='detector_misses'),
                        {k: v for k, v in valid.items() if k != 'scale'},
                        dict(valid, mean=[]), dict(valid, mean=[[0, 0]]), dict(valid, scale=[1]),
                        dict(valid, coefficients=[1]), dict(valid, scale=[0, 1]), dict(valid, scale=[-1, 1]),
                        dict(valid, coefficients=[1, np.inf]), dict(valid, mean=[np.nan, 0]),
                        dict(valid, bias=np.nan), dict(valid, bias=[0])]
        for invalid in invalid_docs:
            with self.subTest(document=repr(invalid)), self.assertRaises(ValueError):
                PixelModel.from_dict(invalid)
        model = PixelModel.from_dict(valid)
        for invalid in [[], [[1]], [[1, 2, 3]], [[1, np.nan]], [[1, np.inf]]]:
            with self.subTest(features=repr(invalid)), self.assertRaises(ValueError):
                model.scores(invalid)

    def test_metric_and_threshold_inputs_require_finite_probabilities(self):
        for labels, scores in [([], []), ([0, 1], [.2]), ([[0, 1]], [[.2, .9]]),
                               ([0, 2], [.2, .9]), ([0, np.nan], [.2, .9]),
                               ([0, 1], [.2, np.nan]), ([0, 1], [.2, np.inf]),
                               ([0, 1], [-.1, .9]), ([0, 1], [.2, 1.01])]:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                classifier_metrics(labels, scores, .8)
            with self.assertRaises(ValueError):
                select_threshold(labels, scores)
        for threshold in [np.nan, np.inf, -.01, 1.01]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                classifier_metrics([0, 1], [.2, .9], threshold)


if __name__ == '__main__':
    unittest.main()
