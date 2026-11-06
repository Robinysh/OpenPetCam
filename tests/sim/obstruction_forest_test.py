"""Unseen nonlinear prediction, bounded trees and robust model snapshots."""
import copy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.obstruction_forest import IMAGE_SCHEMA, ForestModel, fit
from sim.obstruction_scene import image_features


class ObstructionForestTest(unittest.TestCase):
    def data(self, n=300, seed=39):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, (n, 2))
        return x, (x[:, 0] * x[:, 1] > 0).astype(int)

    def model(self, **kw):
        options = dict(seed=8, n_trees=8, max_depth=5, min_leaf=4)
        options.update(kw)
        return fit(*self.data(), **options)

    def test_unseen_xor_generalization_needs_nonlinear_interactions(self):
        model = fit(*self.data(1600, 1), seed=2)
        unseen, labels = self.data(2000, 902)
        self.assertGreater(np.mean((model.scores(unseen) >= .5) == labels), .97)
        # All four quadrants must work; a single-threshold rule cannot pass.
        probes = np.array([[-.8, -.8], [-.8, .8], [.8, -.8], [.8, .8]])
        np.testing.assert_array_equal(model.scores(probes) >= .5, [True, False, False, True])

    def test_fit_and_prediction_are_deterministic_and_do_not_mutate_inputs(self):
        x, y = self.data()
        originals = x.copy(), y.copy()
        a, b = fit(x, y, seed=19), fit(x, y, seed=19)
        self.assertEqual(a.to_dict(), b.to_dict())
        np.testing.assert_array_equal(a.scores(x), b.scores(x))
        self.assertFalse(np.array_equal(a.scores(x), fit(x, y, seed=20).scores(x)))
        np.testing.assert_array_equal(x, originals[0])
        np.testing.assert_array_equal(y, originals[1])

    def test_bootstrap_leaf_scores_are_bounded_and_defined_on_constant_features(self):
        x, y = np.ones((80, 3)), np.tile([0, 1], 40)
        model = fit(x, y, seed=31)
        scores = model.scores(np.array([[1, 1, 1], [2, 3, 4]]))
        self.assertTrue(np.all((scores > 0) & (scores < 1)))
        self.assertTrue(all(len(t) == 1 for t in model.trees))
        self.assertEqual(scores[0], scores[1])
        self.assertAlmostEqual(scores[0], .5, delta=.08)

    def test_tree_depth_size_bootstrap_count_and_leaf_bounds(self):
        model = self.model(max_depth=3, min_leaf=8)
        self.assertEqual(len(model.trees), 8)
        for tree in model.trees:
            self.assertLessEqual(len(tree), 15)
            self.assertEqual(tree[0].count, 300)
            pending = [(0, 0)]
            while pending:
                index, depth = pending.pop()
                self.assertLessEqual(depth, 3)
                node = tree[index]
                if node.feature >= 0:
                    left, right = tree[node.left], tree[node.right]
                    self.assertGreaterEqual(min(left.count, right.count), 8)
                    self.assertEqual(left.count + right.count, node.count)
                    self.assertEqual(left.positives + right.positives, node.positives)
                    pending.extend([(node.left, depth + 1), (node.right, depth + 1)])

    def test_scores_validate_feature_schema_without_label_or_scene_inputs(self):
        model = self.model()
        for features in [[], [1, 2], [[1, 2, 3]], [[np.nan, 0]], [['1', '2']]]:
            with self.subTest(features=features), self.assertRaises(ValueError):
                model.scores(features)
        with self.assertRaisesRegex(ValueError, 'image feature schema'):
            model.predict(np.zeros((8, 8, 3), dtype=np.uint8))
        with self.assertRaises(TypeError):
            model.scores([[1, 2]], labels=[1])

    def test_raw_rgb_prediction_uses_exactly_the_public_pixel_descriptor(self):
        colors = [(40, 40, 80), (40, 210, 80), (210, 40, 80), (210, 210, 80)]
        images = [np.full((12, 16, 3), color, dtype=np.uint8) for color in colors]
        x = np.array([image_features(image) for image in images] * 8)
        labels = np.tile([1, 0, 0, 1], 8)
        model = fit(x, labels, seed=77, n_trees=8, max_depth=4, min_leaf=1, max_features=129)
        self.assertEqual(model.feature_schema, IMAGE_SCHEMA)
        self.assertEqual(model.n_features, 129)
        for image in images:
            self.assertEqual(model.predict(image), float(model.scores([image_features(image)])[0]))
        np.testing.assert_array_equal([model.predict(image) >= .5 for image in images], [1, 0, 0, 1])
        with self.assertRaises(ValueError):
            model.predict(images[0].astype(float))

    def test_json_roundtrip_and_snapshots_are_detached_from_model(self):
        model = self.model()
        doc = model.to_dict()
        restored = ForestModel.from_dict(json.loads(json.dumps(doc, allow_nan=False)))
        np.testing.assert_array_equal(model.scores(self.data()[0]), restored.scores(self.data()[0]))
        doc['trees'][0][0]['positives'] = 9999
        self.assertNotEqual(model.to_dict(), doc)
        with self.assertRaises(FrozenInstanceError):
            model.seed = 2
        with self.assertRaises(FrozenInstanceError):
            model.trees[0][0].threshold = 9

    def test_corrupted_schema_configuration_counts_and_links_are_rejected(self):
        model = self.model()
        doc = model.to_dict()
        bad = [dict(doc, version=2), dict(doc, model='other'), dict(doc, feature_schema='pixels_v99'),
               dict(doc, feature_schema=IMAGE_SCHEMA), dict(doc, n_features=0),
               dict(doc, max_depth=9), dict(doc, trees=[]), dict(doc, trees=doc['trees'] * 9)]
        for changes in [dict(threshold=float('nan')), dict(feature=999), dict(left=0),
                        dict(count=9999), dict(positives=-1), dict(left=True), dict(extra=1)]:
            mutated = copy.deepcopy(doc)
            mutated['trees'][0][0].update(changes)
            bad.append(mutated)
        unreachable = copy.deepcopy(doc)
        unreachable['trees'][0].append(dict(feature=-1, threshold=0, left=-1, right=-1, count=8, positives=0))
        bad.append(unreachable)
        shallow = copy.deepcopy(doc)
        shallow['max_depth'] = 1
        bad.append(shallow)
        for corrupt in bad:
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                ForestModel.from_dict(corrupt)

    def test_training_labels_and_bounds_are_validated(self):
        x, y = self.data()
        for labels in [y[:-1], np.ones(len(y)), np.full(len(y), np.nan), y.astype(str),
                       np.where(y, .5, 0)]:
            with self.subTest(labels=labels[:3]), self.assertRaises(ValueError):
                fit(x, labels)
        for kw in [dict(seed=-1), dict(seed=True), dict(n_trees=65), dict(n_trees=0),
                   dict(max_depth=9), dict(min_leaf=0), dict(max_features=3),
                   dict(n_thresholds=33), dict(feature_schema=IMAGE_SCHEMA)]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                fit(x, y, **kw)
        with self.assertRaises(ValueError):
            fit([[np.inf, 1], [0, 2]], [0, 1])
        with self.assertRaises(ValueError):
            fit(np.zeros((2, 1025)), [0, 1])


if __name__ == '__main__':
    unittest.main()
