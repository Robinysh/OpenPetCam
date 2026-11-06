"""Geometry, label isolation and split boundaries for injected RGB obstructions."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.obstruction_scene import (ObstructionScene, Occluder, domain_seeds,
                                   image_features, image_quality)


POSE = (1850, 716)


class ObstructionSceneTest(unittest.TestCase):
    def test_render_is_rgb_only_and_repeated_exposure_is_identical(self):
        scene = ObstructionScene(**domain_seeds('test', 0), occluders=[Occluder()])
        first = scene.render(POSE, 2)
        self.assertIsInstance(first, np.ndarray)
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(first.shape, (90, 160, 3))
        np.testing.assert_array_equal(first, scene.render(POSE, 2))
        scene.truth(POSE, 2)
        np.testing.assert_array_equal(first, scene.render(POSE, 2))
        self.assertEqual(image_features(first).shape, (129,))
        self.assertTrue(np.isfinite(image_features(first)).all())

    def test_partial_crop_does_not_exclude_whole_view(self):
        partial = ObstructionScene(occluders=[Occluder(size_uv=(.7, .8))])
        full = ObstructionScene(occluders=[Occluder(size_uv=(2, 2))])
        truth = partial.truth(POSE, 1)
        self.assertGreater(truth.persistent_fraction, .1)
        self.assertLess(truth.persistent_fraction, .85)
        self.assertFalse(truth.blocked)
        self.assertTrue(full.truth(POSE, 1).blocked)
        cropped = ObstructionScene(occluders=[Occluder(center_uv=(-.1, .5), size_uv=(.7, .8))])
        self.assertLess(cropped.truth(POSE, 1).persistent_fraction, truth.persistent_fraction)

    def test_pose_changes_and_mount_shift_change_injected_visibility(self):
        scene = ObstructionScene(occluders=[Occluder()], mount_offsets={1: (0, 40, 0)})
        self.assertGreater(scene.truth(POSE, 1).persistent_fraction, .1)
        self.assertEqual(scene.truth((2500, 716), 1).persistent_fraction, 0)
        self.assertLess(scene.truth(POSE, 1, setup_version=1).persistent_fraction,
                        scene.truth(POSE, 1).persistent_fraction)
        self.assertFalse(np.array_equal(scene.render(POSE, 1), scene.render(POSE, 1, 1)))

    def test_removal_and_object_change_have_separate_setup_versions(self):
        scene = ObstructionScene(occluders=[Occluder(size_uv=(2, 2), active_versions=(0,), color=(20, 40, 60)),
                                           Occluder(size_uv=(2, 2), active_versions=(1,), color=(90, 150, 110))])
        self.assertTrue(scene.truth(POSE, 0, 0).blocked)
        self.assertTrue(scene.truth(POSE, 0, 1).blocked)
        self.assertFalse(scene.truth(POSE, 0, 2).blocked)
        self.assertFalse(np.array_equal(scene.render(POSE, 0, 0), scene.render(POSE, 0, 1)))
        timed = ObstructionScene(occluders=[Occluder(size_uv=(2, 2), end_s=5)])
        self.assertTrue(timed.truth(POSE, 4.9).blocked)
        self.assertFalse(timed.truth(POSE, 5).blocked)

    def test_transient_person_is_not_persistent_blockage(self):
        scene = ObstructionScene(occluders=[Occluder(role='transient', size_uv=(2, 2), start_s=3, end_s=8)])
        self.assertFalse(scene.truth(POSE, 4).blocked)
        self.assertGreater(scene.truth(POSE, 4).transient_fraction, .85)
        self.assertEqual(scene.truth(POSE, 9).total_injected_fraction, 0)

    def test_depth_compositing_uses_nearest_visible_object(self):
        scene = ObstructionScene(occluders=[Occluder(role='persistent', size_uv=(2, 2), distance_m=.6),
                                           Occluder(role='transient', size_uv=(2, 2), distance_m=.2)])
        truth = scene.truth(POSE, 1)
        self.assertEqual(truth.persistent_fraction, 0)
        self.assertGreater(truth.transient_fraction, .85)

    def test_background_and_object_identities_are_disjoint_between_splits(self):
        groups = {split: [domain_seeds(split, i) for i in range(12)] for split in ('train', 'validation', 'test')}
        for key in ('seed', 'background_seed', 'object_seed'):
            all_seeds = [g[key] for values in groups.values() for g in values]
            self.assertEqual(len(set(all_seeds)), 36)
        a = ObstructionScene(**groups['train'][0])
        b = ObstructionScene(**groups['test'][0])
        self.assertFalse(np.array_equal(a.render(POSE, 0), b.render(POSE, 0)))

    def test_low_light_and_blur_are_rgb_quality_counterexamples(self):
        clear = ObstructionScene(seed=1, noise_std=0)
        dark = ObstructionScene(seed=1, noise_std=0, low_light=.05)
        blurred = ObstructionScene(seed=1, noise_std=0, blur=4)
        self.assertFalse(dark.truth(POSE, 1).blocked)
        self.assertFalse(blurred.truth(POSE, 1).blocked)
        self.assertLess(image_quality(dark.render(POSE, 1))['brightness'], .08)
        self.assertLess(image_quality(blurred.render(POSE, 1))['gradient'],
                        image_quality(clear.render(POSE, 1))['gradient'])

    def test_identical_rgb_can_have_different_hidden_depth_labels(self):
        # A flat near cover matching a blank far wall cannot be distinguished
        # from this single image. Labels and features must not leak that depth.
        kwargs = dict(seed=15, background_kind='blank', background_color=(128, 128, 128), noise_std=0)
        clear = ObstructionScene(**kwargs)
        blocked = ObstructionScene(**kwargs, occluders=[Occluder(size_uv=(2, 2), texture='flat', color=(128, 128, 128))])
        a, b = clear.render(POSE, 0), blocked.render(POSE, 0)
        self.assertFalse(clear.truth(POSE, 0).blocked)
        self.assertTrue(blocked.truth(POSE, 0).blocked)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(image_features(a), image_features(b))

    def test_optional_photo_is_sampled_as_unlabelled_background(self):
        yy, xx = np.mgrid[0:32, 0:64]
        photo = np.stack([xx * 4, yy * 8, np.full(xx.shape, 128)], axis=2).astype(np.uint8)
        scene = ObstructionScene(background=photo, noise_std=0)
        frame = scene.render(POSE, 0)
        self.assertEqual(frame.shape, (90, 160, 3))
        self.assertTrue(scene.truth(POSE, 0).injected_only)
        self.assertFalse(scene.truth(POSE, 0).background_labelled_clear)
        self.assertEqual(scene.truth(POSE, 0).persistent_fraction, 0)
        self.assertFalse(np.array_equal(frame, scene.render((2300, 716), 0)))


if __name__ == '__main__':
    unittest.main()
