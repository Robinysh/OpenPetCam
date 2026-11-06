"""Analytic propagation checks, independent of saved room data or fitted echoes.

These verify the implemented approximation, not real-world localization accuracy.
Run: python3 tests/sim/acoustic_localization_model_test.py
"""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.geometry import Rig
from sim.acoustics.localization_model import (
    Surface, RoomModel, load_room, build_paths, render_transfer, render_ir,
)


def panel_x(x=0., y=(-1., 1.), z=(0., 2.), coefficient=.7, ident="panel"):
    return Surface(ident, (x, 0, 0), (1, 0, 0),
                   ((x, y[0], z[0]), (x, y[1], z[0]),
                    (x, y[1], z[1]), (x, y[0], z[1])), coefficient)


def shoebox(lengths=(4., 3., 5.)):
    # Each face is a finite rectangle. Independent image-source expectations
    # below use coordinate sign changes, not the production reflection helper.
    surfaces = []
    for axis, length in enumerate(lengths):
        other = [i for i in range(3) if i != axis]
        for side, coordinate in enumerate((0., length)):
            vertices = []
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                point = np.zeros(3)
                point[axis] = coordinate
                point[other[0]] = u * lengths[other[0]]
                point[other[1]] = v * lengths[other[1]]
                vertices.append(tuple(point))
            normal = np.eye(3)[axis]
            surfaces.append(Surface(f"{axis}_{side}", vertices[0], tuple(normal), tuple(vertices),
                                    .5 + .05 * (2 * axis + side)))
    return RoomModel(surfaces)


class PathTests(unittest.TestCase):
    def test_free_field_known_distance_preserves_fractional_phase_and_amplitude(self):
        paths = build_paths(RoomModel(()), (0, 0, 0), (3, 4, 0), sound_speed=340)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].length_m, 5.)
        self.assertEqual(paths[0].delay_s, 5 / 340)
        self.assertEqual(paths[0].gain, 1 / (20 * math.pi))
        f = np.array([0., 123.45, 3210.])
        expected = np.exp(-2j * np.pi * f * 5 / 340) / (20 * math.pi)
        np.testing.assert_allclose(render_transfer(paths, f), expected, atol=1e-15)
        self.assertFalse(paths.metadata["same_model_synthetic_results_are_independent_validation"])

    def test_first_reflection_matches_independent_triangle_and_finite_panel(self):
        model = RoomModel((panel_x(),))
        paths = build_paths(model, (1, 0, 0), (1, 0, 2))
        self.assertEqual(len(paths), 2)
        reflected = paths[1]
        self.assertEqual(reflected.surface_ids, ("panel",))
        np.testing.assert_allclose(reflected.points_m[1], (0, 0, 1), atol=1e-15)
        self.assertAlmostEqual(reflected.length_m, 2 * math.sqrt(2))
        self.assertAlmostEqual(reflected.gain, .7 / (8 * math.pi * math.sqrt(2)))
        outside = RoomModel((panel_x(z=(2.1, 3)),))
        self.assertEqual(len(build_paths(outside, (1, 0, 0), (1, 0, 2))), 1)
        too_short = RoomModel((panel_x(y=(.1, 1)),))
        self.assertEqual(len(build_paths(too_short, (1, 0, 0), (1, 0, 2))), 1)

    def test_shoebox_all_six_first_order_images_and_reciprocity(self):
        lengths = (4., 3., 5.)
        model = shoebox(lengths)
        source, receiver = np.array((.7, 1.1, 1.4)), np.array((2.2, 1.8, 3.1))
        paths = build_paths(model, source, receiver)
        self.assertEqual(len(paths), 7)
        by_surface = {p.surface_ids: p for p in paths}
        for axis, length in enumerate(lengths):
            for side, coordinate in enumerate((0, length)):
                image = source.copy()
                image[axis] = 2 * coordinate - source[axis]
                distance = np.linalg.norm(image - receiver)
                p = by_surface[(f"{axis}_{side}",)]
                self.assertAlmostEqual(p.length_m, distance, places=13)
                self.assertAlmostEqual(p.gain, (.5 + .05 * (2 * axis + side)) / (4 * math.pi * distance))
        reverse = {p.surface_ids: p for p in build_paths(model, receiver, source)}
        for ident, p in by_surface.items():
            self.assertAlmostEqual(p.delay_s, reverse[ident].delay_s, places=14)
            self.assertAlmostEqual(p.gain, reverse[ident].gain, places=14)
            np.testing.assert_allclose(p.points_m, reverse[ident].points_m[::-1], atol=1e-14)

    def test_blocked_direct_and_absorber_are_not_openings(self):
        wall = panel_x(coefficient=0)
        blocked = build_paths(RoomModel((wall,)), (-1, 0, 1), (1, 0, 1), timing="direct_relative")
        self.assertFalse(blocked.direct_visible)
        self.assertEqual(len(blocked), 0)
        self.assertTrue(blocked.metadata["direct_path_blocked"])
        self.assertFalse(blocked.metadata["direct_reference_observed"])
        np.testing.assert_array_equal(render_transfer(blocked, [0, 1000]), [0j, 0j])
        outside = build_paths(RoomModel((wall,)), (-1, 2, 1), (1, 2, 1))
        self.assertTrue(outside.direct_visible)

    def test_reflection_segment_occlusion_without_blocking_direct(self):
        blocker = Surface("blocker", (0, 0, .5), (0, 0, 1),
                          ((.4, -.2, .5), (.6, -.2, .5), (.6, .2, .5), (.4, .2, .5)), 0)
        model = RoomModel((panel_x(), blocker))
        paths = build_paths(model, (1, 0, 0), (1, 0, 2))
        self.assertTrue(paths.direct_visible)
        self.assertEqual([p.surface_ids for p in paths], [()])

    def test_plane_sign_and_polygon_winding_do_not_change_paths(self):
        original = panel_x()
        flipped = replace(original, normal=(-1, 0, 0), vertices_m=original.vertices_m[::-1])
        for surface in (original, flipped):
            self.assertTrue(surface.contains((0, 0, 1)))
            self.assertFalse(surface.contains((0, 0, 3)))
            self.assertTrue(surface.blocks_segment((-1, 0, 1), (1, 0, 1)))
        a = build_paths(RoomModel((original,)), (1, .2, 0), (2, -.3, 2))
        b = build_paths(RoomModel((flipped,)), (1, .2, 0), (2, -.3, 2))
        np.testing.assert_allclose(render_transfer(a, np.arange(3000)), render_transfer(b, np.arange(3000)))

    def test_coplanar_grazing_is_blocked_only_where_panel_intersects(self):
        surface = panel_x()
        self.assertTrue(surface.blocks_segment((0, 0, -1), (0, 0, 3)))
        self.assertFalse(surface.blocks_segment((0, 2, -1), (0, 2, 3)))
        self.assertFalse(surface.blocks_segment((0, 0, -1), (0, 0, -.1)))
        self.assertFalse(surface.blocks_segment((1, 0, 1), (0, 0, 1)))

    def test_direct_relative_preserves_magnitude_and_only_changes_global_phase(self):
        model = RoomModel((panel_x(coefficient=-.8),))
        absolute = build_paths(model, (1, .2, 0), (1.4, -.1, 2))
        relative = build_paths(model, (1, .2, 0), (1.4, -.1, 2), timing="direct_relative")
        f = np.linspace(0, 7600, 701)
        a, r = render_transfer(absolute, f), render_transfer(relative, f)
        np.testing.assert_allclose(a, r * np.exp(-2j * np.pi * f * absolute.direct_distance_m / 343), atol=2e-15)
        np.testing.assert_allclose(np.abs(a), np.abs(r), atol=2e-15)
        self.assertEqual(relative[0].delay_s, 0.)
        self.assertEqual([p.gain for p in absolute], [p.gain for p in relative])
        self.assertLess(relative[1].gain, 0)
        # Squaring path amplitudes separately would lose this interference.
        incoherent = sum(p.gain ** 2 for p in relative)
        self.assertGreater(np.ptp(np.abs(r) ** 2), incoherent)

    def test_rotated_rig_mic_axis_and_signed_dipole_are_explicit(self):
        pose = Rig().pose(.8, -.2)
        room = RoomModel(())
        front, back = pose.receiver_m + 2 * pose.receiver_axis, pose.receiver_m - 2 * pose.receiver_axis
        cardioid = build_paths(room, front, pose.receiver_m, pose.receiver_axis, .5)
        rejected = build_paths(room, back, pose.receiver_m, pose.receiver_axis, .5)
        dipole = build_paths(room, back, pose.receiver_m, pose.receiver_axis, 1.)
        self.assertAlmostEqual(cardioid[0].gain, 1 / (8 * math.pi))
        self.assertAlmostEqual(rejected[0].gain, 0., places=15)
        self.assertAlmostEqual(dipole[0].gain, -1 / (8 * math.pi))
        self.assertFalse(cardioid.metadata["mic_directivity_calibrated"])
        self.assertGreater(np.linalg.norm(pose.receiver_m - pose.lens_m), .01)

    def test_bad_parameters_do_not_silently_change_model(self):
        for kwargs in ({"sound_speed": 0}, {"sound_speed": float("nan")},
                       {"mic_directivity": .5}, {"mic_directivity": 2},
                       {"timing": "measured"}, {"max_order": 2}, {"max_order": True},
                       {"max_order": 1.0}, {"receiver_axis": (0, 0, 2)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_paths(RoomModel(()), (0, 0, 0), (1, 0, 0), **kwargs)
        with self.assertRaises(ValueError):
            build_paths(RoomModel(()), (0, 0, 0), (0, 0, 0))
        self.assertEqual(len(build_paths(RoomModel((panel_x(),)), (1, 0, 0), (1, 0, 2), max_order=0)), 1)


class RenderTests(unittest.TestCase):
    def test_fractional_ir_matches_exact_complex_sum_below_transition_band(self):
        paths = build_paths(RoomModel((panel_x(coefficient=-.6),)), (1, .1, 0), (1.3, -.2, 2.3))
        fs, onset = 16000, .02
        impulse = render_ir(paths, fs, .125, onset)
        f = np.fft.rfftfreq(len(impulse), 1 / fs)
        expected = render_transfer(paths, f) * np.exp(-2j * np.pi * f * onset)
        keep = f <= .35 * fs
        np.testing.assert_allclose(np.fft.rfft(impulse)[keep], expected[keep], atol=2e-6, rtol=1e-4)
        self.assertAlmostEqual(impulse.sum(), sum(p.gain for p in paths), places=14)
        self.assertTrue(np.any(impulse < 0))
        self.assertGreater(np.count_nonzero(np.abs(impulse) > 1e-6), 8)

    def test_fractional_delay_is_not_rounded_and_window_is_not_silently_cut(self):
        fs, delay = 16000, 123.25 / 16000
        paths = build_paths(RoomModel(()), (0, 0, 0), (343 * delay, 0, 0))
        impulse = render_ir(paths, fs, .05, .01)
        center = (delay + .01) * fs
        measured = np.dot(np.arange(len(impulse)), impulse) / impulse.sum()
        self.assertAlmostEqual(measured, center, delta=.002)
        self.assertNotEqual(round(measured), measured)
        relative = build_paths(RoomModel(()), (0, 0, 0), (1, 0, 0), timing="direct_relative")
        with self.assertRaisesRegex(ValueError, "truncates"):
            render_ir(relative, onset_s=0)
        with self.assertRaisesRegex(ValueError, "truncates"):
            render_ir(paths, duration_s=.005)
        with self.assertRaises(ValueError):
            render_transfer(paths, [float("nan")])


class LoaderTests(unittest.TestCase):
    def manifest(self):
        return {"inputs": {"code": "fixture"}, "camera": {"level_tilt": 700}, "room": {
            "floor_height_m": .8, "ceiling_m": 2.6, "ceiling_assumed": True, "overhead_m": .2,
            "provenance": {"floor": "unvalidated earlier echo estimate"},
            "walls": [{"label": "furniture", "a": [-1, 2], "b": [1, 2], "source": "visual estimate"}]}}

    def test_manifest_axes_bounds_provenance_and_materials_remain_assumptions(self):
        document = self.manifest()
        model = load_room(document, {"*": .5, "furniture": -.2}, {"wall_0": (-.8, .1)})
        self.assertEqual([s.surface_id for s in model.surfaces], ["floor", "ceiling", "wall_0"])
        self.assertEqual(model.surfaces[0].point_m, (0, -.8, 0))
        self.assertAlmostEqual(model.surfaces[1].point_m[1], 1.8)
        np.testing.assert_allclose(model.surfaces[2].vertices_m,
                                   [(-1, -.8, 2), (1, -.8, 2), (1, .1, 2), (-1, .1, 2)])
        self.assertEqual(model.surfaces[2].reflection, -.2)
        self.assertEqual(model.declared_wall_extents[0]["top_y_m"], .1)
        self.assertEqual(model.declared_bounds_m["z_min"], 2)
        self.assertFalse(model.declared_bounds_m["is_enclosed_or_walkable_region"])
        self.assertTrue(model.provenance["geometry_is_uncertain_prior"])
        self.assertFalse(model.provenance["acoustic_materials_calibrated"])
        self.assertFalse(model.provenance["floor_horizontal_extent_known"])
        self.assertTrue(model.provenance["overhang_not_included"])
        self.assertEqual(model.provenance["source_manifest_inputs"], document["inputs"])
        self.assertFalse(model.provenance["legacy_acoustic_fit_is_validation"])

    def test_exact_source_bytes_are_hashed(self):
        raw = json.dumps(self.manifest(), indent=2).encode() + b"\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_bytes(raw)
            model = load_room(path)
        self.assertEqual(model.provenance["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(model.provenance["source"], str(path))

    def test_unknown_overrides_rejected_instead_of_ignored(self):
        for kwargs in ({"reflection_coefficients": {"bogus": .2}},
                       {"wall_height_limits": {"floor": (0, 1)}},
                       {"wall_height_limits": {"wall_0": (1, 0)}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                load_room(self.manifest(), **kwargs)

    def test_invalid_polygon_and_active_reflector_rejected(self):
        for kwargs in ({"normal": (2, 0, 0)}, {"reflection": 1.1}, {"reflection": float("nan")},
                       {"vertices_m": ((0, 0, 0), (0, 1, 0), (1, 1, 1))},
                       {"vertices_m": ((0, 0, 0), (0, 1, 0), (0, .2, .2), (0, 0, 1))}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(panel_x(), **kwargs)
        with self.assertRaises(ValueError):
            RoomModel((panel_x(), panel_x()))


if __name__ == "__main__":
    unittest.main()
