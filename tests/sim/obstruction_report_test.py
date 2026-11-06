"""Offline HTML contracts with synthetic fixtures and an embedded tiny PNG."""
import base64
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.obstruction_report import render


PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aI5sAAAAASUVORK5CYII=')


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts, self.images, self.svgs = [], [], 0

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.scripts.append(dict(attrs))
        if tag == 'img':
            self.images.append(dict(attrs))
        if tag == 'svg':
            self.svgs += 1


class ObstructionReportTest(unittest.TestCase):
    def fixture(self, folder, gate=False):
        counts = dict(opportunities=5, inspections=3, skips=1, useful_skips=1,
                      false_skips=0, deferred_probes=1, missed_pet_opportunities=0)
        family = dict(episodes=1, learned_before_change=1, learned_cases_requiring_recovery=1,
                      learned_cases_recovered_within_bound=1, recovery_bound_s=120,
                      max_first_inspection_after_change_s=30)
        bank = {'policies': {name: dict(counts) for name in ('always_inspect', 'empty_means_blocked', 'visual_memory')},
                'by_family': {'removed': family}}
        metric = dict(n=8, tp=2, fp=1, tn=4, fn=1, recall=2/3, precision=2/3)
        manifest = {'input_sha256': {'sim/obstruction_bench.py': '0' * 64},
                    'lifecycle': {'end_s': 485, 'tick_s': 15}, 'limits': ['Synthetic <depth> limitation']}
        selection = {'model': 'forest<script>bad</script>', 'threshold': .9,
                     'development': dict(metric), 'model_candidates': []}
        summary = {'pixel_test': metric, 'banks': {'lifecycle-heldout': bank, 'photo-sensitivity': bank},
                   'selected_threshold': .9, 'simulation_gate': gate,
                   'simulation_checks': {'heldout_pixel_fp_zero': gate}, 'inputs_unchanged': True}
        samples = [dict(at_s=5, pose=[1850, 600], state='unknown', coverage=0, deferred=False, skip=False),
                   dict(at_s=20, pose=[1850, 600], state='reinspection_due', coverage=0, deferred=True, skip=False)]
        episode = {'family': 'removed', 'id': 'example</summary><script>bad</script>',
                   'samples': samples, 'counts': {'visual_memory': counts}}
        examples = [dict(path='tiny.png', split='test<script>bad</script>', group=0,
                         variant=2, injected_coverage=.3, score=.1)]
        data = {'manifest': manifest, 'selection': selection, 'summary': summary,
                'pixel-results': {'training': [{}], 'development': [{}]},
                'examples': examples, 'lifecycle-heldout': {'episodes': [episode]},
                'photo-sensitivity': {'episodes': [episode]}}
        for name, value in data.items():
            (folder / (name + '.json')).write_text(json.dumps(value))
        (folder / 'tiny.png').write_bytes(PNG)

    def test_self_contained_escaped_report_has_explicit_scope_and_failed_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            text = render(folder, folder / 'report.html').read_text()
            tags = Tags()
            tags.feed(text)
            self.assertEqual(tags.scripts, [])
            self.assertEqual(len(tags.images), 1)
            self.assertTrue(tags.images[0]['src'].startswith('data:image/png;base64,'))
            self.assertGreaterEqual(tags.svgs, 4)
            self.assertNotIn('<script>', text)
            self.assertIn('&lt;script&gt;', text)
            self.assertNotIn('{{', text)
            self.assertNotIn('fetch(', text)
            self.assertIn('finite simulation gate did not pass', text)
            self.assertIn('background unlabelled', text)
            self.assertIn('zero denominator is no recovery evidence', text)
            self.assertIn('Input hashes do not all match', text)
            self.assertIn('scripted symbolic', text)
            self.assertIn('Identical RGB', text)

    def test_passed_synthetic_gate_still_does_not_authorize_deployment(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder, gate=True)
            text = render(folder, folder / 'report.html').read_text()
            self.assertIn('finite simulation gate passed', text)
            self.assertIn('Neither outcome authorizes deployment', text)
            self.assertIn('Real-camera validation is still open', text)

    def test_gallery_cannot_embed_files_outside_the_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            rows = json.loads((folder / 'examples.json').read_text())
            rows[0]['path'] = '../private.png'
            (folder / 'examples.json').write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'inside the experiment'):
                render(folder, folder / 'report.html')

    def test_reference_prerequisites_are_visible_without_claiming_clear_real_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            manifest = json.loads((folder / 'manifest.json').read_text())
            manifest['reference_protocol'] = {'selected': True, 'acquired_at_s': -60}
            (folder / 'manifest.json').write_text(json.dumps(manifest))
            text = render(folder, folder / 'report.html').read_text()
            self.assertIn('selected model requires a registered reference', text)
            self.assertIn('original background is not certified clear', text)
            self.assertIn('obstruction already present in the reference', text)
            self.assertIn('not implemented at the production camera boundary', text)

    def test_reference_gallery_embeds_both_observed_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            rows = json.loads((folder / 'examples.json').read_text())
            rows[0]['reference_path'] = 'tiny.png'
            (folder / 'examples.json').write_text(json.dumps(rows))
            text = render(folder, folder / 'report.html').read_text()
            tags = Tags()
            tags.feed(text)
            self.assertEqual(len(tags.images), 2)
            self.assertTrue(all(image['src'].startswith('data:image/png;base64,') for image in tags.images))
            self.assertIn('Current RGB', text)
            self.assertIn('Supplied reference', text)


if __name__ == '__main__':
    unittest.main()
