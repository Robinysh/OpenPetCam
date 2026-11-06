"""Artifact safety and protocol labels with synthetic, non-household fixtures."""
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.search_report import render


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.current = dict(attrs)
            self.current['text'] = ''
            self.scripts.append(self.current)

    def handle_data(self, data):
        if self.current is not None:
            self.current['text'] += data

    def handle_endtag(self, tag):
        if tag == 'script':
            self.current = None


class SearchReportTest(unittest.TestCase):
    def fixture(self, folder, selected='nearest', passes=False):
        stat = dict(n=6, mean_capped_loss_s=70, findable_within_60s=.5,
                    mean_motor_steps=1000, wrong_finds=0, deadline_count=2,
                    mean_score_s=72, found_right=.5)
        paired = {k: {'candidate_minus_base': -1, 'paired_bootstrap_95': [-2, 0]}
                  for k in ('loss_s', 'motor_steps', 'score_s')}
        compare = {'base': stat, 'candidate': stat, 'paired': paired,
                   'by_family': {'absent': {'incumbent': stat, selected: stat}}}
        scenario = dict(target_start=[100, 200], target_end=[150, 200], move_start_s=0,
                        move_duration_s=5, absent=False, occluded=False)
        row = dict(id='synthetic<script>alert(1)</script><img src=x onerror=alert(1)>', family='prior',
                   trace=[[0, 100, 200, 100]], elapsed_s=1, found_right=True,
                   wrong_find=False, deadline=False)
        bank = {'comparison': compare, 'cases': [{'id': row['id'], 'family': 'prior', 'scenario': scenario}],
                'results': {'incumbent': [row], selected: [row]}}
        manifest = {'policies': ['incumbent', selected], 'budget_s': 120,
                    'input_sha256': {'sim/search_bench.py': '0' * 64},
                    'limits': ['Synthetic <constraint>'], 'split': {'test_seed': 21, 'stress_seed': 31}}
        summary = {'selected': selected, 'inputs_unchanged': True,
                   'cancellation': {'incumbent': {'n': 1, 'stopped': 1, 'max_elapsed_s': 7}},
                   'acceptance': {'passes_simulation_gate': passes,
                                  'checks': {'test': {'correct_find_nonregression': passes}}}}
        for name, data in {'manifest': manifest, 'selection': {'development': {'incumbent': stat, selected: stat}},
                           'summary': summary, 'test': bank, 'stress': bank}.items():
            (folder / (name + '.json')).write_text(json.dumps(data))

    def test_embedded_data_escapes_script_end_and_needs_no_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            path = render(folder, folder / 'report.html')
            text = path.read_text()
            parser = Scripts()
            parser.feed(text)
            self.assertEqual(len(parser.scripts), 2)
            self.assertTrue(all('src' not in script for script in parser.scripts))
            data = json.loads(parser.scripts[0]['text'])
            self.assertEqual(data['cases'][0]['id'],
                             'synthetic<script>alert(1)</script><img src=x onerror=alert(1)>')
            self.assertNotIn('<img src=x', text)
            self.assertNotIn('fetch(', text)
            self.assertNotIn('{{', text)
            self.assertIn('Synthetic &lt;constraint&gt;', text)
            self.assertIn('Current prior overlay omitted', text)

    def test_acceptance_uses_every_recorded_guardrail_and_never_installs(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder, selected='<nearest>', passes=False)
            text = render(folder, folder / 'report.html').read_text()
            self.assertIn('Simulation guardrail failed', text)
            self.assertIn('correct find nonregression', text)
            self.assertIn('&lt;nearest&gt;', text)
            self.assertIn('no policy installed', text)
            self.fixture(folder, passes=True)
            text = render(folder, folder / 'report.html').read_text()
            self.assertIn('Finite simulation gate passed', text)
            self.assertIn('installation are separate steps', text)


if __name__ == '__main__':
    unittest.main()
