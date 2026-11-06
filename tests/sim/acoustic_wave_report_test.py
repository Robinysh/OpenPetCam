"""Self-contained report contracts with tiny, synthetic signed 3-D slices."""
import base64
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.wave_report import render


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__();self.items=[];self.current=None

    def handle_starttag(self, tag, attrs):
        if tag=='script':
            self.current=dict(attrs);self.current['text']='';self.items.append(self.current)

    def handle_data(self,data):
        if self.current is not None:self.current['text']+=data

    def handle_endtag(self,tag):
        if tag=='script':self.current=None


class WaveReportTest(unittest.TestCase):
    def fixture(self,folder):
        methods=['wave_known_source','wave_blind32','arbitrary_source_control']
        rows=[]
        for name in methods:
            rows.append(dict(scene='fine_same_prior',method=name,case_id='case',source_id='source<script>bad</script>',
                             source_kind='observed_room_replay',kind='single',true_position_m=[0,0,0],
                             position_m=None if name=='arbitrary_source_control' else [1,0,0],
                             distance_error_m=None if name=='arbitrary_source_control' else 1,
                             bearing_error_deg=None if name=='arbitrary_source_control' else 20,
                             normalized_residual=None if name=='arbitrary_source_control' else .1,
                             status='conditional',accepted=False))
        data={'protocol':{'input_sha256':{'absent-source.py':'0'*64},'frequencies_hz':[80,220],
                          'source_waveforms':[{}]*4,'assumptions':['An <uncertain> box']},
              'summary':{'by_scene':{'fine_same_prior':{}},'methods':methods,'real_acceptance':False,'input_hashes_match':True},
              'results':rows,'cases':[{'id':'case'}],
              'rooms':{'nominal':{'shape_xyz':[8,9,10],'spacing_m':.125,'sound_speed_m_s':343,
                                   'uniform_damping_s_inv':3,'outer_padding_m':.8,'solid_voxels':10}},
              'physics_sources':{'sources':[{'title':'Primary <math>','url':'https://example.org/math.pdf','use':'equation only'}]}}
        for name,value in data.items():(folder/(name+'.json')).write_text(json.dumps(value))
        self.pressure=np.array([[[0.,-1.],[2.,3.]],[[4.,-5.],[6.,7.]]])
        np.savez(folder/'wavefield.npz',times_s=[0,.01],x_m=[-1,1],z_m=[0,2],pressure=self.pressure,
                 solid=[[False,True],[False,False]],source_xz_m=[0,1],slice_y_m=.2)

    def test_signed_field_roundtrip_global_scale_and_no_external_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.fixture(folder)
            text=render(folder,folder/'report.html').read_text();parser=Scripts();parser.feed(text)
            self.assertEqual(len(parser.items),2)
            self.assertTrue(all('src' not in script for script in parser.items))
            data=json.loads(parser.items[0]['text'])
            raw=np.frombuffer(base64.b64decode(data['pressure_b64']),'<f8').reshape(data['shape'])
            np.testing.assert_array_equal(raw,self.pressure)
            self.assertEqual(data['global_abs_max'],7)
            self.assertLess(raw.min(),0)
            self.assertNotIn('fetch(',text);self.assertNotIn('{{',text)
            self.assertIn('Current source hashes do not all match',text)
            self.assertIn('not measured pascals',text)
            self.assertIn('two-dimensional propagation approximation',text)

    def test_scope_escape_and_missing_predictions_stay_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.fixture(folder)
            text=render(folder,folder/'report.html').read_text()
            self.assertIn('An &lt;uncertain&gt; box',text)
            self.assertIn('Primary &lt;math&gt;',text)
            self.assertNotIn('<script>bad</script>',text)
            self.assertIn('0 / 1',text)
            self.assertIn('already contain reverberation',text)
            self.assertIn('not a completed geometry FWI result',text)
            self.assertIn('physical source coordinates',text)
            self.assertIn('always abstains',text)

    def test_invalid_field_is_rejected_instead_of_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.fixture(folder)
            np.savez(folder/'wavefield.npz',times_s=[0,.01],x_m=[-1,1],z_m=[0,2],pressure=np.zeros((2,3,2)),
                     solid=np.zeros((3,2),bool),source_xz_m=[0,1],slice_y_m=.2)
            with self.assertRaises(ValueError):render(folder,folder/'report.html')

    def test_operational_acceptance_cannot_hide_inside_offline_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.fixture(folder)
            rows=json.loads((folder/'results.json').read_text());rows[0]['accepted']=True
            (folder/'results.json').write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'unaccepted offline'):
                render(folder,folder/'report.html')


if __name__=='__main__':unittest.main()
