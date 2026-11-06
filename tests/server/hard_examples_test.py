"""Mining includes proposal misses and preserves labels, provenance and held-out groups."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
import hard_examples as H


class HardExamplesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frames = self.root/'frames'
        self.frames.mkdir()
        self.day_map = {'2026-01-01':'train', '2026-01-02':'dev', '2026-01-03':'test'}

    def image(self, day='2026-01-01', name='12-00.png', number=0, root=None):
        path = (root or self.frames)/day/name
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new('RGB', (80, 48), (20 + number % 200, 30, 60))
        draw = ImageDraw.Draw(image)
        draw.rectangle((number % 40, 3, number % 40 + 8, 20), fill=(180, 200, 50))
        image.save(path)
        return path

    def manifest(self, records=None):
        if records is None:
            records, problems = H.inputs([self.frames])
        else:
            problems = []
        return H.build_manifest(records, self.day_map, problems=problems)

    def annotation(self, label='visible', boxes=None, source='review'):
        return {'label':label,'boxes':[[.1,.2,.4,.6]] if boxes is None else boxes,
                'source':source,'reviewer':'independent-visual-review'}

    def refreeze(self, manifest):
        manifest.pop('manifest_sha256', None)
        manifest['manifest_sha256'] = hashlib.sha256(H.canonical(manifest)).hexdigest()
        return manifest

    def test_no_proposal_frame_is_mined_as_unreviewed_not_absent(self):
        path = self.image()
        records, _ = H.inputs([self.frames])
        sha = H.digest(path)
        manifest = H.build_manifest(records, self.day_map, scores={sha:{'proposal_count':0}})
        row = manifest['entries'][0]
        self.assertEqual(row['label'], 'unreviewed')
        self.assertIn('no_proposal_needs_visual_review_not_absence', row['reasons'])
        self.assertEqual(len(H.review_queue(manifest)['entries']),1)
        files = H.exports(manifest,{})
        self.assertEqual(files['readiness.json']['omitted'], {'unreviewed':1})
        self.assertFalse(files['readiness.json']['training_ready'])

    def test_report_visibility_is_priority_lead_not_a_box_label(self):
        reports = self.root/'reports';folder=reports/'20260101-130000';folder.mkdir(parents=True)
        self.image(root=reports,day='20260101-130000',name='frame.jpg',number=2)
        (folder/'report.json').write_text(json.dumps({'note':'I can see a nose','client':{'shown_status':'Out of sight'}}))
        manifest = H.build_manifest(H.inputs(report_roots=[reports])[0],self.day_map)
        row=manifest['entries'][0]
        self.assertEqual((row['split'],row['label']),('train','unreviewed'))
        self.assertGreater(row['priority'],100)
        self.assertNotIn('boxes',row)
        self.assertEqual(row['aliases'][0]['reported_status'],'Out of sight')

    def test_unknown_clock_and_conflicting_day_are_quarantined(self):
        first=self.image(name='frame000123.png')
        second=self.image(name='14-01.png',number=1)
        records=[{'path':str(first),'source_type':'recording'}, {'path':str(second),'day':'2026-01-02'}]
        manifest=self.manifest(records)
        self.assertTrue(all(r['split']=='quarantine' for r in manifest['entries']))
        self.assertEqual({reason for r in manifest['entries'] for reason in r['issues']}, {'capture_time_unknown','conflicting_days'})
        self.assertEqual(H.review_queue(manifest)['entries'],[])
        self.assertEqual(len(H.review_queue(manifest,splits=('quarantine',))['entries']),2)

    def test_frozen_day_map_quarantines_unknown_days(self):
        self.image(day='2026-01-04')
        row=self.manifest()['entries'][0]
        self.assertEqual(row['split'],'quarantine')
        self.assertIn('day_missing_from_frozen_map',row['issues'])

    def test_same_day_never_crosses_partitions_and_default_split_is_deterministic(self):
        for i in range(8):self.image(name=f'{12+i:02d}-00.png',number=i)
        records,_=H.inputs([self.frames])
        first=H.build_manifest(records);second=H.build_manifest(list(reversed(records)))
        self.assertEqual(first,second)
        self.assertEqual(len({r['split'] for r in first['entries']}),1)

    def test_exact_duplicate_across_split_days_is_one_quarantined_record(self):
        first=self.image();second=self.frames/'2026-01-03'/'12-00.png';second.parent.mkdir()
        second.write_bytes(first.read_bytes())
        manifest=self.manifest()
        self.assertEqual(len(manifest['entries']),1)
        row=manifest['entries'][0]
        self.assertEqual((row['split'],len(row['aliases'])),('quarantine',2))
        self.assertIn('duplicate_cross_split_conflict',row['issues'])

    def test_same_recording_episode_across_days_cannot_leak(self):
        first=self.image();second=self.image(day='2026-01-03',number=1)
        manifest=self.manifest([{'path':str(path),'group':'cross-midnight-episode'} for path in (first,second)])
        self.assertTrue(all(r['split']=='quarantine' for r in manifest['entries']))
        self.assertTrue(all('recording_group_cross_split_conflict' in r['issues'] for r in manifest['entries']))

    def test_near_similar_frames_are_kept_and_only_known_scene_gets_diversity_hint(self):
        first=self.image();second=self.image(name='12-01.png',number=1)
        plain=self.manifest()
        self.assertEqual(len(plain['entries']),2)
        self.assertFalse(any('near_duplicate_of' in r for r in plain['entries']))
        known=self.manifest([{'path':str(p),'scene':'fixed-counter'} for p in (first,second)])
        self.assertEqual(len(known['entries']),2)
        # dHash differences may exceed the conservative threshold; no labels propagate.
        self.assertTrue(all(r['label']=='unreviewed' for r in known['entries']))

    def test_queue_is_bounded_diverse_and_test_is_sealed_by_default(self):
        for day in self.day_map:
            for i in range(8):self.image(day=day,name=f'{12+i//4:02d}-{i%4:02d}.png',number=i+int(day[-1])*20)
        manifest=self.manifest();queue=H.review_queue(manifest,limit=7,per_group=2,per_source=6)
        self.assertLessEqual(len(queue['entries']),6)
        self.assertTrue(all(r['split']!='test' for r in queue['entries']))
        groups={}
        for row in queue['entries']:groups[row['group']]=groups.get(row['group'],0)+1
        self.assertTrue(all(n<=2 for n in groups.values()))
        self.assertTrue(any(r['selection']=='uniform_control' for r in queue['entries']))
        self.assertEqual(queue,H.review_queue(manifest,limit=7,per_group=2,per_source=6))

    def test_report_lead_has_priority_even_when_group_cap_is_one(self):
        normal=self.image();special=self.image(name='12-01.png',number=1)
        records=[{'path':str(normal)}, {'path':str(special),'source_type':'report','report_note':'nose is visible'}]
        manifest=self.manifest(records)
        queue=H.review_queue(manifest,limit=2,per_group=1,per_source=2)
        self.assertEqual(queue['entries'][0]['path'],str(special.resolve()))

    def test_explicit_report_focus_wins_without_fabricating_truth_or_scores(self):
        ordinary=self.image()
        focused=self.image(name='12-01.png',number=1)
        records=[{'path':str(ordinary),'source_type':'report','report_id':'20260101-120000'},
                 {'path':str(focused),'source_type':'report','report_id':'20260101-120100'}]
        manifest=H.build_manifest(records,self.day_map,focus_reports=['20260101-120100'])
        queue=H.review_queue(manifest,limit=1,per_group=1)
        self.assertEqual(queue['entries'][0]['path'],str(focused.resolve()))
        self.assertEqual(queue['entries'][0]['label'],'unreviewed')
        self.assertIn('explicit_report_focus_not_a_label',queue['entries'][0]['reasons'])
        self.assertTrue(all(not r['mining_signals'] for r in manifest['entries']))

    def test_unreadable_images_and_symlink_escape_do_not_enter_training(self):
        (self.frames/'bad.jpg').write_bytes(b'not an image')
        other=self.image(root=self.root/'outside')
        (self.frames/'escape.jpg').symlink_to(other)
        records,problems=H.inputs([self.frames]);manifest=H.build_manifest(records,self.day_map,problems=problems)
        self.assertEqual(manifest['entries'],[])
        self.assertEqual({r['reason'] for r in manifest['excluded']},{'source_symlink_escape','unreadable_or_oversize_image'})

    def test_mine_and_export_never_change_sources_or_legacy_labels(self):
        image=self.image();labels=self.root/'labels.jsonl';labels.write_text('legacy owner label\n')
        before={p:H.digest(p) for p in [image,labels]};manifest=self.manifest()
        target=self.root/'run'
        H.write_snapshots(target,{'manifest.json':manifest,'review_queue.json':H.review_queue(manifest)})
        H.write_snapshots(target,{'manifest.json':manifest})
        with self.assertRaisesRegex(ValueError,'already exists'):
            H.write_snapshots(target,{'manifest.json':{}})
        self.assertEqual(before,{p:H.digest(p) for p in [image,labels]})
        self.assertEqual(json.loads((target/'manifest.json').read_text()),manifest)

    def test_uncertain_and_boxless_positive_are_excluded_from_localizer(self):
        for i in range(4):self.image(name=f'12-0{i}.png',number=i)
        manifest=self.manifest();ids=[r['id'] for r in manifest['entries']]
        labels={ids[0]:self.annotation('visible',[]),ids[1]:self.annotation('uncertain',[]),ids[2]:self.annotation('absent',[])}
        export=H.exports(manifest,labels)
        self.assertEqual(len(export['train.json']['entries']),1)
        self.assertEqual(export['readiness.json']['omitted'],{'unreviewed':1,'visible_without_box':1,'uncertain':1})
        presence=H.exports(manifest,labels,task='presence')
        self.assertEqual(len(presence['train.json']['entries']),2)

    def test_positive_boxes_and_negative_labels_export_with_original_review_provenance(self):
        for day in self.day_map:
            self.image(day=day,number=int(day[-1])*10)
            self.image(day=day,name='12-01.png',number=int(day[-1])*10+1)
        manifest=self.manifest();labels={}
        for row in manifest['entries']:
            label='visible' if row['capture_time'].endswith('00:00') else 'absent'
            labels[row['id']]=self.annotation(label,[[.1,.2,.4,.6]] if label=='visible' else [])
        export=H.exports(manifest,labels)
        self.assertTrue(export['readiness.json']['training_ready'])
        self.assertTrue(export['readiness.json']['independent_test_has_both_classes'])
        for split in ('train','dev','test'):
            self.assertEqual(len(export[split+'.json']['entries']),2)
            for row in export[split+'.json']['entries']:
                self.assertEqual(row['annotation'],labels[row['id']])
                self.assertEqual(row['annotation']['source'],'review')
                self.assertEqual(row['source'],'review')
                self.assertEqual(row['split'],split)
                self.assertEqual(row['reviewer'],row['annotation']['reviewer'])

    def test_invalid_or_self_attributed_owner_labels_fail_closed(self):
        self.image();manifest=self.manifest();ident=manifest['entries'][0]['id']
        bad=[self.annotation('absent'), self.annotation(boxes=[[0,0,float('nan'),1]]),
             {**self.annotation(),'source':'detector'}, {**self.annotation(),'reviewer':''},
             {**self.annotation(),'source':'owner'},
             {**self.annotation(),'source':'owner','reviewer':'codex','attestation':'owner-submitted'}]
        for annotation in bad:
            with self.subTest(annotation=annotation),self.assertRaises(ValueError):H.validate(manifest,{ident:annotation})
        owner={**self.annotation(),'source':'owner','reviewer':'local-owner','attestation':'owner-submitted'}
        self.assertTrue(H.validate(manifest,{ident:owner})[0]['valid'])

    def test_changed_source_or_frozen_manifest_cannot_be_exported(self):
        path=self.image();manifest=self.manifest();path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'source image changed'):H.validate(manifest,{})
        tampered=copy.deepcopy(manifest);tampered['entries'][0]['split']='test'
        with self.assertRaisesRegex(ValueError,'changed after freeze'):H.validate(tampered,{},verify_files=False)

    def test_refrozen_leaking_group_and_day_are_detected(self):
        self.image();self.image(name='13-00.png',number=1)
        manifest=self.manifest();manifest['entries'][0]['split']='test';self.refreeze(manifest)
        with self.assertRaisesRegex(ValueError,'day leaks'):H.validate(manifest,{},verify_files=False)

    def test_annotation_binding_and_unknown_frame_cannot_be_silently_reused(self):
        self.image();manifest=self.manifest();ident=manifest['entries'][0]['id']
        for annotations in [{'wrong':self.annotation()}, {'manifest_sha256':'wrong','annotations':{}},
                            {ident:{**self.annotation(),'sha256':'wrong'}}, {ident:{**self.annotation(),'split':'test'}}]:
            with self.subTest(annotations=annotations),self.assertRaises(ValueError):H.validate(manifest,annotations)

    def test_append_only_review_jsonl_preserves_owner_precedence_and_latest_owner_edit(self):
        self.image();manifest=self.manifest();row=manifest['entries'][0];common={k:row[k] for k in ['id','sha256','group','split']};common['manifest_sha256']=manifest['manifest_sha256']
        owner={**common,**self.annotation(),'source':'owner','reviewer':'local-owner','attestation':'owner-submitted'}
        review={**common,**self.annotation('absent',[])}
        uncertain={**owner,'label':'uncertain','boxes':[]}
        path=self.root/'labels.jsonl';path.write_text('\n'.join(json.dumps(r) for r in [owner,review,uncertain])+'\n')
        labels=H.load_annotations(path,manifest)
        self.assertEqual(labels['annotations'][row['id']]['label'],'uncertain')
        path.write_text(path.read_text()+'{broken')
        with self.assertRaises(ValueError):H.load_annotations(path,manifest)

    def test_source_manifest_relative_paths_and_metadata_have_explicit_provenance(self):
        path=self.image();source=self.root/'source.json'
        source.write_text(json.dumps({'frames':[{'path':str(path.relative_to(self.root)),'scene':'view-a','group':'session-a'}]}))
        records,_=H.inputs(source_manifest=source);manifest=self.manifest(records)
        self.assertEqual(manifest['entries'][0]['path'],str(path.resolve()))
        self.assertEqual(manifest['entries'][0]['scene'],'view-a')

    def test_cli_round_trip_is_deterministic_and_private(self):
        self.image();daymap=self.root/'days.json';daymap.write_text(json.dumps(self.day_map));out=self.root/'run'
        command=[sys.executable,str(ROOT/'tools/hard_examples.py'),'mine','--frames',str(self.frames),'--day-map',str(daymap),'--out',str(out)]
        first=subprocess.run(command,capture_output=True,text=True,check=True)
        second=subprocess.run(command,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(first.stdout),json.loads(second.stdout))
        self.assertEqual(json.loads(first.stdout)['frames'],1)
        labels=self.root/'review.json';labels.write_text('{}')
        result=subprocess.run([sys.executable,str(ROOT/'tools/hard_examples.py'),'export','--manifest',str(out/'manifest.json'),'--annotations',str(labels),'--out',str(self.root/'export')],capture_output=True,text=True,check=True)
        self.assertFalse(json.loads(result.stdout)['training_ready'])


if __name__ == '__main__':
    unittest.main()
