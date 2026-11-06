"""Independent synthetic checks for supervised partitions and frozen calibration.

No pretrained downloads, GPU, live models, camera or private examples are used.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
import train_detector as td


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.image = self.directory/'example.jpg'; self.image.write_bytes(b'synthetic image bytes')
        self.row = {'id': 'image-one', 'path': 'example.jpg', 'sha256': td.file_hash(self.image),
                    'day': '2026-01-01', 'group': 'episode-one', 'split': 'train',
                    'label': 'visible', 'boxes': [[.1,.2,.5,.7]], 'source': 'owner',
                    'reviewer': 'label_ui', 'scope': 'reviewed_crop_not_whole_frame', 'tags': ['occluded']}

    def load(self, rows, split='train', wrapper=None):
        path = self.directory/'manifest.json'
        path.write_text(json.dumps({wrapper: rows} if wrapper else rows))
        return td.load_records(path, split)

    def test_portable_relative_images_and_export_entries(self):
        for wrapper in (None, 'records', 'entries'):
            rows, receipt = self.load([self.row], wrapper=wrapper)
            self.assertEqual(rows[0]['path'], str(self.image))
            self.assertEqual(rows[0]['scope'], 'reviewed_crop_not_whole_frame')
            self.assertEqual(receipt['excluded'], [])

    def test_modified_image_cannot_reuse_old_label_hash(self):
        self.image.write_bytes(b'changed image')
        with self.assertRaisesRegex(ValueError, 'Image content changed'):
            self.load([self.row])

    def test_unknown_rows_are_excluded_and_counted_without_becoming_negatives(self):
        rows, receipt = self.load([self.row, {'id':'unknown','label':'uncertain'}])
        self.assertEqual(len(rows), 1)
        self.assertEqual(receipt['excluded'], [{'id':'unknown','label':'uncertain'}])
        self.assertEqual(td.partition_summary(rows)['absent'], 0)

    def test_pseudo_labels_and_unattributed_labels_fail(self):
        for change in ({'source':'model'}, {'reviewer':''}, {'label_origin':'teacher'},
                       {'annotation':{'source':'model'}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load([{**self.row, **change}])

    def test_wrong_partition_and_duplicate_ids_fail(self):
        with self.assertRaisesRegex(ValueError, 'split'):
            self.load([self.row], 'dev')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.load([self.row, self.row])

    def test_box_targets_cannot_fabricate_visible_or_empty_examples(self):
        for change in ({'boxes':[]}, {'label':'absent'}, {'boxes':[[.5,.2,.1,.7]]},
                       {'boxes':[[False,0,1,1]]}, {'boxes':[[0,0,float('nan'),1]]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load([{**self.row, **change}])
        rows, _ = self.load([{**self.row, 'label':'absent', 'boxes':[]}])
        self.assertEqual(rows[0]['label'], 'absent')

    def test_partition_groups_days_parent_frames_ids_and_content_are_protected(self):
        left = [self.row]
        fresh = {**self.row, 'id':'new', 'sha256':'b'*64, 'day':'2026-01-02', 'group':'new-episode'}
        td.assert_disjoint(left, [fresh])
        for key in ('id','sha256','day','group'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                td.assert_disjoint(left, [{**fresh,key:self.row[key]}])
        with self.assertRaisesRegex(ValueError, 'parent_frame'):
            td.assert_disjoint([{**self.row, 'parent_frame':'video-frame'}], [{**fresh, 'parent_frame':'video-frame'}])

    def test_summary_binds_labels_boxes_and_source_not_just_image_names(self):
        first = td.partition_summary([{**self.row,'path':str(self.image)}])
        changed = td.partition_summary([{**self.row,'boxes':[[.2,.2,.5,.7]],'path':str(self.image)}])
        self.assertNotEqual(first['members_sha256'], changed['members_sha256'])
        self.assertEqual(first['scopes'], ['reviewed_crop_not_whole_frame'])
        self.assertEqual(first['unique_groups'], 1)

    def test_immutable_output_never_overwrites_a_candidate_or_live_models(self):
        path = self.directory/'candidate'
        self.assertEqual(td.immutable_output(path), path)
        with self.assertRaisesRegex(ValueError, 'new directory'):
            td.immutable_output(path)
        with self.assertRaisesRegex(ValueError, 'live model'):
            td.immutable_output(ROOT/'models'/'forbidden-test-candidate')


    def test_evaluation_receipts_are_new_only_and_cannot_write_live_models(self):
        output = self.directory/'receipt.json'
        self.assertEqual(td.evaluation_output(output), output)
        self.assertFalse(output.exists(), 'Validation must not create a receipt before inference')
        output.write_text('retained')
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            td.evaluation_output(output)
        self.assertEqual(output.read_text(), 'retained')
        with self.assertRaisesRegex(ValueError, 'live model'):
            td.evaluation_output(ROOT/'models'/'forbidden-experiment-receipt.json')


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        self.visible = {'label':'visible','boxes':[[.1,.1,.4,.4]]}
        self.absent = {'label':'absent','boxes':[]}
        self.good = {'box':[.1,.1,.4,.4], 'score':.8}

    def test_localized_recall_cannot_pass_a_wrong_object_in_a_positive_frame(self):
        metrics = td.score_predictions([self.visible], [[{'box':[.6,.6,.9,.9], 'score':.9}]], .5)
        self.assertEqual(metrics['localized_frame_recall_iou_0_5'], 0)
        self.assertEqual(metrics['unmatched_predictions_on_visible_images'], 1)
        self.assertIsNone(metrics['false_alarm_rate'])

    def test_duplicate_predictions_cannot_match_a_target_twice(self):
        metrics = td.score_predictions([self.visible], [[self.good, self.good]], .5)
        self.assertEqual(metrics['matched_boxes_iou_0_5'], 1)
        self.assertEqual(metrics['unmatched_predictions_on_visible_images'], 1)

    def test_dev_calibration_uses_absent_examples_and_keeps_maximum_safe_recall(self):
        predictions = [[self.good], [{'box':[.2,.2,.6,.6], 'score':.7}]]
        selected = td.calibrate_threshold([self.visible,self.absent], predictions)
        self.assertEqual(selected['threshold'], .8)
        self.assertEqual(selected['metrics']['false_alarm_images'], 0)
        self.assertEqual(selected['metrics']['localized_frame_recall_iou_0_5'], 1)

    def test_no_safe_positive_is_reported_as_reject_all_not_improvement(self):
        result = td.calibrate_threshold([self.visible,self.absent],
                                       [[{**self.good,'score':.2}], [{**self.good,'score':.9}]])
        self.assertTrue(result['reject_all'])
        self.assertEqual(result['metrics']['matched_boxes_iou_0_5'], 0)
        self.assertEqual(result['metrics']['false_alarm_images'], 0)

    def test_calibration_rejects_single_class_and_bad_probability(self):
        with self.assertRaisesRegex(ValueError, 'both reviewed'):
            td.calibrate_threshold([self.visible], [[self.good]])
        for score in (float('nan'), -1, True, 1.1):
            with self.subTest(score=score), self.assertRaises(ValueError):
                td.select_predictions([{**self.good,'score':score}], .5)

    def test_missing_prediction_rows_cannot_silently_skip_examples(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            td.score_predictions([self.visible,self.absent], [[self.good]], .5)


class FrozenEvaluationTest(unittest.TestCase):
    def member(self, id, day, split):
        return {'id':id,'sha256':hashlib.sha256(id.encode()).hexdigest(),'day':day,'group':id,
                'split':split,'label':'visible','boxes':[[0,0,1,1]],'source':'owner',
                'reviewer':'label_ui','scope':'reviewed_image','tags':[]}

    def metadata(self):
        train = self.member('train','2026-01-01','train'); dev = self.member('dev','2026-01-02','dev')
        return {'partitions':{'train':td.partition_summary([train]),'dev':td.partition_summary([dev])}}

    def test_test_partition_rejects_train_or_dev_scene_reuse(self):
        metadata = self.metadata()
        for partition in ('train','dev'):
            row = metadata['partitions'][partition]['members'][0]
            with self.subTest(partition=partition), self.assertRaisesRegex(ValueError,'overlap'):
                td.verify_evaluation_partition([{**row,'split':'test'}], metadata)
        test = self.member('test','2026-01-03','test')
        self.assertEqual(td.verify_evaluation_partition([test], metadata), 'test')

    def test_development_diagnostic_must_use_exact_frozen_membership(self):
        metadata = self.metadata(); dev = metadata['partitions']['dev']['members']
        self.assertEqual(td.verify_evaluation_partition(dev, metadata), 'dev')
        with self.assertRaisesRegex(ValueError, 'frozen development'):
            td.verify_evaluation_partition([{**dev[0],'boxes':[[.1,.1,.9,.9]]}], metadata)

    def test_metadata_and_checkpoint_mutations_are_detected_without_loading_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); checkpoint = path/'checkpoint.pt'; checkpoint.write_bytes(b'frozen-weight-fixture')
            metadata = {**self.metadata(), 'schema':td.SCHEMA,
                        'architecture':'torchvision.ssdlite320_mobilenet_v3_large',
                        'checkpoint_sha256':td.file_hash(checkpoint)}
            (path/'metadata.json').write_text(json.dumps(td.seal_metadata(metadata)))
            self.assertEqual(td.load_candidate(path)['checkpoint_sha256'], td.file_hash(checkpoint))
            checkpoint.write_bytes(b'different')
            with self.assertRaisesRegex(ValueError, 'weights changed'):
                td.load_candidate(path)
            checkpoint.write_bytes(b'frozen-weight-fixture')
            broken = td.seal_metadata(metadata); broken['schema'] = 999
            (path/'metadata.json').write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, 'metadata integrity'):
                td.load_candidate(path)

    def test_training_cli_has_no_test_input_or_deploy_operation(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            td.parser().parse_args(['train','--train','a','--dev','b','--out','c','--test','secret-test'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            td.parser().parse_args(['deploy'])
        args = td.parser().parse_args(['evaluate','--data','d','--candidate','c','--baseline-model','m',
                                     '--baseline-verifier','v','--out','r','--diagnostic'])
        self.assertTrue(args.diagnostic)


if __name__ == '__main__':
    unittest.main()
