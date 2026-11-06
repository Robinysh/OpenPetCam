"""Explicit local-owner reviews bind to frozen frames and append without hidden writes."""
import hashlib
import http.client
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
import hard_examples as H
import hard_examples_ui as UI


class ReviewUiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frames = self.root/'frames'
        for i, day in enumerate(['2026-01-01', '2026-01-02', '2026-01-03']):
            folder = self.frames/day
            folder.mkdir(parents=True)
            Image.new('RGB', (80, 48), (30+i*40, 50, 90)).save(folder/'12-00.png')
        records, problems = H.inputs([self.frames])
        self.manifest = H.build_manifest(records, {'2026-01-01':'train','2026-01-02':'dev','2026-01-03':'test'}, problems=problems)
        self.manifest_path = self.root/'manifest.json'
        self.manifest_path.write_bytes(H.canonical(self.manifest))
        self.queue = H.review_queue(self.manifest, splits=('train','dev','test'))
        self.queue_path = self.root/'queue.json'
        self.queue_path.write_bytes(H.canonical(self.queue))
        self.answers = self.root/'new-labels'/'owner.jsonl'
        self.review = UI.Review(self.manifest_path, self.queue_path, self.answers)

    def serve(self, review=None):
        server = UI.ThreadingHTTPServer(('127.0.0.1', 0), UI.handler(review or self.review))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        return server.server_port

    def request(self, port, path='/api/state', data=None, headers=None, raw=None):
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
        base = {'Origin':f'http://127.0.0.1:{port}', 'Content-Type':'application/json',
                'X-Hard-Examples-Token':self.review.token}
        if headers:
            base.update(headers)
        body = raw if raw is not None else json.dumps(data).encode() if data is not None else None
        try:
            connection.request('POST' if body is not None else 'GET', path, body=body, headers=base)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def visible(self, ident=None):
        return {'id':ident or self.review.items[0]['id'], 'label':'visible', 'boxes':[[.1,.2,.4,.6]], 'tags':['occluded']}

    def test_opening_state_and_frames_never_creates_annotations_or_directories(self):
        port = self.serve()
        code, _, body = self.request(port)
        state = json.loads(body)
        self.assertEqual((code,state['reviewed'],state['total'],state['excluded']), (200,0,2,1))
        self.assertFalse(self.answers.parent.exists())
        self.assertNotIn('path', state['items'][0])
        ident = state['items'][0]['id']
        code, headers, body = self.request(port, '/api/frame?id='+ident)
        self.assertEqual(code, 200)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(hashlib.sha256(body).hexdigest(), state['items'][0]['sha256'])
        self.assertFalse(self.answers.exists())

    def test_only_queued_frames_are_served_and_changed_bytes_fail_closed(self):
        port = self.serve()
        for path in ['/api/frame?id=../../etc/passwd','/api/frame?path=/etc/passwd','/manifest.json','/api/frame?id=unknown']:
            self.assertEqual(self.request(port,path)[0], 404)
        item=self.review.items[0]
        Path(item['path']).write_bytes(b'changed')
        self.assertEqual(self.request(port,'/api/frame?id='+item['id'])[0],409)
        with self.assertRaises(UI.Conflict):
            self.review.submit(self.visible())
        self.assertFalse(self.answers.exists())

    def test_image_links_and_annotation_history_links_are_rejected(self):
        item=self.review.items[0];path=Path(item['path']);copy=self.root/'copy.png'
        copy.write_bytes(path.read_bytes());path.unlink();path.symlink_to(copy)
        with self.assertRaises(FileNotFoundError):
            self.review.image(item['id'])
        self.answers.parent.mkdir();self.answers.symlink_to(self.root/'missing.jsonl')
        with self.assertRaises(ValueError):
            self.review.state()

    def test_sealed_test_frames_are_excluded_unless_explicitly_requested(self):
        sealed=next(r for r in self.manifest['entries'] if r['split']=='test')
        with self.assertRaises(FileNotFoundError):
            self.review.image(sealed['id'])
        with self.assertRaises(ValueError):
            self.review.submit(self.visible(sealed['id']))
        explicit=UI.Review(self.manifest_path,self.queue_path,self.answers,include_sealed=True)
        result=explicit.submit({'id':sealed['id'],'label':'uncertain','boxes':[],'tags':[]})
        self.assertEqual(result['annotation']['split'],'test')
        self.assertEqual(explicit.state()['total'],3)

    def test_host_origin_csrf_and_content_type_reject_before_writes(self):
        port=self.serve()
        cases=[({'Host':'attacker.invalid'},403),({'Origin':'http://attacker.invalid'},403),
               ({'Origin':'null'},403),({'Origin':''},403),({'X-Hard-Examples-Token':'wrong'},403),
               ({'Content-Type':'text/plain'},415),({'Content-Type':'application/x-www-form-urlencoded'},415)]
        for headers,status in cases:
            with self.subTest(headers=headers):
                self.assertEqual(self.request(port,'/api/annotation',self.visible(),headers)[0],status)
        self.assertEqual(self.request(port,headers={'Host':'attacker.invalid'})[0],403)
        self.assertFalse(self.answers.exists())

    def test_visible_label_requires_finite_positive_visible_part_boxes(self):
        bad=[[],[[.4,.2,.1,.6]],[[.1,.2,float('nan'),.6]],[[.1,.2,1.1,.6]],[[.1,.2,True,.6]],[[.1,.2,.1,.6]]]
        for boxes in bad:
            with self.subTest(boxes=boxes),self.assertRaises(ValueError):
                self.review.submit({**self.visible(),'boxes':boxes})
        self.assertFalse(self.answers.exists())

    def test_visible_part_boxes_and_owner_provenance_are_bound_to_original(self):
        item=self.review.items[0]
        result=self.review.submit(self.visible())
        annotation=result['annotation']
        self.assertEqual((annotation['source'],annotation['reviewer'],annotation['attestation']),('owner','local-owner','owner-submitted'))
        for name in ['id','sha256','group','split']:
            self.assertEqual(annotation[name],item[name])
        self.assertEqual(annotation['manifest_sha256'],self.manifest['manifest_sha256'])
        self.assertEqual(annotation['boxes'],[[.1,.2,.4,.6]])
        self.assertEqual(self.review.state()['reviewed'],1)
        self.assertEqual(len(self.answers.read_text().splitlines()),1)
        self.assertEqual(H.load_annotations(self.answers,self.manifest)['annotations'][item['id']],annotation)

    def test_absent_and_uncertain_are_explicit_and_cannot_keep_positive_boxes(self):
        for label in ['absent','uncertain']:
            with self.assertRaises(ValueError):
                self.review.submit({**self.visible(),'label':label})
        first,second=self.review.items
        a=self.review.submit({'id':first['id'],'label':'absent','boxes':[]})
        u=self.review.submit({'id':second['id'],'label':'uncertain','boxes':[]})
        self.assertEqual((a['annotation']['label'],u['annotation']['label']),('absent','uncertain'))
        self.assertEqual(self.review.state()['reviewed'],2)

    def test_replacement_requires_explicit_opt_in_and_latest_revision_and_is_append_only(self):
        result=self.review.submit(self.visible())
        old=self.answers.read_bytes()
        absent={'id':self.review.items[0]['id'],'label':'absent','boxes':[]}
        with self.assertRaises(UI.Conflict):
            self.review.submit(absent)
        with self.assertRaises(UI.Conflict):
            self.review.submit({**absent,'replace':True,'revision':'stale'})
        self.assertEqual(self.answers.read_bytes(),old)
        replacement=self.review.submit({**absent,'replace':True,'revision':result['revision']})
        self.assertTrue(self.answers.read_bytes().startswith(old))
        self.assertEqual(len(self.answers.read_text().splitlines()),2)
        self.assertEqual(self.review.state()['items'][0]['annotation']['label'],'absent')
        self.assertEqual(replacement['annotation']['replaces'],result['revision'])

    def test_appending_after_a_valid_last_line_without_newline_preserves_history(self):
        self.review.submit(self.visible())
        self.answers.write_bytes(self.answers.read_bytes().rstrip(b'\n'))
        before=self.answers.read_bytes()
        self.review.submit({'id':self.review.items[1]['id'],'label':'uncertain','boxes':[]})
        self.assertTrue(self.answers.read_bytes().startswith(before+b'\n'))
        self.assertEqual(len(H.load_annotations(self.answers,self.manifest)['annotations']),2)

    def test_client_cannot_spoof_provenance_and_tags_are_bounded(self):
        for extra in [{'source':'review'},{'reviewer':'agent'},{'sha256':'other'},{'tags':['day','ir']},{'tags':[['bad']]},{'tags':['occluded','occluded']}]:
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                self.review.submit({**self.visible(),**extra})
        self.assertFalse(self.answers.exists())

    def test_bad_or_stale_history_is_not_silently_accepted(self):
        self.review.submit(self.visible())
        with self.answers.open('a') as handle:handle.write('{truncated\n')
        with self.assertRaises(ValueError):
            self.review.state()
        with self.assertRaises(ValueError):
            self.review.submit(self.visible(self.review.items[1]['id']))

    def test_http_rejects_malformed_json_and_preserves_review_history(self):
        port=self.serve()
        self.assertEqual(self.request(port,'/api/annotation',raw=b'{bad')[0],400)
        self.assertEqual(self.request(port,'/api/annotation',raw=b'{"label":NaN}')[0],400)
        self.assertEqual(self.request(port,'/api/annotation',raw=b'x'*65537)[0],413)
        code,_,body=self.request(port,'/api/annotation',self.visible())
        self.assertEqual(code,200)
        self.assertEqual(json.loads(body)['annotation']['label'],'visible')

    def test_real_page_load_has_no_post_or_default_label_submission(self):
        page=UI.PAGE.read_text()
        js=re.search(r'<script>(.*?)</script>',page,re.S)[1]
        program=r'''const calls=[];const elements=new Map();
const element=()=>({hidden:false,textContent:'',value:'',tagName:'DIV',style:{},classList:{toggle(){}},setAttribute(){},replaceChildren(){},append(){},getContext(){return {clearRect(){}}}});
global.document={getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);},querySelectorAll(){return[];},addEventListener(){}};
global.fetch=async(path,options)=>{calls.push({path,method:options.method||'GET'});return{ok:true,json:async()=>({items:[],reviewed:0,total:0,excluded:0,token:'fixture'})};};
''' + js + r'''
setTimeout(()=>{console.log(JSON.stringify(calls));},0);'''
        result=subprocess.run(['node','-e',program],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),[{'path':'/api/state','method':'GET'}])
        self.assertNotIn('autoplay',page.lower())
        self.assertIn('No visible pet',page)
        self.assertIn('Uncertain',page)
        self.assertIn('button.disabled=!imageReady||saving',page)
        self.assertIn("$('frame').hidden=true;$('zoom')",page)


if __name__=='__main__':
    unittest.main()
