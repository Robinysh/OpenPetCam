"""Run the shipped Kanban store in Node: complete history, truthful counts and safe moves."""
import json
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'server/requests.js'


class RequestHistoryFrontendTest(unittest.TestCase):
    def run_board(self, scenario):
        js = 'const {LANES,laneDescriptions,Store,laneOf,progress,matches,components,reviewReady,componentState,componentLabel,readyReplyReview,reviewQuestion,needsReason} = require(' + json.dumps(str(MODULE)) + ');\n' + r'''
const records=Array.from({length:50},(_,i)=>({id:'20260101-'+String(50-i).padStart(6,'0'),ts:1000-i,
 by:i===25?'QA probe (Claude)':'Owner',note:'request '+(50-i),revision:'revision-'+i,
 reply:{status:'answered',text:'Historical answer',fix:{status:'live'}},
 status:{state:i===0?'in_progress':i===1?'needs_review':i===2?'monitoring':'not_started',reason:'No verified result yet'},
 resolution:{components:[{id:'part',title:'Remaining work',state:i===1?'needs_review':i===2?'monitoring':'not_started',evidence:['Investigation notes'],
 ...(i===1?{review:{question:'Accept the framing shown in this example?',evidence:['Two annotated frames are ready']}}:{})}]}}));
const requests=[];
const api=async(path,body)=>{
 requests.push({path,body});if(body) throw Error('Done requires verification evidence for every requested component.');
 const q=new URL(path,'https://camera.invalid').searchParams,before=q.get('before');
 const start=before?records.findIndex(r=>r.id===before)+1:0,page=records.slice(start,start+20);
 return {reports:page,total:records.length,next_before:start+20<records.length?page.at(-1).id:null};
};
const store=new Store(api);
(async()=>{
''' + scenario + r'''
})().catch(e=>{console.error(e.stack);process.exit(1)});
'''
        result = subprocess.run(['node', '-e', js], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_all_fifty_load_without_an_older_button_and_survive_polling(self):
        result = self.run_board('''await store.refresh();await store.refresh();
console.log(JSON.stringify({count:store.items.length,total:store.total,oldest:store.items.at(-1).note,requests}));''')
        self.assertEqual(result['count'], 50)
        self.assertEqual(result['total'], 50)
        self.assertEqual(result['oldest'], 'request 1')
        self.assertTrue(any('before=' in x['path'] for x in result['requests']))

    def test_counts_are_disjoint_and_include_old_qa_requests(self):
        result = self.run_board('''await store.refresh();
console.log(JSON.stringify({counts:store.counts(),qa:store.items.filter(r=>matches(r,'QA probe')).length}));''')
        self.assertEqual(sum(result['counts'].values()), 50)
        self.assertEqual(result['counts']['needs_review'], 1)
        self.assertEqual(result['counts']['monitoring'], 1)
        self.assertEqual(result['counts']['not_started'], 47)
        self.assertEqual(result['counts']['in_progress'], 1)
        self.assertEqual(result['qa'], 1)

    def test_legacy_live_claim_is_never_done_and_hardware_maps_to_blocked(self):
        result = self.run_board('''console.log(JSON.stringify({legacy:laneOf({reply:{fix:{status:'live'}}}),
hardware:laneOf({status:{state:'needs_hardware'}}),empty:laneOf({}),bad:laneOf({status:{state:'toString'}}),array:laneOf({status:{state:['monitoring']}})}));''')
        self.assertEqual(result, {'legacy': 'not_started', 'hardware': 'blocked', 'empty': 'not_started', 'bad': 'not_started', 'array': 'not_started'})

    def test_monitoring_and_immediate_review_are_distinct_lanes(self):
        result = self.run_board("""console.log(JSON.stringify({lanes:LANES,description:laneDescriptions.monitoring,
monitor:laneOf({status:{state:'monitoring'}}),review:laneOf({status:{state:'needs_review'}})}));""")
        self.assertEqual(len(result['lanes']), 6)
        self.assertIn(['monitoring', 'Monitoring'], result['lanes'])
        self.assertIn('No decision needed now', result['description'])
        self.assertEqual((result['monitor'], result['review']), ('monitoring', 'needs_review'))

    def test_review_component_requires_question_and_review_material(self):
        result = self.run_board("""const valid={state:'needs_review',review:{question:'Accept the framing shown?',evidence:['Two annotated frames']}};
console.log(JSON.stringify({ready:reviewReady(valid),readyState:componentState(valid),readyLabel:componentLabel(valid),
legacy:componentLabel({state:'needs_review',evidence:['Tests remain to be run']}),
missingQuestion:reviewReady({review:{evidence:['An artifact']}}),
missingEvidence:reviewReady({review:{question:'Accept?',evidence:[]}}),
objectEvidence:reviewReady({review:{question:'Accept?',evidence:{path:'artifact'}}}),
monitor:componentLabel({state:'monitoring'})}));""")
        self.assertTrue(result['ready'])
        self.assertEqual(result['readyState'], 'needs_review')
        self.assertEqual(result['readyLabel'], 'Needs review')
        self.assertEqual(result['legacy'], 'Backlog')
        self.assertFalse(result['missingQuestion'])
        self.assertFalse(result['missingEvidence'])
        self.assertFalse(result['objectEvidence'])
        self.assertEqual(result['monitor'], 'Monitoring')

    def test_malformed_and_unverified_components_do_not_claim_review_or_done(self):
        result = self.run_board("""console.log(JSON.stringify({labels:[null,[],{state:[]},{state:'constructor'},{state:'__proto__'},
{state:'done'},{state:'done',evidence:[]}].map(componentLabel)}));""")
        self.assertEqual(result['labels'], ['Backlog'] * 7)

    def test_review_move_requires_a_question_before_sending_any_request(self):
        result = self.run_board("""await store.refresh();let message='';
try{await store.move(records[0].id,'needs_review','   ')}catch(e){message=e.message}
console.log(JSON.stringify({message,posted:requests.filter(r=>r.body).length,lane:laneOf(store.items[0])}));""")
        self.assertIn('review or decide now', result['message'])
        self.assertEqual(result['posted'], 0)
        self.assertEqual(result['lane'], 'in_progress')

    def test_review_question_and_monitoring_move_preserve_the_displayed_revision(self):
        result = self.run_board("""await store.refresh();const sent=[];
store.api=async(path,body)=>{sent.push(body);return {...records[0],revision:'new',status:{state:body.state},workflow:{reason:body.reason}}};
await store.move(records[0].id,'needs_review','Approve the illustrated framing?','shown-review');
await store.move(records[0].id,'monitoring','','shown-monitor');
console.log(JSON.stringify({sent,lane:laneOf(store.items[0])}));""")
        self.assertEqual(result['sent'][0]['reason'], 'Approve the illustrated framing?')
        self.assertEqual(result['sent'][0]['revision'], 'shown-review')
        self.assertEqual(result['sent'][1]['revision'], 'shown-monitor')
        self.assertEqual(result['lane'], 'monitoring')

    def test_typed_reply_review_packet_is_available_without_a_resolution(self):
        result = self.run_board("""const packet={question:'Accept the illustrated framing?',evidence:['The comparison is ready below']};
const row={status:{state:'needs_review'},reply:{status:'answered',fix:{status:'planned',review:packet}}};
console.log(JSON.stringify({packet:readyReplyReview(row),question:reviewQuestion(row),
legacy:readyReplyReview({reply:{status:'answered',fix:{status:'planned',needs_owner:true}}}),
failed:readyReplyReview({...row,reply:{status:'answered',fix:{status:'failed_checks',review:packet}}}),
looking:readyReplyReview({...row,reply:{status:'looking',fix:{status:'planned',review:packet}}}),
unknown:readyReplyReview({...row,reply:{fix:{status:'planned',review:packet}}}),
resolved:readyReplyReview({...row,resolution:{components:[]}})}));""")
        self.assertEqual(result['question'], 'Accept the illustrated framing?')
        self.assertEqual(result['packet']['evidence'], ['The comparison is ready below'])
        for key in ['legacy', 'failed', 'looking', 'unknown', 'resolved']:
            self.assertIsNone(result[key], key)

    def test_review_card_uses_the_question_and_monitoring_does_not_request_a_decision(self):
        result = self.run_board("""const part={id:'framing',state:'needs_review',review:{question:'Keep this framing?',evidence:['The comparison below']}};
console.log(JSON.stringify({question:reviewQuestion({status:{state:'needs_review'},resolution:{components:[part]}}),
manual:reviewQuestion({status:{state:'needs_review'},workflow:{state:'needs_review',reason:'Choose between the two examples'}}),
monitoring:reviewQuestion({status:{state:'monitoring'},resolution:{components:[part]}})}));""")
        self.assertEqual(result['question'], 'Keep this framing?')
        self.assertEqual(result['manual'], 'Choose between the two examples')
        self.assertIsNone(result['monitoring'])

    def reason_form(self, state):
        source = MODULE.read_text()
        function = re.search(r'    function showMoveReason\(\) \{.*?^    }', source, re.M | re.S).group(0)
        js = 'const {needsReason}=require(' + json.dumps(str(MODULE)) + ');\n' + r"""
const nodes={
 'request-state':{value:STATE},'request-reason':{value:''},'request-reason-label':{},
 'request-move-save':{},'request-block-reason':{classList:{toggle:(name,on)=>{nodes.hidden=on}}}};
const $=id=>nodes[id];let saving=false;
""".replace('STATE', json.dumps(state)) + function + r"""
showMoveReason();const blank={hidden:nodes.hidden,required:nodes['request-reason'].required,
label:nodes['request-reason-label'].textContent,disabled:nodes['request-move-save'].disabled};
nodes['request-reason'].value='A concrete question';showMoveReason();
console.log(JSON.stringify({blank,filledDisabled:nodes['request-move-save'].disabled}));
"""
        result = subprocess.run(['node', '-e', js], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_review_form_requires_a_decision_question_and_monitoring_note_is_optional(self):
        review = self.reason_form('needs_review')
        self.assertEqual(review['blank']['label'], 'What should I review or decide?')
        self.assertTrue(review['blank']['required'])
        self.assertTrue(review['blank']['disabled'])
        self.assertFalse(review['filledDisabled'])
        monitor = self.reason_form('monitoring')
        self.assertFalse(monitor['blank']['required'])
        self.assertFalse(monitor['blank']['disabled'])
        self.assertFalse(monitor['blank']['hidden'])
        self.assertIn('keep an eye on', monitor['blank']['label'])
        self.assertTrue(self.reason_form('not_started')['blank']['hidden'])

    def test_dropping_into_review_opens_a_question_instead_of_sending_a_blank_move(self):
        source = MODULE.read_text()
        body = re.search(r"lane\.addEventListener\('drop', event => \{(.*?)^          }\);", source, re.M | re.S).group(1)
        js = 'const {needsReason}=require(' + json.dumps(str(MODULE)) + ');\n' + r"""
const state='needs_review',selected={};let opened=null,moved=0,focused=false,shown=false;
const store={items:[{id:'example'}]},lane={classList:{remove(){}}};
const $=id=>id==='request-state'?selected:{focus(){focused=true}};
const openCard=id=>opened=id,showMoveReason=()=>shown=true,move=()=>moved++;
const event={preventDefault(){},dataTransfer:{getData:()=>'example'}};
function drop(){
""" + body + r"""
}drop();console.log(JSON.stringify({opened,moved,focused,shown,state:selected.value}));
"""
        result = subprocess.run(['node', '-e', js], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'opened': 'example', 'moved': 0, 'focused': True,
                                                   'shown': True, 'state': 'needs_review'})

    def test_component_progress_requires_evidence_and_tolerates_malformed_parts(self):
        result = self.run_board('''console.log(JSON.stringify({
p:progress({resolution:{components:[null,{id:'a',state:'done'}, {id:'b',state:'done',evidence:['checked']}, {id:'c',state:'not_started'}]}}),
bad:progress({resolution:{components:{a:1}}})}));''')
        self.assertEqual(result['p'], {'done': 1, 'total': 3})
        self.assertEqual(result['bad'], {'done': 0, 'total': 0})

    def test_rejected_move_does_not_move_card_or_drop_component_evidence(self):
        result = self.run_board('''await store.refresh();let message='';
try{await store.move(records[0].id,'done')}catch(e){message=e.message}
console.log(JSON.stringify({lane:laneOf(store.items[0]),parts:store.items[0].resolution.components,message}));''')
        self.assertEqual(result['lane'], 'in_progress')
        self.assertEqual(result['parts'][0]['title'], 'Remaining work')
        self.assertIn('verification evidence', result['message'])

    def test_move_uses_the_revision_the_reader_saw(self):
        result = self.run_board('''await store.refresh();let sent;
store.api=async(path,body)=>{sent=body;return {...records[0],revision:'new',status:{state:body.state}}};
await store.move(records[0].id,'blocked','Waiting for hardware','shown-before-poll');
console.log(JSON.stringify({sent,lane:laneOf(store.items[0])}));''')
        self.assertEqual(result['sent']['revision'], 'shown-before-poll')
        self.assertEqual(result['sent']['reason'], 'Waiting for hardware')
        self.assertEqual(result['lane'], 'blocked')

    def test_late_refresh_cannot_undo_a_confirmed_move(self):
        result = self.run_board('''store.items=[records[0]];let finish;
store.api=async(path,body)=>body?{...records[0],revision:'new',status:{state:'not_started'}}:new Promise(r=>finish=r);
const pending=store.refresh();await store.move(records[0].id,'not_started');
finish({reports:[records[0]],total:1,next_before:null});await pending;
console.log(JSON.stringify({lane:laneOf(store.items[0]),revision:store.items[0].revision}));''')
        self.assertEqual(result, {'lane': 'not_started', 'revision': 'new'})

    def test_failed_history_load_preserves_existing_cards_and_exposes_retry_error(self):
        result = self.run_board('''await store.refresh();store.api=async()=>{throw Error('Offline')};await store.refresh();
console.log(JSON.stringify({count:store.items.length,error:store.error,loading:store.loading}));''')
        self.assertEqual(result['count'], 50)
        self.assertFalse(result['loading'])
        self.assertIn('Offline', result['error'])

    def test_search_finds_typed_reply_review_question_and_evidence(self):
        result = self.run_board("""records[0].resolution=null;
records[0].reply={status:'answered',text:'Proposal ready',fix:{status:'planned',review:{question:'Which crop should we keep?',evidence:['Inspect the copper comparison']}}};
await store.refresh();console.log(JSON.stringify({question:store.items.filter(r=>matches(r,'Which crop')).length,
evidence:store.items.filter(r=>matches(r,'copper comparison')).length}));""")
        self.assertEqual(result, {'question': 1, 'evidence': 1})

    def test_search_uses_requested_work_and_old_discussion(self):
        result = self.run_board('''await store.refresh();console.log(JSON.stringify({
work:store.items.filter(r=>matches(r,'Remaining work')).length,
discussion:store.items.filter(r=>matches(r,'historical answer')).length,
zero:store.items.filter(r=>matches(r,'nonexistent')).length}));''')
        self.assertEqual(result, {'work': 50, 'discussion': 50, 'zero': 0})


if __name__ == '__main__':
    unittest.main()
