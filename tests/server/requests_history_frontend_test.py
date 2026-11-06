"""Run the shipped Kanban store in Node: complete history, truthful counts and safe moves."""
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'server/requests.js'


class RequestHistoryFrontendTest(unittest.TestCase):
    def run_board(self, scenario):
        js = 'const {Store,laneOf,progress,matches,components} = require(' + json.dumps(str(MODULE)) + ');\n' + r'''
const records=Array.from({length:50},(_,i)=>({id:'20260101-'+String(50-i).padStart(6,'0'),ts:1000-i,
 by:i===25?'QA probe (Claude)':'Owner',note:'request '+(50-i),revision:'revision-'+i,
 reply:{status:'answered',text:'Historical answer',fix:{status:'live'}},
 status:{state:i===0?'in_progress':'needs_review',label:i===0?'In progress':'Needs review',reason:'No verified result yet'},
 resolution:{components:[{id:'part',title:'Remaining work',state:'not_started',evidence:['Still needs review']}]}}));
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
        self.assertEqual(result['counts']['needs_review'], 49)
        self.assertEqual(result['counts']['in_progress'], 1)
        self.assertEqual(result['qa'], 1)

    def test_legacy_live_claim_is_never_done_and_hardware_maps_to_blocked(self):
        result = self.run_board('''console.log(JSON.stringify({legacy:laneOf({reply:{fix:{status:'live'}}}),
hardware:laneOf({status:{state:'needs_hardware'}}),empty:laneOf({}),bad:laneOf({status:{state:'toString'}})}));''')
        self.assertEqual(result, {'legacy': 'needs_review', 'hardware': 'blocked', 'empty': 'not_started', 'bad': 'not_started'})

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

    def test_search_uses_requested_work_and_old_discussion(self):
        result = self.run_board('''await store.refresh();console.log(JSON.stringify({
work:store.items.filter(r=>matches(r,'Remaining work')).length,
discussion:store.items.filter(r=>matches(r,'historical answer')).length,
zero:store.items.filter(r=>matches(r,'nonexistent')).length}));''')
        self.assertEqual(result, {'work': 50, 'discussion': 50, 'zero': 0})


if __name__ == '__main__':
    unittest.main()
