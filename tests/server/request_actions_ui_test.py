"""Owner tasks need a bounded protocol; unresolved engineering remains agent work."""
import json
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'server/requests.js'


def function(name):
    match = re.search(r'    function ' + name + r'\(.*?^    }\n', MODULE.read_text(), re.M | re.S)
    assert match, name
    return match.group(0)


def node(scenario, dom=False):
    js = 'const {actionPlans,actionSummary,reportFrame,nonempty,laneOf,isGoal,goalTitle,goalHealth,goalValue,measurementAge,components,componentState,componentLabel,reviewReady,reviewQuestion,readyReplyReview,progress,hasEvidence,LANES}=require(' + json.dumps(str(MODULE)) + ');\n'
    js += r'''
const owner={assignee:'owner',task:'Observe three ordinary uses.',steps:['Use the app normally.','Count each clear match or mismatch.'],
 measure:'Matches and mismatches out of three eligible uses.',done_when:'Three uses recorded, or stop at the first mismatch.',on_failure:'File one report while the mismatch is visible.',record_template:'Check result: __ matches / __ uses; mismatch: __'};
const agent={assignee:'agent',task:'Compare a frozen correction on visible and absent examples.',steps:['Replay the reported frame.','Measure independent examples and runtime.'],
 measure:'Visible detections, false positives and elapsed inference time.',done_when:'Publish before/after evidence against the frozen acceptance criteria.'};
const report={id:'20261101-120000',ts:1,note:'Pet visible but status says out of sight',shown:'Pet · out of sight',revision:'r1',status:{state:'not_started'},
 resolution:{components:[{id:'tracking',title:'Tracking',state:'not_started',evidence:['An unresolved recorded miss.'],next_action:agent}]}};
'''
    if dom:
        js += r'''
class El {
 constructor(tag='div'){this.tag=tag;this.children=[];this.dataset={};this.attrs={};this.style={};this.scrollTop=0;this.textContent='';this.value='';this.classes=new Set();
  this.classList={add:c=>this.classes.add(c),remove:c=>this.classes.delete(c),toggle:(c,v)=>v?this.classes.add(c):this.classes.delete(c)};}
 append(...items){this.children.push(...items);}replaceChildren(){this.children=[];}setAttribute(k,v){this.attrs[k]=v;}
 querySelectorAll(){return [];}contains(){return false;}focus(){this.focused=true;}addEventListener(){}
}
const make=(tag,cls,text)=>{const el=new El(tag);el.className=cls||'';if(text!==undefined)el.textContent=text;return el;};
const detail=new El(),document={activeElement:null},flatten=el=>[String(el.textContent||''),...el.children.flatMap(flatten)],elements=el=>[el,...el.children.flatMap(elements)];
const controls={'report-note':new El('textarea'),'report-new':new El('button')};let compose=0,posted=0,messages=[];
controls['report-new'].onclick=()=>compose++;
const $=id=>controls[id]||elements(detail).find(el=>el.id===id),say=text=>messages.push(text),backToBoard=()=>{},showMoveReason=()=>{},when=ts=>'time:'+ts;
let detailRevision='',detailDirty=false;const revisionOf=row=>row.revision;
'''
        js += ''.join(function(name) for name in ['evidence', 'renderActionPlan', 'renderNextActions', 'renderSavedFrame', 'restoreDetail', 'renderDetail', 'card'])
    js += '\n(async()=>{\n' + scenario + '\n})().catch(e=>{console.error(e.stack);process.exit(1)});'
    result = subprocess.run(['node', '-e', js], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


class ActionProtocolTest(unittest.TestCase):
    def test_agent_task_does_not_become_owner_review_or_completion(self):
        r = node("console.log(JSON.stringify({plans:actionPlans(report),summary:actionSummary(report),lane:laneOf(report),question:reviewQuestion(report)}));")
        self.assertEqual(r['plans'][0]['assignee'], 'agent')
        self.assertTrue(r['summary'].startswith('Agent: '))
        self.assertEqual(r['lane'], 'not_started')
        self.assertIsNone(r['question'])

    def test_complete_owner_protocol_is_available_without_becoming_review(self):
        r = node("const row={...report,status:{state:'monitoring'},resolution:{components:[{id:'watch',state:'monitoring',next_action:owner}]}};console.log(JSON.stringify({plans:actionPlans(row),summary:actionSummary(row),lane:laneOf(row)}));")
        self.assertEqual(r['plans'][0]['steps'], ['Use the app normally.', 'Count each clear match or mismatch.'])
        self.assertEqual(r['summary'], 'Observe three ordinary uses.')
        self.assertEqual(r['lane'], 'monitoring')

    def test_vague_or_malformed_owner_asks_are_not_instructions(self):
        r = node("""const bad=[null,{},'watch', {...owner,assignee:'person'}, {...owner,task:' '}, {...owner,measure:null},
{...owner,done_when:''}, {...owner,steps:[]}, {...owner,steps:[null,3,' ']}, {...owner,on_failure:''}];
console.log(JSON.stringify(bad.map(next_action=>actionPlans({...report,resolution:{components:[{id:'x',state:'monitoring',next_action}]}}).length)));""")
        self.assertEqual(r, [0] * 10)

    def test_completed_components_and_automatic_goals_cannot_assign_chores(self):
        r = node("""console.log(JSON.stringify({donePart:actionPlans({...report,resolution:{components:[{id:'x',state:'done',evidence:['Verified'],next_action:owner}]}}),
doneRow:actionPlans({...report,status:{state:'done'}}),goal:actionPlans({...report,id:'goal:cpu',goal_id:'cpu',kind:'goal'})}));""")
        self.assertEqual(r, {'donePart': [], 'doneRow': [], 'goal': []})

    def test_existing_ready_review_question_and_evidence_are_preserved(self):
        r = node("""const row={...report,status:{state:'needs_review'},resolution:{components:[{id:'x',state:'needs_review',review:{question:'Which example should we keep?',evidence:['Two annotated views.']}}]}};
console.log(JSON.stringify({summary:actionSummary(row),question:reviewQuestion(row),lane:laneOf(row),evidence:row.resolution.components[0].review.evidence}));""")
        self.assertEqual(r['summary'], 'Which example should we keep?')
        self.assertEqual(r['lane'], 'needs_review')
        self.assertEqual(r['evidence'], ['Two annotated views.'])

    def test_frame_url_must_be_exact_same_origin_endpoint_for_this_report(self):
        r = node("""const expected='/api/reports/frame?id='+report.id;
const values=[expected,'https://other.invalid'+expected,'//other.invalid'+expected,'/media/frame.jpg','/api/reports/frame?id=20260101-000000',expected+'&other=1',null];
console.log(JSON.stringify({values:values.map(frame=>reportFrame({...report,evidence:{frame}})),badId:reportFrame({...report,id:'../private',evidence:{frame:'/api/reports/frame?id=../private'}})}));""")
        self.assertEqual(r['values'], ['/api/reports/frame?id=20261101-120000'] + [None] * 6)
        self.assertIsNone(r['badId'])

    def test_record_template_is_optional_bounded_and_does_not_change_the_source(self):
        r = node("""const original={...owner,record_template:'  A result  '};const row={...report,resolution:{components:[{id:'x',state:'monitoring',next_action:original}]}};
const plan=actionPlans(row)[0];console.log(JSON.stringify({trimmed:plan.record_template,original:original.record_template,
long:actionPlans({...report,resolution:{components:[{id:'x',state:'monitoring',next_action:{...owner,record_template:'x'.repeat(2001)}}]}})[0].record_template}));""")
        self.assertEqual(r, {'trimmed': 'A result', 'original': '  A result  ', 'long': None})


class ActionDomTest(unittest.TestCase):
    def test_unresolved_miss_puts_no_owner_action_and_agent_criteria_first(self):
        r = node("""renderDetail(report);const text=flatten(detail).join(' ');console.log(JSON.stringify({text,
order:elements(detail).filter(el=>el.tag==='section').map(el=>el.className)}));""", dom=True)
        self.assertIn('Your next step No action needed from you now.', r['text'])
        self.assertIn('Agent’s next step', r['text'])
        self.assertIn('Visible detections, false positives and elapsed inference time.', r['text'])
        self.assertIn('Ready for review when', r['text'])
        self.assertLess(r['text'].index('Your next step'), r['text'].index('Requested work'))
        self.assertLess(r['text'].index('Agent’s next step'), r['text'].index('An unresolved recorded miss.'))

    def test_monitoring_protocol_is_actionable_and_bounded(self):
        r = node("""renderNextActions({...report,status:{state:'monitoring'},resolution:{components:[{id:'watch',state:'monitoring',next_action:owner}]}});
console.log(JSON.stringify({text:flatten(detail).join(' '),tags:elements(detail).map(el=>el.tag)}));""", dom=True)
        for text in ['Observe three ordinary uses.', 'How to check', 'Measure', 'Check complete when', 'Report when', 'Record result']:
            self.assertIn(text, r['text'])
        self.assertIn('ol', r['tags'])
        self.assertNotIn('Ready for your review', r['text'])

    def test_monitoring_without_protocol_exposes_agent_preparation(self):
        r = node("""renderNextActions({...report,status:{state:'monitoring'},resolution:{components:[]}});console.log(JSON.stringify(flatten(detail).join(' ')));""", dom=True)
        self.assertIn('No action needed from you now.', r)
        self.assertIn('The agent still needs to define a concrete check.', r)

    def test_saved_frame_is_labelled_historical_with_separate_phone_status(self):
        r = node("""renderSavedFrame({...report,evidence:{frame:'/api/reports/frame?id='+report.id}});const image=elements(detail).find(el=>el.tag==='img');
const before=flatten(detail).join(' ');image.onerror();console.log(JSON.stringify({before,after:flatten(detail).join(' '),src:image.src,hidden:image.hidden,tags:elements(detail).map(el=>el.tag)}));""", dom=True)
        self.assertIn('Saved report frame.', r['before'])
        self.assertIn('may be from different moments', r['before'])
        self.assertIn('Status recorded on the phone: Pet · out of sight', r['before'])
        self.assertEqual(r['src'], '/api/reports/frame?id=20261101-120000')
        self.assertTrue(r['hidden'])
        self.assertIn('saved frame is unavailable', r['after'])
        self.assertNotIn('video', r['tags'])

    def test_record_result_opens_composer_and_never_sends_automatically(self):
        r = node("""renderNextActions({...report,status:{state:'monitoring'},resolution:{components:[{id:'watch',state:'monitoring',next_action:owner}]}});
elements(detail).find(el=>el.textContent==='Record result').onclick();console.log(JSON.stringify({compose,posted,note:controls['report-note'].value,focused:controls['report-note'].focused}));""", dom=True)
        self.assertEqual(r, {'compose': 1, 'posted': 0, 'note': 'Check result: __ matches / __ uses; mismatch: __', 'focused': True})

    def test_record_result_preserves_an_existing_draft(self):
        r = node("""controls['report-note'].value='An unfinished report';renderActionPlan(detail,owner);
elements(detail).find(el=>el.textContent==='Record result').onclick();console.log(JSON.stringify({compose,note:controls['report-note'].value,messages,posted}));""", dom=True)
        self.assertEqual(r['note'], 'An unfinished report')
        self.assertEqual(r['compose'], 1)
        self.assertEqual(r['posted'], 0)
        self.assertIn('existing draft is preserved', r['messages'][0])

    def test_action_content_is_plain_text_and_does_not_create_markup(self):
        r = node("""const task='<img src=x onerror=alert(1)>';renderActionPlan(detail,{...owner,task});console.log(JSON.stringify({text:flatten(detail).join(' '),images:elements(detail).filter(el=>el.tag==='img').length}));""", dom=True)
        self.assertIn('<img src=x onerror=alert(1)>', r['text'])
        self.assertEqual(r['images'], 0)

    def test_card_snippet_identifies_the_next_agent_or_owner_task(self):
        r = node("""const agentText=flatten(card(report)).join(' '),ownerText=flatten(card({...report,status:{state:'monitoring'},resolution:{components:[{id:'x',state:'monitoring',next_action:owner}]}})).join(' ');
console.log(JSON.stringify({agentText,ownerText}));""", dom=True)
        self.assertIn('Agent: Compare a frozen correction', r['agentText'])
        self.assertIn('Observe three ordinary uses.', r['ownerText'])


if __name__ == '__main__':
    unittest.main()
