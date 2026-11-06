"""Startup goals distinguish page-open measurements from connection diagnostics."""
import json
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'server/requests.js'


def function(name):
    match = re.search(r'    (?:async )?function ' + name + r'\(.*?^    }\n', MODULE.read_text(), re.M | re.S)
    assert match, name
    return match.group(0)


def node(scenario, setup=''):
    code = 'const {Store,LANES,isGoal,laneOf,components,componentState,componentLabel,reviewReady,reviewQuestion,readyReplyReview,progress,hasEvidence,goalTitle,goalHealth,goalValue,startupStats,startupSeconds,startupReading,measurementAge,trendSegments,actionSummary,nonempty}=require(' + json.dumps(str(MODULE)) + ');\n'
    code += r'''
const stats={from:'open',p50:1.755,p90:17.748,n:25,ready:true,observed_at:100,min_samples:5,window_days:7,
 connect_p50:.3,connect_p90:.9,connect_n:250,timeouts:2,reconnects:38};
const goal={id:'goal:video_start',kind:'goal',goal_id:'video_start',title:'Opening the app shows live video quickly',target:'p50 <= 1, p90 <= 2',
 health:{state:'off_target',label:'Off target',reason:'Startup is too slow.',expires_at:9999999999},sample:{value:1.755,ts:110,observed_at:100,startup:stats},series:[],work:[]};
const report={id:'20260101-120000',revision:'r1',note:'A slow live startup',status:{state:'not_started'},resolution:{components:[{id:'startup',state:'not_started',goal_id:'video_start'}]}};
''' + setup + '\n(async()=>{\n' + scenario + '\n})().catch(e=>{console.error(e.stack);process.exit(1)});'
    result = subprocess.run(['node', '-e', code], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


DOM = r'''
class El {
 constructor(tag='div'){this.tag=tag;this.children=[];this.dataset={};this.attrs={};this.classes=new Set();this.style={};this.scrollTop=0;this.textContent='';
 this.classList={add:c=>this.classes.add(c),remove:c=>this.classes.delete(c),toggle:(c,v)=>v?this.classes.add(c):this.classes.delete(c)};}
 append(...items){this.children.push(...items);}replaceChildren(){this.children=[];}setAttribute(k,v){this.attrs[k]=v;}
 addEventListener(){}querySelectorAll(){return [];}contains(){return false;}focus(){}
}
const make=(tag,cls,text)=>{const el=new El(tag);el.className=cls||'';if(text!==undefined)el.textContent=text;return el;};
const detail=new El(),document={activeElement:null,createElementNS:(ns,tag)=>new El(tag)},$=()=>null;
let detailRevision='',detailDirty=false;const revisionOf=row=>JSON.stringify(row),backToBoard=()=>{},when=ts=>'time:'+ts;
const flatten=el=>[String(el.textContent||''),...el.children.flatMap(flatten)],elements=el=>[el,...el.children.flatMap(elements)];
'''


class StartupGoalFormattingTest(unittest.TestCase):
    def test_card_reading_includes_both_percentiles_and_page_open_denominator(self):
        r = node('console.log(JSON.stringify(startupReading(goal)));')
        self.assertEqual(r, {'value': 'Median 1.755 s', 'detail': 'P90 17.748 s · 25 page openings'})

    def test_missing_legacy_or_non_open_stats_cannot_create_page_open_percentiles(self):
        r = node('''const legacy={...goal,sample:{value:.1}};
const connect={...goal,sample:{value:.1,startup:{...stats,from:'connect',p50:.1,p90:.1}}};
console.log(JSON.stringify({legacy:startupReading(legacy),connect:startupReading(connect),legacyValue:legacy.sample.value}));''')
        self.assertEqual(r['legacy']['value'], 'Median —')
        self.assertIn('unavailable', r['legacy']['detail'])
        self.assertEqual(r['connect']['value'], 'Median —')
        self.assertNotIn('0.1 s', r['connect']['detail'])
        self.assertEqual(r['legacyValue'], .1)

    def test_insufficient_samples_show_count_without_assessed_percentiles(self):
        r = node('console.log(JSON.stringify(startupReading({...goal,sample:{value:null,startup:{...stats,n:2,ready:false}}})));')
        self.assertEqual(r, {'value': 'Median —', 'detail': 'P90 — · 2 page openings · need 5 valid samples'})

    def test_malformed_values_stay_unavailable_instead_of_becoming_zero(self):
        r = node('''const bad=[NaN,Infinity,-1,'0',null,true];
console.log(JSON.stringify({seconds:bad.map(startupSeconds),summaries:bad.map(p90=>startupReading({...goal,sample:{startup:{...stats,p90}}})),
 count:startupStats({...goal,sample:{startup:{...stats,n:'25'}}}),other:startupStats({...goal,goal_id:'cpu'})}));''')
        self.assertEqual(r['seconds'], ['—'] * 6)
        self.assertTrue(all(item['value'] == 'Median —' for item in r['summaries']))
        self.assertIsNone(r['count']['n'])
        self.assertFalse(r['count']['ready'])
        self.assertIsNone(r['other'])

    def test_minimum_cannot_be_lowered_by_malformed_stats(self):
        r = node('''console.log(JSON.stringify([0,1,-1,null,true,NaN].map(min_samples=>startupStats({...goal,sample:{startup:{...stats,n:2,min_samples}}}))));''')
        self.assertTrue(all(item['min_samples'] == 5 and not item['ready'] for item in r))

    def test_zero_is_displayed_only_when_it_is_an_actual_valid_number(self):
        r = node('console.log(JSON.stringify([startupSeconds(0),startupSeconds(10),startupSeconds(1.2)]));')
        self.assertEqual(r, ['0 s', '10 s', '1.2 s'])


class StartupGoalDomTest(unittest.TestCase):
    def render(self, scenario):
        return node(scenario, DOM + ''.join(function(name) for name in ['evidence', 'restoreDetail', 'renderGoalDetail', 'card']))

    def test_card_shows_p90_and_sample_count_without_becoming_movable(self):
        r = self.render('''const el=card(goal).children[0];console.log(JSON.stringify({text:flatten(el).join(' '),draggable:el.draggable}));''')
        for value in ['Median 1.755 s', 'P90 17.748 s', '25 page openings', 'Off target']:
            self.assertIn(value, r['text'])
        self.assertFalse(r['draggable'])

    def test_detail_separates_success_percentiles_failure_context_and_connection_diagnostics(self):
        r = self.render('''renderGoalDetail(goal);console.log(JSON.stringify({text:flatten(detail).join(' '),tags:elements(detail).map(el=>el.tag)}));''')
        for value in ['Page-open median 1.755 s', 'Page-open p90 17.748 s', 'Page-open samples 25', 'Last 7 days',
                      'Failed starts are outside these percentiles', 'foreground resumes are not measured', 'Startup timeout events 2',
                      'Reconnect events 38', 'Connection-only samples 250', 'Connection-only median 0.3 s',
                      'Connection-only p90 0.9 s', 'do not satisfy the page-open target']:
            self.assertIn(value, r['text'])
        self.assertNotIn('select', r['tags'])

    def test_detail_never_displays_missing_legacy_p90_as_zero(self):
        r = self.render('''renderGoalDetail({...goal,sample:{value:.1}});console.log(JSON.stringify(flatten(detail).join(' ')));''')
        self.assertIn('Page-open p90 Not assessed', r)
        self.assertIn('Page-open samples Unavailable', r)
        self.assertIn('Legacy recorded value 0.1 (measurement basis unverified)', r)
        self.assertNotIn('Page-open p90 0', r)

    def test_report_links_to_its_existing_goal_without_duplicating_it(self):
        setup = DOM + '''
let links=[];const openMeasuredGoal=(id,reportId)=>links.push({id,reportId}),renderNextActions=()=>{},renderSavedFrame=()=>{},showMoveReason=()=>{};
''' + ''.join(function(name) for name in ['evidence', 'restoreDetail', 'renderDetail'])
        r = node('''const row={...report,resolution:{components:[...report.resolution.components,{id:'duplicate',state:'not_started',goal_id:'video_start'},{id:'bad',state:'not_started',goal_id:'../escape'}]}};
renderDetail(row);const buttons=elements(detail).filter(el=>el.dataset.goal);buttons[0].onclick();
console.log(JSON.stringify({buttons:buttons.map(el=>el.textContent),links}));''', setup)
        self.assertEqual(r['buttons'], ['View measured goal'])
        self.assertEqual(r['links'], [{'id': 'video_start', 'reportId': '20260101-120000'}])


NAV = r'''
let active=true,selectedId=report.id,openEpoch=1,goalNavigation=0,saving=false,detailDirty=false;
let messages=[],opened=[],lanes=[],calls=[],hold=null;
const say=text=>messages.push(text),chooseLane=lane=>lanes.push(lane),openCard=id=>{selectedId=id;opened.push(id);};
const store=new Store(async(path,body)=>{calls.push({path,body:body??null});return {available:true,goals:[goal]};});
store.items=[report];
'''


class StartupGoalNavigationTest(unittest.TestCase):
    def run_navigation(self, scenario):
        return node(scenario, NAV + function('openMeasuredGoal'))

    def test_link_refreshes_read_only_goals_and_opens_same_monitoring_board(self):
        r = self.run_navigation('''await openMeasuredGoal('video_start',report.id);
console.log(JSON.stringify({calls,opened,lanes,reportRevision:store.items[0].revision}));''')
        self.assertEqual(r['calls'], [{'path': '/api/monitoring', 'body': None}])
        self.assertEqual(r['opened'], ['goal:video_start'])
        self.assertEqual(r['lanes'], ['monitoring'])
        self.assertEqual(r['reportRevision'], 'r1')

    def test_failed_refresh_keeps_report_and_never_opens_cached_goal(self):
        r = self.run_navigation('''store.goals=[goal];store.api=async()=>{throw Error('Offline');};await openMeasuredGoal('video_start',report.id);
console.log(JSON.stringify({selectedId,opened,messages,error:store.goalError}));''')
        self.assertEqual(r['selectedId'], '20260101-120000')
        self.assertEqual(r['opened'], [])
        self.assertIn('Your request is still open', r['messages'][-1])
        self.assertIn('Offline', r['error'])

    def test_slow_report_history_does_not_block_goal_navigation(self):
        r = self.run_navigation('''store.pending=new Promise(()=>{});await openMeasuredGoal('video_start',report.id);
console.log(JSON.stringify({calls,opened}));''')
        self.assertEqual(r['calls'], [{'path': '/api/monitoring', 'body': None}])
        self.assertEqual(r['opened'], ['goal:video_start'])

    def test_dirty_move_draft_or_save_blocks_navigation_without_fetch(self):
        r = self.run_navigation('''detailDirty=true;await openMeasuredGoal('video_start',report.id);detailDirty=false;saving=true;await openMeasuredGoal('video_start',report.id);
console.log(JSON.stringify({calls,opened,selectedId,messages}));''')
        self.assertEqual(r['calls'], [])
        self.assertEqual(r['opened'], [])
        self.assertEqual(r['selectedId'], '20260101-120000')
        self.assertTrue(all('Save your request changes' in message for message in r['messages']))

    def test_closed_or_changed_report_cannot_be_reopened_by_late_goal_response(self):
        r = self.run_navigation('''let finish;store.api=()=>new Promise(resolve=>finish=resolve);const pending=openMeasuredGoal('video_start',report.id);
active=false;openEpoch++;finish({available:true,goals:[goal]});await pending;
console.log(JSON.stringify({opened,lanes}));''')
        self.assertEqual(r, {'opened': [], 'lanes': []})

    def test_unknown_goal_and_invalid_identity_keep_current_request(self):
        r = self.run_navigation('''await openMeasuredGoal('../private',report.id);await openMeasuredGoal('absent_goal',report.id);
console.log(JSON.stringify({calls,opened,selectedId,messages}));''')
        self.assertEqual(len(r['calls']), 1)
        self.assertEqual(r['opened'], [])
        self.assertEqual(r['selectedId'], '20260101-120000')
        self.assertIn('unavailable', r['messages'][-1])


if __name__ == '__main__':
    unittest.main()
