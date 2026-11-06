"""Run real Monitoring store/DOM helpers: automatic goals do not become movable tasks."""
import json
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'server/requests.js'


def node(scenario, setup=''):
    js = 'const {Store,LANES,laneOf,isGoal,compareRows,normalizeGoals,goalTitle,goalHealth,goalValue,measurementAge,trendSegments,matches,hasEvidence}=require(' + json.dumps(str(MODULE)) + ');\n'
    js += r'''
const report={id:'20261101-120000',note:'Follow needs watching',revision:'report-r1',status:{state:'monitoring'},resolution:{components:[]}};
const goal={id:'goal:cpu',goal_id:'cpu',kind:'goal',title:'Camera services CPU',metric:'Share of one core',target:'<= 8',source:'systemd accounting',
 note:'CPU counters reset after service restart.',health:{state:'on_target',label:'On target',reason:'Within target.',expires_at:200},
 sample:{ts:110,observed_at:100,value:4,target:'<= 8',ok:true,detail:'Measured over an hour'},series:[[10,3,true],[20,null,null],[30,4,true]],
 work:[{id:'goal:cpu',title:'Reduce CPU use',status:'open',summary:'Investigation queued',updated:100}]};
const page={reports:[report],total:1,next_before:null};let posted=[];
const api=async(path,body)=>{if(body){posted.push(body);return {...report,revision:'r2',status:{state:body.state}};}return path==='/api/monitoring'?{available:true,goals:[goal]}:page;};
''' + setup + '\n(async()=>{\n' + scenario + '\n})().catch(e=>{console.error(e.stack);process.exit(1)});'
    result = subprocess.run(['node', '-e', js], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


def function(name):
    match = re.search(r'    function ' + name + r'\(.*?^    }\n', MODULE.read_text(), re.M | re.S)
    assert match, name
    return match.group(0)


class GoalStoreTest(unittest.TestCase):
    def test_goals_join_monitoring_without_inflating_report_history_or_completion(self):
        r = node('''const store=new Store(api);await store.refresh();
console.log(JSON.stringify({reports:store.items.length,total:store.total,goals:store.goals.length,counts:store.counts(),goalLane:laneOf({...store.goals[0],status:{state:'done'}})}));''')
        self.assertEqual((r['reports'], r['total'], r['goals']), (1, 1, 1))
        self.assertEqual(r['counts']['monitoring'], 2)
        self.assertEqual(r['counts']['done'], 0)
        self.assertEqual(r['goalLane'], 'monitoring')

    def test_fresh_goal_batches_cannot_bury_owner_observations_or_reorder_other_lanes(self):
        r = node('''const rows=[{...goal,ts:1000},{...report,id:'older',ts:1},{...report,id:'newer',ts:2},
{...report,id:'backlog-old',ts:3,status:{state:'not_started'}},{...report,id:'backlog-new',ts:4,status:{state:'not_started'}}].sort(compareRows);
console.log(JSON.stringify({monitoring:rows.filter(row=>laneOf(row)==='monitoring').map(row=>row.id),backlog:rows.filter(row=>laneOf(row)==='not_started').map(row=>row.id)}));''')
        self.assertEqual(r['monitoring'], ['newer', 'older', 'goal:cpu'])
        self.assertEqual(r['backlog'], ['backlog-new', 'backlog-old'])

    def test_goal_move_is_rejected_before_any_post(self):
        r = node('''const store=new Store(api);await store.refresh();let error;
try{await store.move('goal:cpu','done')}catch(e){error=e.message}
console.log(JSON.stringify({posted,error,lane:laneOf(store.goals[0])}));''')
        self.assertEqual(r['posted'], [])
        self.assertIn('stay in Monitoring', r['error'])
        self.assertEqual(r['lane'], 'monitoring')

    def test_goal_refresh_failure_preserves_reports_and_marks_cached_health_unknown(self):
        r = node('''const store=new Store(api);await store.refresh();
store.api=async(path)=>{if(path==='/api/monitoring')throw Error('Unavailable');return page;};await store.refresh();
const cached=store.allItems().find(isGoal);console.log(JSON.stringify({reports:store.items.length,goals:store.goals.length,error:store.goalError,health:goalHealth(cached,120)}));''')
        self.assertEqual((r['reports'], r['goals']), (1, 1))
        self.assertIn('Unavailable', r['error'])
        self.assertEqual(r['health']['state'], 'unavailable')
        self.assertNotEqual(r['health']['label'], 'On target')

    def test_report_fetch_failure_does_not_hide_goal_monitors(self):
        r = node('''const store=new Store(async(path)=>{if(path!=='/api/monitoring')throw Error('Reports offline');return {goals:[goal],available:true};});
await store.refresh();console.log(JSON.stringify({goals:store.goals.length,count:store.counts().monitoring,error:store.error,goalError:store.goalError}));''')
        self.assertEqual((r['goals'], r['count']), (1, 1))
        self.assertIn('Reports offline', r['error'])
        self.assertEqual(r['goalError'], '')

    def test_goal_timeout_is_bounded_and_does_not_pin_report_loading(self):
        r = node('''let aborted=false;const store=new Store(async(path,body,options)=>{if(path==='/api/monitoring'){options.signal.addEventListener('abort',()=>aborted=true);return new Promise(()=>{});}return page;});
store.monitorTimeoutMs=10;await store.refresh();console.log(JSON.stringify({aborted,reports:store.items.length,pending:store.pending,loading:store.loading,goalsLoading:store.goalsLoading,error:store.goalError}));''')
        self.assertTrue(r['aborted'])
        self.assertEqual(r['reports'], 1)
        self.assertIsNone(r['pending'])
        self.assertFalse(r['loading'] or r['goalsLoading'])
        self.assertIn('timed out', r['error'])

    def test_recovery_clears_transport_error_without_moving_reports(self):
        r = node('''const store=new Store(api);await store.refresh();store.api=async(path)=>{if(path==='/api/monitoring')throw Error('Down');return page;};await store.refresh();store.api=api;await store.refresh();
console.log(JSON.stringify({error:store.goalError,health:goalHealth(store.allItems().find(isGoal),120),report:store.items[0]}));''')
        self.assertEqual(r['error'], '')
        self.assertEqual(r['health']['state'], 'on_target')
        self.assertEqual(r['report']['revision'], 'report-r1')

    def test_malformed_or_duplicate_goal_data_cannot_silently_choose_a_healthy_value(self):
        r = node('''const store=new Store(api);await store.refresh();store.api=async(path)=>path==='/api/monitoring'?{goals:[goal,{...goal,health:{state:'off_target'}}]}:page;await store.refresh();
let badIdentity=false;try{normalizeGoals([{...goal,goal_id:'../report'}])}catch(e){badIdentity=true}
console.log(JSON.stringify({goals:store.goals.length,error:store.goalError,badIdentity,health:goalHealth(store.allItems().find(isGoal),120)}));''')
        self.assertEqual(r['goals'], 1)
        self.assertIn('more than once', r['error'])
        self.assertTrue(r['badIdentity'])
        self.assertEqual(r['health']['state'], 'unavailable')

    def test_goal_identity_is_namespaced_and_cannot_override_lane_or_movability(self):
        r = node('''const row=normalizeGoals([{...goal,id:report.id,movable:true,status:{state:'done'}}])[0];console.log(JSON.stringify({id:row.id,movable:row.movable,lane:laneOf(row),isGoal:isGoal(row)}));''')
        self.assertEqual(r, {'id': 'goal:cpu', 'movable': False, 'lane': 'monitoring', 'isGoal': True})

    def test_goal_fetch_cannot_undo_an_in_flight_report_move(self):
        r = node('''const store=new Store(api);store.items=[report];let finishGoals,finishReports;
store.api=async(path,body)=>body?{...report,revision:'r2',status:{state:'in_progress'}}:path==='/api/monitoring'?new Promise(r=>finishGoals=r):new Promise(r=>finishReports=r);
const refresh=store.refresh();await store.move(report.id,'in_progress','','report-r1');finishGoals({goals:[goal]});finishReports(page);await refresh;
console.log(JSON.stringify({revision:store.items[0].revision,lane:laneOf(store.items[0]),goals:store.goals.length}));''')
        self.assertEqual(r, {'revision': 'r2', 'lane': 'in_progress', 'goals': 1})

    def test_off_target_stale_unmeasured_and_invalid_are_never_ready_review_or_done(self):
        r = node('''const values=['off_target','stale','unmeasured','invalid'].map(state=>{const row={...goal,health:{state,label:state,expires_at:200}};return {state:goalHealth(row,120).state,lane:laneOf(row)};});
console.log(JSON.stringify(values));''')
        self.assertEqual([x['state'] for x in r], ['off_target', 'stale', 'unmeasured', 'invalid'])
        self.assertTrue(all(x['lane'] == 'monitoring' for x in r))

    def test_cached_health_expires_and_missing_values_cannot_look_on_target(self):
        r = node('''console.log(JSON.stringify({expired:goalHealth(goal,201),missing:goalHealth({...goal,sample:{...goal.sample,value:null}},120),unknown:goalHealth({...goal,health:{state:'constructor'}},120),noExpiry:goalHealth({...goal,health:{state:'on_target'}},120)}));''')
        self.assertEqual(r['expired']['state'], 'stale')
        self.assertEqual(r['missing']['state'], 'invalid')
        self.assertEqual(r['unknown']['state'], 'invalid')
        self.assertEqual(r['noExpiry']['state'], 'invalid')

    def test_goal_search_finds_sources_limits_and_linked_work(self):
        r = node('''console.log(JSON.stringify(['systemd','counters reset','investigation queued','<= 8'].map(q=>matches(goal,q))));''')
        self.assertEqual(r, [True] * 4)

    def test_sample_age_uses_source_measurement_time_and_handles_missing_or_future_times(self):
        r = node('''console.log(JSON.stringify([measurementAge(goal,160),measurementAge(goal,3700),measurementAge(goal,86500),
measurementAge({...goal,sample:{ts:110}},120),measurementAge(goal,0)]));''')
        self.assertEqual(r, ['Measured 1m ago', 'Measured 1h ago', 'Measured 1d ago',
                             'Collected just now', 'Measurement time is ahead of this device'])

    def test_empty_measurement_uses_checked_age_instead_of_claiming_a_measured_value(self):
        r = node('''const empty={...goal,sample:{...goal.sample,value:null},health:{state:'unmeasured',label:'Not measured'}};
console.log(JSON.stringify({age:measurementAge(empty,160),value:goalValue(empty),label:goalHealth(empty,160).label}));''')
        self.assertEqual(r, {'age': 'Checked 1m ago', 'value': '—', 'label': 'Not measured'})

    def test_find_display_titles_do_not_leak_catalog_placeholders_or_mutate_definitions(self):
        r = node('''const row={...goal,goal_id:'search_success',title:'Find <pet> finds her when she is home'};
console.log(JSON.stringify({title:goalTitle(row),original:row.title,find:goalTitle({...row,goal_id:'find_time'}),other:goalTitle({...goal,goal_id:'constructor'})}));''')
        self.assertEqual(r['title'], 'Find locates your pet')
        self.assertEqual(r['find'], 'Find locates your pet quickly')
        self.assertIn('<pet>', r['original'])
        self.assertEqual(r['other'], 'Camera services CPU')

    def test_trend_gaps_are_broken_and_non_numeric_data_is_ignored(self):
        r = node('''console.log(JSON.stringify({paths:trendSegments([[1,1],[2,null],[3,3],[4,'4'],[5,5]]),one:trendSegments([[1,1]]),allMissing:trendSegments([[1,null],[2,null]])}));''')
        self.assertEqual(len(r['paths']), 3)
        self.assertTrue(all('L' not in p for p in r['paths']))
        self.assertEqual(r['one'], [])
        self.assertEqual(r['allMissing'], [])

    def test_trend_does_not_connect_over_omitted_long_gaps(self):
        r = node('''console.log(JSON.stringify(trendSegments([[1,1],[1801,2],[10801,3],[12601,4]])));''')
        self.assertEqual(len(r), 2)
        self.assertTrue(all(p.count('L') == 1 for p in r))


class GoalDomTest(unittest.TestCase):
    def render(self, scenario):
        setup = r'''
class El {
 constructor(tag='div'){this.tag=tag;this.children=[];this.dataset={};this.attrs={};this.classes=new Set();this.style={};this.scrollTop=0;this.textContent='';
 this.classList={add:c=>this.classes.add(c),remove:c=>this.classes.delete(c),toggle:(c,v)=>v?this.classes.add(c):this.classes.delete(c)};}
 append(...items){this.children.push(...items);}replaceChildren(){this.children=[];}setAttribute(k,v){this.attrs[k]=v;}
 addEventListener(){}querySelectorAll(){return [];}contains(){return false;}focus(){}
}
const make=(tag,cls,text)=>{const el=new El(tag);el.className=cls||'';if(text!==undefined)el.textContent=text;return el;};
const detail=new El(),document={activeElement:null,createElementNS:(ns,tag)=>new El(tag)},$=()=>null;
let detailRevision='',detailDirty=false;const revisionOf=row=>JSON.stringify(row),backToBoard=()=>{},when=ts=>'time:'+ts;
const flatten=el=>[String(el.textContent||''),...el.children.flatMap(flatten)],tags=el=>[el.tag,...el.children.flatMap(tags)];
''' + ''.join(function(name) for name in ['evidence', 'restoreDetail', 'renderGoalDetail', 'renderDetail', 'card'])
        return node(scenario, setup)

    def test_goal_detail_has_no_move_controls_and_keeps_measurement_vs_collection_times_distinct(self):
        r = self.render('''renderDetail({...goal,health:{state:'stale',label:'Stale',reason:'Source is old'}});
console.log(JSON.stringify({text:flatten(detail).join(' '),tags:tags(detail),revision:detailRevision}));''')
        self.assertNotIn('select', r['tags'])
        self.assertNotIn('textarea', r['tags'])
        self.assertIn('Measured at time:100', r['text'])
        self.assertIn('Collected at time:110', r['text'])
        self.assertIn('systemd accounting', r['text'])
        self.assertIn('counters reset', r['text'])
        self.assertIn('Linked work', r['text'])
        self.assertIn('Investigation queued', r['text'])
        self.assertIn('Stale', r['text'])
        self.assertIn('skipped collections are not shown', r['text'])
        self.assertIn('checked periodically', r['text'])

    def test_goal_card_is_not_draggable_and_does_not_show_verified_task_progress(self):
        r = self.render('''const item=card(goal),button=item.children[0];console.log(JSON.stringify({draggable:button.draggable,text:flatten(button).join(' ')}));''')
        self.assertFalse(r['draggable'])
        self.assertIn('Automatic goal', r['text'])
        self.assertNotIn('verified', r['text'])

    def navigation(self, scenario):
        src = MODULE.read_text()
        body = re.search(r'return \{ store, close, async open\(options = \{\}\) \{(.*?)^    } };', src, re.M | re.S).group(1)
        setup = r'''
let active=false,openEpoch=0,selectedId=null,selectedLane='needs_review',firstLoad=true,timer=null;
const root={classList:{remove(){},add(){}}},node={classList:{add(){},remove(){}},focus(){}};
const $=()=>node,store={goals:[goal]};let selected=[],opened=[],messages=[],resolvers=[];
const chooseLane=state=>selected.push(state),openCard=id=>opened.push(id),say=text=>messages.push(text),backToBoard=()=>{};
const refresh=()=>new Promise(resolve=>resolvers.push(resolve));
''' + function('close') + '\nconst open=async function(options={}){' + body + '\n};\n'
        return node(scenario, setup)

    def test_deep_link_selects_monitoring_and_opens_the_named_goal(self):
        r = self.navigation("const pending=open({lane:'monitoring',goal:'cpu'});resolvers[0]();await pending;console.log(JSON.stringify({selected,opened}));")
        self.assertEqual(r, {'selected': ['monitoring'], 'opened': ['goal:cpu']})

    def test_closed_or_superseded_goal_navigation_cannot_reopen_after_refresh(self):
        r = self.navigation("const first=open({lane:'monitoring',goal:'cpu'});close();resolvers[0]();await first;const second=open({lane:'monitoring',goal:'cpu'});const third=open({lane:'monitoring'});resolvers[1]();resolvers[2]();await Promise.all([second,third]);console.log(JSON.stringify({opened}));")
        self.assertEqual(r['opened'], [])


if __name__ == '__main__':
    unittest.main()
