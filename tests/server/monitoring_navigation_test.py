"""Exercise shipped Monitoring navigation and request cancellation in Node."""
import json
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / 'server/index.html'
AGENT = ROOT / 'server/agent.html'


def between(source, start, end):
    """Keep execution tied to the deployed source rather than a copied handler."""
    return source[source.index(start):source.index(end, source.index(start))]


def run_node(source):
    result = subprocess.run(['node', '-e', source], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


def main_fixture(scenario, search=''):
    navigation = between(INDEX.read_text(), 'function openMonitoring(goal)', '// Live status is pushed')
    return run_node(r'''
const vm = require('vm');
const opens = [], tracks = [], listeners = {}, agentWindow = {};
const controls = {'monitoring-open': {}, 'agent-frame': {contentWindow: agentWindow}};
const context = {
  URLSearchParams, location: {origin: 'https://camera.test', search: SEARCH},
  $: id => controls[id], track: (...args) => tracks.push(args),
  requestBoard: {open: options => {opens.push(options); return Promise.resolve();}},
  window: {addEventListener: (type, handler) => {listeners[type] = handler;}}
};
vm.runInNewContext(NAVIGATION, context);
const message = changes => listeners.message({origin: context.location.origin,
  source: agentWindow, data: {type: 'openpetcam-monitoring', goal: 'cpu'}, ...changes});
'''.replace('SEARCH', json.dumps(search)).replace('NAVIGATION', json.dumps(navigation)) + scenario)


def agent_fixture(scenario, search='', embedded=False):
    source = AGENT.read_text()
    declaration = re.search(r'^const selectedThread = .*?;$', source, re.M)
    if declaration is None:
        raise AssertionError('Agent page no longer declares its selected thread')
    navigation = declaration.group(0) + '\n' + between(
        source, 'if (selectedThread?.startsWith("goal:"))', '// Inside the app\'s Agent tab')
    anchor = re.search(r'<a href="([^"]+)" id="open-monitoring"', source)
    if anchor is None:
        raise AssertionError('Agent page no longer supplies a standalone Monitoring link')
    return run_node(r'''
const vm = require('vm');
const posts = [], control = {href: HREF}; let prevented = 0;
const location = {origin: 'https://camera.test', search: SEARCH};
const window = {}; window.parent = EMBEDDED ? {postMessage: (...args) => posts.push(args)} : window;
const context = {URLSearchParams, encodeURIComponent, location, window, $: () => control};
vm.runInNewContext(NAVIGATION, context);
const click = () => control.onclick({preventDefault: () => {prevented++;}});
'''.replace('HREF', json.dumps(anchor.group(1))).replace('SEARCH', json.dumps(search))
        .replace('EMBEDDED', json.dumps(embedded)).replace('NAVIGATION', json.dumps(navigation)) + scenario)


class MonitoringNavigationTest(unittest.TestCase):
    def test_only_trusted_agent_frame_messages_open_the_board(self):
        result = main_fixture('''
for (const invalid of [
  {origin: 'https://untrusted.test'}, {source: {}}, {source: null},
  {data: {type: 'other', goal: 'cpu'}}, {data: null}, {data: {goal: 'cpu'}}
]) message(invalid);
const ignored = opens.length;
message({});
console.log(JSON.stringify({ignored, opens}));
''')
        self.assertEqual(result['ignored'], 0)
        self.assertEqual(result['opens'], [{'lane': 'monitoring', 'goal': 'cpu'}])

    def test_trusted_messages_bound_and_validate_selected_goal(self):
        result = main_fixture('''
for (const goal of ['follow_retention', '../report', '<script>', 'x'.repeat(101), 7, null])
  message({data: {type: 'openpetcam-monitoring', goal}});
console.log(JSON.stringify(opens));
''')
        self.assertEqual(result[0], {'lane': 'monitoring', 'goal': 'follow_retention'})
        self.assertEqual(result[1:], [{'lane': 'monitoring'}] * 5)

    def test_direct_deep_link_opens_monitoring_with_selected_goal(self):
        result = main_fixture('console.log(JSON.stringify(opens));',
                              '?monitoring=goals&goal=app_status_fresh')
        self.assertEqual(result, [{'lane': 'monitoring', 'goal': 'app_status_fresh'}])

    def test_unrelated_query_does_not_open_monitoring(self):
        result = main_fixture('console.log(JSON.stringify(opens));', '?goal=cpu&monitoring=other')
        self.assertEqual(result, [])

    def test_main_monitoring_button_opens_same_lane(self):
        self.assertRegex(INDEX.read_text(), r'<button[^>]+id="monitoring-open"[^>]+type="button">Monitoring</button>')
        result = main_fixture('''
controls['monitoring-open'].onclick();
console.log(JSON.stringify({opens, tracks}));
''')
        self.assertEqual(result['opens'], [{'lane': 'monitoring'}])
        self.assertEqual(result['tracks'], [['sheet', {'name': 'monitoring'}]])

    def test_standalone_agent_link_preserves_normal_navigation(self):
        result = agent_fixture('click();console.log(JSON.stringify({href:control.href,prevented,posts}));')
        self.assertEqual(result, {'href': '/?monitoring=goals', 'prevented': 0, 'posts': []})

    def test_standalone_selected_goal_link_survives_round_trip(self):
        result = agent_fixture('click();console.log(JSON.stringify({href:control.href,prevented,posts}));',
                               '?thread=goal%3Afollow_retention')
        self.assertEqual(result, {'href': '/?monitoring=goals&goal=follow_retention',
                                  'prevented': 0, 'posts': []})
        opened = main_fixture('console.log(JSON.stringify(opens));', '?' + result['href'].split('?', 1)[1])
        self.assertEqual(opened, [{'lane': 'monitoring', 'goal': 'follow_retention'}])

    def test_embedded_agent_uses_same_origin_message_instead_of_navigating(self):
        result = agent_fixture('click();console.log(JSON.stringify({prevented,posts}));',
                               '?embed=1&thread=goal%3Acpu', embedded=True)
        self.assertEqual(result, {'prevented': 1, 'posts': [
            [{'type': 'openpetcam-monitoring', 'goal': 'cpu'}, 'https://camera.test']]})

    def test_embedded_activity_without_goal_opens_monitoring_without_selection(self):
        for search in ('?embed=1', '?embed=1&thread=report%3A20260101-120000'):
            with self.subTest(search=search):
                result = agent_fixture('click();console.log(JSON.stringify({prevented,posts}));',
                                       search, embedded=True)
                self.assertEqual(result, {'prevented': 1, 'posts': [
                    [{'type': 'openpetcam-monitoring'}, 'https://camera.test']]})


class RequestSignalTest(unittest.TestCase):
    def test_real_api_helper_forwards_abort_signal_for_get_and_post(self):
        api = between(INDEX.read_text(), 'async function api(path, body, options = {})', 'const fmtTime =')
        result = run_node(r'''
const vm = require('vm'), calls = [], signal = new AbortController().signal;
const context = {location: {}, fetch: async (path, options) => {
  calls.push({path, options}); return {status:200,ok:true,json:async()=>({accepted:true})};
}};
vm.runInNewContext(API, context);
(async()=>{
  const get = await context.api('/api/monitoring', undefined, {signal});
  const post = await context.api('/api/report', {note:'A request'}, {signal});
  const defaultGet = await context.api('/api/monitoring');
  console.log(JSON.stringify({get,post,defaultGet,
    getSignal:calls[0].options.signal===signal,postSignal:calls[1].options.signal===signal,
    getOptions:calls[0].options,postOptions:calls[1].options,defaultSignal:calls[2].options.signal===undefined}));
})().catch(e=>{console.error(e.stack);process.exit(1)});
'''.replace('API', json.dumps(api)))
        self.assertTrue(result['getSignal'] and result['postSignal'] and result['defaultSignal'])
        self.assertEqual(result['getOptions']['cache'], 'no-store')
        self.assertNotIn('method', result['getOptions'])
        self.assertEqual(result['postOptions']['method'], 'POST')
        self.assertEqual(result['postOptions']['headers'], {'Content-Type': 'application/json'})
        self.assertEqual(json.loads(result['postOptions']['body']), {'note': 'A request'})
        self.assertEqual([result[key] for key in ('get', 'post', 'defaultGet')], [{'accepted': True}] * 3)


if __name__ == '__main__':
    unittest.main()
