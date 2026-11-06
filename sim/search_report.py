"""Render a private, self-contained report from a completed search experiment.

    python3 -m sim.search_report eval/route_search/experiment-v1

Only the requested HTML artifact is written. No policy, models, camera APIs or
network services are touched. Embedded data makes the report work via file://.
"""
import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path

from .search_prior import load_prior


ROOT = Path(__file__).resolve().parents[1]


def _read(folder, name):
    return json.loads((folder / name).read_text())


def _number(value, digits=1):
    return f'{value:,.{digits}f}'


def _percent(value):
    return f'{100 * value:.1f}%'


def _table(headers, rows, selected=None):
    head = ''.join(f'<th scope="col">{escape(str(h))}</th>' for h in headers)
    body = []
    for index, row in enumerate(rows):
        cells = ''.join(f'<td>{escape(str(c))}</td>' for c in row)
        body.append(f'<tr{" class=selected" if index == selected else ""}>{cells}</tr>')
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def render(folder, output=None):
    folder = Path(folder).resolve()
    output = Path(output).resolve() if output else folder.parent / 'index.html'
    manifest = _read(folder, 'manifest.json')
    selection = _read(folder, 'selection.json')
    summary = _read(folder, 'summary.json')
    test, stress = _read(folder, 'test.json'), _read(folder, 'stress.json')
    selected = summary['selected']
    tests = test['comparison']
    stresses = stress['comparison']
    policies = manifest['policies']
    development = selection['development']
    ndev = development['incumbent']['n']
    budget = manifest['budget_s']
    tb, tc = tests['base'], tests['candidate']
    sb, sc = stresses['base'], stresses['candidate']
    travel_improvement = (tb['mean_motor_steps'] - tc['mean_motor_steps']) / tb['mean_motor_steps']
    wrong_guard = tc['wrong_finds'] <= tb['wrong_finds'] and sc['wrong_finds'] <= sb['wrong_finds']
    acceptance = summary.get('acceptance', {})
    passes_gate = acceptance.get('passes_simulation_gate', wrong_guard)
    guard_label = ('Wrong-find guardrail failed' if not wrong_guard else
                   'Finite simulation gate passed' if passes_gate else 'Simulation guardrail failed')
    wrong_counts = (f'The selected route, <b>{escape(selected.replace("_", " "))}</b>, records '
                    f'<b>{tc["wrong_finds"]} vs {tb["wrong_finds"]}</b> wrong finds in the test bank '
                    f'and <b>{sc["wrong_finds"]} vs {sb["wrong_finds"]}</b> under stress (candidate vs incumbent). ')
    if not wrong_guard:
        guard_body = wrong_counts + 'The development rule allowed no increase. Faster searches do not clear that rule.'
        lede = ('Going to the nearest useful stop first saves time and movement in this simulator. '
                'It also produces more wrong confirmations. That makes it a candidate to investigate, '
                'with a failed guardrail still attached.')
    elif not passes_gate:
        failed = [f'{bank}: {check.replace("_", " ")}'
                  for bank, checks in acceptance.get('checks', {}).items()
                  for check, passed in checks.items() if not passed]
        guard_body = wrong_counts + 'Other acceptance checks failed: ' + escape('; '.join(failed)) + '. The route remains offline.'
        lede = 'This finite experiment compares route ordering, search time, movement and confirmation errors. Its acceptance checks still leave work to do.'
    else:
        guard_body = wrong_counts + 'The declared finite-bank checks passed. This remains an offline candidate; real-world validation and installation are separate steps.'
        lede = 'This route cleared the declared checks in the simulator. The next question is whether independently labelled real searches support the same result.'

    comparisons = []
    for label, report in [('Test', tests), ('Stress', stresses)]:
        for role, name in [('base', 'Incumbent'), ('candidate', selected.replace('_', ' '))]:
            r = report[role]
            comparisons.append([label, name, str(r['n']), _number(r['mean_capped_loss_s']) + ' s',
                                _percent(r['findable_within_60s']), _number(r['mean_motor_steps'], 0),
                                str(r['wrong_finds']), str(r['deadline_count'])])
    comparison_table = _table(['Bank', 'Route', 'Trials', 'Capped time ↓', 'Findable ≤60 s ↑',
                               'Motor steps ↓', 'Wrong finds', '120 s cutoffs'], comparisons)
    dev_rows = [[name.replace('_', ' '), _number(development[name]['mean_score_s']) + ' s',
                 _number(development[name]['mean_capped_loss_s']) + ' s',
                 _number(development[name]['mean_motor_steps'], 0),
                 _percent(development[name]['found_right']), str(development[name]['wrong_finds'])]
                for name in policies]
    dev_table = _table(['Development route', 'Selection score ↓', 'Capped time ↓',
                       'Motor steps ↓', 'Correct finds', 'Wrong finds'], dev_rows,
                      selected=policies.index(selected))
    intervals = []
    names = {'loss_s': 'Capped time, seconds', 'motor_steps': 'Motor steps', 'score_s': 'Selection score, seconds'}
    for label, report in [('Test', tests), ('Stress', stresses)]:
        for metric, readable in names.items():
            delta = report['paired'][metric]
            lo, hi = delta['paired_bootstrap_95']
            intervals.append([label, readable, _number(delta['candidate_minus_base']),
                              f'{_number(lo)} to {_number(hi)}'])
    interval_table = _table(['Bank', 'Candidate minus incumbent', 'Mean change', 'Paired bootstrap 95% interval'], intervals)
    family_rows = []
    for family, report in tests['by_family'].items():
        base, candidate = report['incumbent'], report[selected]
        family_rows.append([family.replace('_', ' '), str(base['n']),
                            f'{_percent(base["found_right"])} → {_percent(candidate["found_right"])}',
                            f'{_number(base["mean_capped_loss_s"])} → {_number(candidate["mean_capped_loss_s"])} s',
                            f'{base["wrong_finds"]} → {candidate["wrong_finds"]}'])
    family_table = _table(['Test family', 'Paired cases', 'Correct finds', 'Capped time', 'Wrong finds'], family_rows)
    wrong_rows = []
    for row in test['results'][selected]:
        if row['wrong_find']:
            evidence = row.get('confirmation') or {}
            pose = evidence.get('camera_pose')
            wrong_rows.append([row['id'], row['family'], _number(row['elapsed_s']) + ' s',
                               _number(evidence.get('captured_at', 0)) + ' s',
                               ', '.join(_number(x, 0) for x in pose) if pose else '—',
                               'False detector box' if evidence.get('false_detection') else 'Unverified'])
    wrong_table = _table(['Test case', 'Family', 'Returned at', 'Frame exposed at',
                         'Optical pan, tilt', 'Confirmation truth'], wrong_rows) if wrong_rows else '<p>No wrong finds recorded in this test bank.</p>'

    current_hashes_match = all((ROOT / name).exists() and
                               hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value
                               for name, value in manifest['input_sha256'].items())
    prior_data = {'targets': [], 'frame': [723.6, 431.1], 'limits': [3700, 1000]}
    if current_hashes_match:
        prior = load_prior(ROOT / 'sim/params.json', ROOT / 'models/room_map.json')
        prior_data = {'targets': prior.targets.tolist(), 'frame': list(prior.frame),
                      'limits': list(prior.limits)}
    base_by_id = {r['id']: r for r in test['results']['incumbent']}
    candidate_by_id = {r['id']: r for r in test['results'][selected]}
    cases = []
    for case in test['cases']:
        a, b = base_by_id[case['id']], candidate_by_id[case['id']]
        if 'trace' in a and 'trace' in b:
            cases.append({'id': case['id'], 'family': case['family'], 'scenario': case['scenario'],
                          'base': {k: a[k] for k in ('trace', 'elapsed_s', 'found_right', 'wrong_find', 'deadline')},
                          'candidate': {k: b[k] for k in ('trace', 'elapsed_s', 'found_right', 'wrong_find', 'deadline')}})
    embedded = json.dumps({'cases': cases, 'prior': prior_data, 'budget': budget,
                           'candidate': selected}, separators=(',', ':')).replace('<', '\\u003c')
    evidence_links = []
    for filename, label in [('manifest.json', 'Frozen inputs and assumptions'), ('selection.json', 'Development selection'),
                            ('test.json', 'Every test case'), ('stress.json', 'Every stress case'),
                            ('cancellation.json', 'Cancellation checks'), ('routes.json', 'Planned routes'),
                            ('summary.json', 'Machine-readable summary')]:
        href = escape(os.path.relpath(folder / filename, output.parent), quote=True)
        evidence_links.append(f'<a href="{href}">{escape(label)}</a>')
    limitations = ''.join(f'<li>{escape(item)}</li>' for item in manifest['limits'])
    historical = ''
    earlier = folder.parent / 'experiment-v1' / 'summary.json'
    if earlier.exists() and earlier.parent != folder:
        previous = json.loads(earlier.read_text())['comparisons']
        old_test, old_stress = previous['test'], previous['stress']
        href = escape(os.path.relpath(earlier, output.parent), quote=True)
        historical = (f'<p class="note">An <a href="{href}">earlier diagnostic run</a> recorded '
                      f'{old_test["candidate"]["wrong_finds"]} vs {old_test["base"]["wrong_finds"]} wrong test finds '
                      f'and {old_stress["candidate"]["wrong_finds"]} vs {old_stress["base"]["wrong_finds"]} under stress. '
                      'A recorder edge case was then fixed: confirmation is accepted only after the controller’s '
                      'found handler returns. The routes and development protocol stayed fixed; this report uses '
                      'the later run with fresh test and stress seeds.</p>')
    cancellation = '; '.join(f'{name}: {r["stopped"]}/{r["n"]} stopped, latest return {r["max_elapsed_s"]:.2f} s'
                              for name, r in summary['cancellation'].items())
    hash_text = ('All experiment input hashes matched at completion, and still match at render time.'
                 if summary['inputs_unchanged'] and current_hashes_match else
                 'Input hashes no longer all match; the report preserves recorded results. Current prior overlay omitted.')
    content = TEMPLATE
    replacements = {
        'TITLE': 'A shorter search. A failed guardrail.' if not passes_gate else 'Comparing the camera’s next move.',
        'SELECTED': escape(selected.replace('_', ' ')), 'GUARD_LABEL': guard_label,
        'GUARD_BODY': guard_body, 'LEDE': lede,
        'TEST_N': str(tb['n']), 'STRESS_N': str(sb['n']), 'DEV_N': str(ndev), 'POLICY_N': str(len(policies)),
        'TIME_GAIN': _number(tb['mean_capped_loss_s'] - tc['mean_capped_loss_s']),
        'TRAVEL_GAIN': _number(100 * travel_improvement, 1),
        'TEST_WRONG': f'{tc["wrong_finds"]} vs {tb["wrong_finds"]}',
        'STRESS_WRONG': f'{sc["wrong_finds"]} vs {sb["wrong_finds"]}',
        'COMPARISON': comparison_table, 'DEVELOPMENT': dev_table, 'INTERVALS': interval_table,
        'FAMILIES': family_table, 'WRONG_TABLE': wrong_table, 'LIMITATIONS': limitations,
        'CANCELLATION': escape(cancellation), 'HASH_TEXT': escape(hash_text),
        'HISTORICAL': historical, 'RUN_NAME': escape(folder.name),
        'TEST_SEED': str(manifest['split']['test_seed']), 'STRESS_SEED': str(manifest['split']['stress_seed']),
        'BENCH_HASH': manifest['input_sha256']['sim/search_bench.py'],
        'EVIDENCE_LINKS': ''.join(evidence_links), 'EMBEDDED': embedded,
    }
    for key, value in replacements.items():
        content = content.replace('{{' + key + '}}', value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    return output


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><title>OpenPetCam · Search routes in simulation</title>
<style>
:root{--paper:#f5f3ec;--ink:#253632;--muted:#5f7069;--line:#d5dcd3;--green:#217864;--orange:#ad6224;--blue:#526f9a}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1180px;margin:auto;padding:60px 38px 90px}a{color:var(--green);text-underline-offset:3px}p{max-width:76ch;margin:12px 0}h1,h2,h3{font-family:Georgia,"Times New Roman",serif;font-weight:400;letter-spacing:-.025em;line-height:1.15}h1{font-size:clamp(42px,6vw,75px);max-width:850px;margin:22px 0}h2{font-size:34px;margin:0 0 20px}h3{font-size:24px}.kicker{text-transform:uppercase;letter-spacing:.16em;font-size:12px;font-weight:700;color:var(--muted)}.lede{font-size:21px;line-height:1.55;max-width:780px}.subtle{color:var(--muted);font-size:14px}.scope{display:inline-block;border:1px solid var(--line);padding:4px 10px;font-size:12px;letter-spacing:.06em;text-transform:uppercase}.guard{margin:32px 0;padding:20px 24px;background:#f4e7d5;border-left:4px solid var(--orange)}.guard strong{display:block;font-size:20px;margin-bottom:6px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin:32px 0 4px}.stat{border-top:1px solid var(--line);padding-top:16px}.value{font:42px/1.15 Georgia,serif;letter-spacing:-.04em}.label{display:block;font-size:14px;color:var(--muted);margin-top:8px}section{padding-top:54px;margin-top:38px;border-top:1px solid var(--line)}.table-scroll{overflow-x:auto;margin:24px 0}table{border-collapse:collapse;min-width:680px;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th{text-align:left;color:var(--muted);font-size:12px;letter-spacing:.02em;font-weight:600;padding:12px 14px;border-bottom:2px solid var(--line)}td{padding:12px 14px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{padding-left:0}tr.selected{background:#e5ede3}tr.selected td:first-child{padding-left:10px}.note{border-left:2px solid var(--line);padding-left:18px;color:var(--muted);font-size:14px}.plot-shell{background:#fcfcf7;border:1px solid var(--line);margin:25px 0}.plot-head{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:18px 20px;border-bottom:1px solid var(--line)}.legend{font-size:13px;display:flex;gap:18px;flex-wrap:wrap}.swatch{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:7px}.base{background:var(--blue)}.candidate{background:var(--green)}.target{background:var(--orange)}canvas{display:block;width:100%;height:420px}button,select{font:inherit;font-size:14px;background:#fcfcf7;color:var(--ink);border:1px solid #a6b7ac;border-radius:4px;padding:8px 12px}button{cursor:pointer;min-width:102px}button:hover{background:#e5ede3}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #d8ae65;outline-offset:3px}.controls{padding:16px 20px;border-top:1px solid var(--line);display:grid;grid-template-columns:auto 1fr auto;gap:18px;align-items:center}input[type=range]{width:100%;accent-color:var(--green)}#time{font-size:13px;min-width:86px;font-variant-numeric:tabular-nums}.trace-note{padding:0 20px 18px;font-size:13px;color:var(--muted)}.trace-note span{display:block}.assumptions{columns:2;column-gap:40px;padding-left:20px}.assumptions li{break-inside:avoid;margin-bottom:12px;font-size:14px}.evidence{display:flex;flex-wrap:wrap;gap:10px 24px;font-size:14px;margin-top:25px}code{font:12px ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}.footer{margin-top:40px;font-size:13px;color:var(--muted)}@media(max-width:700px){main{padding:34px 20px 60px}.stats{gap:16px}.value{font-size:32px}.lede{font-size:19px}.plot-head{align-items:flex-start;flex-direction:column}.controls{grid-template-columns:auto 1fr;gap:12px}#time{grid-column:1/-1}.assumptions{columns:1}canvas{height:330px}section{margin-top:28px;padding-top:34px}}
</style></head><body><main>
<header><div class="kicker">OpenPetCam / Find-route experiment</div><h1>{{TITLE}}</h1>
<p class="lede">{{LEDE}}</p>
<p class="subtle">{{RUN_NAME}} · {{DEV_N}} development cases × {{POLICY_N}} routes · {{TEST_N}} independent-seed test cases · {{STRESS_N}} stress cases</p><span class="scope">Offline only · no policy installed</span></header>
<aside class="guard"><strong>{{GUARD_LABEL}}</strong>{{GUARD_BODY}}</aside>
<div class="stats"><div class="stat"><div class="value">{{TIME_GAIN}} s</div><span class="label">less mean capped search time, test bank</span></div><div class="stat"><div class="value">{{TRAVEL_GAIN}}%</div><span class="label">less integrated motor travel, test bank</span></div><div class="stat"><div class="value">120 s</div><span class="label">fixed budget; every miss pays the full cap</span></div></div>
<section><h2>What the comparison measures</h2><p>The production <code>Tracker.search()</code> runs against a controlled camera and target. Every route gets the same target trajectory and frame-keyed detector draws. Searches include the current view, confirmation, glimpse retries, moving-frame checks and the final return after failure.</p>
{{COMPARISON}}<p class="subtle">“Findable” excludes the deliberately absent and persistently occluded families. Each family has equal weight; these percentages do not estimate household performance. Capped time assigns 120 seconds to every miss or wrong find. Motor travel is integrated absolute pan + tilt movement until the search returns or confirms; a pending centering move after confirmation is outside this measurement.</p></section>
<section><h2>Watch the actual routes</h2><p>The paths below are optical camera positions from the executed controller, including reversals and reacquisition. Pale points mark floor-bearing <em>view centres</em> used by the angular prior. This is pan/tilt motor space, not a room floor plan.</p>
<div class="plot-shell"><div class="plot-head"><label for="case">Test case <select id="case" aria-label="Choose a recorded test case"></select></label><div class="legend"><span><i class="swatch base"></i>Incumbent</span><span><i class="swatch candidate"></i>{{SELECTED}}</span><span><i class="swatch target"></i>Target</span></div></div><canvas id="routes" role="img" aria-label="Animated optical pan and tilt routes for both search policies"></canvas><div class="controls"><button id="play" type="button">Play ×4</button><input id="scrub" type="range" min="0" max="120" value="0" step="0.1" aria-label="Simulation time in seconds"><output id="time">0.0 / 120 s</output></div><div class="trace-note"><span id="state"></span><span>Both routes share one clock. A path freezes when that search returns; no unrecorded motion is invented. Rectangles show the measured field of view, with the inner dashed box marking the simulator’s 90% detection footprint.</span></div></div>
<p class="subtle">The first twelve test cases are shown without choosing for appearance. The orange cross is a prescribed angular trajectory. An absent target is omitted; an occluded target is shown dashed. The same simulated frame always gives the same detector draw, regardless of how often it is queried.</p></section>
<section><h2>Selection happened before the test</h2><p>The development bank selected <b>{{SELECTED}}</b> using capped time plus a movement preference: two seconds per 1,000 motor steps. Eligible routes could have no more wrong finds and at most a two percentage point reduction in correct or 60-second finds. The selected name was saved before the independent-seed test and stress banks ran.</p>{{DEVELOPMENT}}
<p class="note">“Exact” means the best subset/order in the supplied small action set under binary, static visibility and a declared planning cost. It does not establish a globally optimal room route. The controller simulation, which includes motion and detector errors, is a separate experiment.</p></section>
<section><h2>How much does the simulator support?</h2><p>Paired intervals compare the same scenarios under both routes. Negative changes favour the candidate. They quantify Monte Carlo uncertainty within these fixed assumptions; they do not cover errors in the room model, target prior or detector.</p>{{INTERVALS}}<p class="subtle">Percentile paired bootstrap, 2,000 resamples. Test seed {{TEST_SEED}}; stress seed {{STRESS_SEED}}. Independent simulation seeds share the same all-day source prior and floor mask. They are not held-out household days or independently labelled real searches. Rare wrong-find counts remain uncertain; an observed zero does not establish a zero error rate.</p>
<h3>Different kinds of searches</h3>{{FAMILIES}}<p class="subtle">The “hard” family uses independent 20% per-frame recall. Persistent complete occlusion is a separate family. Partial occlusion and pose-dependent correlated detector failures have not been validated.</p></section>
<section><h2>The wrong confirmations stay visible</h2><p>Correctness comes from the box returned to the controller, at that frame’s exposure time. A target being somewhere in the current view cannot turn a false detector box into a correct confirmation.</p>{{WRONG_TABLE}}<p class="subtle">A false confirmation can occur even while the simulated target is in view. Scoring only the final camera view would hide that error.</p>{{HISTORICAL}}
<p class="note">Cancellation was checked separately at 7 seconds with an absent target and false positives disabled. {{CANCELLATION}}. These twelve checks per route cover this timing and setup, not every possible interruption.</p></section>
<section><h2>What remains open</h2><ul class="assumptions">{{LIMITATIONS}}</ul><p>The useful result is a measured offline lead: nearest-stop ordering is faster in the declared scenes. The next acceptance step needs the false-confirmation guardrail and independently labelled real evidence. This report makes no deployment recommendation.</p></section>
<section><h2>Inspect the evidence</h2><p class="subtle">{{HASH_TEXT}} The artifact embeds its animation data and makes no network requests.</p><div class="evidence">{{EVIDENCE_LINKS}}</div><p class="footer">Benchmark source SHA-256<br><code>{{BENCH_HASH}}</code><br>Private experiment artifact. Household data stays local.</p></section>
</main><script id="data" type="application/json">{{EMBEDDED}}</script><script>
const D=JSON.parse(document.getElementById('data').textContent),canvas=document.getElementById('routes'),ctx=canvas.getContext('2d'),choice=document.getElementById('case'),slider=document.getElementById('scrub'),play=document.getElementById('play');
let running=false,last=0;
for(let i=0;i<D.cases.length;i++){const opt=document.createElement('option');opt.value=i;opt.textContent=D.cases[i].id+' · '+D.cases[i].family.replaceAll('_',' ');choice.appendChild(opt)}
slider.max=D.budget;
function camera(trace,t){if(!trace.length)return[0,0];if(t<=trace[0][0])return[trace[0][3],trace[0][2]];for(let i=1;i<trace.length;i++){if(trace[i][0]>=t){const a=trace[i-1],b=trace[i],f=(t-a[0])/(b[0]-a[0]);return[a[3]+f*(b[3]-a[3]),a[2]+f*(b[2]-a[2])]}}const a=trace[trace.length-1];return[a[3],a[2]]}
function target(s,t){let f=t<s.move_start_s?0:s.move_duration_s===0?1:Math.max(0,Math.min(1,(t-s.move_start_s)/s.move_duration_s));return s.target_start.map((a,i)=>a+f*(s.target_end[i]-a))}
function state(r,t){const kind=r.found_right?'correct confirmation':r.wrong_find?'wrong confirmation':r.deadline?'budget reached':'search ended';return t>=r.elapsed_s?kind+' at '+r.elapsed_s.toFixed(1)+' s · frozen':'searching · returns at '+r.elapsed_s.toFixed(1)+' s'}
function draw(){const w=canvas.clientWidth,h=canvas.clientHeight,dpr=window.devicePixelRatio||1;if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);const t=Number(slider.value),c=D.cases[Number(choice.value)||0];document.getElementById('time').textContent=t.toFixed(1)+' / '+D.budget+' s';if(!c){ctx.fillText('No recorded traces available.',30,40);return}const m={left:58,right:28,top:28,bottom:46},pw=w-m.left-m.right,ph=h-m.top-m.bottom,X=x=>m.left+x/D.prior.limits[0]*pw,Y=y=>m.top+(1-y/D.prior.limits[1])*ph;ctx.lineWidth=1;ctx.font='11px system-ui';ctx.fillStyle='#64776b';ctx.textAlign='right';for(let y=0;y<=1000;y+=200){ctx.strokeStyle='#e4e9df';ctx.beginPath();ctx.moveTo(m.left,Y(y));ctx.lineTo(w-m.right,Y(y));ctx.stroke();ctx.fillText(y,m.left-10,Y(y)+4)}ctx.textAlign='center';for(let x=0;x<=3500;x+=500){ctx.strokeStyle='#e4e9df';ctx.beginPath();ctx.moveTo(X(x),m.top);ctx.lineTo(X(x),h-m.bottom);ctx.stroke();ctx.fillText(x,X(x),h-m.bottom+21)}ctx.fillText('Pan motor steps →',w/2,h-8);ctx.save();ctx.translate(15,h/2);ctx.rotate(-Math.PI/2);ctx.fillText('Tilt motor steps →',0,0);ctx.restore();ctx.save();ctx.beginPath();ctx.rect(m.left,m.top,pw,ph);ctx.clip();ctx.fillStyle='#d9e1d4';for(const p of D.prior.targets){ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),1.6,0,Math.PI*2);ctx.fill()}
function route(r,color){ctx.strokeStyle=color;ctx.lineWidth=2.2;ctx.setLineDash([]);ctx.beginPath();let started=false;for(let i=0;i<r.trace.length;i+=3){const p=r.trace[i];if(p[0]>t)break;if(!started){ctx.moveTo(X(p[3]),Y(p[2]));started=true}else ctx.lineTo(X(p[3]),Y(p[2]))}const p=camera(r.trace,t);if(started)ctx.lineTo(X(p[0]),Y(p[1]));ctx.stroke();const fw=D.prior.frame[0],fh=D.prior.frame[1];ctx.globalAlpha=.09;ctx.fillStyle=color;ctx.fillRect(X(p[0]-fw/2),Y(p[1]+fh/2),fw/D.prior.limits[0]*pw,fh/D.prior.limits[1]*ph);ctx.globalAlpha=.6;ctx.lineWidth=1;ctx.strokeRect(X(p[0]-fw/2),Y(p[1]+fh/2),fw/D.prior.limits[0]*pw,fh/D.prior.limits[1]*ph);ctx.setLineDash([4,4]);ctx.strokeRect(X(p[0]-.45*fw),Y(p[1]+.45*fh),.9*fw/D.prior.limits[0]*pw,.9*fh/D.prior.limits[1]*ph);ctx.setLineDash([]);ctx.globalAlpha=1;ctx.fillStyle=color;ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),4,0,Math.PI*2);ctx.fill()}
route(c.base,'#526f9a');route(c.candidate,'#217864');if(!c.scenario.absent){const p=target(c.scenario,t);ctx.strokeStyle='#ad6224';ctx.lineWidth=2;ctx.setLineDash(c.scenario.occluded?[3,3]:[]);ctx.beginPath();ctx.moveTo(X(p[0])-7,Y(p[1]));ctx.lineTo(X(p[0])+7,Y(p[1]));ctx.moveTo(X(p[0]),Y(p[1])-7);ctx.lineTo(X(p[0]),Y(p[1])+7);ctx.stroke();ctx.setLineDash([])}ctx.restore();document.getElementById('state').textContent='Incumbent: '+state(c.base,t)+' | '+D.candidate+': '+state(c.candidate,t)+(c.scenario.absent?' | target absent':c.scenario.occluded?' | target persistently occluded':'')}
function stop(){running=false;play.textContent='Play ×4'}
function animate(now){if(!running)return;if(last){slider.value=Math.min(D.budget,Number(slider.value)+(now-last)/1000*4)}last=now;draw();if(Number(slider.value)>=D.budget){stop();return}requestAnimationFrame(animate)}
play.addEventListener('click',()=>{if(running){stop();return}if(Number(slider.value)>=D.budget)slider.value=0;running=true;last=0;play.textContent='Pause';requestAnimationFrame(animate)});slider.addEventListener('input',()=>{stop();draw()});choice.addEventListener('change',()=>{stop();slider.value=0;draw()});window.addEventListener('resize',draw);document.addEventListener('visibilitychange',()=>{if(document.hidden)stop()});draw();
</script></body></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    print(render(args.folder, args.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
