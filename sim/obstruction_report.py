"""Render a private obstruction-learning report, with embedded PNGs and SVG.

    python3 -m sim.obstruction_report eval/obstruction_learning/experiment-v1

Reads a completed experiment and writes only the requested HTML artifact. The
artifact needs no JavaScript, file fetches, network access or external assets.
"""
import argparse
import base64
import hashlib
from html import escape
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLORS = {'unknown': '#e0e3da', 'suspected': '#e9c77d',
          'temporarily_excluded': '#43816a', 'reinspection_due': '#789ac0'}


def _read(folder, name):
    return json.loads((folder / name).read_text())


def _num(value, digits=0):
    return '—' if value is None else f'{value:,.{digits}f}'


def _pct(value):
    return '—' if value is None else f'{100 * value:.1f}%'


def _table(headers, rows):
    head = ''.join(f'<th scope="col">{escape(str(x))}</th>' for x in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(str(x))}</td>' for x in row) + '</tr>' for row in rows)
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _png_data(folder, filename):
    path = (folder / filename).resolve()
    if not path.is_relative_to(folder) or path.suffix.lower() != '.png':
        raise ValueError('example PNG must be inside the experiment directory')
    content = path.read_bytes()
    if not content.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('example is not a PNG')
    return 'data:image/png;base64,' + base64.b64encode(content).decode('ascii')


def _timeline(episode, end_s, tick_s):
    samples = episode.get('samples') or []
    if not samples:
        return '<p class="muted">No detailed observations saved for this episode.</p>'
    poses = sorted(set(tuple(s['pose']) for s in samples))
    end = max(float(end_s), max(float(s['at_s']) for s in samples))
    width, left, right, top, row_h = 940, 110, 20, 30, 29
    height = top + len(poses) * row_h + 40
    scale = (width - left - right) / max(end, 1)
    x = lambda t: left + float(t) * scale
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Observation states by camera pose" xmlns="http://www.w3.org/2000/svg">',
             '<style>text{font:12px system-ui,sans-serif;fill:#53675b}</style>']
    for t in range(0, int(end) + 1, 60):
        parts.append(f'<path d="M{x(t):.2f} {top-8}V{height-28}" stroke="#dce2d6"/><text x="{x(t):.2f}" y="{height-8}" text-anchor="middle">{t}s</text>')
    for j, pose in enumerate(poses):
        y = top + j * row_h
        parts.append(f'<text x="{left-12}" y="{y+14}" text-anchor="end">{pose[0]:g}, {pose[1]:g}</text>')
        rows = sorted((s for s in samples if tuple(s['pose']) == pose), key=lambda s: s['at_s'])
        for k, s in enumerate(rows):
            finish = rows[k+1]['at_s'] if k+1 < len(rows) else min(end, s['at_s'] + tick_s)
            span = max(.8, (finish - s['at_s']) * scale)
            color = COLORS.get(s['state'], COLORS['unknown'])
            description = (f'{s["at_s"]:g}s · {s["state"]} · '
                           f'injected coverage {100*s.get("coverage", 0):.1f}% · '
                           f'{"deferred" if s.get("deferred") else "skipped" if s.get("skip") else "inspection opportunity"}')
            parts.append(f'<rect x="{x(s["at_s"]):.2f}" y="{y}" width="{span:.2f}" height="20" fill="{color}"><title>{escape(description)}</title></rect>')
            if s.get('deferred'):
                parts.append(f'<path d="M{x(s["at_s"]):.2f} {y+20}l{min(span,9):.2f} -20" stroke="#a65c36" stroke-width="2"/>')
    family = episode['family']
    marker = 245 if family == 'long_transient' else 185
    if family in ('removed', 'changed_setup', 'unannounced_move', 'long_transient'):
        name = {'removed': 'cover removed', 'changed_setup': 'setup announced',
                'unannounced_move': 'object moved; no setup signal', 'long_transient': 'temporary object leaves'}[family]
        parts.append(f'<path d="M{x(marker):.2f} 20V{height-28}" stroke="#a65c36" stroke-dasharray="4 4"/><text x="{min(x(marker)+5,width-210):.2f}" y="13">{escape(name)}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def render(folder, output=None):
    folder = Path(folder).resolve()
    output = Path(output).resolve() if output else folder.parent / 'index.html'
    manifest = _read(folder, 'manifest.json')
    summary, selection = _read(folder, 'summary.json'), _read(folder, 'selection.json')
    pixels, examples = _read(folder, 'pixel-results.json'), _read(folder, 'examples.json')
    metrics = summary['pixel_test']
    banks = summary['banks']
    synthetic = [(name, bank) for name, bank in banks.items() if name != 'photo-sensitivity']
    primary = next(((name, bank) for name, bank in synthetic if name.startswith('lifecycle')), synthetic[0])
    baseline = primary[1]['policies']['always_inspect']
    learned = primary[1]['policies']['visual_memory']
    failed = [name for name, passed in summary['simulation_checks'].items() if not passed]
    gate_title = 'The finite simulation gate passed.' if summary['simulation_gate'] else 'The finite simulation gate did not pass.'
    gate_text = ('The declared checks passed on these synthetic scenes. Real-camera validation is still open.'
                 if not failed else 'Failed checks: ' + '; '.join(name.replace('_', ' ') for name in failed) + '.')
    check_table = _table(['Declared check', 'Recorded result'],
                         [(name.replace('_', ' '), 'PASS' if passed else 'FAIL')
                          for name, passed in summary['simulation_checks'].items()])
    confusion = _table(['Held-out images', 'True positives', 'False positives', 'True negatives', 'False negatives', 'Recall', 'Precision'],
                       [[_num(metrics['n']), _num(metrics['tp']), _num(metrics['fp']), _num(metrics['tn']),
                         _num(metrics['fn']), _pct(metrics['recall']), _pct(metrics['precision'])]])
    model_rows = []
    for candidate in selection.get('model_candidates', [selection]):
        d = candidate['development']
        model_rows.append([candidate.get('model', 'linear'), _num(d['n']), _num(d['tp']),
                           _num(d['fp']), _pct(d['recall']), _num(candidate['threshold'], 3),
                           'Eligible' if candidate.get('enabled', True) else 'Rejected'])
    model_table = _table(['Development model', 'Images', 'True positives', 'False positives', 'Recall', 'Threshold', 'Calibration'], model_rows)
    reference = manifest.get('reference_protocol')
    reference_scope = ''
    if reference:
        selected_reference = bool(reference.get('selected', 'reference' in selection.get('model', '')))
        disposition = ('The selected model requires a registered reference image.' if selected_reference else
                       'A registered-reference model was compared on development data; it was not selected.')
        reference_scope = ('<aside class="notice"><strong>Reference images are an explicit prerequisite.</strong>'
                           + escape(disposition) +
                           ' The simulation acquires a previous RGB view before the injected object appears. '
                           'The original background is not certified clear. This measures changes relative to '
                           'that supplied view; an obstruction already present in the reference can remain invisible to the comparison. '
                           'Reference acquisition and registration are not implemented at the production camera boundary.</aside>'
                           '<p class="muted">Announced remounts invalidate stored references; the experiment does not manufacture '
                           'a new clean reference afterward. Missing or invalid references fail open. Registration noise and '
                           'photographic backgrounds are sensitivity conditions, not proof of real alignment accuracy.</p>')
    scheduled_rows, recovery_rows, timeline_sections = [], [], []
    for name, bank in banks.items():
        photo = name == 'photo-sensitivity'
        display_name = name + (' · injected-mask diagnostic' if photo else '')
        for policy, counts in bank['policies'].items():
            scheduled_rows.append([display_name, policy.replace('_', ' '), _num(counts['opportunities']),
                                   _num(counts['inspections']), _num(counts['skips']), _num(counts['useful_skips']),
                                   _num(counts['false_skips']), _num(counts.get('deferred_probes', 0)),
                                   _num(counts.get('missed_pet_opportunities', 0))])
        for family, stats in bank.get('by_family', {}).items():
            denominator = stats.get('learned_cases_requiring_recovery', 0)
            if stats.get('recovery_bound_s') is not None:
                recovery_rows.append([display_name, family.replace('_', ' '), _num(stats.get('episodes')),
                                      _num(stats.get('learned_before_change', 0)),
                                      f'{stats.get("learned_cases_recovered_within_bound", 0)} / {denominator}',
                                      _num(stats.get('recovery_bound_s'), 1) + ' s',
                                      _num(stats.get('max_first_inspection_after_change_s'), 1) + ' s'])
        source = folder / (name + '.json')
        if source.exists():
            episodes = json.loads(source.read_text()).get('episodes', [])
            details = []
            for i, episode in enumerate(e for e in episodes if e.get('samples')):
                if i >= 10:
                    break
                v = episode['counts']['visual_memory']
                title = f'{episode["family"].replace("_", " ")} · {episode["id"]}'
                details.append(f'<details{" open" if i == 0 and name == primary[0] else ""}><summary>{escape(title)}</summary>'
                               f'<p class="muted">{v["skips"]} temporary exclusions; {v["false_skips"]} skips outside the injected blocking mask; '
                               f'{v.get("deferred_probes",0)} deferred mandatory probes.</p>'
                               + _timeline(episode, manifest['lifecycle']['end_s'], manifest['lifecycle']['tick_s']) + '</details>')
            timeline_sections.append(f'<h3>{escape(display_name)}</h3>' + ''.join(details))
    schedule = _table(['Bank', 'Policy', 'View opportunities', 'Inspected', 'Skipped',
                       'Injected-blocked skips', 'Outside-mask skips¹', 'Deferred probes', 'Scripted pet misses²'], scheduled_rows)
    recovery = _table(['Bank', 'Changed family', 'Episodes', 'Learned before change', 'Recovered / requiring recovery',
                      'Declared bound', 'Latest first inspection'], recovery_rows)
    gallery = []
    for row in examples:
        uri = _png_data(folder, row['path'])
        caption = (f'{row.get("split", "test")} group {row.get("group", "?")} · variant {row.get("variant", "?")} · '
                   f'injected coverage {100*row.get("injected_coverage", 0):.1f}% · score {row.get("score", 0):.3f}')
        picture = f'<img src="{uri}" alt="{escape(caption, quote=True)}" loading="lazy">'
        if row.get('reference_path'):
            ref_uri = _png_data(folder, row['reference_path'])
            picture = ('<div class="pair"><div><span>Current RGB</span>' + picture + '</div><div>'
                       f'<span>Supplied reference</span><img src="{ref_uri}" alt="Supplied earlier RGB reference" loading="lazy"></div></div>')
        gallery.append(f'<figure>{picture}<figcaption>{escape(caption)}</figcaption></figure>')
    links = []
    for name, label in [('manifest.json', 'Inputs and scope'), ('selection.json', 'Frozen development selection'),
                        ('pixel-results.json', 'Every labelled pixel example'), ('model.json', 'Offline fitted model'),
                        ('summary.json', 'Results and acceptance'), ('examples.json', 'Image metadata')]:
        if (folder / name).exists():
            href = escape(os.path.relpath(folder / name, output.parent), quote=True)
            links.append(f'<a href="{href}">{escape(label)}</a>')
    review = folder.parent / 'independent-review.json'
    if review.exists():
        href = escape(os.path.relpath(review, output.parent), quote=True)
        links.append(f'<a href="{href}">Independent review</a>')
    current_match = all((ROOT / name).exists() and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                        for name, digest in manifest['input_sha256'].items())
    hashes = ('Input hashes matched at experiment completion and still match at render time.'
              if summary['inputs_unchanged'] and current_match else
              'Input hashes do not all match the current source. Recorded results are preserved; inspect the manifest before reuse.')
    training_n = len(pixels.get('training', []))
    development_n = len(pixels.get('development', []))
    replacements = {'RUN': escape(folder.name), 'GATE_TITLE': gate_title, 'GATE_TEXT': escape(gate_text),
                    'MODEL': escape(selection.get('model', 'linear')), 'THRESHOLD': _num(summary['selected_threshold'], 3),
                    'TRAIN_N': _num(training_n), 'DEV_N': _num(development_n), 'TEST_N': _num(metrics['n']),
                    'TP': _num(metrics['tp']), 'FP': _num(metrics['fp']), 'RECALL': _pct(metrics['recall']),
                    'SKIPS': _num(learned['skips']), 'PRIMARY_OPPS': _num(baseline['opportunities']),
                    'CHECKS': check_table, 'CONFUSION': confusion, 'MODEL_TABLE': model_table,
                    'REFERENCE_SCOPE': reference_scope,
                    'SCHEDULE': schedule, 'RECOVERY': recovery, 'TIMELINES': ''.join(timeline_sections),
                    'GALLERY': ''.join(gallery), 'LINKS': ''.join(links), 'HASHES': escape(hashes),
                    'LIMITS': ''.join(f'<li>{escape(item)}</li>' for item in manifest.get('limits', []))}
    html = TEMPLATE
    for key, value in replacements.items():
        html = html.replace('{{' + key + '}}', value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return output


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>OpenPetCam · Learning obstructed views</title><style>
:root{--paper:#f5f3ec;--ink:#253a30;--muted:#607267;--line:#d2dacf;--green:#356e59;--amber:#a75c32}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1180px;margin:auto;padding:58px 36px 80px}h1,h2,h3{font-family:Georgia,"Times New Roman",serif;font-weight:400;line-height:1.16;letter-spacing:-.025em}h1{font-size:clamp(40px,6vw,70px);max-width:870px;margin:24px 0}h2{font-size:33px;margin:0 0 20px}h3{font-size:23px;margin-top:30px}p{max-width:79ch;margin:12px 0}a{color:var(--green);text-underline-offset:3px}.kicker{text-transform:uppercase;letter-spacing:.16em;color:var(--muted);font-size:12px;font-weight:700}.lede{font-size:21px;max-width:780px}.muted{font-size:14px;color:var(--muted)}.scope{display:inline-block;border:1px solid var(--line);padding:4px 10px;font-size:12px;letter-spacing:.07em;text-transform:uppercase}.notice{border-left:4px solid var(--amber);background:#f2e6d6;padding:18px 24px;margin:30px 0}.notice strong{font-size:21px;display:block;margin-bottom:6px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin:32px 0}.stat{border-top:1px solid var(--line);padding-top:15px}.value{font:40px/1.2 Georgia,serif;letter-spacing:-.04em}.stat span{display:block;font-size:14px;color:var(--muted);margin-top:8px}section{border-top:1px solid var(--line);padding-top:42px;margin-top:42px}.table-scroll{overflow-x:auto;margin:24px 0}table{border-collapse:collapse;min-width:660px;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th{text-align:left;color:var(--muted);font-size:12px;font-weight:600;padding:11px 13px;border-bottom:2px solid var(--line)}td{padding:12px 13px;border-bottom:1px solid var(--line)}td:first-child,th:first-child{padding-left:0}.note{padding-left:18px;border-left:2px solid var(--line);color:var(--muted);font-size:14px}.alias{display:grid;grid-template-columns:1fr 1fr;gap:22px;max-width:680px;margin:22px 0}.alias figure{margin:0}.alias svg{width:100%;height:90px;display:block}.alias figcaption{font-size:14px;color:var(--muted);margin-top:8px}.legend{display:flex;flex-wrap:wrap;gap:12px 20px;font-size:13px;margin:20px 0}.chip{width:16px;height:12px;display:inline-block;margin-right:7px;vertical-align:middle}details{border-top:1px solid var(--line);padding:12px 0}details:last-child{border-bottom:1px solid var(--line)}summary{cursor:pointer;font-size:15px}details>svg{width:100%;height:auto;display:block;margin:12px 0}.gallery{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px;margin-top:28px}.gallery figure{margin:0}.gallery img{display:block;width:100%;height:auto;background:#e0e3da}.gallery figcaption{font-size:12px;line-height:1.5;color:var(--muted);margin-top:8px}.limits{padding-left:20px;columns:2;column-gap:40px}.limits li{font-size:14px;break-inside:avoid;margin-bottom:12px}.evidence{display:flex;flex-wrap:wrap;gap:10px 24px;margin:24px 0;font-size:14px}.footer{font-size:13px;color:var(--muted);margin-top:30px}code{font:12px ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}@media(max-width:720px){main{padding:34px 20px 60px}.lede{font-size:19px}.stats{gap:14px}.value{font-size:31px}.gallery{grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.limits{columns:1}.alias{gap:14px}h2{font-size:29px}section{margin-top:30px;padding-top:32px}}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:7px}.pair span{display:block;font-size:11px;color:var(--muted);margin-bottom:5px}
</style></head><body><main><header><div class="kicker">OpenPetCam / Visual obstruction experiment</div><h1>Learning which views are blocked.</h1><p class="lede">The camera needs visual evidence before it stops checking a view. This experiment fits that evidence from rendered RGB, then tests a memory that keeps every exclusion temporary.</p><p class="muted">{{RUN}} · {{TRAIN_N}} training images · {{DEV_N}} development images · {{TEST_N}} held-out synthetic images</p><span class="scope">Private offline experiment · no live suppression installed</span></header>
<aside class="notice"><strong>{{GATE_TITLE}}</strong>{{GATE_TEXT}} Neither outcome authorizes deployment or establishes real-camera accuracy.</aside><div class="stats"><div class="stat"><div class="value">{{RECALL}}</div><span>held-out recall for injected ≥85% coverage</span></div><div class="stat"><div class="value">{{FP}}</div><span>held-out pixel false positives</span></div><div class="stat"><div class="value">{{SKIPS}}</div><span>temporary exclusions among {{PRIMARY_OPPS}} scheduled views in the main lifecycle bank</span></div></div>
<section><h2>One image cannot prove depth or permanence</h2><p>A smooth object close to the lens and a matching blank wall farther away can produce exactly the same pixels. They therefore produce the same classifier score, even when the hidden physical labels disagree. More confidence cannot recover information absent from the image.</p><div class="alias"><figure><svg viewBox="0 0 220 90" role="img" aria-label="Uniform grey camera image"><rect width="220" height="90" fill="#808080"/></svg><figcaption>Far blank wall. RGB = (128, 128, 128).</figcaption></figure><figure><svg viewBox="0 0 220 90" role="img" aria-label="The identical uniform grey camera image"><rect width="220" height="90" fill="#808080"/></svg><figcaption>Near cover with the same appearance. Identical RGB.</figcaption></figure></div><p class="note">This alias is tested with identical rendered arrays and opposite hidden depth labels. The model learns appearance within the simulation. Repeated visits establish repeated evidence, not a permanent physical fact.</p></section>
<section><h2>What was learned from pixels</h2><p>The selected model is <b>{{MODEL}}</b>, with its threshold frozen at <b>{{THRESHOLD}}</b> on development data. Training and development choose the model and threshold before the held-out images are generated. Background, object and sensor identities are split by whole scene group; test views also occupy a separate angular sector.</p>{{REFERENCE_SCOPE}}{{MODEL_TABLE}}{{CONFUSION}}<p class="muted">Prediction receives RGB-derived colour, spatial and texture features, and paired RGB differences when a reference model is used. Pose IDs, setup versions, timestamps, random seeds, masks, coverage labels and empty pet detections are excluded from feature values. Supplied reference validity is an explicit input contract. The positive pixel label is total <em>injected</em> near-object coverage of at least 85%. Scores have not been calibrated on real camera recordings. Images within a scene group are related; the image count is not a count of independent physical worlds.</p>{{CHECKS}}</section>
<section><h2>Temporary memory, with a route back in</h2><p>A single positive frame cannot exclude a view. The memory requires fresh high-quality observations separated in time and spanning a minimum duration. Pet sightings or clear and ambiguous visual evidence reopen it. Setup changes invalidate the memory; probe deadlines and expiry make it eligible for inspection even before a successful probe arrives.</p>{{SCHEDULE}}<p class="muted">Counts are scheduled view opportunities, not measured motor movement, wall-clock search latency or real pet recall. The probe budget limits mandatory reinspections; ordinary unknown-view observations are separate. Deferred probes reduce actual inspections, but are not counted as exclusions learned from visual evidence.</p><p class="note">¹ “Outside-mask skips” means coverage below the injected 85% threshold in synthetic scenes. In the photo-composite bank it means only <b>skips outside the injected mask—background unlabelled</b>; it is not a real-image accuracy measurement. Persistent and transient injected objects are counted separately in the source JSON. A temporarily present object can justify a temporary exclusion without proving permanence.<br>² Pet sightings are scripted symbolic positive-evidence controls. Misses include skipped and deferred uninspected opportunities; these are not RGB-rendered pet detections.</p>
<h3>Recovery after a change</h3>{{RECOVERY}}<p class="muted">The recovery denominator includes only cases actually learned before the change. A zero denominator is no recovery evidence. Outside-mask skips can include a previously correct exclusion that becomes stale after an unannounced removal, before the next probe. Meeting a recovery bound does not erase a failed zero-false-exclusion check. Bounds apply to the declared scheduled poses, budget and valid observations; they do not establish live-camera latency.</p></section>
<section><h2>Read the state over time</h2><p>The first recorded episodes are shown without choosing for appearance. Each row is one exact pan/tilt pose. Colours show the memory state before the scheduled observation; orange slashes mark deferred mandatory probes. Open an episode to inspect its timeline.</p><div class="legend"><span><i class="chip" style="background:#e0e3da"></i>Unknown / open</span><span><i class="chip" style="background:#e9c77d"></i>Suspected / open</span><span><i class="chip" style="background:#43816a"></i>Temporarily excluded</span><span><i class="chip" style="background:#789ac0"></i>Reinspection due / open</span></div>{{TIMELINES}}</section>
<section><h2>The held-out RGB examples</h2><p>These saved examples expose what the model actually sees: partial covers, textured and empty backgrounds, flat surfaces and blur. Captions contain evaluator labels and predictions; that metadata never enters image features.</p><div class="gallery">{{GALLERY}}</div></section>
<section><h2>Limits that remain</h2><ul class="limits">{{LIMITS}}</ul><p>The result is an offline measurement of appearance learning and temporary memory. Real suppression needs independently labelled recordings and the same recovery guarantees at the camera boundary.</p></section>
<section><h2>Inspect the experiment</h2><p class="muted">{{HASHES}} Images and timelines are embedded. This artifact makes no network requests and requires no JavaScript.</p><div class="evidence">{{LINKS}}</div><p class="footer">Private household experiment. Production models and camera behaviour were not changed.</p></section></main></body></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    print(render(args.folder, args.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
